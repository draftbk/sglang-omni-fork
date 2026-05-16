# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Voxtral-TTS V1 pipeline."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import torch

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.proto import StagePayload

logger = logging.getLogger(__name__)


_VOXTRAL_MISTRAL_COMMON_HINT = (
    "Voxtral TTS requires the `mistral-common` package (Tekken speech "
    "tokenizer). Install via `uv pip install 'mistral-common[audio]>=1.8.0'`."
)


def _import_mistral_common():
    try:
        from mistral_common.protocol.speech.request import SpeechRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise RuntimeError(_VOXTRAL_MISTRAL_COMMON_HINT) from exc
    return SpeechRequest, MistralTokenizer


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_state(payload: StagePayload) -> VoxtralTTSState:
    return VoxtralTTSState.from_dict(payload.data)


def _store_state(payload: StagePayload, state: VoxtralTTSState) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = 8,
):
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    checkpoint_dir = _resolve_checkpoint(model_path)
    SpeechRequest, MistralTokenizer = _import_mistral_common()
    tekken_path = os.path.join(checkpoint_dir, "tekken.json")
    tokenizer = MistralTokenizer.from_file(tekken_path)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}

        if isinstance(inputs, str):
            text = inputs
        elif isinstance(inputs, dict):
            text = inputs.get("text", "")
        else:
            text = str(inputs) if inputs else ""

        tts_params = metadata.get("tts_params", {})
        voice = tts_params.get("voice") or params.get("voice") or "cheerful_female"

        encoded = tokenizer.encode_speech_request(
            SpeechRequest(input=text, voice=voice)
        )

        max_new_tokens = params.get("max_new_tokens", 4096)
        if isinstance(max_new_tokens, dict):
            max_new_tokens = max_new_tokens.get("max_new_tokens", 4096)

        state = VoxtralTTSState(
            input_ids=list(encoded.tokens),
            voice=voice,
            max_new_tokens=max_new_tokens,
        )
        return _store_state(payload, state)

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_generation_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 4096,
    context_length: int = 8192,
    server_args_overrides: dict | None = None,
):
    from sglang_omni.models.voxtral_tts_v1.bootstrap import create_voxtral_scheduler

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0
    return create_voxtral_scheduler(
        model_path,
        gpu_id=gpu_id,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
        server_args_overrides=server_args_overrides,
    )


def _load_audio_tokenizer(checkpoint_dir: str, device: str):
    import glob

    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralTTSAudioTokenizer
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

    config = VoxtralModelConfig.from_model_path(checkpoint_dir)
    tokenizer = VoxtralTTSAudioTokenizer(
        audio_tokenizer_args=config.audio_tokenizer_args,
        audio_config={
            "audio_model_args": config.audio_model_args.acoustic_transformer_args,
        },
    )

    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not safetensors_files:
        raise RuntimeError(f"No .safetensors files in {checkpoint_dir}")

    t0 = time.perf_counter()
    remapping_rules = [
        (r"^audio_tokenizer\.(.*)$", r"\1"),
        (
            r"^mm_audio_embeddings\.audio_codebook_embeddings\.embeddings\.(weight|bias)",
            r"audio_token_embedding.embeddings.\1",
        ),
    ]
    for name, tensor in safetensors_weights_iterator(safetensors_files):
        is_audio_tokenizer = name.startswith(
            "mm_audio_embeddings.audio_codebook_embeddings"
        ) or name.startswith("audio_tokenizer.")
        if not is_audio_tokenizer:
            continue
        remapped = name
        for pattern, repl in remapping_rules:
            if re.fullmatch(pattern, remapped):
                remapped = re.sub(pattern, repl, remapped)
        tokenizer.load_weight((remapped, tensor))

    tokenizer = tokenizer.to(dtype=torch.bfloat16, device=device).eval()
    logger.info("Audio tokenizer loaded in %.2fs", time.perf_counter() - t0)
    return tokenizer


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info("Loading Voxtral audio tokenizer for vocoding...")
    audio_tokenizer = _load_audio_tokenizer(checkpoint_dir, device)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = _load_state(payload)
        audio_codes = state.audio_codes

        if audio_codes is None or (
            isinstance(audio_codes, torch.Tensor) and audio_codes.numel() == 0
        ):
            state.audio_samples = []
            payload = _store_state(payload, state)
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = 24000
            payload.data["modality"] = "audio"
            return payload

        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = torch.tensor(audio_codes)

        # Warmup context for the causal decoder; trimmed after decode.
        n_warmup = 2
        warmup_samples = 0
        if audio_codes.shape[0] > 0:
            first_frame = audio_codes[0:1]
            warmup = first_frame.repeat(n_warmup, 1)
            codes_with_warmup = torch.cat([warmup, audio_codes], dim=0)
            warmup_samples = n_warmup * audio_tokenizer.downsample_factor
        else:
            codes_with_warmup = audio_codes

        results = audio_tokenizer.decode_helper_batch_async([codes_with_warmup])
        audio_np = results[0]
        if warmup_samples > 0 and len(audio_np) > warmup_samples:
            audio_np = audio_np[warmup_samples:]

        fade_in_ms = 10
        fade_samples = min(
            int(fade_in_ms * audio_tokenizer.sampling_rate / 1000),
            len(audio_np),
        )
        if fade_samples > 0:
            fade_in = torch.linspace(
                0, 1, fade_samples, device=audio_np.device, dtype=audio_np.dtype,
            )
            audio_np[:fade_samples] = audio_np[:fade_samples] * fade_in

        state.audio_samples = audio_np
        state.sample_rate = audio_tokenizer.sampling_rate
        payload = _store_state(payload, state)
        payload.data["audio_data"] = audio_np.tolist()
        payload.data["sample_rate"] = audio_tokenizer.sampling_rate
        payload.data["modality"] = "audio"

        if state.prompt_tokens or state.completion_tokens:
            payload.data["usage"] = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }
        return payload

    return SimpleScheduler(_vocode)
