# SPDX-License-Identifier: Apache-2.0
"""Voxtral-TTS SGLang scheduler bootstrap."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


def _load_voice_embeddings(checkpoint_dir: str, device: str) -> dict[str, torch.Tensor]:
    voice_dir = os.path.join(checkpoint_dir, "voice_embedding")
    if not os.path.isdir(voice_dir):
        raise FileNotFoundError(
            f"Voxtral checkpoint at {checkpoint_dir!r} has no voice_embedding/ dir"
        )
    voice_embeddings: dict[str, torch.Tensor] = {}
    for fname in sorted(os.listdir(voice_dir)):
        if not fname.endswith(".pt"):
            continue
        name = fname.removesuffix(".pt")
        emb = torch.load(
            os.path.join(voice_dir, fname),
            map_location=device,
            weights_only=True,
        )
        voice_embeddings[name] = emb.to(dtype=torch.bfloat16)
    if not voice_embeddings:
        raise FileNotFoundError(
            f"No .pt voice embeddings under {voice_dir!r}"
        )
    logger.info(
        "Loaded %d voice embeddings: %s",
        len(voice_embeddings),
        list(voice_embeddings.keys()),
    )
    return voice_embeddings


def _load_acoustic_and_embedding(
    checkpoint_dir: str,
    voxtral_config,
    device: str,
):
    import glob

    from safetensors import safe_open

    from sglang_omni.models.voxtral_tts.acoustic_transformer import (
        FlowMatchingAudioTransformer,
    )
    from sglang_omni.models.voxtral_tts_v1.audio_modules import MultiVocabEmbeddings

    full_audio_args = {
        "semantic_codebook_size": voxtral_config.audio_model_args.semantic_codebook_size,
        "acoustic_codebook_size": voxtral_config.audio_model_args.acoustic_codebook_size,
        "n_acoustic_codebook": voxtral_config.audio_model_args.n_acoustic_codebook,
        "acoustic_transformer_args": voxtral_config.audio_model_args.acoustic_transformer_args,
        "audio_encoding_args": voxtral_config.audio_model_args.audio_encoding_args,
    }

    acoustic_transformer = FlowMatchingAudioTransformer(full_audio_args)
    audio_token_embedding = MultiVocabEmbeddings(
        audio_model_args=full_audio_args,
        embedding_dim=voxtral_config.text_config.dim,
    )

    shard_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not shard_paths:
        raise FileNotFoundError(
            f"No .safetensors shards in {checkpoint_dir} for acoustic load"
        )

    n_loaded_acoustic = 0
    embedding_loaded = False
    audio_embedding_key = (
        "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight"
    )
    acoustic_prefix = "acoustic_transformer."

    for path in shard_paths:
        with safe_open(path, framework="pt", device="cpu") as fp:
            for ckpt_key in fp.keys():
                if ckpt_key == audio_embedding_key:
                    tensor = fp.get_tensor(ckpt_key)
                    audio_token_embedding.embeddings.weight.data.copy_(tensor)
                    embedding_loaded = True
                    continue
                if ckpt_key.startswith(acoustic_prefix):
                    tensor = fp.get_tensor(ckpt_key)
                    param_name = ckpt_key[len(acoustic_prefix) :]
                    acoustic_transformer.load_weight((param_name, tensor))
                    n_loaded_acoustic += 1

    logger.info(
        "Voxtral acoustic_transformer: %d weights | audio_token_embedding: %s",
        n_loaded_acoustic,
        embedding_loaded,
    )

    acoustic_transformer = acoustic_transformer.to(
        dtype=torch.bfloat16, device=device
    ).eval()
    audio_token_embedding = audio_token_embedding.to(
        dtype=torch.bfloat16, device=device
    ).eval()
    return acoustic_transformer, audio_token_embedding


def materialize_voxtral_config_json(checkpoint_dir: str) -> str:
    """Synthesize a HF-style ``config.json`` from Mistral ``params.json``.

    SGLang's AutoConfig path requires config.json; Voxtral ships only
    params.json. If the checkpoint dir is read-only (HF cache snapshot),
    fall back to a sibling working dir with symlinks to the shards.
    """
    if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        return checkpoint_dir

    params_path = os.path.join(checkpoint_dir, "params.json")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"{checkpoint_dir} has neither config.json nor params.json"
        )

    with open(params_path) as f:
        params = json.load(f)

    hf_config = {
        "model_type": "llama",
        "architectures": ["VoxtralSGLangTextModel"],
        "hidden_size": int(params["dim"]),
        "num_attention_heads": int(params["n_heads"]),
        "num_key_value_heads": int(params["n_kv_heads"]),
        "num_hidden_layers": int(params["n_layers"]),
        "intermediate_size": int(params["hidden_dim"]),
        "head_dim": int(params["head_dim"]),
        "vocab_size": int(params["vocab_size"]),
        "max_position_embeddings": 32768,
        "rope_theta": float(params["rope_theta"]),
        "rms_norm_eps": float(params["norm_eps"]),
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": bool(params.get("tied_embeddings", True)),
        "rope_scaling": None,
    }

    config_path = os.path.join(checkpoint_dir, "config.json")
    try:
        with open(config_path, "w") as f:
            json.dump(hf_config, f, indent=2)
        logger.info(
            "Wrote synthesized config.json into Voxtral checkpoint at %s",
            checkpoint_dir,
        )
        return checkpoint_dir
    except OSError as exc:
        logger.info(
            "Cannot write to %s (%s); creating a sibling working dir with "
            "symlinked checkpoint shards.",
            checkpoint_dir,
            exc,
        )

    work_dir = tempfile.mkdtemp(prefix="voxtral_sglang_")
    for entry in os.listdir(checkpoint_dir):
        src = os.path.join(checkpoint_dir, entry)
        dst = os.path.join(work_dir, entry)
        os.symlink(src, dst)
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        json.dump(hf_config, f, indent=2)
    logger.info("Voxtral working dir at %s", work_dir)
    return work_dir


def create_voxtral_scheduler(
    model_path: str,
    *,
    gpu_id: int = 0,
    max_new_tokens: int = 4096,
    context_length: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
):
    """Build an ``OmniScheduler`` for Voxtral-TTS generation."""
    from sglang_omni.models.voxtral_tts.acoustic_transformer import (
        AudioSpecialTokens,
    )
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig
    from sglang_omni.models.voxtral_tts_v1.model_runner import VoxtralModelRunner
    from sglang_omni.models.voxtral_tts_v1.request_builders import (
        VoxtralAdapterContext,
        make_voxtral_scheduler_adapters,
    )
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    working_dir = materialize_voxtral_config_json(checkpoint_dir)
    voxtral_config = VoxtralModelConfig.from_model_path(checkpoint_dir)
    audio_token_id = int(voxtral_config.audio_model_args.audio_token_id)
    end_audio_token_id = int(AudioSpecialTokens.id(AudioSpecialTokens.end_audio))

    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "dtype": "bfloat16",
        # Per-step feedback writes into _feedback_buffer happen synchronously
        # in prepare_decode; SGLang's overlap_schedule would prefetch the
        # next forward concurrently and race against those writes.
        "disable_overlap_schedule": True,
        # Radix prefix cache + voice-injected ``Req.input_embeds`` is a
        # shape-mismatch crash: a 2nd matching prompt hits the cache,
        # SGLang shrinks the extend slice, but Req.input_embeds still
        # spans the full prompt → set_kv_buffer fails. Qwen3 talker
        # disables radix for the same reason.
        "disable_radix_cache": True,
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        working_dir,
        context_length=context_length,
        **overrides,
    )
    # Default to fa3 only when the user hasn't picked a backend (mirrors
    # fishaudio_s2_pro).
    if getattr(server_args, "attention_backend", None) is None:
        server_args.attention_backend = "fa3"

    # CUDA graph must be off during worker construction; re-enabled and
    # captured after setup_feedback_buffer() so the captured decode
    # branch reads a stable tensor identity.
    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="VoxtralSGLangTextModel",
    )

    model_worker.model_runner.model.setup_feedback_buffer(
        int(server_args.max_running_requests)
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False
        model_worker.model_runner.init_device_graphs()

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )

    device_str = f"cuda:{gpu_id}"
    voice_embeddings = _load_voice_embeddings(checkpoint_dir, device=device_str)
    embed_tokens = model_worker.model_runner.model.embed_tokens
    acoustic_transformer, audio_token_embedding = _load_acoustic_and_embedding(
        checkpoint_dir,
        voxtral_config=voxtral_config,
        device=device_str,
    )

    model_runner = VoxtralModelRunner(
        model_worker,
        output_proc,
        acoustic_transformer=acoustic_transformer,
        audio_token_embedding=audio_token_embedding,
        end_audio_token_id=end_audio_token_id,
    )

    ctx = VoxtralAdapterContext(
        embed_tokens=embed_tokens,
        voice_embeddings=voice_embeddings,
        audio_token_id=audio_token_id,
        end_audio_token_id=end_audio_token_id,
        vocab_size=int(voxtral_config.text_config.vocab_size),
        max_new_tokens=max_new_tokens,
    )
    request_builder, result_adapter = make_voxtral_scheduler_adapters(ctx)

    return OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
    )
