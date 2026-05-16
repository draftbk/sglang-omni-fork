# SPDX-License-Identifier: Apache-2.0
"""StagePayload ↔ OmniScheduler adapters for Voxtral-TTS."""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData


@dataclass
class VoxtralSGLangRequestData(SGLangARRequestData):
    # acoustic_codebook_residuals accumulates the 36 non-semantic codes per
    # step; combined with the semantic column at result time to form
    # [T, 37] for the vocoder.
    voice: str | None = None
    end_audio_token_id: int | None = None
    acoustic_codebook_residuals: list[torch.Tensor] = field(default_factory=list)
    pending_feedback_queue: collections.deque = field(default_factory=collections.deque)
    max_new_tokens: int = 4096


@dataclass
class VoxtralAdapterContext:
    """Per-checkpoint adapter wiring shared between the request builder and
    the result adapter. Constructed once at bootstrap time."""

    embed_tokens: Any
    voice_embeddings: dict[str, torch.Tensor]
    audio_token_id: int
    end_audio_token_id: int
    vocab_size: int
    max_new_tokens: int = 4096


def build_voxtral_request(
    state: VoxtralTTSState,
    *,
    request_id: str,
    ctx: VoxtralAdapterContext,
) -> VoxtralSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    input_ids_list = list(state.input_ids or [])
    if not input_ids_list:
        raise ValueError("Voxtral request has no input_ids")
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    max_new_tokens = state.max_new_tokens or ctx.max_new_tokens

    stop_token_ids = [ctx.end_audio_token_id]

    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        stop_token_ids=stop_token_ids,
    )
    # Voxtral ships tekken.json (mistral-common), not a HF tokenizer;
    # passing None bypasses SamplingParams' tokenizer-vocab assertion.
    sampling_params.normalize(None)
    sampling_params.verify(None)

    req = Req(
        rid=request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        vocab_size=ctx.vocab_size,
        eos_token_ids=set(stop_token_ids),
    )
    # SGLang's Req has no default for these; the base sampler
    # (_apply_codec_suppress_tokens) and prefill (_input_embeds_are_projected
    # branch) read them unconditionally. Voxtral never suppresses tokens at
    # the sampler — its semantic code is chosen in the acoustic step — and
    # always provides hidden-size input_embeds (voice injection below).
    req._codec_suppress_tokens = None
    req._input_embeds_are_projected = True

    voice = state.voice or "cheerful_female"
    voice_emb = ctx.voice_embeddings.get(voice)
    if voice_emb is None:
        raise ValueError(
            f"Voxtral voice {voice!r} not found; available: "
            f"{sorted(ctx.voice_embeddings)}"
        )

    input_ids_dev = input_ids.to(ctx.embed_tokens.weight.device)
    with torch.no_grad():
        embeds = ctx.embed_tokens(input_ids_dev).to(dtype=voice_emb.dtype)
        audio_positions = (input_ids_dev == ctx.audio_token_id).nonzero(as_tuple=True)[0]
        n_voice_frames = min(int(audio_positions.shape[0]), int(voice_emb.shape[0]))
        if n_voice_frames > 0:
            embeds[audio_positions[:n_voice_frames]] = voice_emb[:n_voice_frames].to(
                embeds.dtype
            )
    # schedule_batch.prepare_for_extend expects nested Python lists.
    req.input_embeds = embeds.float().cpu().tolist()

    return VoxtralSGLangRequestData(
        input_ids=input_ids,
        req=req,
        voice=voice,
        max_new_tokens=max_new_tokens,
        end_audio_token_id=ctx.end_audio_token_id,
    )


def apply_voxtral_result(
    state: VoxtralTTSState, data: VoxtralSGLangRequestData
) -> None:
    output_ids = data.req.output_ids
    state.prompt_tokens = int(data.input_ids.shape[0])
    if not output_ids:
        state.audio_codes = torch.empty(0, 37, dtype=torch.long)
        state.completion_tokens = 0
        return

    # Single per-request GPU→CPU sync of the acoustic residuals.
    residuals = torch.stack(data.acoustic_codebook_residuals, dim=0).cpu()  # [T, 36]
    semantic = torch.tensor(output_ids, dtype=torch.long).unsqueeze(1)  # [T, 1]
    # SGLang stop-token logic appends end_audio to output_ids but the
    # acoustic step never produces a residual row for it; trim to match.
    n = min(semantic.shape[0], residuals.shape[0])
    state.audio_codes = torch.cat([semantic[:n], residuals[:n]], dim=1)
    state.completion_tokens = n


def make_voxtral_scheduler_adapters(ctx: VoxtralAdapterContext):
    """Return ``(request_builder, result_adapter)`` for OmniScheduler."""

    def request_builder(payload: StagePayload) -> VoxtralSGLangRequestData:
        state = VoxtralTTSState.from_dict(payload.data)
        req_data = build_voxtral_request(
            state, request_id=payload.request_id, ctx=ctx
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: VoxtralSGLangRequestData) -> StagePayload:
        payload = data.stage_payload
        state = VoxtralTTSState.from_dict(payload.data)
        apply_voxtral_result(state, data)
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data=state.to_dict(),
        )

    return request_builder, result_adapter
