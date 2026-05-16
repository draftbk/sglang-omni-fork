# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts_v1.config import VoxtralTTSV1PipelineConfig
from sglang_omni.models.voxtral_tts_v1.request_builders import (
    VoxtralAdapterContext,
    VoxtralSGLangRequestData,
    apply_voxtral_result,
    build_voxtral_request,
    make_voxtral_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload


def test_voxtral_pipeline_config_topology_and_validation() -> None:
    """Preserves the 3-stage non-streaming topology under the V1 flat StageConfig schema."""
    config = VoxtralTTSV1PipelineConfig(model_path="model")

    assert [stage.name for stage in config.stages] == [
        "preprocessing",
        "generation",
        "vocoder",
    ]
    assert config.resolved_entry_stage == "preprocessing"
    assert config.terminal_stages == ["vocoder"]
    assert config.gpu_placement == {"generation": 0, "vocoder": 0}

    preprocessing, generation, vocoder = config.stages
    assert preprocessing.next == "generation" and not preprocessing.terminal
    assert generation.next == "vocoder" and not generation.terminal
    assert generation.factory_args == {"device": "cuda:0", "max_new_tokens": 4096}
    assert vocoder.terminal and vocoder.next is None
    assert vocoder.factory_args == {"device": "cuda:0"}


def test_voxtral_state_round_trips_through_dict() -> None:
    """VoxtralTTSState survives the cross-stage dict serialization that
    StagePayload.data uses. Tensor audio_codes survives; audio_samples is
    serialized as a list for msgpack compatibility."""
    audio_codes = torch.tensor(
        [[100, 101, 102], [200, 201, 202]], dtype=torch.long
    )
    audio_samples = torch.tensor([0.1, -0.1, 0.2], dtype=torch.float32)
    state = VoxtralTTSState(
        input_ids=[1, 2, 3],
        voice="cheerful_female",
        max_new_tokens=512,
        audio_codes=audio_codes,
        prompt_tokens=10,
        completion_tokens=7,
        audio_samples=audio_samples,
        sample_rate=24000,
    )

    restored = VoxtralTTSState.from_dict(state.to_dict())

    assert restored.input_ids == [1, 2, 3]
    assert restored.voice == "cheerful_female"
    assert restored.max_new_tokens == 512
    assert restored.prompt_tokens == 10
    assert restored.completion_tokens == 7
    assert restored.sample_rate == 24000
    assert torch.equal(restored.audio_codes, audio_codes)
    assert restored.audio_samples == audio_samples.tolist()


def test_voxtral_stage_factory_dotted_paths_resolve_to_callables() -> None:
    """Every StageConfig.factory must importlib-resolve to a callable, so the
    pipeline compiler can wire stages without instantiating models."""
    config = VoxtralTTSV1PipelineConfig(model_path="model")
    for stage in config.stages:
        module_path, _, attr = stage.factory.rpartition(".")
        module = importlib.import_module(module_path)
        fn = getattr(module, attr)
        assert callable(fn), f"{stage.factory} did not resolve to a callable"


@pytest.fixture(autouse=True)
def _stub_sampling_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """SamplingParams.normalize/verify require an HF tokenizer; bypass."""
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda self, tokenizer: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda self, vocab_size: None,
    )


def _make_ctx(
    *,
    audio_token_id: int = 10,
    end_audio_token_id: int = 99,
    vocab_size: int = 128,
    voices: tuple[str, ...] = ("cheerful_female", "warm_male"),
    hidden: int = 4,
) -> VoxtralAdapterContext:
    embed = nn.Embedding(vocab_size, hidden)
    nn.init.zeros_(embed.weight)
    return VoxtralAdapterContext(
        embed_tokens=embed,
        voice_embeddings={
            name: torch.full((2, hidden), float(idx + 1), dtype=torch.bfloat16)
            for idx, name in enumerate(voices)
        },
        audio_token_id=audio_token_id,
        end_audio_token_id=end_audio_token_id,
        vocab_size=vocab_size,
        max_new_tokens=64,
    )


def test_build_voxtral_request_injects_voice_at_audio_positions() -> None:
    """audio_token_id positions in input_ids are overwritten with the voice
    embedding; other positions come from embed_tokens."""
    ctx = _make_ctx(audio_token_id=10, hidden=4)
    state = VoxtralTTSState(input_ids=[1, 10, 10, 2], voice="cheerful_female")

    data = build_voxtral_request(state, request_id="r1", ctx=ctx)

    assert data.voice == "cheerful_female"
    assert data.end_audio_token_id == 99
    embeds = torch.tensor(data.req.input_embeds, dtype=torch.float32)
    assert embeds.shape == (4, 4)
    # cheerful_female voice embedding is filled with 1.0; positions 1 and 2
    # are audio_token_id and should be overwritten.
    assert torch.allclose(embeds[1:3], torch.ones(2, 4))
    # Non-audio positions still come from the zero-initialized embed_tokens.
    assert torch.all(embeds[0] == 0) and torch.all(embeds[3] == 0)


def test_build_voxtral_request_raises_on_unknown_voice() -> None:
    """Voice not in the checkpoint dict is a hard error, not a silent fallback."""
    ctx = _make_ctx()
    state = VoxtralTTSState(input_ids=[1, 2, 3], voice="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        build_voxtral_request(state, request_id="r2", ctx=ctx)


def test_build_voxtral_request_raises_on_empty_input_ids() -> None:
    ctx = _make_ctx()
    state = VoxtralTTSState(input_ids=[], voice="cheerful_female")
    with pytest.raises(ValueError, match="input_ids"):
        build_voxtral_request(state, request_id="r3", ctx=ctx)


def test_apply_voxtral_result_concats_semantic_and_residuals() -> None:
    """Result adapter stacks per-step residuals once and concatenates with
    the semantic column to form the [T, 37] code grid for the vocoder."""
    state = VoxtralTTSState(input_ids=[1, 2, 3, 4], voice="cheerful_female")
    req = MagicMock()
    req.output_ids = [5, 6, 7]
    data = VoxtralSGLangRequestData(
        input_ids=torch.tensor([1, 2, 3, 4]),
        req=req,
        voice="cheerful_female",
        acoustic_codebook_residuals=[
            torch.full((36,), float(t), dtype=torch.long) for t in (5, 6, 7)
        ],
    )

    apply_voxtral_result(state, data)

    assert state.prompt_tokens == 4
    assert state.completion_tokens == 3
    assert state.audio_codes.shape == (3, 37)
    assert state.audio_codes[:, 0].tolist() == [5, 6, 7]
    assert torch.all(state.audio_codes[:, 1:] == state.audio_codes[:, :1])


def test_apply_voxtral_result_trims_dangling_end_audio_token() -> None:
    """SGLang's stop-token logic appends end_audio to output_ids but the
    acoustic step never produces a matching residual row; trim to match."""
    state = VoxtralTTSState(input_ids=[1, 2], voice="cheerful_female")
    req = MagicMock()
    req.output_ids = [5, 6, 99]  # 99 = end_audio, has no residual
    data = VoxtralSGLangRequestData(
        input_ids=torch.tensor([1, 2]),
        req=req,
        voice="cheerful_female",
        acoustic_codebook_residuals=[torch.zeros(36, dtype=torch.long) for _ in range(2)],
    )

    apply_voxtral_result(state, data)

    assert state.completion_tokens == 2
    assert state.audio_codes.shape == (2, 37)
    assert state.audio_codes[:, 0].tolist() == [5, 6]


def test_apply_voxtral_result_empty_output_yields_empty_codes() -> None:
    state = VoxtralTTSState(input_ids=[1, 2], voice="cheerful_female")
    req = MagicMock()
    req.output_ids = []
    data = VoxtralSGLangRequestData(
        input_ids=torch.tensor([1, 2]),
        req=req,
        voice="cheerful_female",
    )

    apply_voxtral_result(state, data)

    assert state.prompt_tokens == 2
    assert state.completion_tokens == 0
    assert state.audio_codes.shape == (0, 37)


def test_scheduler_adapters_round_trip_payload() -> None:
    """make_voxtral_scheduler_adapters returns a (builder, adapter) pair
    that preserves StagePayload identity and serializes audio_codes back."""
    ctx = _make_ctx()
    state = VoxtralTTSState(input_ids=[1, 10, 2], voice="cheerful_female")
    payload = StagePayload(
        request_id="req-x",
        request=OmniRequest(inputs="hello", params={}),
        data=state.to_dict(),
    )

    request_builder, result_adapter = make_voxtral_scheduler_adapters(ctx)
    data = request_builder(payload)
    assert data.stage_payload is payload
    data.req.output_ids = [42, 43]
    data.acoustic_codebook_residuals = [
        torch.full((36,), float(t), dtype=torch.long) for t in (42, 43)
    ]

    out = result_adapter(data)
    assert out.request_id == "req-x"
    assert out.request is payload.request
    assert out.data["completion_tokens"] == 2
    assert len(out.data["audio_codes"]) == 2
    assert out.data["audio_codes"][0][0] == 42
