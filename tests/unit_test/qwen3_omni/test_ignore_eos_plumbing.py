# SPDX-License-Identifier: Apache-2.0
"""Regression test: ``ignore_eos`` threads through the request stack.

The MMMU decode-throughput-parity lane needs the upstream SGLang sampler to
keep emitting tokens until ``max_new_tokens`` regardless of EOS. The flag
flows: ``ChatCompletionRequest.ignore_eos`` →
``_build_chat_generate_request`` → ``GenerateRequest.sampling.ignore_eos``
→ ``SamplingParams.to_dict()`` (consumed by
``build_sglang_thinker_request``) → upstream
``sglang.srt.sampling.sampling_params.SamplingParams(ignore_eos=...)``.

These tests cover the three handoffs that live in pure Python without GPU
dependencies. The final hop into the upstream SGLang ``SamplingParams``
constructor is exercised indirectly: ``build_sglang_thinker_request``
reads the value from the dict, so any plumbing break above this layer
shows up here.
"""
from __future__ import annotations

import pytest

# sglang_omni.serve depends on fastapi which is not always installed on dev
# machines; skip the module-level imports cleanly when fastapi is missing
# so the rest of the unit-test suite still collects.
pytest.importorskip("fastapi")

from sglang_omni.client.types import SamplingParams as ClientSamplingParams  # noqa: E402
from sglang_omni.serve.openai_api import _build_chat_generate_request  # noqa: E402
from sglang_omni.serve.protocol import ChatCompletionRequest, ChatMessage  # noqa: E402


def _make_request(**kwargs) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="ping")],
        **kwargs,
    )


def test_ignore_eos_default_false() -> None:
    gen_req = _build_chat_generate_request(_make_request())
    assert gen_req.sampling.ignore_eos is False


def test_ignore_eos_threads_through_to_generate_request() -> None:
    gen_req = _build_chat_generate_request(_make_request(ignore_eos=True))
    assert gen_req.sampling.ignore_eos is True


def test_ignore_eos_in_sampling_params_to_dict() -> None:
    sp = ClientSamplingParams(ignore_eos=True)
    d = sp.to_dict()
    assert d["ignore_eos"] is True


def test_ignore_eos_default_omitted_from_payload() -> None:
    # Default-False ignore_eos still serializes to False so build_sglang_thinker_request's
    # params.get("ignore_eos", False) cast yields a stable False.
    sp = ClientSamplingParams()
    d = sp.to_dict()
    assert d["ignore_eos"] is False


def test_build_sglang_thinker_request_passes_ignore_eos_to_upstream() -> None:
    """AC-10 final-hop spy test: build_sglang_thinker_request forwards
    ``ignore_eos=True`` into the upstream SGLang ``SamplingParams(...)``
    instantiation at ``request_builders.py:347``. Mocks the upstream
    import so the test runs without the SGLang runtime."""
    sglang_mod = pytest.importorskip("sglang")
    pytest.importorskip("torch")
    pytest.importorskip("xxhash")

    import sys
    from unittest.mock import MagicMock, patch

    import torch

    # The function imports ``SamplingParams`` and ``MultimodalInputs, Req``
    # at call time from inside sglang.srt.*. Patch both import sites.
    fake_sampling_module = MagicMock()
    fake_sampling_module.SamplingParams = MagicMock(name="SamplingParams")
    fake_schedule_module = MagicMock()
    fake_schedule_module.MultimodalInputs = MagicMock(name="MultimodalInputs")
    fake_schedule_module.Req = MagicMock(name="Req")

    state = MagicMock()
    state.prompt = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": None,
    }
    state.thinker_inputs = None
    tokenizer = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "sglang.srt.sampling.sampling_params": fake_sampling_module,
            "sglang.srt.managers.schedule_batch": fake_schedule_module,
        },
    ):
        from sglang_omni.models.qwen3_omni import request_builders

        try:
            request_builders.build_sglang_thinker_request(
                state,
                params={"ignore_eos": True, "max_new_tokens": 256},
                tokenizer=tokenizer,
                vocab_size=152064,
            )
        except Exception:
            # We don't care if downstream Req construction trips on the
            # mock; only the SamplingParams call matters here.
            pass

    fake_sampling_module.SamplingParams.assert_called()
    kwargs = fake_sampling_module.SamplingParams.call_args.kwargs
    assert kwargs.get("ignore_eos") is True


def test_build_sglang_thinker_request_defaults_ignore_eos_false() -> None:
    """When params dict omits ignore_eos, upstream SGLang gets False."""
    sglang_mod = pytest.importorskip("sglang")
    pytest.importorskip("torch")
    pytest.importorskip("xxhash")

    import sys
    from unittest.mock import MagicMock, patch

    import torch

    fake_sampling_module = MagicMock()
    fake_sampling_module.SamplingParams = MagicMock(name="SamplingParams")
    fake_schedule_module = MagicMock()
    fake_schedule_module.MultimodalInputs = MagicMock(name="MultimodalInputs")
    fake_schedule_module.Req = MagicMock(name="Req")

    state = MagicMock()
    state.prompt = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": None,
    }
    state.thinker_inputs = None
    tokenizer = MagicMock()

    with patch.dict(
        sys.modules,
        {
            "sglang.srt.sampling.sampling_params": fake_sampling_module,
            "sglang.srt.managers.schedule_batch": fake_schedule_module,
        },
    ):
        from sglang_omni.models.qwen3_omni import request_builders

        try:
            request_builders.build_sglang_thinker_request(
                state,
                params={"max_new_tokens": 2048},
                tokenizer=tokenizer,
                vocab_size=152064,
            )
        except Exception:
            pass

    fake_sampling_module.SamplingParams.assert_called()
    kwargs = fake_sampling_module.SamplingParams.call_args.kwargs
    assert kwargs.get("ignore_eos") is False
