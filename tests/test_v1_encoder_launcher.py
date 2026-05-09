# SPDX-License-Identifier: Apache-2.0
"""Unit tests for v1 launcher / preflight changes — Phase 0 of #375.

These exercise the StageProcessSpec / mp_runner / compile_pipeline /
launcher predicates without spawning subprocesses or touching CUDA.
"""

from __future__ import annotations

import inspect

import pytest

from sglang_omni_v1.config.compiler import (
    _SGLANG_ENCODER_BACKENDS,
    _resolve_factory_args,
    compile_pipeline,
)
from sglang_omni_v1.config.schema import PipelineConfig, StageConfig
from sglang_omni_v1.models.qwen3_omni.stages import (
    create_audio_encoder_executor,
    create_image_encoder_executor,
)
from sglang_omni_v1.pipeline.mp_runner import _build_stage_groups
from sglang_omni_v1.pipeline.stage_process import (
    StageProcessSpec,
    get_stage_process_env,
)
from sglang_omni_v1.scheduling.sglang_backend.encoder_server_args import (
    _ENCODER_PROTECTED_KEYS,
    build_sglang_encoder_server_args,
)

# ---------------------------------------------------------------------------
# Lightweight fake factories so we don't construct the real SGLang worker
# during unit tests. The factory module + symbol live here so dotted-import
# resolution works.
# ---------------------------------------------------------------------------

def _factory_no_tp_params(model_path: str, *, gpu_id: int = 0, **kwargs):
    return ("no-tp", model_path, gpu_id, kwargs)


def _factory_tp_aware(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    **kwargs,
):
    return ("tp-aware", tp_rank, tp_size, nccl_port, kwargs)


def _factory_encoder_local_default(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    backend: str = "local",
    **kwargs,
):
    return ("encoder", backend, tp_rank, tp_size)


# Make these importable as
# tests.test_v1_encoder_launcher._factory_*
# so StageConfig.factory accepts the dotted path.


def _stage(
    name: str,
    factory_dotted: str,
    *,
    tp_size: int = 1,
    gpu: int | list[int] | None = None,
    factory_args: dict | None = None,
) -> StageConfig:
    return StageConfig(
        name=name,
        factory=factory_dotted,
        tp_size=tp_size,
        gpu=gpu,
        factory_args=factory_args or {},
        next="sink",
    )


def _config_simple(stage: StageConfig, *, runtime_overrides=None) -> PipelineConfig:
    """Build a 2-stage pipeline: ``stage`` -> terminal sink."""
    sink = StageConfig(
        name="sink",
        factory="tests.test_v1_encoder_launcher._factory_no_tp_params",
        terminal=True,
    )
    return PipelineConfig(
        model_path="fake/model",
        stages=[stage, sink],
        runtime_overrides=runtime_overrides or {},
    )


# ---------------------------------------------------------------------------
# StageProcessSpec.single_visible_device + get_stage_process_env
# ---------------------------------------------------------------------------


def test_single_visible_device_default_off_at_tp1():
    spec = StageProcessSpec(stage_name="x", role="single", tp_size=1, gpu_id=4)
    assert get_stage_process_env(spec) == {}


def test_single_visible_device_remaps_at_tp1_when_flag_set():
    spec = StageProcessSpec(
        stage_name="x",
        role="single",
        tp_size=1,
        gpu_id=4,
        single_visible_device=True,
    )
    env = get_stage_process_env(spec, env={})
    assert env["CUDA_VISIBLE_DEVICES"] == "4"
    assert env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] == "true"


# ---------------------------------------------------------------------------
# Launcher: _build_stage_groups sets single_visible_device from resolved
# backend; runtime_overrides flip the flag.
# ---------------------------------------------------------------------------


def test_build_stage_groups_sets_single_visible_for_sglang_backend():
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        gpu=4,
        factory_args={"backend": "sglang"},
    )
    groups = _build_stage_groups(_config_simple(stage))
    image_group = next(g for g in groups if g.stage_name == "image_encoder")
    assert image_group.specs[0].single_visible_device is True


def test_build_stage_groups_runtime_override_flips_backend():
    # factory_args has no backend; runtime_overrides flips it on.
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        gpu=2,
    )
    cfg = _config_simple(
        stage,
        runtime_overrides={"image_encoder": {"backend": "sglang"}},
    )
    groups = _build_stage_groups(cfg)
    spec = next(g for g in groups if g.stage_name == "image_encoder").specs[0]
    assert spec.single_visible_device is True


def test_build_stage_groups_local_backend_keeps_flag_off():
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        gpu=2,
        factory_args={"backend": "local"},
    )
    groups = _build_stage_groups(_config_simple(stage))
    spec = next(g for g in groups if g.stage_name == "image_encoder").specs[0]
    assert spec.single_visible_device is False


# ---------------------------------------------------------------------------
# Two-layer TP preflight
# ---------------------------------------------------------------------------


def test_preflight_layer1_rejects_factory_without_tp_params():
    # SimpleScheduler-style factory missing tp_rank/tp_size/nccl_port.
    stage = _stage(
        "agg",
        "tests.test_v1_encoder_launcher._factory_no_tp_params",
        tp_size=2,
        gpu=[0, 1],
    )
    with pytest.raises(ValueError, match="not accept TP launch parameters"):
        _build_stage_groups(_config_simple(stage))


def test_preflight_layer2_rejects_encoder_with_local_backend():
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        tp_size=2,
        gpu=[0, 1],
        # No backend in factory_args → resolver picks default "local".
    )
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        _build_stage_groups(_config_simple(stage))


def test_preflight_layer2_rejects_encoder_with_auto_backend():
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        tp_size=2,
        gpu=[0, 1],
        factory_args={"backend": "auto"},
    )
    with pytest.raises(ValueError, match="requires backend='sglang'"):
        _build_stage_groups(_config_simple(stage))


def test_preflight_layer2_passes_with_sglang_backend():
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        tp_size=2,
        gpu=[0, 1],
        factory_args={"backend": "sglang"},
    )
    # Should not raise — but actual subprocess isn't spawned.
    groups = _build_stage_groups(_config_simple(stage))
    image_group = next(g for g in groups if g.stage_name == "image_encoder")
    assert len(image_group.specs) == 2
    assert image_group.specs[0].single_visible_device is True
    assert image_group.specs[1].single_visible_device is True


def test_preflight_does_not_regress_thinker_tp():
    # Thinker-style: TP params present, no backend kwarg.
    stage = _stage(
        "thinker",
        "tests.test_v1_encoder_launcher._factory_tp_aware",
        tp_size=2,
        gpu=[0, 1],
    )
    groups = _build_stage_groups(_config_simple(stage))
    thinker_group = next(g for g in groups if g.stage_name == "thinker")
    assert len(thinker_group.specs) == 2


# ---------------------------------------------------------------------------
# compile_pipeline rejects multiprocess-only stages directly
# ---------------------------------------------------------------------------


def test_compile_pipeline_rejects_tp_size_gt_1():
    stage = _stage(
        "thinker",
        "tests.test_v1_encoder_launcher._factory_tp_aware",
        tp_size=2,
        gpu=[0, 1],
    )
    with pytest.raises(ValueError, match="tp_size=2"):
        compile_pipeline(_config_simple(stage))


def test_compile_pipeline_rejects_sglang_backend():
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        gpu=4,
        factory_args={"backend": "sglang"},
    )
    with pytest.raises(ValueError, match="per-process CUDA"):
        compile_pipeline(_config_simple(stage))


# ---------------------------------------------------------------------------
# Backend resolution contract: launcher reads only factory_args /
# runtime_overrides, never the factory's signature default.
# ---------------------------------------------------------------------------


def test_backend_resolution_ignores_signature_default():
    # Factory's signature default IS "local" — but a hypothetical flip of
    # the default to "sglang" should NOT affect launcher decisions. We
    # simulate this by checking the resolver returns "local" when neither
    # factory_args nor runtime_overrides set backend, regardless of what
    # the signature default is.
    stage = _stage(
        "image_encoder",
        "tests.test_v1_encoder_launcher._factory_encoder_local_default",
        gpu=2,
    )
    cfg = _config_simple(stage)
    args = _resolve_factory_args(cfg.stages[0], cfg)
    # No "backend" key present in resolved args — factory signature
    # default would supply it at call time, but launcher must not see
    # it here.
    assert "backend" not in args
    # Defaulting to "local" via .get() in launcher matches the contract.
    assert args.get("backend", "local") == "local"


def test_real_factory_signature_defaults_to_local():
    """Lock the production factory defaults so a code-review miss can't
    silently flip the launcher decision via signature defaults."""
    image_sig = inspect.signature(create_image_encoder_executor)
    audio_sig = inspect.signature(create_audio_encoder_executor)
    assert image_sig.parameters["backend"].default == "local"
    assert audio_sig.parameters["backend"].default == "local"


# ---------------------------------------------------------------------------
# Encoder ServerArgs helper protected keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "tp_size",
        "tp_rank",
        "gpu_id",
        "nccl_port",
        "rank",
        "world_size",
        "encoder_only",
        "language_only",
        "mm_enable_dp_encoder",
        "mem_fraction_static",
        "max_running_requests",
        "context_length",
        "chunked_prefill_size",
    ],
)
def test_protected_keys_in_overrides_raise(key):
    with pytest.raises(ValueError, match="protected keys"):
        build_sglang_encoder_server_args(
            model_path="fake/model",
            tp_size=1,
            base_gpu_id=0,
            dist_init_addr="127.0.0.1:0",
            **{key: 1},
        )


def test_protected_keys_set_is_complete():
    # Every key in the docstring claim must be in the set, so a future
    # editor can't trim the set without updating the test.
    must_protect = {
        "tp_size",
        "tp_rank",
        "gpu_id",
        "nccl_port",
        "encoder_only",
        "language_only",
        "mm_enable_dp_encoder",
        "mem_fraction_static",
        "max_running_requests",
        "max_prefill_tokens",
        "chunked_prefill_size",
        "context_length",
    }
    assert must_protect.issubset(_ENCODER_PROTECTED_KEYS)


# ---------------------------------------------------------------------------
# Public _SGLANG_ENCODER_BACKENDS constant matches launcher / compiler
# ---------------------------------------------------------------------------


def test_sglang_encoder_backends_constant():
    assert _SGLANG_ENCODER_BACKENDS == frozenset({"sglang", "auto"})
