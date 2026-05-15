# SPDX-License-Identifier: Apache-2.0
"""Resolve declarative stage config into runtime process settings."""

from __future__ import annotations

import inspect
from typing import Any

from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.utils import import_string


def resolve_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    """Resolve factory args, injecting model_path and gpu_id when appropriate."""
    args = dict(stage_cfg.factory_args)
    stage_overrides = global_cfg.runtime_overrides.get(stage_cfg.name, {})
    if stage_overrides:
        args.update(stage_overrides)
    factory = import_string(stage_cfg.factory)
    sig = inspect.signature(factory)

    if "model_path" in sig.parameters and "model_path" not in args:
        args["model_path"] = global_cfg.model_path

    if "gpu_id" in sig.parameters and "gpu_id" not in args:
        placement = global_cfg.gpu_placement.get(stage_cfg.name)
        if placement is not None:
            args["gpu_id"] = placement[0] if isinstance(placement, list) else placement

    return args


def build_relay_config(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    relay_cfg = stage_cfg.relay
    if relay_cfg is not None:
        return {
            "relay_type": global_cfg.relay_backend,
            "slot_size_mb": relay_cfg.slot_size_mb,
            "credits": relay_cfg.credits,
            "rank": relay_cfg.rank,
            "world_size": relay_cfg.world_size,
            "gpu_id": parse_gpu_id(relay_cfg.device),
        }

    if global_cfg.relay_backend == "shm":
        gpu_id = None
    else:
        gpu = stage_cfg.gpu
        if gpu is None:
            gpu_id = None
        elif isinstance(gpu, list):
            gpu_id = gpu[0]
        else:
            gpu_id = gpu

    return {
        "relay_type": global_cfg.relay_backend,
        "slot_size_mb": 512,
        "credits": 2,
        "rank": None,
        "world_size": None,
        "gpu_id": gpu_id,
    }


def parse_gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Unsupported device string: {device}")


def detect_same_gpu_targets(
    sender_cfg: StageConfig,
    targets: list[str],
    *,
    gpu_placement: dict[str, int | list[int]] | None = None,
    cfg_map: dict[str, StageConfig] | None = None,
) -> set[str]:
    if not gpu_placement or not cfg_map:
        return set()
    sender_gpu = primary_gpu(sender_cfg, gpu_placement)
    if sender_gpu is None:
        return set()
    same: set[str] = set()
    for target_name in targets:
        receiver_cfg = cfg_map.get(target_name)
        if receiver_cfg is None:
            continue
        receiver_gpu = primary_gpu(receiver_cfg, gpu_placement)
        if receiver_gpu is not None and receiver_gpu == sender_gpu:
            same.add(target_name)
    return same


def primary_gpu(
    stage_cfg: StageConfig,
    gpu_placement: dict[str, int | list[int]],
) -> int | None:
    """Return the primary (rank 0) GPU id for a stage, or None for CPU stages."""
    raw = gpu_placement.get(stage_cfg.name)
    if raw is None:
        return None
    return raw[0] if isinstance(raw, list) else raw
