# SPDX-License-Identifier: Apache-2.0
"""Shared compilation helpers for the v1 multi-process pipeline."""

from __future__ import annotations

import inspect
import socket
from pathlib import Path
from typing import Any

from sglang_omni_v1.config.schema import PipelineConfig, StageConfig
from sglang_omni_v1.utils import import_string

# ------------------------------------------------------------------
# Factory args
# ------------------------------------------------------------------


def _resolve_factory_args(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    """Resolve factory args, injecting model_path and gpu_id if accepted."""
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
        gpu_id = placement[0] if isinstance(placement, list) else placement
        args["gpu_id"] = gpu_id

    return args


# ------------------------------------------------------------------
# Relay config
# ------------------------------------------------------------------


def _build_relay_config(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict[str, Any]:
    relay_cfg = stage_cfg.relay
    if relay_cfg is not None:
        # Explicit relay config
        return {
            "relay_type": global_cfg.relay_backend,
            "slot_size_mb": relay_cfg.slot_size_mb,
            "credits": relay_cfg.credits,
            "rank": relay_cfg.rank,
            "world_size": relay_cfg.world_size,
            "gpu_id": _parse_gpu_id(relay_cfg.device),
        }

    # Auto-infer from gpu field. For shm, keep transport buffers on CPU:
    # ShmRelay copies tensors to host shared memory anyway, so CUDA staging
    # only inflates GPU allocator pressure.
    if global_cfg.relay_backend == "shm":
        gpu_id = None
    else:
        gpu = stage_cfg.gpu
        if gpu is None:
            gpu_id = None  # CPU stage
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


def _parse_gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Unsupported device string: {device}")


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


def _find_free_tcp_ports(start: int, count: int) -> list[int]:
    """Find *count* available TCP ports starting from *start*.

    Does NOT set SO_REUSEADDR so that TIME_WAIT ports are skipped.
    """
    ports: list[int] = []
    port = start
    while len(ports) < count:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                ports.append(port)
        except OSError:
            pass
        port += 1
    return ports


def _allocate_endpoints(
    config: PipelineConfig,
    *,
    stages: list[StageConfig],
) -> dict[str, str]:
    endpoints: dict[str, str] = {}

    if config.completion_endpoint:
        endpoints["completion"] = config.completion_endpoint
    if config.abort_endpoint:
        endpoints["abort"] = config.abort_endpoint

    if config.endpoints.scheme == "ipc":
        base_dir = Path(config.endpoints.base_path) / config.name
        base_dir.mkdir(parents=True, exist_ok=True)
        endpoints.setdefault("completion", f"ipc://{base_dir}/completion.sock")
        endpoints.setdefault("abort", f"ipc://{base_dir}/abort.sock")
        for s in stages:
            endpoints[f"stage_{s.name}"] = f"ipc://{base_dir}/stage_{s.name}.sock"
        return endpoints

    if config.endpoints.scheme == "tcp":
        needed = 2 + len(stages)
        ports = _find_free_tcp_ports(config.endpoints.base_port, needed)
        idx = 0
        if "completion" not in endpoints:
            endpoints["completion"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        if "abort" not in endpoints:
            endpoints["abort"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        for s in stages:
            endpoints[f"stage_{s.name}"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        return endpoints

    raise ValueError(f"Unknown endpoint scheme: {config.endpoints.scheme}")


def _detect_same_gpu_targets(
    sender_cfg: StageConfig,
    targets: list[str],
    *,
    gpu_placement: dict[str, int | list[int]] | None = None,
    cfg_map: dict[str, StageConfig] | None = None,
) -> set[str]:
    if not gpu_placement or not cfg_map:
        return set()
    sender_gpu = _primary_gpu(sender_cfg, gpu_placement)
    if sender_gpu is None:
        return set()
    same: set[str] = set()
    for target_name in targets:
        receiver_cfg = cfg_map.get(target_name)
        if receiver_cfg is None:
            continue
        receiver_gpu = _primary_gpu(receiver_cfg, gpu_placement)
        if receiver_gpu is not None and receiver_gpu == sender_gpu:
            same.add(target_name)
    return same


def _primary_gpu(
    stage_cfg: StageConfig,
    gpu_placement: dict[str, int | list[int]],
) -> int | None:
    """Return the primary (rank 0) GPU id for a stage, or None for CPU stages."""
    raw = gpu_placement.get(stage_cfg.name)
    if raw is None:
        return None
    return raw[0] if isinstance(raw, list) else raw
