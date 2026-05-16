# SPDX-License-Identifier: Apache-2.0
"""Run-metadata block emitted alongside each MMMU result JSON.

Captures code SHAs, sglang version, model + dataset revisions, sampling
config, container identity + digest, host info, per-rep bookkeeping.
Shell-out helpers degrade to None when their inputs are missing.
"""

from __future__ import annotations

import importlib.metadata
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunMetadata:
    commit_sha: str | None = None
    branch: str | None = None
    sglang_version: str | None = None

    backend: str = "omni"
    model_id: str | None = None
    dataset_revisions: dict[str, str] = field(default_factory=dict)

    seed: int | None = None
    ignore_eos: bool = False
    lane: str = "A"
    stream: bool = False
    max_tokens: int | None = None
    max_concurrency: int = 1
    temperature: float = 0.0
    warmup: int = 0
    request_rate: float | None = None
    timeout_s: int = 300
    repo_id: str | None = None
    max_samples: int | None = None

    host: str | None = None
    container_name: str | None = None
    container_image: str | None = None
    container_image_digest: str | None = None
    server_port: int | None = None
    gpu_topology: str | None = None

    repetition_index: int = 0
    failure_count: int = 0


def get_commit_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_current_branch(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_sglang_version() -> str | None:
    try:
        return importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        return None


def get_container_image_digest(container_name: str) -> str | None:
    """Resolve the running container's image digest. None when docker is unavailable."""
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.check_output(
            ["docker", "inspect", container_name, "--format", "{{index .Image}}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except subprocess.CalledProcessError:
        return None


def get_gpu_topology() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        return subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError:
        return None


def to_dict(meta: RunMetadata) -> dict[str, Any]:
    return asdict(meta)
