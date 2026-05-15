# SPDX-License-Identifier: Apache-2.0
"""Build the run-metadata block emitted alongside each MMMU sweep result.

The metadata block captures every reproducibility-relevant knob: code SHAs,
serving stack versions, model + dataset revisions, sampling configuration,
mem-fraction and KV-cache settings, the configured (not just observed)
prefix-cache policy, encoder-patch activation state, the H200 host the
cell ran on, container identity and image digest, and per-rep bookkeeping.

This module is callable both from the eval harness (to inline metadata
into each result JSON) and from the sweep script (to stamp a single
metadata blob across paired reps). All shell-out helpers (``docker
inspect``, ``nvidia-smi``, ``importlib.metadata``) gracefully degrade to
``None`` when their inputs are missing so the same emitter works on
non-GPU dev boxes and inside the H200 containers.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunMetadata:
    """All fields required by the run-metadata block of the issue #379 closure plan."""

    # Code SHAs
    commit_sha: str | None = None
    branch: str | None = None
    sglang_version: str | None = None

    # Backend + model identity
    backend: str = "omni"
    model_id: str | None = None
    model_revision: str | None = None
    dataset_revisions: dict[str, str] = field(default_factory=dict)

    # Sampling + lane config
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

    # Memory + cache policy (POLICY enforced via launch flag, not just observed)
    mem_fraction_static_configured: float | None = None
    kv_cache_capacity_tokens: int | None = None
    steady_state_gpu_gb: list[float] = field(default_factory=list)
    prefix_cache_disabled: bool = True

    # Encoder-patch activation state (PR #436 dormant unless a production
    # caller wires apply_qwen3_vl_hf_parity_patches into the load path).
    encoder_patches_active: bool = False

    # Hardware + container identity
    host: str | None = None
    container_name: str | None = None
    container_image: str | None = None
    container_image_digest: str | None = None
    server_port: int | None = None
    gpu_topology: str | None = None

    # Per-rep bookkeeping
    repetition_index: int = 0
    failure_count: int = 0


def get_commit_sha(repo_root: Path) -> str | None:
    """Return the git HEAD SHA, or None outside a git checkout."""
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
    """Resolve the sha256 image digest a running container is using.

    Returns None when docker is not available or the container does not
    exist. The fmt string yields one of:
      - ``sha256:abc...`` when the image is a content-addressed reference
      - the image name (e.g. ``lmsysorg/sglang:latest``) when no digest
        is known
    """
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.check_output(
            [
                "docker",
                "inspect",
                container_name,
                "--format",
                "{{index .Image}}",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except subprocess.CalledProcessError:
        return None


def sample_gpu_memory_used_gb() -> list[float]:
    """Sample per-GPU memory.used in GB via nvidia-smi.

    Returns an empty list when nvidia-smi is unavailable (e.g. on a macOS
    dev box). Callers that need authoritative numbers should call this
    at ``warmup_complete + 30s`` per the plan's steady-state contract.
    """
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return []
    values: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(round(int(line) / 1024.0, 3))
        except ValueError:
            continue
    return values


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


_KV_POOL_LINE = re.compile(
    r"KV[- ]?[Cc]ache.*?(?:tokens?|capacity).*?(\d[\d,]*)",
)


def scrape_kv_cache_capacity_from_log(log_path: Path) -> int | None:
    """Find the SGLang KV pool init log line and return its token capacity.

    The exact wording shifts between SGLang releases; the regex above
    matches "KV-cache pool ... capacity X tokens" and similar. Returns
    None when no line matches.
    """
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    match = _KV_POOL_LINE.search(text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def to_dict(meta: RunMetadata) -> dict[str, Any]:
    return asdict(meta)


def write_metadata(meta: RunMetadata, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(to_dict(meta), indent=2, ensure_ascii=False))


REQUIRED_FIELDS = tuple(RunMetadata.__dataclass_fields__.keys())


def validate(meta_dict: dict[str, Any]) -> list[str]:
    """Return a list of missing required keys (empty list = valid)."""
    return [key for key in REQUIRED_FIELDS if key not in meta_dict]
