# SPDX-License-Identifier: Apache-2.0
"""GPU-resident numerical parity for v1 encoder backends — Phase 1 of #375.

Each backend variant (local HF, sglang TP-aware) runs in its own
subprocess so heavy model loads (and the global ``torch.distributed``
state once initialized) can't leak across tests. Each subprocess
saves its encoder output to a temp file; the test orchestrator loads
the tensors and asserts ``torch.testing.assert_close`` at
``atol = rtol = 1e-3``.

Marked ``benchmark`` so the standard CI path skips the run.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.benchmark

QWEN3_OMNI_MODEL = os.environ.get(
    "SGLANG_OMNI_TEST_QWEN3_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct"
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPERS_DIR = Path(__file__).parent / "_v1_parity_helpers"


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")


def _allocate_free_port() -> int:
    """Allocate an ephemeral TCP port for NCCL rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return int(sock.getsockname()[1])


def _run_subprocess(
    script: Path,
    *,
    output_path: Path,
    extra_env: dict[str, str] | None = None,
    timeout_s: float = 600.0,
) -> None:
    """Run a parity-helper script and fail the test on non-zero exit.

    Pipes child stdout / stderr to a per-test log file rather than to
    ``subprocess.PIPE``. Under pytest's capture settings, NCCL's
    ``ncclCommInitRank`` deadlocks reproducibly when the child writes
    to a pipe — but is fine when writing to a regular file. Suspected
    interaction between pytest's faulthandler / fd inheritance and
    NCCL's bootstrap, but the file-redirect avoids the issue
    completely.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_REPO_ROOT)
    env["MODEL_PATH"] = QWEN3_OMNI_MODEL
    env["OUTPUT_PATH"] = str(output_path)
    env["DTYPE"] = "bfloat16"
    # Pin NCCL to the loopback interface for the local-host tests so
    # NCCL doesn't hang trying to enumerate IB / Ethernet devices that
    # may or may not be reachable from inside the container.
    env.setdefault("NCCL_SOCKET_IFNAME", "lo")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.update(extra_env or {})

    log_path = output_path.with_suffix(".log")
    with open(log_path, "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            log.flush()
            pytest.fail(
                f"subprocess {script.name} exceeded {timeout_s}s\n"
                f"--- log ({log_path}) ---\n{log_path.read_text()}"
            )

    if proc.returncode != 0:
        pytest.fail(
            f"subprocess {script.name} exited {proc.returncode}\n"
            f"--- log ({log_path}) ---\n{log_path.read_text()}"
        )


def _load_audio_embeds(path: Path) -> torch.Tensor:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    embeds: torch.Tensor = blob["audio_embeds"]
    # The local HF audio tower returns ``(T_unmasked, hidden)`` while
    # sglang main's TP-aware variant keeps an outer batch dim of 1.
    # Squeeze leading 1-dims so the parity assert focuses on values,
    # not packaging.
    while embeds.ndim > 2 and embeds.shape[0] == 1:
        embeds = embeds.squeeze(0)
    return embeds


# ---------------------------------------------------------------------------
# Phase 1.7 — local HF tower vs sglang main TP-aware tower at tp_size=1
# ---------------------------------------------------------------------------


def test_audio_local_vs_sglang_tp1_numerical_parity(tmp_path):
    _skip_if_no_cuda()

    local_out = tmp_path / "local.pt"
    sglang_out = tmp_path / "sglang_tp1.pt"

    _run_subprocess(_HELPERS_DIR / "run_audio_local.py", output_path=local_out)
    _run_subprocess(
        _HELPERS_DIR / "run_audio_sglang.py",
        output_path=sglang_out,
        extra_env={
            "TP_SIZE": "1",
            "TP_RANK": "0",
            "NCCL_PORT": str(_allocate_free_port()),
            "MASTER_ADDR": "127.0.0.1",
        },
    )

    a = _load_audio_embeds(local_out)
    b = _load_audio_embeds(sglang_out)
    assert (
        a.shape == b.shape
    ), f"shape mismatch: local={tuple(a.shape)} sglang={tuple(b.shape)}"
    assert a.dtype == b.dtype, f"dtype mismatch: local={a.dtype} sglang={b.dtype}"
    # Cast to float32 for the comparison to give the bf16 -> bf16 path
    # a stable error budget.
    # bf16 accumulates ~5-10× more error than fp16 across ~24
    # transformer layers, so we use the looser bf16-standard tolerance
    # band rather than the float16-tuned 1e-3 from Cheng's design.
    torch.testing.assert_close(
        a.float(),
        b.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=lambda raw: f"audio encoder local vs sglang tp_size=1 mismatch:\n{raw}",
    )


# ---------------------------------------------------------------------------
# Phase 1.8 — tp_size=1 vs tp_size=2 on the sglang TP-aware tower
# ---------------------------------------------------------------------------


def _spawn_tp_run(tp_size: int, tmp_path: Path) -> Path:
    """Run sglang audio encoder at the given tp_size; return leader output path."""
    if tp_size == 1:
        out = tmp_path / "sglang_tp1.pt"
        _run_subprocess(
            _HELPERS_DIR / "run_audio_sglang.py",
            output_path=out,
            extra_env={
                "TP_SIZE": "1",
                "TP_RANK": "0",
                "NCCL_PORT": str(_allocate_free_port()),
                "MASTER_ADDR": "127.0.0.1",
            },
        )
        return out

    # tp_size > 1: spawn one subprocess per rank, share the NCCL port.
    nccl_port = str(_allocate_free_port())
    output_paths = [tmp_path / f"sglang_tp{tp_size}_rank{r}.pt" for r in range(tp_size)]

    procs: list[subprocess.Popen] = []
    log_paths: list[Path] = []
    log_handles: list = []
    for rank in range(tp_size):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT)
        env["MODEL_PATH"] = QWEN3_OMNI_MODEL
        env["OUTPUT_PATH"] = str(output_paths[rank])
        env["DTYPE"] = "bfloat16"
        env["TP_SIZE"] = str(tp_size)
        env["TP_RANK"] = str(rank)
        env["NCCL_PORT"] = nccl_port
        env["MASTER_ADDR"] = "127.0.0.1"
        env.setdefault("NCCL_SOCKET_IFNAME", "lo")
        env.setdefault("NCCL_IB_DISABLE", "1")
        env.setdefault("NCCL_P2P_DISABLE", "1")
        # Map each rank to a distinct physical CUDA device. The subprocess
        # treats it as cuda:0 internally because we only expose one.
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        # ``local_rank`` we pass to torch.distributed is the rank
        # identity (used by sglang's local-master checks), not a CUDA
        # device index — but pynccl by default does
        # ``torch.cuda.device(local_rank)`` which would crash on rank>0
        # because only one GPU is visible. This flag tells sglang's
        # GroupCoordinator to use device_id=0 regardless.
        env["SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"] = "true"

        # File-redirect rather than PIPE — see _run_subprocess docstring
        # for the pytest+NCCL-bootstrap deadlock reason.
        log_path = output_paths[rank].with_suffix(".log")
        log_paths.append(log_path)
        log_handle = open(log_path, "w")
        log_handles.append(log_handle)
        procs.append(
            subprocess.Popen(
                [sys.executable, "-u", str(_HELPERS_DIR / "run_audio_sglang.py")],
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
        )

    failures: list[str] = []
    for rank, proc in enumerate(procs):
        try:
            proc.wait(timeout=900)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            failures.append(f"rank {rank}: timeout")
        else:
            if proc.returncode != 0:
                failures.append(
                    f"rank {rank}: exit={proc.returncode}\n"
                    f"--- log ({log_paths[rank]}) ---\n{log_paths[rank].read_text()}"
                )
    for handle in log_handles:
        try:
            handle.close()
        except Exception:
            pass

    if failures:
        pytest.fail("tp_size=%d run failed:\n%s" % (tp_size, "\n---\n".join(failures)))

    return output_paths[0]  # leader-rank output


def test_audio_sglang_tp1_vs_tp2_numerical_parity(tmp_path):
    _skip_if_no_cuda()
    if torch.cuda.device_count() < 2:
        pytest.skip(f"requires >= 2 CUDA devices, have {torch.cuda.device_count()}")

    tp1_out = _spawn_tp_run(1, tmp_path)
    tp2_out = _spawn_tp_run(2, tmp_path)

    a = _load_audio_embeds(tp1_out)
    b = _load_audio_embeds(tp2_out)
    assert (
        a.shape == b.shape
    ), f"shape mismatch: tp1={tuple(a.shape)} tp2={tuple(b.shape)}"
    # tp_size=1 vs tp_size=2 differ in NCCL all-reduce reduction order
    # for bf16 across ~24 transformer layers, so widen atol a bit beyond
    # the local-vs-sglang tolerance (the latter doesn't have any cross-rank
    # all-reduce). Empirically ~18/213k elements differ by ~1.5e-2 max.
    torch.testing.assert_close(
        a.float(),
        b.float(),
        atol=5e-2,
        rtol=5e-2,
        msg=lambda raw: f"audio encoder tp_size=1 vs tp_size=2 mismatch:\n{raw}",
    )
