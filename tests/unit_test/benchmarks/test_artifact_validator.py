# SPDX-License-Identifier: Apache-2.0
"""Artifact gate tests: validate_mmmu_artifacts.py is a hard sweep gate.

The validator drives from sweep-status.jsonl. For every status row it
requires the cell directory to exist with the full bundle (mmmu_results
.json, preflight.json, launcher.log, stderr.log), the run_metadata to
carry all REQUIRED_FIELDS, the live metadata fields to be non-empty for
successful rows, and the status row's container_image_digest to be
non-empty and equal to the cell's run_metadata digest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


VALIDATOR = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "scripts"
    / "validate_mmmu_artifacts.py"
)


def _run_validator(out_root: Path, status_log: Path) -> tuple[int, str]:
    res = subprocess.run(
        [sys.executable, str(VALIDATOR), str(out_root), str(status_log)],
        capture_output=True,
        text=True,
    )
    return res.returncode, res.stdout + res.stderr


_DEFAULT_LAUNCH_CMD = [
    "docker", "run", "-d", "--name", "sglang-omni-hayden-benchmark",
    "frankleeeee/sglang-omni:dev",
    "sgl-omni", "serve", "--model-path", "/snapshot",
    "--text-only", "--port", "30000",
    "--mem-fraction-static", "0.9",
    "--disable-radix-cache",
]


def _write_complete_cell(
    cell_dir: Path,
    digest: str = "sha256:abc",
    *,
    launch_command: list[str] | None = _DEFAULT_LAUNCH_CMD,
    container_name: str = "sglang-omni-hayden-benchmark",
) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "summary": {},
        "speed": {},
        "config": {},
        "run_metadata": {
            "commit_sha": "deadbeef",
            "branch": "feat/mmmu-streaming-benchmark",
            "sglang_version": "0.5.8",
            "backend": "omni",
            "model_id": "qwen3-omni",
            "model_revision": "abc123",
            "dataset_revisions": {"MMMU/MMMU": "rev1"},
            "seed": 42,
            "ignore_eos": False,
            "lane": "A",
            "stream": True,
            "max_tokens": 2048,
            "max_concurrency": 8,
            "temperature": 0.0,
            "warmup": 5,
            "request_rate": None,
            "timeout_s": 300,
            "repo_id": None,
            "max_samples": None,
            "mem_fraction_static_configured": 0.9,
            "kv_cache_capacity_tokens": 123456,
            "steady_state_gpu_gb": [80.5],
            "prefix_cache_disabled": True,
            "encoder_patches_active": False,
            "host": "ion8-omni",
            "container_name": container_name,
            "container_image": "frankleeeee/sglang-omni:dev",
            "container_image_digest": digest,
            "server_port": 30000,
            "gpu_topology": "fake",
            "repetition_index": 0,
            "failure_count": 0,
        },
        "per_sample": [],
    }
    (cell_dir / "mmmu_results.json").write_text(json.dumps(result))
    preflight = {
        "containers": {
            container_name: (
                {
                    "container_image_digest": digest,
                    "container_image": "frankleeeee/sglang-omni:dev",
                    "launch_command": list(launch_command),
                }
                if launch_command is not None
                else {
                    "container_image_digest": digest,
                    "container_image": "frankleeeee/sglang-omni:dev",
                }
            )
        }
    }
    (cell_dir / "preflight.json").write_text(json.dumps(preflight))
    (cell_dir / "launcher.log").write_text("ready /snapshot/abc123\n")
    (cell_dir / "stderr.log").write_text("")


def _write_status(
    status_log: Path, rows: list[dict]
) -> None:
    # The validator requires every status row to carry a failure_log_path
    # key (the path to its stderr capture). Inject a default so each test
    # case doesn't have to repeat the field; tests that exercise the
    # missing/empty failure_log_path contract override it explicitly.
    normalized = []
    for row in rows:
        if "failure_log_path" not in row:
            row = {**row, "failure_log_path": str(Path(row.get("cell_dir", ".")) / "stderr.log")}
        normalized.append(row)
    status_log.write_text("\n".join(json.dumps(r) for r in normalized) + "\n")


def test_validator_passes_on_complete_bundle(tmp_path) -> None:
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "container_name": "sglang-omni-hayden-benchmark",
                "container_image": "frankleeeee/sglang-omni:dev",
                "server_port": 30000,
                "failure_count": 0,
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 0, out


def test_validator_fails_when_status_log_missing(tmp_path) -> None:
    out_root = tmp_path / "sweep"
    out_root.mkdir()
    status = tmp_path / "sweep-status.jsonl"
    # Status log empty / missing.
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "status log" in out


def test_validator_fails_on_missing_bundle_file(tmp_path) -> None:
    """Status row points at a cell that is missing preflight.json."""
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell)
    (cell / "preflight.json").unlink()  # remove a required file
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "preflight.json" in out


def test_validator_fails_on_empty_digest(tmp_path) -> None:
    """A successful cell with an empty status-row digest is rejected."""
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "digest" in out


def test_validator_fails_on_digest_mismatch(tmp_path) -> None:
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:OTHER",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "digest" in out


def test_validator_fails_when_preflight_missing_launch_command(tmp_path) -> None:
    """Launch-evidence enforcement: a cell whose retained preflight.json
    is missing `launch_command` for its container is rejected. This closes
    the evidence-loss path where the eval could silently fall back to CLI
    declarations.
    """
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc", launch_command=None)
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "container_name": "sglang-omni-hayden-benchmark",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "launch_command" in out


def test_validator_fails_when_launch_command_missing_disable_radix_cache(tmp_path) -> None:
    """preflight has launch_command but is missing --disable-radix-cache → fail."""
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    weak_cmd = [
        "docker", "run", "-d", "--name", "sglang-omni-hayden-benchmark",
        "frankleeeee/sglang-omni:dev",
        "sgl-omni", "serve", "--model-path", "/snapshot",
        "--mem-fraction-static", "0.9",
        # --disable-radix-cache absent
    ]
    _write_complete_cell(cell, digest="sha256:abc", launch_command=weak_cmd)
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "container_name": "sglang-omni-hayden-benchmark",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "disable-radix-cache" in out


def test_validator_fails_when_launch_command_missing_mem_fraction(tmp_path) -> None:
    """preflight launch_command is missing --mem-fraction-static → fail."""
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    weak_cmd = [
        "docker", "run", "-d", "--name", "sglang-omni-hayden-benchmark",
        "frankleeeee/sglang-omni:dev",
        "sgl-omni", "serve", "--model-path", "/snapshot",
        "--disable-radix-cache",
        # --mem-fraction-static absent
    ]
    _write_complete_cell(cell, digest="sha256:abc", launch_command=weak_cmd)
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "container_name": "sglang-omni-hayden-benchmark",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "mem-fraction-static" in out


def test_validator_fails_when_failed_row_has_empty_failure_log_path(tmp_path) -> None:
    """A failed status row with empty `failure_log_path` is rejected — the
    plan requires failed reps to surface their failure log path so
    silently-dropped failures get caught at the contract layer.
    """
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "failed",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "failure_count": 0,
                "failure_log_path": "",  # explicit empty
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "failure_log_path" in out


def test_validator_fails_when_failure_count_disagrees_between_row_and_metadata(tmp_path) -> None:
    """Status row failure_count != run_metadata failure_count → fail.

    This pattern indicates silently-retried failures that overwrote the
    failure evidence in run_metadata but left a stale count in the
    status row, or vice versa. The plan requires the two to agree.
    """
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "container_name": "sglang-omni-hayden-benchmark",
                "failure_count": 3,  # disagrees with metadata (which is 0)
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "failure_count" in out


def test_validator_fails_when_metadata_failure_count_disagrees_with_per_sample(tmp_path) -> None:
    """run_metadata.failure_count != count(per_sample where is_success=False) → fail.

    Catches the silently-retried-failure pattern at a different layer:
    the metadata claims k failures, but the persisted per-sample records
    show a different number not-successful.
    """
    out_root = tmp_path / "sweep"
    cell = out_root / "lane_A" / "omni" / "rep_0"
    _write_complete_cell(cell, digest="sha256:abc")
    # Inject inconsistent per_sample: 2 failed, but metadata.failure_count is 0.
    result_path = cell / "mmmu_results.json"
    data = json.loads(result_path.read_text())
    data["per_sample"] = [
        {"sample_id": "s0", "lane": "A", "is_success": False},
        {"sample_id": "s1", "lane": "A", "is_success": False},
        {"sample_id": "s2", "lane": "A", "is_success": True},
    ]
    result_path.write_text(json.dumps(data))
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell),
                "container_image_digest": "sha256:abc",
                "container_name": "sglang-omni-hayden-benchmark",
                "failure_count": 0,
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "per_sample" in out


def test_validator_fails_on_duplicate_dispatch_tuple(tmp_path) -> None:
    """Two status rows sharing (host, backend, lane, rep) → fail.

    Silently-retried reps that overwrote prior status rows would produce
    this shape. The validator must catch it.
    """
    out_root = tmp_path / "sweep"
    cell_a = out_root / "lane_A" / "omni" / "rep_0"
    cell_b = out_root / "lane_A" / "omni" / "rep_0_retry"
    _write_complete_cell(cell_a, digest="sha256:abc")
    _write_complete_cell(cell_b, digest="sha256:abc")
    status = tmp_path / "sweep-status.jsonl"
    base = {
        "host": "ion8-omni",
        "backend": "omni",
        "lane": "A",
        "rep": 0,
        "status": "success",
        "container_image_digest": "sha256:abc",
        "container_name": "sglang-omni-hayden-benchmark",
        "failure_count": 0,
    }
    _write_status(
        status,
        [
            {**base, "cell_dir": str(cell_a)},
            {**base, "cell_dir": str(cell_b)},  # duplicate tuple
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "duplicate" in out


def test_validator_fails_on_orphan_cell_with_no_status_row(tmp_path) -> None:
    """A cell on disk that no status row references is rejected."""
    out_root = tmp_path / "sweep"
    cell_known = out_root / "lane_A" / "omni" / "rep_0"
    cell_orphan = out_root / "lane_A" / "omni" / "rep_1"
    _write_complete_cell(cell_known, digest="sha256:abc")
    _write_complete_cell(cell_orphan, digest="sha256:def")
    status = tmp_path / "sweep-status.jsonl"
    _write_status(
        status,
        [
            {
                "host": "ion8-omni",
                "backend": "omni",
                "lane": "A",
                "rep": 0,
                "status": "success",
                "cell_dir": str(cell_known),
                "container_image_digest": "sha256:abc",
            }
        ],
    )
    rc, out = _run_validator(out_root, status)
    assert rc == 1
    assert "orphan" in out
