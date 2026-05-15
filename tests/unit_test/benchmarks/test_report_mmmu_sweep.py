# SPDX-License-Identifier: Apache-2.0
"""Report generator tests: single-lane required, mixed-lane rejected.

The original plan's AC-10 Negative Test specifies that mixing Lane A and
Lane B per-sample records in the same input file must be rejected by the
report generator with a clear error. These tests lock that contract at
both the record-layer guard (``assert_single_lane``) and the file-layer
cell loader (``load_cell``).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.scripts.report_mmmu_sweep import (
    MixedLaneError,
    assert_single_lane,
    load_cell,
)


REPORT_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "scripts"
    / "report_mmmu_sweep.py"
)


def _record(sample_id: str, lane: str, is_success: bool = True) -> dict:
    return {"sample_id": sample_id, "lane": lane, "is_success": is_success}


def test_assert_single_lane_accepts_uniform_lane_a_records() -> None:
    records = [_record("s0", "A"), _record("s1", "A"), _record("s2", "A")]
    assert assert_single_lane(records) == "A"


def test_assert_single_lane_accepts_uniform_lane_b_records() -> None:
    records = [_record("s0", "B"), _record("s1", "B")]
    assert assert_single_lane(records) == "B"


def test_assert_single_lane_rejects_mixed_a_and_b() -> None:
    records = [_record("s0", "A"), _record("s1", "B"), _record("s2", "A")]
    with pytest.raises(MixedLaneError) as excinfo:
        assert_single_lane(records)
    msg = str(excinfo.value)
    assert "mixed lane" in msg
    # The error must name both offending lanes so the operator can find
    # the source of the mix quickly.
    assert "'A'" in msg and "'B'" in msg
    # The error must name the index of the first conflicting record.
    assert "index 1" in msg


def test_assert_single_lane_rejects_record_missing_lane_field() -> None:
    records = [_record("s0", "A"), {"sample_id": "s1", "is_success": True}]
    with pytest.raises(MixedLaneError) as excinfo:
        assert_single_lane(records)
    assert "missing the 'lane' field" in str(excinfo.value)


def test_assert_single_lane_rejects_empty_input() -> None:
    with pytest.raises(MixedLaneError):
        assert_single_lane([])


def test_load_cell_reads_lane_from_per_sample_and_cross_checks_metadata(tmp_path) -> None:
    cell = tmp_path / "lane_A" / "omni" / "rep_0"
    cell.mkdir(parents=True)
    data = {
        "run_metadata": {"lane": "A"},
        "per_sample": [_record("s0", "A"), _record("s1", "A")],
    }
    (cell / "mmmu_results.json").write_text(json.dumps(data))
    lane, records = load_cell(cell / "mmmu_results.json")
    assert lane == "A"
    assert len(records) == 2


def test_load_cell_rejects_metadata_vs_record_lane_disagreement(tmp_path) -> None:
    """``run_metadata.lane`` and per-sample lane disagree → reject.

    Catches the case where a file was hand-merged: the metadata says
    Lane A but the records were copied from a Lane B run.
    """
    cell = tmp_path / "lane_A" / "omni" / "rep_0"
    cell.mkdir(parents=True)
    data = {
        "run_metadata": {"lane": "A"},
        "per_sample": [_record("s0", "B"), _record("s1", "B")],
    }
    (cell / "mmmu_results.json").write_text(json.dumps(data))
    with pytest.raises(MixedLaneError) as excinfo:
        load_cell(cell / "mmmu_results.json")
    assert "disagrees with" in str(excinfo.value)


def test_build_mmmu_result_records_stamps_lane_on_each_record() -> None:
    """The record builder must persist the lane on every per-sample
    record so the report generator's mixed-lane guard can run at the
    record layer (not just the file-level metadata).
    """
    from benchmarks.benchmarker.data import RequestResult
    from benchmarks.dataset.mmmu import MMMUSample
    from benchmarks.tasks.visual_understand import build_mmmu_result_records

    samples = [
        MMMUSample(
            sample_id="s0",
            question="?",
            options=["x", "y", "z", "w"],
            answer="A",
            images=[],
            subject="art",
            prompt="?",
            all_choices=["A", "B", "C", "D"],
            index2ans={"A": "x", "B": "y", "C": "z", "D": "w"},
            question_type="multiple-choice",
        ),
    ]
    results = [
        RequestResult(
            is_success=True,
            text="A",
            prompt_tokens=10,
            completion_tokens=1,
            latency_s=0.5,
        )
    ]
    records = build_mmmu_result_records(samples, results, lane="B")
    assert len(records) == 1
    assert records[0]["lane"] == "B"


def test_report_cli_accepts_single_lane_bundle(tmp_path) -> None:
    """The CLI entry point reads a list of cells, validates lane purity,
    and exits 0 on a clean bundle."""
    cell = tmp_path / "lane_A" / "omni" / "rep_0"
    cell.mkdir(parents=True)
    data = {
        "run_metadata": {"lane": "A"},
        "per_sample": [_record("s0", "A"), _record("s1", "A")],
    }
    (cell / "mmmu_results.json").write_text(json.dumps(data))
    result = subprocess.run(
        [sys.executable, str(REPORT_SCRIPT), str(cell / "mmmu_results.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "lane='A'" in result.stdout


def test_report_cli_rejects_mixed_lane_bundle(tmp_path) -> None:
    """The CLI entry point exits non-zero when two cells from different
    lanes are loaded into one report payload — exactly the case the plan
    specifies must be rejected."""
    cell_a = tmp_path / "lane_A" / "omni" / "rep_0"
    cell_a.mkdir(parents=True)
    (cell_a / "mmmu_results.json").write_text(
        json.dumps(
            {
                "run_metadata": {"lane": "A"},
                "per_sample": [_record("s0", "A")],
            }
        )
    )
    cell_b = tmp_path / "lane_B" / "omni" / "rep_0"
    cell_b.mkdir(parents=True)
    (cell_b / "mmmu_results.json").write_text(
        json.dumps(
            {
                "run_metadata": {"lane": "B"},
                "per_sample": [_record("s0", "B")],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPORT_SCRIPT),
            str(cell_a / "mmmu_results.json"),
            str(cell_b / "mmmu_results.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mixed lane" in result.stderr
