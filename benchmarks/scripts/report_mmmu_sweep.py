#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Single report-generation entry point for an MMMU sweep bundle.

The original plan's AC-10 Negative Test specifies that mixing Lane A and
Lane B per-sample records in the same input file must be rejected by the
report generator with a clear error. This module owns that guard and
exposes it as both a library function (``assert_single_lane``) and a
CLI.

The report generator is intentionally narrow: it reads retained
``mmmu_results.json`` cells, asserts that the per-sample records all
carry the same lane (and that the file-level ``run_metadata.lane``
agrees), and surfaces the loaded records to the caller. Anything more
elaborate (accuracy tables, latency tables, the issue #379 follow-up
comment body) is intentionally caller-driven so the strictly-required
plan obligation lives in a tiny, testable surface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


class MixedLaneError(ValueError):
    """Raised when per-sample MMMU records span more than one lane.

    Tracks the offending lanes and the first conflicting record so the
    error message can name what went wrong without dumping the full
    payload.
    """


def assert_single_lane(records: Iterable[dict]) -> str:
    """Assert that every per-sample record carries the same lane label.

    Returns the (single) lane string when records are consistent. Raises
    ``MixedLaneError`` when records carry more than one distinct lane,
    or when any record is missing the ``lane`` field entirely.

    The check is on the record layer, not the file-level metadata, so
    even a hand-merged file containing per-sample records from two
    different lanes is rejected.
    """
    records = list(records)
    if not records:
        raise MixedLaneError(
            "no per-sample records supplied; cannot validate lane purity"
        )
    seen: set[str] = set()
    for idx, record in enumerate(records):
        if "lane" not in record:
            raise MixedLaneError(
                f"per-sample record at index {idx} is missing the 'lane' field "
                f"(sample_id={record.get('sample_id', '<unknown>')!r}); "
                f"cannot prove lane purity"
            )
        seen.add(record["lane"])
        if len(seen) > 1:
            raise MixedLaneError(
                f"mixed lane records detected: input contains both "
                f"{sorted(seen)!r}; report generator refuses mixed Lane A / "
                f"Lane B per-sample input. First conflict at record index "
                f"{idx} (sample_id={record.get('sample_id', '<unknown>')!r}, "
                f"lane={record['lane']!r})"
            )
    return next(iter(seen))


def load_cell(result_path: Path) -> tuple[str, list[dict]]:
    """Load a retained ``mmmu_results.json`` and return ``(lane, per_sample)``.

    Cross-checks ``run_metadata.lane`` against the per-sample lane labels.
    Raises ``MixedLaneError`` if either layer is inconsistent.
    """
    data = json.loads(result_path.read_text())
    meta_lane = (data.get("run_metadata") or {}).get("lane")
    per_sample = data.get("per_sample") or []
    record_lane = assert_single_lane(per_sample) if per_sample else meta_lane
    if meta_lane and record_lane and meta_lane != record_lane:
        raise MixedLaneError(
            f"{result_path}: run_metadata.lane={meta_lane!r} disagrees with "
            f"per-sample record lane={record_lane!r}; refusing inconsistent input"
        )
    return (record_lane or meta_lane or ""), per_sample


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "cells",
        nargs="+",
        help="Paths to retained mmmu_results.json files to validate as one report input.",
    )
    args = parser.parse_args(argv)

    all_records: list[dict] = []
    for cell_path in args.cells:
        _, per_sample = load_cell(Path(cell_path))
        all_records.extend(per_sample)
    lane = assert_single_lane(all_records)
    print(f"[report] OK — {len(all_records)} records, lane={lane!r}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MixedLaneError as exc:
        print(f"[report] FAILED — {exc}", file=sys.stderr)
        sys.exit(2)
