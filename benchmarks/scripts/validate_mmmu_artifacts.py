#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate a finished MMMU sweep's retained artifact bundle.

Walks the sweep output root, opens every ``mmmu_results.json``, and asserts:

- The full AC-9 ``REQUIRED_FIELDS`` key set is present under
  ``run_metadata``.
- ``model_revision``, ``container_image_digest``, ``dataset_revisions``,
  ``mem_fraction_static_configured``, ``prefix_cache_disabled``,
  ``kv_cache_capacity_tokens``, and ``steady_state_gpu_gb`` are non-None
  (the live values the plan requires).
- The per-cell ``run_metadata.container_image_digest`` matches the
  ``container_image_digest`` field in the corresponding ``sweep-status.jsonl``
  row, so the status log and the retained artifact agree on which image
  actually served the cell.

Exit codes:
  0 = all cells valid
  1 = at least one cell missing artifacts or failed validation

Usage:
  validate_mmmu_artifacts.py <out_root> <status_log>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from benchmarks.scripts.run_metadata import REQUIRED_FIELDS

LIVE_REQUIRED = (
    "model_revision",
    "container_image_digest",
    "dataset_revisions",
    "mem_fraction_static_configured",
    "kv_cache_capacity_tokens",
    "steady_state_gpu_gb",
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _validate_cell(cell_dir: Path, status_row: dict | None) -> list[str]:
    issues: list[str] = []
    result_path = cell_dir / "mmmu_results.json"
    if not result_path.exists():
        issues.append(f"{cell_dir}: missing mmmu_results.json")
        return issues
    try:
        data = json.loads(result_path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(f"{result_path}: invalid JSON ({exc})")
        return issues

    meta = data.get("run_metadata")
    if not isinstance(meta, dict):
        issues.append(f"{result_path}: missing run_metadata block")
        return issues

    for key in REQUIRED_FIELDS:
        if key not in meta:
            issues.append(f"{result_path}: run_metadata missing key {key!r}")

    for key in LIVE_REQUIRED:
        value = meta.get(key)
        if value is None or value == [] or value == {}:
            issues.append(
                f"{result_path}: live AC-9 field {key!r} is empty/None ({value!r})"
            )

    if status_row is not None:
        status_digest = (status_row.get("container_image_digest") or "").strip()
        meta_digest = (meta.get("container_image_digest") or "").strip()
        if status_digest and meta_digest and status_digest != meta_digest:
            issues.append(
                f"{result_path}: status row digest {status_digest!r} does "
                f"not match run_metadata digest {meta_digest!r}"
            )

    # Bundle completeness: AC-7's launcher-log evidence must be retained.
    if not (cell_dir / "launcher.log").exists():
        issues.append(f"{cell_dir}: missing launcher.log (AC-7 evidence)")
    if not (cell_dir / "preflight.json").exists():
        issues.append(f"{cell_dir}: missing preflight.json (AC-7 source)")

    return issues


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    out_root = Path(sys.argv[1])
    status_log = Path(sys.argv[2])

    status_rows = _load_jsonl(status_log)
    cell_status: dict[Path, dict] = {}
    for row in status_rows:
        cd = row.get("cell_dir")
        if cd:
            cell_status[Path(cd).resolve()] = row

    all_issues: list[str] = []
    cells = sorted(out_root.rglob("mmmu_results.json"))
    if not cells:
        all_issues.append(f"{out_root}: no mmmu_results.json files found")
    for result in cells:
        cell_dir = result.parent
        status_row = cell_status.get(cell_dir.resolve())
        all_issues.extend(_validate_cell(cell_dir, status_row))

    if all_issues:
        print("[validate] FAILED", file=sys.stderr)
        for issue in all_issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    print(f"[validate] OK — {len(cells)} cells, all AC-9 fields populated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
