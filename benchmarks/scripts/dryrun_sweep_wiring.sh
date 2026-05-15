#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# End-to-end dry-run of the MMMU sweep orchestration.
#
# Invokes the REAL `run_mmmu_sweep.sh` with PATH-prepended `ssh` and `scp`
# shims that simulate a remote H200 by writing synthetic preflight.json /
# launcher.log / mmmu_results.json into a tempdir acting as the remote
# filesystem. The shimmed `scp` then copies those files back into the
# sweep's local output tree, exactly as a real H200 run would. The sweep
# script finally invokes the real `validate_mmmu_artifacts.py` and exits
# non-zero on any bundle defect.
#
# This exercises the full chain end-to-end:
#   run_mmmu_sweep.sh -> ssh preflight -> scp preflight back ->
#   ssh benchmark per cell -> scp results + preflight + launcher.log back ->
#   sweep-status.jsonl row build -> real validate_mmmu_artifacts.py gate.
#
# Runs in ~1s on any dev box. No docker, no GPU, no real SSH required.
# Tests post-tamper the produced bundle and re-run the validator to
# prove failure detection on real defects (missing launcher.log, mangled
# digest), not on a stubbed validator.
#
# Usage:
#   bash benchmarks/scripts/dryrun_sweep_wiring.sh [--keep] [--out <path>]
#
# Flags:
#   --keep         do not clean up the tempdir on exit. Useful for tests
#                  that need to inspect or tamper with the resulting bundle.
#   --out <path>   write the sweep output to <path> instead of a tempdir.
#                  When set, the bundle survives the script's exit.
#
# Exit codes:
#   0 = orchestration succeeded end-to-end and the validator gate passed
#   non-zero = the real sweep script or validator reported failure;
#              the wiring is broken (look at stderr for the validator's
#              `[validate] FAILED` block).

set -euo pipefail

KEEP=0
OUT_ROOT_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep) KEEP=1; shift;;
        --out) OUT_ROOT_ARG="$2"; shift 2;;
        -h|--help)
            sed -n '2,40p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ORCH_ROOT="$(mktemp -d)"
if [[ "$KEEP" -eq 0 ]]; then
    trap 'rm -rf "$ORCH_ROOT"' EXIT
fi

FAKE_FS="$ORCH_ROOT/fake-remote-fs"
SHIM_BIN="$ORCH_ROOT/bin"
SYNTH_PY="$ORCH_ROOT/synth.py"

if [[ -n "$OUT_ROOT_ARG" ]]; then
    SWEEP_OUT="$OUT_ROOT_ARG"
    mkdir -p "$(dirname "$SWEEP_OUT")"
else
    SWEEP_OUT="$ORCH_ROOT/sweep-out"
fi

mkdir -p "$FAKE_FS/tmp" "$SHIM_BIN"

# ----------------------------------------------------- synthetic-bundle helper
#
# Emits the same JSON shape the real eval harness emits, with all
# REQUIRED_FIELDS populated and non-empty LIVE_REQUIRED values so the
# validator's success-row contract passes. Kept in Python because hand-rolling
# the run_metadata block in shell is more error-prone than a JSON dump.

cat > "$SYNTH_PY" << 'PY_EOF'
"""Synthesize fake preflight.json or mmmu_results.json for the orchestration dry-run."""

from __future__ import annotations

import json
import os
import sys

DIGEST = "sha256:dryrun0000000000000000000000000000000000000000000000000000000000"


def _container_record(name: str, image: str, port: int) -> dict:
    if "omni" in name:
        launch = [
            "docker", "run", "-d", "--name", name, image,
            "sgl-omni", "serve", "--model-path", "/snapshot",
            "--text-only", "--port", str(port),
            "--mem-fraction-static", "0.9",
            "--disable-radix-cache",
        ]
    else:
        launch = [
            "docker", "run", "-d", "--name", name, image,
            "python", "-m", "sglang.launch_server", "--model-path", "/snapshot",
            "--port", str(port),
            "--mem-fraction-static", "0.9",
            "--disable-radix-cache",
        ]
    return {
        "container_image_digest": DIGEST,
        "container_image": image,
        "launch_command": launch,
    }


def write_preflight(out_path: str) -> None:
    body = {
        "ok": True,
        "containers": {
            "sglang-omni-hayden-benchmark": _container_record(
                "sglang-omni-hayden-benchmark",
                "frankleeeee/sglang-omni:dev",
                30000,
            ),
            "sglang-hayden-benchmark": _container_record(
                "sglang-hayden-benchmark",
                "lmsysorg/sglang",
                30001,
            ),
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(body, f, indent=2)


def write_results(out_dir: str, backend: str, lane: str, rep_idx: int, host: str) -> None:
    if backend == "omni":
        container_name = "sglang-omni-hayden-benchmark"
        container_image = "frankleeeee/sglang-omni:dev"
        port = 30000
        model_id = "qwen3-omni"
    else:
        container_name = "sglang-hayden-benchmark"
        container_image = "lmsysorg/sglang"
        port = 30001
        model_id = "qwen3-vl"
    body = {
        "summary": {},
        "speed": {},
        "config": {},
        "run_metadata": {
            "commit_sha": "dryrun-deadbeef",
            "branch": "dryrun-branch",
            "sglang_version": "0.5.8",
            "backend": backend,
            "model_id": model_id,
            "model_revision": "dryrun-model-rev",
            "dataset_revisions": {"MMMU/MMMU": "dryrun-dataset-rev"},
            "seed": 42,
            "ignore_eos": lane == "B",
            "lane": lane,
            "stream": True,
            "max_tokens": 256 if lane == "B" else 2048,
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
            "host": host,
            "container_name": container_name,
            "container_image": container_image,
            "container_image_digest": DIGEST,
            "server_port": port,
            "gpu_topology": "dryrun-topology",
            "repetition_index": rep_idx,
            "failure_count": 0,
        },
        # Two synthetic successful per-sample records so the validator's
        # failure-accounting cross-check (failure_count == count(is_success
        # is false)) is exercised end-to-end against a non-empty list, not
        # just skipped on the empty path.
        "per_sample": [
            {
                "sample_id": f"dryrun-{lane}-{backend}-{rep_idx}-0",
                "lane": lane,
                "is_success": True,
            },
            {
                "sample_id": f"dryrun-{lane}-{backend}-{rep_idx}-1",
                "lane": lane,
                "is_success": True,
            },
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "mmmu_results.json"), "w") as f:
        json.dump(body, f, indent=2)
    open(os.path.join(out_dir, "stderr.log"), "w").close()


def main() -> int:
    mode = sys.argv[1]
    if mode == "preflight":
        write_preflight(sys.argv[2])
    elif mode == "results":
        out_dir = sys.argv[2]
        backend = sys.argv[3]
        lane = sys.argv[4]
        rep_idx = int(sys.argv[5])
        host = sys.argv[6]
        write_results(out_dir, backend, lane, rep_idx, host)
    else:
        print(f"unknown mode: {mode}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
PY_EOF

# ------------------------------------------------------------------- shim: ssh
#
# Recognizes the exact patterns `run_mmmu_sweep.sh` emits and writes the
# expected output files into FAKE_FS at the absolute paths the orchestrator
# would expect on the H200 host. Falls back to error on anything else, so a
# future refactor that emits a new ssh pattern fails the dry-run loudly.

cat > "$SHIM_BIN/ssh" << 'SSH_EOF'
#!/usr/bin/env bash
set -euo pipefail
# Strip ssh options. `-o OPT`, `-i KEY`, `-p PORT`, `-F CFG`, `-l USER`,
# `-L/-R/-D/-J/-P` all take a value; bare flags consume one slot.
while [[ "${1:-}" == -* ]]; do
    case "$1" in
        -o|-i|-p|-F|-l|-L|-R|-D|-J|-P) shift 2;;
        *) shift;;
    esac
done
host="${1:-}"; shift || true
cmd="${*:-}"

# Reachability probe: `ssh ... <host> true`.
if [[ "$cmd" == "true" ]]; then exit 0; fi

# Extract the token immediately following `<key>` in cmd by word iteration.
extract_arg() {
    local key="$1"
    local found=0
    local tok
    for tok in $cmd; do
        if [[ "$found" -eq 1 ]]; then
            echo "$tok"
            return 0
        fi
        if [[ "$tok" == "$key" ]]; then
            found=1
        fi
    done
    return 0
}

case "$cmd" in
    *preflight_mmmu_sweep*)
        out_path=$(extract_arg "--output")
        log_omni=$(extract_arg "--launcher-log-omni")
        log_sglang=$(extract_arg "--launcher-log-sglang")
        if [[ -z "$out_path" ]]; then
            echo "fake-ssh: preflight invocation has no --output" >&2
            exit 3
        fi
        python3 "$DRYRUN_HELPER" preflight "${DRYRUN_FAKE_FS}${out_path}"
        # The orchestrator scps these launcher log paths back after the
        # cells run, so they must exist on the fake remote FS.
        for log_path in "$log_omni" "$log_sglang"; do
            if [[ -n "$log_path" ]]; then
                mkdir -p "${DRYRUN_FAKE_FS}$(dirname "$log_path")"
                printf '[dryrun] launcher.log: server ready\n[dryrun] KV cache capacity: 123456 tokens\n' \
                    > "${DRYRUN_FAKE_FS}${log_path}"
            fi
        done
        ;;
    *benchmark_omni_mmmu*)
        out_dir=$(extract_arg "--output-dir")
        backend=$(extract_arg "--backend")
        lane=$(extract_arg "--lane")
        rep_idx=$(extract_arg "--repetition-index")
        if [[ -z "$out_dir" || -z "$backend" || -z "$lane" || -z "$rep_idx" ]]; then
            echo "fake-ssh: benchmark invocation missing one of --output-dir/--backend/--lane/--repetition-index" >&2
            exit 3
        fi
        python3 "$DRYRUN_HELPER" results \
            "${DRYRUN_FAKE_FS}${out_dir}" "$backend" "$lane" "$rep_idx" "$host"
        ;;
    *"mkdir -p"*)
        # Allow `mkdir -p <dir>` and chained `mkdir -p <dir> && rm -rf <dir>/*`.
        # Rewrite absolute paths into the fake FS and exec locally.
        rewritten=$(printf '%s' "$cmd" | sed "s|/tmp/|${DRYRUN_FAKE_FS}/tmp/|g")
        bash -c "$rewritten" 2>/dev/null || true
        ;;
    *"rm -rf"*)
        rewritten=$(printf '%s' "$cmd" | sed "s|/tmp/|${DRYRUN_FAKE_FS}/tmp/|g")
        bash -c "$rewritten" 2>/dev/null || true
        ;;
    *)
        echo "fake-ssh: unsupported command pattern: $cmd" >&2
        exit 4
        ;;
esac
SSH_EOF
chmod +x "$SHIM_BIN/ssh"

# ------------------------------------------------------------------- shim: scp
#
# `scp -q -r <host>:/p/. <local>/`  → copy fake-remote tree to local
# `scp -q <host>:/p/file <local>`   → copy fake-remote file to local
# Discards options, then copies from FAKE_FS+<path> to dst.

cat > "$SHIM_BIN/scp" << 'SCP_EOF'
#!/usr/bin/env bash
set -euo pipefail
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -q|-r|-3|-4|-6|-B|-C|-O|-p|-T|-v) shift;;
        -o|-i|-c|-F|-J|-l|-P|-S|-X) shift 2;;
        *) args+=("$1"); shift;;
    esac
done
if [[ ${#args[@]} -lt 2 ]]; then
    echo "fake-scp: need src + dst args; got ${args[*]}" >&2
    exit 2
fi
src="${args[0]}"
dst="${args[1]}"
remote_path="${src#*:}"
case "$remote_path" in
    */.)
        # Directory-with-trailing-dot copy.
        base="${remote_path%/.}"
        mkdir -p "$dst" 2>/dev/null || true
        if [[ -d "${DRYRUN_FAKE_FS}${base}" ]]; then
            cp -r "${DRYRUN_FAKE_FS}${base}"/. "${dst}/" 2>/dev/null || true
        fi
        ;;
    *)
        mkdir -p "$(dirname "$dst")" 2>/dev/null || true
        if [[ -e "${DRYRUN_FAKE_FS}${remote_path}" ]]; then
            cp -r "${DRYRUN_FAKE_FS}${remote_path}" "$dst" 2>/dev/null || true
        fi
        ;;
esac
SCP_EOF
chmod +x "$SHIM_BIN/scp"

# Export the locations the shims read from, then prepend the shim dir to PATH
# so `ssh` and `scp` resolve to our shims for the duration of the sweep run.
export DRYRUN_FAKE_FS="$FAKE_FS"
export DRYRUN_HELPER="$SYNTH_PY"
export PATH="$SHIM_BIN:$PATH"

# Invoke the real sweep script. `--serial --lanes both --reps 1` produces
# 2 lanes * 2 backends * 1 rep = 4 cells, exercising both backends and both
# lane semantics without needing the parallel-by-lane branch (which would
# fork two background processes and race on the fake FS).
echo "[dryrun] invoking real run_mmmu_sweep.sh against shim ssh/scp..."
bash "$REPO_ROOT/benchmarks/scripts/run_mmmu_sweep.sh" \
    --reps 1 --lanes both --serial \
    --out "$SWEEP_OUT" \
    --host-lane-a fake-host-a --host-lane-b fake-host-b

echo "[dryrun] sweep + validator both succeeded against shimmed environment"
echo "[dryrun] bundle at $SWEEP_OUT"
echo "[dryrun] wiring OK"
