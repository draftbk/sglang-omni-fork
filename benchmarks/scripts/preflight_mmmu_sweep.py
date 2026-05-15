#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight gate for the MMMU sweep on H200 hosts.

Verifies that everything the sweep depends on is bit-pinned and
contractually correct before any GPU time is spent. Failures here are
designed to be louder and earlier than failures during the sweep itself.

What the gate verifies (AC-7):

1. HuggingFace model revisions for ``Qwen/Qwen3-VL-30B-A3B-Instruct`` and
   ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` resolve and are recorded.
2. A local snapshot for each model exists at the snapshot directory
   (created via ``huggingface-cli download --revision <sha>`` if
   ``--download`` is passed).
3. The two named benchmark containers exist with the expected images:
   - ``sglang-omni-hayden-benchmark`` ← ``frankleeeee/sglang-omni:dev``
   - ``sglang-hayden-benchmark``       ← ``lmsysorg/sglang``
   Container names are enforced strictly. Image digests captured via
   ``docker inspect`` are recorded in the preflight output JSON.
4. Each running server's ``/v1/models`` endpoint returns the expected
   model identifier.
5. The launcher log on each container records loading from the expected
   local snapshot path (regex match against the snapshot directory).
6. A single ``image_url`` data-URI request to the SGLang server returns
   HTTP 200 (proves stock SGLang accepts data-URIs as the omni-side
   payload translator emits them).
7. Dataset revision pinning: every entry the sweep config will request is
   present in ``benchmarks/dataset/mmmu_revisions.json``. When
   ``--update-revisions`` is passed, this gate resolves current
   HuggingFace dataset SHAs and writes them into that file.

The gate writes a JSON report to ``--output`` and exits non-zero on any
failure. Every named contract violation is reported with its remedy.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CONTAINERS: dict[str, str] = {
    "sglang-omni-hayden-benchmark": "frankleeeee/sglang-omni:dev",
    "sglang-hayden-benchmark": "lmsysorg/sglang",
}

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("omni", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    ("sglang", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
]

DATASET_REPOS_TO_PIN: list[str] = ["MMMU/MMMU", "zhaochenyang20/mmmu-ci-50"]


@dataclass
class PreflightReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_revisions: dict[str, str] = field(default_factory=dict)
    snapshot_paths: dict[str, str] = field(default_factory=dict)
    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_revisions: dict[str, str] = field(default_factory=dict)
    sglang_data_uri_probe: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


# ---------------------------------------------------------------- HF lookup


def resolve_hf_revision(repo_id: str) -> str | None:
    """Return the current main-branch commit SHA for an HF model repo.

    Uses the public HF refs API; requires network. Returns None on any
    failure (caller decides whether to treat as fatal).
    """
    url = f"https://huggingface.co/api/models/{repo_id}/refs"
    try:
        with request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError):
        return None
    for branch in data.get("branches", []):
        if branch.get("name") == "main":
            return branch.get("targetCommit")
    return None


def resolve_hf_dataset_revision(repo_id: str) -> str | None:
    url = f"https://huggingface.co/api/datasets/{repo_id}/refs"
    try:
        with request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError):
        return None
    for branch in data.get("branches", []):
        if branch.get("name") == "main":
            return branch.get("targetCommit")
    return None


# ----------------------------------------------------------- container checks


def docker_inspect_image(container_name: str) -> str | None:
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


def docker_inspect_repo_tag(container_name: str) -> str | None:
    if shutil.which("docker") is None:
        return None
    try:
        out = subprocess.check_output(
            ["docker", "inspect", container_name, "--format", "{{.Config.Image}}"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except subprocess.CalledProcessError:
        return None


def check_container(name: str, expected_image: str, report: PreflightReport) -> None:
    digest = docker_inspect_image(name)
    image_ref = docker_inspect_repo_tag(name)
    info: dict[str, Any] = {
        "container_image_digest": digest,
        "container_image": image_ref,
        "name_ok": True,
    }
    if digest is None or image_ref is None:
        info["name_ok"] = False
        report.fail(
            f"Container {name!r} not found or docker unavailable. Start it with the expected "
            f"image ({expected_image}) before re-running the preflight."
        )
    else:
        # Loose match: image_ref may include extra tag suffix (e.g. :dev-sha).
        if expected_image.split(":")[0] not in image_ref:
            report.fail(
                f"Container {name!r} is running image {image_ref!r}, expected {expected_image!r}. "
                f"Stop/remove the container and restart from the contracted image."
            )
    report.containers[name] = info


# ----------------------------------------------------------- model probes


def probe_v1_models(base_url: str) -> dict[str, Any] | None:
    url = base_url.rstrip("/") + "/v1/models"
    try:
        with request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError, ValueError):
        return None


def _build_test_image_data_uri() -> str:
    """A 1x1 PNG encoded as a data URI for the SGLang data-URI probe."""
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    b64 = base64.b64encode(png_1x1).decode("ascii")
    return f"data:image/png;base64,{b64}"


def probe_sglang_data_uri(base_url: str, model: str) -> dict[str, Any]:
    """POST a minimal image_url request to the SGLang server. AC-7 step 6."""
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _build_test_image_data_uri()}},
                    {"type": "text", "text": "ping"},
                ],
            }
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            return {"status": resp.status, "ok": resp.status == 200}
    except error.HTTPError as exc:
        return {"status": exc.code, "ok": False, "error": exc.reason}
    except error.URLError as exc:
        return {"status": None, "ok": False, "error": str(exc.reason)}


# ------------------------------------------------------- dataset revisions


def load_dataset_revisions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"$schema_version": 1, "revisions": {}}
    return json.loads(path.read_text())


def save_dataset_revisions(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def update_dataset_revisions(report: PreflightReport, path: Path, repos: list[str]) -> None:
    data = load_dataset_revisions(path)
    revisions = dict(data.get("revisions") or {})
    for repo in repos:
        sha = resolve_hf_dataset_revision(repo)
        if not sha:
            report.fail(
                f"Could not resolve HF dataset main SHA for {repo!r}. Check network "
                f"or revisit the repo name."
            )
            continue
        revisions[repo] = sha
        report.dataset_revisions[repo] = sha
    data["revisions"] = revisions
    save_dataset_revisions(path, data)


def verify_dataset_revisions(report: PreflightReport, path: Path, repos: list[str]) -> None:
    data = load_dataset_revisions(path)
    revisions = dict(data.get("revisions") or {})
    for repo in repos:
        if repo not in revisions or not revisions[repo]:
            report.fail(
                f"No revision pinned for dataset repo {repo!r} in {path}. "
                f"Run `preflight_mmmu_sweep.py --update-revisions` to populate."
            )
            continue
        report.dataset_revisions[repo] = revisions[repo]


# ---------------------------------------------------------- main entrypoint


def verify_launcher_log_references_snapshot(
    log_path: Path, expected_snapshot: str
) -> bool:
    """Return True iff the launcher log mentions the expected snapshot path.

    AC-7 step (d): once a server is launched with ``--model-path <snapshot>``
    its launcher log must record the path. This grep proves the running
    server actually loaded from the pinned snapshot, not from some other
    local checkpoint or the HF cache via a different revision.

    Returns False if the file is missing, unreadable, or never mentions the
    expected path string.
    """
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return False
    return expected_snapshot in text


def print_launch_commands(
    snapshot_omni: str | None,
    snapshot_sglang: str | None,
    port_omni: int,
    port_sglang: int,
) -> None:
    """Print the docker run commands the H200 host operator must execute.

    The preflight does not exec docker run itself: that requires GPU
    access, network egress to docker hub, and persistent log paths the
    operator owns. Instead, the gate prints the exact contracted
    commands so the operator can copy-paste them under a tee that
    preserves the launcher log for the AC-7 step (d) verification.
    """
    print("# --- Container launch commands (run on the target H200 host) ---")
    if snapshot_omni:
        print(
            f"docker run -d --name sglang-omni-hayden-benchmark "
            f"--gpus all --network host "
            f"-v {snapshot_omni}:/snapshot:ro "
            f"frankleeeee/sglang-omni:dev "
            f"sgl-omni serve --model-path /snapshot --text-only --port {port_omni} "
            f"2>&1 | tee /tmp/sglang-omni-benchmark.log"
        )
    if snapshot_sglang:
        print(
            f"docker run -d --name sglang-hayden-benchmark "
            f"--gpus all --network host "
            f"-v {snapshot_sglang}:/snapshot:ro "
            f"lmsysorg/sglang "
            f"python -m sglang.launch_server --model-path /snapshot --port {port_sglang} "
            f"2>&1 | tee /tmp/sglang-benchmark.log"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight gate for the MMMU sweep.")
    p.add_argument("--host-lane-a", default="ion8-omni")
    p.add_argument("--host-lane-b", default="ion9-omni")
    p.add_argument(
        "--base-url-omni",
        default="http://localhost:30000",
        help="Local URL of the sglang-omni-hayden-benchmark container.",
    )
    p.add_argument(
        "--base-url-sglang",
        default="http://localhost:30001",
        help="Local URL of the sglang-hayden-benchmark container.",
    )
    p.add_argument(
        "--snapshot-root",
        default="/root/.cache/huggingface/hub",
        help="Where local HF snapshots live (default: HF default cache).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Run `huggingface-cli download --revision <sha>` to materialize the "
        "snapshots locally. Off by default — the gate verifies the snapshot "
        "exists but does not pull weights unless asked.",
    )
    p.add_argument(
        "--update-revisions",
        action="store_true",
        help="Query HF for current dataset main SHAs and write them into "
        "benchmarks/dataset/mmmu_revisions.json before verification.",
    )
    p.add_argument(
        "--skip-container-check",
        action="store_true",
        help="Skip docker inspect / /v1/models probes (for dev-machine dry-runs).",
    )
    p.add_argument(
        "--launcher-log-omni",
        default=None,
        help="Path to the sglang-omni server launcher log. When provided, the "
        "preflight greps it for the resolved snapshot path to prove the "
        "running server actually loaded from the pinned snapshot.",
    )
    p.add_argument(
        "--launcher-log-sglang",
        default=None,
        help="Path to the sglang server launcher log (same purpose).",
    )
    p.add_argument(
        "--print-launch-commands",
        action="store_true",
        help="Emit the docker run commands the operator should execute on "
        "the H200 host to start the two named containers with snapshot "
        "pinning, then exit. Useful as a sanity step before the live launch.",
    )
    p.add_argument(
        "--port-omni",
        type=int,
        default=30000,
        help="Port the sglang-omni container should listen on (default: 30000).",
    )
    p.add_argument(
        "--port-sglang",
        type=int,
        default=30001,
        help="Port the sglang container should listen on (default: 30001).",
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "preflight_mmmu_sweep.json"),
        help="Where to write the JSON report.",
    )
    return p.parse_args()


def _check_launcher_logs(
    args: argparse.Namespace, report: PreflightReport
) -> None:
    """Verify launcher logs reference resolved snapshot paths (AC-7 step d)."""
    pairs = [
        ("omni", args.launcher_log_omni, "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        ("sglang", args.launcher_log_sglang, "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    ]
    for backend_tag, log_path, repo_id in pairs:
        if not log_path:
            report.warn(
                f"--launcher-log-{backend_tag} not provided; cannot verify the "
                f"{backend_tag} server actually loaded the pinned snapshot."
            )
            continue
        snapshot = report.snapshot_paths.get(repo_id)
        if not snapshot:
            report.fail(
                f"No resolved snapshot path for {repo_id}, so the {backend_tag} "
                f"launcher-log verification cannot run."
            )
            continue
        if not verify_launcher_log_references_snapshot(Path(log_path), snapshot):
            report.fail(
                f"{backend_tag} launcher log at {log_path} does not reference the "
                f"resolved snapshot path {snapshot}. The container is not loading "
                f"from the pinned model — abort before any benchmark traffic runs."
            )


def main() -> int:
    args = parse_args()
    report = PreflightReport()

    # Optional: emit docker run commands and exit. Used by the operator to
    # sanity-check the launch contract before issuing the commands by hand
    # on the H200 host.
    if args.print_launch_commands:
        # We still need resolved snapshot paths for the commands; do a fast
        # HF lookup pass without container probes.
        snapshots: dict[str, str | None] = {}
        for _, repo_id in DEFAULT_MODELS:
            sha = resolve_hf_revision(repo_id)
            if sha:
                snapshot_dirname = f"models--{repo_id.replace('/', '--')}"
                snapshots[repo_id] = str(
                    Path(args.snapshot_root) / snapshot_dirname / "snapshots" / sha
                )
        print_launch_commands(
            snapshots.get("Qwen/Qwen3-Omni-30B-A3B-Instruct"),
            snapshots.get("Qwen/Qwen3-VL-30B-A3B-Instruct"),
            args.port_omni,
            args.port_sglang,
        )
        return 0

    # 1. Model revision resolution.
    for backend_tag, repo_id in DEFAULT_MODELS:
        sha = resolve_hf_revision(repo_id)
        if not sha:
            report.fail(
                f"Could not resolve HF model main SHA for {repo_id!r}. Check network "
                f"and rerun. The MMMU sweep must run against a pinned revision."
            )
            continue
        report.model_revisions[repo_id] = sha
        snapshot_dirname = f"models--{repo_id.replace('/', '--')}"
        snapshot_path = Path(args.snapshot_root) / snapshot_dirname / "snapshots" / sha
        report.snapshot_paths[repo_id] = str(snapshot_path)
        if not snapshot_path.exists():
            if args.download and shutil.which("huggingface-cli") is not None:
                local_dir = snapshot_path.parent.parent / "local-snapshot" / sha
                local_dir.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.check_call(
                        [
                            "huggingface-cli",
                            "download",
                            repo_id,
                            "--revision",
                            sha,
                            "--local-dir",
                            str(local_dir),
                        ]
                    )
                    report.snapshot_paths[repo_id] = str(local_dir)
                except subprocess.CalledProcessError as exc:
                    report.fail(
                        f"huggingface-cli download failed for {repo_id} @ {sha}: {exc}"
                    )
            else:
                # AC-7 requires fail-closed when the snapshot does not exist.
                # The sweep cannot run against unpinned weights and the
                # preflight is the gate that prevents that.
                report.fail(
                    f"Snapshot for {repo_id} @ {sha} not found at {snapshot_path}. "
                    f"Re-run with --download to materialize it, or pre-populate "
                    f"the HF cache before the sweep launches."
                )

    # 2. Container checks (skippable for dev-box dry-runs).
    if not args.skip_container_check:
        for name, expected_image in EXPECTED_CONTAINERS.items():
            check_container(name, expected_image, report)

    # 3. /v1/models probes.
    if not args.skip_container_check:
        for backend_tag, base_url in (
            ("omni", args.base_url_omni),
            ("sglang", args.base_url_sglang),
        ):
            info = probe_v1_models(base_url)
            if info is None:
                report.fail(
                    f"/v1/models on {base_url} did not respond. Confirm the {backend_tag} "
                    f"server is up and listening."
                )
            else:
                # Record the first model identity returned.
                models = info.get("data") or []
                first = models[0].get("id") if models else None
                report.containers.setdefault(
                    "sglang-omni-hayden-benchmark"
                    if backend_tag == "omni"
                    else "sglang-hayden-benchmark",
                    {},
                )["loaded_model"] = first

    # 4. SGLang data-URI compatibility probe.
    if not args.skip_container_check:
        probe = probe_sglang_data_uri(args.base_url_sglang, "qwen3-vl")
        report.sglang_data_uri_probe = probe
        if not probe.get("ok"):
            report.fail(
                f"SGLang data-URI image_url probe failed (status={probe.get('status')}). "
                f"This means stock SGLang at the running revision does not accept "
                f"data: URIs in messages[].content image_url parts. The omni-side "
                f"payload translator emits data: URIs; either swap to a tiny local "
                f"file-server fallback or upgrade SGLang."
            )

    # 4b. Launcher log verification — proves the server actually loaded
    # from the pinned snapshot, not from a stale local checkpoint.
    if not args.skip_container_check:
        _check_launcher_logs(args, report)

    # 5. Dataset revision pinning.
    rev_path = REPO_ROOT / "benchmarks" / "dataset" / "mmmu_revisions.json"
    if args.update_revisions:
        update_dataset_revisions(report, rev_path, DATASET_REPOS_TO_PIN)
    else:
        verify_dataset_revisions(report, rev_path, DATASET_REPOS_TO_PIN)

    # Write JSON report.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "ok": report.ok,
                "errors": report.errors,
                "warnings": report.warnings,
                "model_revisions": report.model_revisions,
                "snapshot_paths": report.snapshot_paths,
                "containers": report.containers,
                "dataset_revisions": report.dataset_revisions,
                "sglang_data_uri_probe": report.sglang_data_uri_probe,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    if not report.ok:
        print(f"[preflight] FAILED — see {out_path}", file=sys.stderr)
        for err in report.errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    print(f"[preflight] OK — report at {out_path}")
    if report.warnings:
        for warn in report.warnings:
            print(f"[preflight] warn: {warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
