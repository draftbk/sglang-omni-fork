# SPDX-License-Identifier: Apache-2.0
"""Preflight remote-routing regression: every post-launch verification stage
must route over SSH when ``--host`` is set, never read orchestrator-local
state. ``probe_v1_models``, ``probe_sglang_data_uri``, and
``verify_launcher_log_references_snapshot`` all need the same host
parameter so the gate actually verifies the machine that launched the
container, not the orchestrator.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


def _import_preflight():
    """Import the preflight module via importlib to avoid argparse parse."""
    import importlib.util
    import sys
    from pathlib import Path

    name = "preflight_mmmu_sweep_test"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks"
        / "scripts"
        / "preflight_mmmu_sweep.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass lookups via cls.__module__ need this
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# probe_v1_models
# ---------------------------------------------------------------------------


def test_probe_v1_models_remote_uses_ssh_curl() -> None:
    mod = _import_preflight()
    captured = {}

    def fake_check_output(cmd, **kwargs):
        captured["cmd"] = cmd
        return json.dumps({"data": [{"id": "qwen3-omni"}]})

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        with patch.object(mod.request, "urlopen") as fake_urlopen:
            result = mod.probe_v1_models(
                "http://localhost:30000", host="ion8-omni"
            )

    assert result == {"data": [{"id": "qwen3-omni"}]}
    fake_urlopen.assert_not_called(), (
        "remote mode must not fall back to orchestrator-local urlopen"
    )
    cmd = captured["cmd"]
    assert cmd[0] == "ssh" and cmd[1] == "ion8-omni"
    assert "curl" in cmd
    assert any(arg.endswith("/v1/models") for arg in cmd)


def test_probe_v1_models_local_uses_urlopen() -> None:
    """Without --host, the local path is used (no SSH)."""
    mod = _import_preflight()

    class _FakeResp:
        def __init__(self, body: str) -> None:
            self._body = body.encode("utf-8")

        def read(self) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch.object(mod.subprocess, "check_output") as fake_check_output:
        with patch.object(
            mod.request,
            "urlopen",
            return_value=_FakeResp(json.dumps({"data": []})),
        ):
            result = mod.probe_v1_models("http://localhost:30000")

    assert result == {"data": []}
    fake_check_output.assert_not_called()


# ---------------------------------------------------------------------------
# probe_sglang_data_uri
# ---------------------------------------------------------------------------


def test_probe_sglang_data_uri_remote_uses_ssh_curl() -> None:
    mod = _import_preflight()
    captured = {}

    def fake_check_output(cmd, **kwargs):
        captured["cmd"] = cmd
        return "200"

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        with patch.object(mod.request, "urlopen") as fake_urlopen:
            result = mod.probe_sglang_data_uri(
                "http://localhost:30001", "qwen3-vl", host="ion9-omni"
            )

    assert result == {"status": 200, "ok": True}
    fake_urlopen.assert_not_called()
    cmd = captured["cmd"]
    assert cmd[0] == "ssh" and cmd[1] == "ion9-omni"
    assert "POST" in cmd
    assert any("/v1/chat/completions" in arg for arg in cmd)


# ---------------------------------------------------------------------------
# verify_launcher_log_references_snapshot
# ---------------------------------------------------------------------------


def test_verify_launcher_log_remote_uses_ssh_cat() -> None:
    mod = _import_preflight()
    captured = {}

    def fake_check_output(cmd, **kwargs):
        captured["cmd"] = cmd
        return "loaded /snapshot/abc123 ready"

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        ok = mod.verify_launcher_log_references_snapshot(
            "/tmp/sglang-omni-benchmark.log", "/snapshot/abc123", host="ion8-omni"
        )

    assert ok is True
    cmd = captured["cmd"]
    assert cmd[0] == "ssh" and cmd[1] == "ion8-omni"
    assert "cat" in cmd
    assert "/tmp/sglang-omni-benchmark.log" in cmd


def test_verify_launcher_log_remote_fails_when_snapshot_absent() -> None:
    """Remote-mode False path: SSH returns log content that does NOT mention the snapshot."""
    mod = _import_preflight()

    def fake_check_output(cmd, **kwargs):
        return "loaded /some/other/path ready"

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        ok = mod.verify_launcher_log_references_snapshot(
            "/tmp/launcher.log", "/snapshot/abc123", host="ion8-omni"
        )
    assert ok is False


def test_verify_launcher_log_remote_fails_when_ssh_errors() -> None:
    """If the SSH cat fails, the verification returns False (not crash)."""
    import subprocess

    mod = _import_preflight()

    def fake_check_output(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd)

    with patch.object(mod.subprocess, "check_output", side_effect=fake_check_output):
        ok = mod.verify_launcher_log_references_snapshot(
            "/tmp/missing.log", "/snapshot/abc123", host="ion8-omni"
        )
    assert ok is False


# ---------------------------------------------------------------------------
# Launch-evidence preservation regression. A prior version used
# `report.containers[name] = info` which wholesale-overwrote any
# `launch_command` written by an earlier `launch_named_container` call,
# silently dropping the evidence the eval needs to derive launch policy.
# This test runs the real serialization shape and asserts the field
# persists across the inspection pass.
# ---------------------------------------------------------------------------


def test_check_container_preserves_launch_command_from_earlier_launch() -> None:
    """Real flow: launch records launch_command → check_container merges
    digest/image fields onto the same record → serialized report still
    carries launch_command for downstream metadata derivation.
    """
    mod = _import_preflight()

    report = mod.PreflightReport()
    container_name = "sglang-omni-hayden-benchmark"
    expected_launch_cmd = [
        "docker", "run", "-d", "--name", container_name,
        "frankleeeee/sglang-omni:dev",
        "sgl-omni", "serve", "--model-path", "/snapshot",
        "--mem-fraction-static", "0.9",
        "--disable-radix-cache",
    ]

    # Simulate what `launch_named_container(..., record=record)` writes
    # into the per-container record before any later inspect pass runs.
    record = report.containers.setdefault(container_name, {})
    record["launch_command"] = list(expected_launch_cmd)
    record["snapshot_path"] = "/snapshots/abc123"

    # Now run the inspect-style update path. The real check_container calls
    # docker_inspect; patch those to return known values so we exercise the
    # *update* logic specifically.
    with patch.object(
        mod, "docker_inspect_image", return_value="sha256:deadbeef"
    ), patch.object(
        mod, "docker_inspect_repo_tag", return_value="frankleeeee/sglang-omni:dev"
    ):
        mod.check_container(
            container_name, "frankleeeee/sglang-omni:dev", report, host=None
        )

    # The post-check record must still carry the launch_command + snapshot.
    final = report.containers[container_name]
    assert final.get("launch_command") == expected_launch_cmd, (
        "check_container wholesale-overwrote launch_command (evidence-loss bug)"
    )
    assert final.get("snapshot_path") == "/snapshots/abc123"
    # The inspect-stage fields are also present.
    assert final.get("container_image_digest") == "sha256:deadbeef"
    assert final.get("container_image") == "frankleeeee/sglang-omni:dev"


def test_check_container_round_trip_via_json_keeps_launch_command(tmp_path) -> None:
    """Serializing the report to JSON and reloading must preserve launch_command."""
    import json
    from dataclasses import asdict

    mod = _import_preflight()

    report = mod.PreflightReport()
    name = "sglang-hayden-benchmark"
    report.containers.setdefault(name, {})["launch_command"] = [
        "docker", "run", "-d", "--name", name, "lmsysorg/sglang",
        "python", "-m", "sglang.launch_server",
        "--mem-fraction-static", "0.85",
        "--disable-radix-cache",
    ]

    with patch.object(
        mod, "docker_inspect_image", return_value="sha256:1234"
    ), patch.object(
        mod, "docker_inspect_repo_tag", return_value="lmsysorg/sglang"
    ):
        mod.check_container(name, "lmsysorg/sglang", report, host=None)

    # Write the report and reload from disk.
    out = tmp_path / "preflight.json"
    out.write_text(json.dumps(asdict(report)))
    reloaded = json.loads(out.read_text())

    container = reloaded["containers"][name]
    assert "launch_command" in container
    assert "--disable-radix-cache" in container["launch_command"]
    assert "--mem-fraction-static" in container["launch_command"]
