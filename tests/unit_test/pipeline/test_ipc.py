# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import sglang_omni.pipeline.mp_runner as mp_runner
import sglang_omni.serve.launcher as launcher
from sglang_omni.config.schema import EndpointsConfig, PipelineConfig, StageConfig
from sglang_omni.pipeline.endpoints import allocate_endpoints, create_ipc_runtime_dir


def noop_factory():
    return None


def _make_config(base_path: Path, *, scheme: str = "ipc") -> PipelineConfig:
    return PipelineConfig(
        model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        entry_stage="preprocessing",
        stages=[
            StageConfig(
                name="preprocessing",
                factory=f"{__name__}.noop_factory",
                terminal=True,
            )
        ],
        endpoints=EndpointsConfig(scheme=scheme, base_path=str(base_path)),
    )


class FakeCoordinator:
    def __init__(
        self,
        completion_endpoint: str,
        abort_endpoint: str,
        entry_stage: str,
        terminal_stages: list[str] | None = None,
    ) -> None:
        del abort_endpoint, entry_stage, terminal_stages
        self.control_plane = SimpleNamespace(completion_endpoint=completion_endpoint)
        self.registered: dict[str, str] = {}
        self.stopped = False

    async def start(self) -> None:
        return None

    async def run_completion_loop(self) -> None:
        await asyncio.Event().wait()

    def register_stage(self, name: str, endpoint: str) -> None:
        self.registered[name] = endpoint

    async def shutdown_stages(self) -> None:
        return None

    async def stop(self) -> None:
        self.stopped = True


class FakeGroup:
    stage_name = "preprocessing"
    tp_size = 1
    processes: list[object] = []

    def __init__(self, leader_endpoint: str) -> None:
        self.leader_endpoint = leader_endpoint
        self.shutdown_called = False

    def spawn(self, ctx) -> None:
        del ctx

    async def wait_ready(self, timeout: float) -> None:
        del timeout

    def any_dead(self) -> bool:
        return False

    def dead_summary(self) -> str:
        return "(none)"

    async def shutdown(self) -> None:
        self.shutdown_called = True


def _fake_groups_from_endpoints(*args, **kwargs) -> list[FakeGroup]:
    del args
    endpoints = kwargs["endpoints"]
    return [FakeGroup(endpoints["stage_preprocessing"])]


def test_ipc_runtime_dir_creation_and_close_contracts(tmp_path: Path) -> None:
    """Preserves IPC runtime directory creation, uniqueness, and idempotent cleanup."""
    ipc_config = _make_config(tmp_path)
    tcp_config = _make_config(tmp_path, scheme="tcp")

    assert create_ipc_runtime_dir(tcp_config) is None

    runtime_a = create_ipc_runtime_dir(ipc_config)
    runtime_b = create_ipc_runtime_dir(ipc_config)
    assert runtime_a is not None
    assert runtime_b is not None
    assert runtime_a.path != runtime_b.path

    runtime_path = runtime_a.path
    runtime_a.close()
    runtime_a.close()
    runtime_b.close()
    assert not runtime_path.exists()
    assert list(tmp_path.iterdir()) == []


def test_allocate_ipc_endpoints_requires_runtime_dir(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    stages, _, _ = config.apply_fusion()

    with pytest.raises(ValueError, match="requires an IPC runtime dir"):
        allocate_endpoints(config, stages=stages)


@pytest.mark.asyncio
async def test_mp_runner_starts_same_model_instances_with_unique_ipc_endpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp_runner, "Coordinator", FakeCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", _fake_groups_from_endpoints)

    config = _make_config(tmp_path)
    runner_a = mp_runner.MultiProcessPipelineRunner(config)
    runner_b = mp_runner.MultiProcessPipelineRunner(config)

    try:
        await runner_a.start()
        await runner_b.start()

        runtime_dirs = [path for path in tmp_path.iterdir() if path.is_dir()]
        assert len(runtime_dirs) == 2
        assert (
            runner_a.coordinator.control_plane.completion_endpoint
            != runner_b.coordinator.control_plane.completion_endpoint
        )
        assert (
            runner_a.stage_endpoints["preprocessing"]
            != runner_b.stage_endpoints["preprocessing"]
        )
    finally:
        await runner_b.stop()
        await runner_a.stop()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_cleans_runtime_dir_on_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingCoordinator(FakeCoordinator):
        async def start(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(mp_runner, "Coordinator", FailingCoordinator)
    monkeypatch.setattr(mp_runner, "_build_stage_groups", _fake_groups_from_endpoints)
    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))

    with pytest.raises(RuntimeError, match="boom"):
        await runner.start()

    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_mp_runner_stop_cleans_runtime_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp_runner, "Coordinator", FakeCoordinator)
    group = FakeGroup("ipc://stage.sock")
    monkeypatch.setattr(mp_runner, "_build_stage_groups", lambda *a, **k: [group])

    runner = mp_runner.MultiProcessPipelineRunner(_make_config(tmp_path))
    await runner.start()
    assert len([path for path in tmp_path.iterdir() if path.is_dir()]) == 1

    await runner.stop()

    assert group.shutdown_called
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_launcher_stops_runner_when_server_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stopped = False

    class FakeRunner:
        def __init__(self, config: PipelineConfig):
            self.config = config
            self.stage_endpoints = {"preprocessing": "ipc://stage.sock"}
            self.coordinator = object()

        async def start(self, timeout: float = 120.0) -> None:
            del timeout

        async def stop(self) -> None:
            nonlocal stopped
            stopped = True

    async def fail_serve(self) -> None:
        del self
        raise RuntimeError("server failed")

    monkeypatch.setattr(launcher, "_find_available_port", lambda host, port: port)
    monkeypatch.setattr(launcher, "MultiProcessPipelineRunner", FakeRunner)
    monkeypatch.setattr(launcher, "create_app", lambda *a, **k: FastAPI())
    monkeypatch.setattr(launcher.uvicorn.Server, "serve", fail_serve)

    with pytest.raises(RuntimeError, match="server failed"):
        await launcher._run_server(_make_config(tmp_path), port=8000)

    assert stopped


def test_profiler_route_requires_dir_without_explicit_template() -> None:
    class FakeProfiler:
        async def broadcast_start(self, **kwargs) -> None:
            del kwargs

        async def broadcast_stop(self, **kwargs) -> None:
            del kwargs

    app = FastAPI()
    launcher._mount_profiler_routes(app, FakeProfiler(), profiler_dir=None)

    with TestClient(app) as client:
        response = client.post("/start_profile", json={})
        assert response.status_code == 400

        response = client.post(
            "/start_profile",
            json={"trace_path_template": "/tmp/profile/{run_id}/{stage}"},
        )
        assert response.status_code == 200
