# SPDX-License-Identifier: Apache-2.0
"""Endpoint allocation and IPC runtime directory ownership."""

from __future__ import annotations

import logging
import re
import shutil
import socket
import tempfile
from pathlib import Path

from sglang_omni.config.schema import PipelineConfig, StageConfig

logger = logging.getLogger(__name__)


class IpcRuntimeDir:
    """Runtime-owned IPC directory for one pipeline instance."""

    def __init__(self, path: Path):
        self.path = path
        self._closed = False

    def __enter__(self) -> IpcRuntimeDir:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"IpcRuntimeDir(path={self.path!r}, closed={self._closed})"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to remove IPC runtime dir %s: %s", self.path, exc)


def create_ipc_runtime_dir(config: PipelineConfig) -> IpcRuntimeDir | None:
    """Create a per-run IPC namespace for one pipeline instance."""
    if config.endpoints.scheme != "ipc":
        return None

    base_root = Path(config.endpoints.base_path)
    base_root.mkdir(parents=True, exist_ok=True)

    namespace_prefix = re.sub(r"[^0-9a-z]+", "-", config.name.lower()).strip("-")
    if not namespace_prefix:
        namespace_prefix = "pipeline"
    path = Path(tempfile.mkdtemp(prefix=f"{namespace_prefix}-", dir=base_root))
    return IpcRuntimeDir(path)


def _find_free_tcp_ports(start: int, count: int) -> list[int]:
    """Find *count* available TCP ports starting from *start*."""
    ports: list[int] = []
    port = start
    while len(ports) < count:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                ports.append(port)
        except OSError:
            pass
        port += 1
    return ports


def allocate_endpoints(
    config: PipelineConfig,
    *,
    stages: list[StageConfig],
    ipc_base_dir: Path | None = None,
) -> dict[str, str]:
    endpoints: dict[str, str] = {}

    if config.completion_endpoint:
        endpoints["completion"] = config.completion_endpoint
    if config.abort_endpoint:
        endpoints["abort"] = config.abort_endpoint

    if config.endpoints.scheme == "ipc":
        if ipc_base_dir is None:
            raise ValueError("IPC endpoint allocation requires an IPC runtime dir")
        endpoints.setdefault("completion", f"ipc://{ipc_base_dir}/completion.sock")
        endpoints.setdefault("abort", f"ipc://{ipc_base_dir}/abort.sock")
        for stage in stages:
            endpoints[f"stage_{stage.name}"] = (
                f"ipc://{ipc_base_dir}/stage_{stage.name}.sock"
            )
        return endpoints

    if config.endpoints.scheme == "tcp":
        needed = 2 + len(stages)
        ports = _find_free_tcp_ports(config.endpoints.base_port, needed)
        idx = 0
        if "completion" not in endpoints:
            endpoints["completion"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        if "abort" not in endpoints:
            endpoints["abort"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        for stage in stages:
            endpoints[f"stage_{stage.name}"] = f"tcp://127.0.0.1:{ports[idx]}"
            idx += 1
        return endpoints

    raise ValueError(f"Unknown endpoint scheme: {config.endpoints.scheme}")
