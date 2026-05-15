# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import queue as queue_mod
import threading
from types import SimpleNamespace

import torch

from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from tests.unit_test.pipeline.helpers import run_scheduler


def test_simple_scheduler_batch_and_error_contracts() -> None:
    """Preserves batched success output and per-request batch failure emission."""
    good = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: [payload.upper() for payload in payloads],
        max_batch_size=2,
        max_batch_wait_ms=10,
    )
    outputs = run_scheduler(
        good,
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
    )
    assert {out.data for out in outputs} == {"A", "B"}

    bad = SimpleScheduler(
        lambda payload: payload,
        batch_compute_fn=lambda payloads: ["only-one"],
        max_batch_size=2,
        max_batch_wait_ms=10,
    )
    outputs = run_scheduler(
        bad,
        [
            IncomingMessage("req-1", "new_request", "a"),
            IncomingMessage("req-2", "new_request", "b"),
        ],
        output_count=2,
    )
    assert {out.request_id for out in outputs} == {"req-1", "req-2"}
    assert all(
        out.type == "error" and isinstance(out.data, ValueError) for out in outputs
    )


def test_threaded_simple_scheduler_runs_requests_concurrently() -> None:
    """Covers concurrent worker execution before result emission."""
    started: list[str] = []
    lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    def compute(payload: str) -> str:
        with lock:
            started.append(payload)
            if len(started) == 2:
                both_started.set()
        assert release.wait(timeout=2.0)
        return payload

    def wait_for_both_started() -> None:
        try:
            assert both_started.wait(timeout=2.0)
        finally:
            release.set()

    outputs = run_scheduler(
        ThreadedSimpleScheduler(compute, max_concurrency=2),
        [
            IncomingMessage("req-1", "new_request", "one"),
            IncomingMessage("req-2", "new_request", "two"),
        ],
        output_count=2,
        before_collect=wait_for_both_started,
    )

    assert {output.request_id for output in outputs} == {"req-1", "req-2"}
    assert {output.data for output in outputs} == {"one", "two"}


def test_threaded_simple_scheduler_reports_worker_errors() -> None:
    """Covers worker exception emission as scheduler errors."""

    def compute(payload: str) -> str:
        raise RuntimeError(payload)

    outputs = run_scheduler(
        ThreadedSimpleScheduler(compute, max_concurrency=1),
        [IncomingMessage("req-err", "new_request", "boom")],
        output_count=1,
    )

    assert outputs[0].request_id == "req-err"
    assert outputs[0].type == "error"
    assert isinstance(outputs[0].data, RuntimeError)


def test_omni_scheduler_default_stream_chunk_buffers_raw_chunks() -> None:
    """Preserves generic stream chunk buffering when no custom handler exists."""
    req_data = SimpleNamespace()
    chunk = SimpleNamespace(data="chunk-data", metadata={"token_id": 1})

    OmniScheduler._append_stream_chunk_default(req_data, chunk)

    assert list(req_data.stream_chunks) == [chunk]


def test_omni_scheduler_default_stream_done_sets_generic_flag() -> None:
    """Preserves generic stream completion state when no custom handler exists."""
    scheduler = object.__new__(OmniScheduler)
    scheduler._stream_done_handler = None
    req_data = SimpleNamespace()

    scheduler._mark_stream_done(req_data)

    assert req_data.stream_done is True


def test_omni_scheduler_run_batch_failure_emits_error_and_aborts() -> None:
    """Forward failures are owned by the scheduler, not model executors."""

    class BoomModelRunner:
        def execute(self, sched_output):
            assert [req.request_id for req in sched_output.requests] == [
                "req-1",
                "req-2",
            ]
            raise RuntimeError("cuda out of memory")

    scheduler = object.__new__(OmniScheduler)
    scheduler._model_runner = BoomModelRunner()
    scheduler._stream_output_builder = None
    scheduler.outbox = queue_mod.Queue()
    scheduler.inbox = queue_mod.Queue()
    scheduler.is_entry_rank = True
    scheduler._aborted_request_ids = set()
    scheduler._pending_stream_chunks = {"req-1": ["stale"]}
    scheduler._pending_stream_done = {"req-2"}
    scheduler._deferred_request_payloads = {"req-1": object()}
    scheduler.waiting_queue = []
    scheduler.last_batch = None

    batch = SimpleNamespace(
        reqs=[
            SimpleNamespace(rid="req-1", _omni_data=SimpleNamespace()),
            SimpleNamespace(rid="req-2", _omni_data=SimpleNamespace()),
        ],
        batch_is_full=True,
    )
    scheduler.running_batch = batch
    scheduler.cur_batch = batch

    result = scheduler.run_batch(batch)

    assert result is omni_scheduler_module._FAILED_BATCH_RESULT
    outputs = [scheduler.outbox.get_nowait(), scheduler.outbox.get_nowait()]
    assert {output.request_id for output in outputs} == {"req-1", "req-2"}
    assert all(output.type == "error" for output in outputs)
    assert all(isinstance(output.data, RuntimeError) for output in outputs)
    assert all("cuda out of memory" in str(output.data) for output in outputs)
    assert scheduler._aborted_request_ids == {"req-1", "req-2"}
    assert batch.reqs == []
    assert scheduler._pending_stream_chunks == {}
    assert scheduler._pending_stream_done == set()
    assert scheduler._deferred_request_payloads == {}


def test_stage_output_cache_eviction_uses_lru_order() -> None:
    cache = StageOutputCache(max_size=2)

    cache.put("a", torch.tensor([1]))
    cache.put("b", torch.tensor([2]))
    assert torch.equal(cache.get("a"), torch.tensor([1]))

    cache.put("c", torch.tensor([3]))

    assert cache.get("b") is None
    assert torch.equal(cache.get("a"), torch.tensor([1]))
    assert torch.equal(cache.get("c"), torch.tensor([3]))


def test_stage_output_cache_tracks_bytes_and_detaches() -> None:
    cache = StageOutputCache(max_bytes=8, cache_device="cpu")

    cache.put("fit", {"x": torch.ones(2, dtype=torch.float32, requires_grad=True)})
    cached = cache.get("fit")

    assert cache.current_bytes == 8
    assert cached["x"].device.type == "cpu"
    assert cached["x"].requires_grad is False

    cache.put("too-large", torch.ones(3, dtype=torch.float32))

    assert cache.get("too-large") is None
    assert cache.current_bytes == 8
