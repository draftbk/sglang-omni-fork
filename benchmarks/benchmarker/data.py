# SPDX-License-Identifier: Apache-2.0
"""Shared data structures for the benchmark framework."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestResult:
    request_id: str = ""
    text: str = ""
    is_success: bool = False
    latency_s: float = 0.0
    audio_duration_s: float = 0.0
    rtf: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # engine_time_s is the true server-reported engine time. Currently only
    # populated by TTS send_fns that read the X-Engine-Time response header or
    # streamed usage_data["engine_time_s"]. send_fns that only have client-side
    # wall-clock visibility (MMMU, audio understanding, video understanding)
    # populate client_wall_time_s instead and leave engine_time_s at 0.0.
    engine_time_s: float = 0.0

    # client_wall_time_s is perf_counter elapsed around session.post(). Used by
    # send_fns that do not get server-side timing back from the engine. None
    # signals "not measured" so JSON consumers can tell apart a zero-time
    # request (impossible) from a request that never set the field.
    client_wall_time_s: float | None = None

    # timing_source declares which of the two fields above is authoritative for
    # tok_per_s and aggregate throughput computation for this request:
    # "engine_time_s" | "client_wall_time_s" | "" (legacy / unset).
    timing_source: str = ""

    tok_per_s: float = 0.0
    wav_path: str = ""
    error: str = ""

    # Streaming-mode fields (populated only when the send_fn consumed an SSE
    # stream and timestamped per-content-chunk arrival).
    # ttft_s is wall-clock from request send to first non-empty delta.content
    # frame. None when not streaming or no content frame ever arrived.
    ttft_s: float | None = None

    # content_chunk_offsets_ms is the list of relative offsets from request
    # send to each non-empty delta.content frame arrival. Length equals
    # content_chunk_count. Stored as relative offsets (not absolute
    # perf_counter values) so the data is portable across hosts.
    content_chunk_offsets_ms: list[float] = field(default_factory=list)

    # Number of non-empty delta.content frames received. Role-only frames,
    # the final finish chunk, and the [DONE] sentinel are not counted.
    content_chunk_count: int = 0
