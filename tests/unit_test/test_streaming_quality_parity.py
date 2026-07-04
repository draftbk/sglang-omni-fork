# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the streaming quality-parity verdict (CPU-only, no model/server)."""

import pytest

from tests.utils import (
    MetricCheckCollector,
    assert_streaming_quality_parity,
    mismatched,
)


def test_pass_when_quality_matches():
    ns = {f"s{i}": {"wer": 0.05, "speaker_sim": 0.85} for i in range(5)}
    assert_streaming_quality_parity(ns, dict(ns))  # identical per utterance -> pass


def test_no_false_alarm_when_both_modes_equally_bad():
    # high absolute WER but identical across modes -> PASS (paired, not absolute)
    ns = {"hard": {"wer": 0.333, "speaker_sim": 0.80}}
    assert_streaming_quality_parity(ns, dict(ns))


def test_fail_on_localized_wer_regression():
    # one utterance cut off in streaming -> caught, even though the other is fine
    ns = {"a": {"wer": 0.0}, "b": {"wer": 0.0}}
    st = {"a": {"wer": 0.0}, "b": {"wer": 0.40}}
    with pytest.raises(AssertionError):
        assert_streaming_quality_parity(ns, st)


def test_fail_on_speaker_sim_regression():
    ns = {"a": {"wer": 0.0, "speaker_sim": 0.88}}
    st = {"a": {"wer": 0.0, "speaker_sim": 0.70}}
    with pytest.raises(AssertionError):
        assert_streaming_quality_parity(ns, st)


def test_collector_defers_raising_to_caller():
    collector = MetricCheckCollector("streaming quality")
    assert_streaming_quality_parity({"a": {"wer": 0.0}}, {"a": {"wer": 0.5}}, collector=collector)
    assert collector.failures


def test_mismatched_walls():
    # determinism wall: identical hashes -> ok; differing -> flagged
    assert mismatched({"a": "h", "b": "h2"}, {"a": "h", "b": "h2"}) == []
    assert mismatched({"a": "h1"}, {"a": "DIFFERENT"}) == ["a"]
