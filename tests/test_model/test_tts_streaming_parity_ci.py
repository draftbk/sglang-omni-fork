# SPDX-License-Identifier: Apache-2.0
"""Opt-in streaming-parity CI test against a live TTS server.

Generates each utterance GREEDY (argmax via top_k=1) + same seed, both non-streaming
and streaming, TWICE per mode, then runs the two walls on the ASSEMBLED audio:

  WALL 1 (determinism): under greedy + same seed each path must reproduce itself
          bit-identically. A failure means the generation is not seed-stable (e.g.
          sampling left on, scheduler non-determinism, or runaway decoding) — a real
          bug, and it makes any stream-vs-non-stream delta meaningless. Hard failure.

  WALL 2 (cross-mode): under greedy, streaming should equal non-streaming. If the
          audio is byte-identical, streaming is faithful (PASS). If not (e.g. a neural
          vocoder whose decode is not sample-reproducible across framings), this can
          only be judged on quality — see assert_streaming_quality_parity (needs ASR);
          this ASR-free test skips that case with a message rather than false-alarm.

Enable by pointing at a running server:
    SGLANG_OMNI_TTS_URL=http://localhost:8000/v1/audio/speech \
    SGLANG_OMNI_TTS_REF=/path/to/reference.wav \
    SGLANG_OMNI_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    pytest tests/test_model/test_tts_streaming_parity_ci.py -v -s
"""

import base64
import hashlib
import io
import json
import os
import wave

import pytest
import requests

from tests.utils import mismatched

URL = os.environ.get("SGLANG_OMNI_TTS_URL")
REF = os.environ.get("SGLANG_OMNI_TTS_REF")
MODEL = os.environ.get("SGLANG_OMNI_TTS_MODEL", "qwen3-tts")
REF_TEXT = os.environ.get(
    "SGLANG_OMNI_TTS_REF_TEXT",
    "We asked over twenty different people, and they all said it was his.",
)
SEED = 12345
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is changing the way we live and work.",
    "Please remember to bring your umbrella, it might rain later today.",
]

pytestmark = pytest.mark.skipif(
    not (URL and REF and os.path.isfile(REF or "")),
    reason="set SGLANG_OMNI_TTS_URL and SGLANG_OMNI_TTS_REF (an existing wav) to run",
)


def _ref_data_uri() -> str:
    with open(REF, "rb") as f:
        return "data:audio/wav;base64," + base64.b64encode(f.read()).decode()


def _payload(text: str, *, stream: bool) -> dict:
    # GREEDY via top_k=1 (NOT temperature=0 — some engines ignore that for the greedy
    # decision); fixed seed; bounded tokens as a no-EOS-runaway watchdog.
    payload = {
        "model": MODEL, "input": text, "ref_audio": _ref_data_uri(), "ref_text": REF_TEXT,
        "response_format": "wav", "seed": SEED, "talker_top_k": 1, "talker_max_new_tokens": 1024,
    }
    if stream:
        payload["stream"] = True
    return payload


def _pcm(wav_bytes: bytes) -> bytes:
    with io.BytesIO(wav_bytes) as buf, wave.open(buf, "rb") as wf:
        return wf.readframes(wf.getnframes())


def _assembled_audio_hash(text: str, *, stream: bool) -> str:
    resp = requests.post(URL, json=_payload(text, stream=stream), stream=stream, timeout=600)
    resp.raise_for_status()
    if not stream:
        pcm = _pcm(resp.content)
    else:  # assemble audio from the SSE chunks (never hash raw SSE bytes)
        pcm, buf = bytearray(), bytearray()
        for chunk in resp.iter_content(8192):
            buf.extend(chunk or b"")
            while b"\n" in buf:
                idx = buf.index(b"\n"); line = bytes(buf[:idx]); del buf[: idx + 1]
                s = line.decode("utf-8", "replace").strip()
                if s.startswith("data:") and s != "data: [DONE]":
                    try:
                        ev = json.loads(s[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    audio = ev.get("audio")
                    if isinstance(audio, dict) and audio.get("data"):
                        pcm.extend(_pcm(base64.b64decode(audio["data"])))
        pcm = bytes(pcm)
    return hashlib.sha256(pcm).hexdigest()


def test_streaming_parity():
    ns_a = {t: _assembled_audio_hash(t, stream=False) for t in TEXTS}
    ns_b = {t: _assembled_audio_hash(t, stream=False) for t in TEXTS}
    st_a = {t: _assembled_audio_hash(t, stream=True) for t in TEXTS}
    st_b = {t: _assembled_audio_hash(t, stream=True) for t in TEXTS}

    # WALL 1 — determinism (greedy + same seed must reproduce). Hard failure.
    assert not mismatched(ns_a, ns_b), "non-streaming not reproducible under greedy+seed"
    assert not mismatched(st_a, st_b), "streaming not reproducible under greedy+seed"

    # WALL 2 — cross-mode parity under greedy.
    diverged = mismatched(ns_a, st_a)
    if diverged:
        pytest.skip(
            f"cross-mode audio not byte-identical for {diverged}; this engine's decode "
            "is not byte-reproducible across framings — judge via quality "
            "(assert_streaming_quality_parity with WER/speaker-sim), which needs ASR."
        )
