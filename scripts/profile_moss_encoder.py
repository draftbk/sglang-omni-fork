#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Standalone microbenchmark: is ``torch.compile`` worth it on the
MOSS-Transcribe-Diarize Whisper encoder?

Reproduces the analysis in ``docs/moss_td_encoder_compile_analysis.md``. It
instantiates the ``WhisperEncoder`` *alone* (random weights — timing is
weight-independent), then for each batch size compares **eager** vs
``torch.compile(encoder, dynamic=True)`` and reports:

  * CUDA-event warm latency over N reps -> mean +/- std (not profiler wall);
  * ``torch._dynamo.explain`` -> graphs / graph_breaks / ops;
  * ``torch.profiler`` kernel breakdown + chrome trace + GPU-busy vs wall;
  * first-compile wall time (run with ``--clear-inductor-cache`` for the cold
    number, without it for the warm-cache number);
  * a recompile probe: reuse one compiled object across batch 1 -> 16 to show
    whether ``dynamic=True`` generalizes the batch dim.

This isolates the encoder (the exact ``is_encoder`` SDPA path the server runs);
it excludes the VQ adaptor / time-merge / scheduler, which is the intent.

Usage:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
      python scripts/profile_moss_encoder.py \
        --model-path OpenMOSS-Team/MOSS-Transcribe-Diarize \
        --batches 1 16 --iters 50 --warmup 10 --trace-dir docs/traces
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import statistics
import sys
import time

# allow `python scripts/profile_moss_encoder.py` from the repo root to import sglang_omni
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _init_tp() -> None:
    """Bring up a single-process TP=1 group (required to construct the
    tensor-parallel Linear layers inside WhisperEncoder). Mirrors the fixture
    in tests/test_model/conftest.py."""
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.cuda.set_device(0)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29551")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_IB_DISABLE", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")

    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
        model_parallel_is_initialized,
    )

    if not dist.is_initialized():
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{os.environ['MASTER_PORT']}",
            local_rank=0,
            backend="nccl",
        )
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size=1)


def _build_encoder(model_path: str):
    import torch
    from transformers import AutoConfig

    # registers the "moss_transcribe_diarize" AutoConfig (gives .audio_config)
    import sglang_omni.models.moss_transcribe_diarize.hf_config  # noqa: F401
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.models.whisper import WhisperEncoder

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    audio_cfg = cfg.audio_config
    with set_default_torch_dtype(torch.bfloat16):
        with torch.device("cuda:0"):
            enc = WhisperEncoder(audio_cfg, None)
    enc.eval()
    return enc, audio_cfg


def _make_inputs(batch: int, n_mel: int):
    import torch

    feats = torch.randn(batch, n_mel, 3000, device="cuda:0", dtype=torch.bfloat16)
    encoder_len = (3000 - 1) // 2 + 1  # whisper stride-2 conv over the time axis
    pos_ids = torch.arange(encoder_len, device="cuda:0", dtype=torch.long)
    return feats, pos_ids


def _timed(fn, warmup: int, iters: int):
    """Per-iter CUDA-event timings (ms). Returns list so we can report std."""
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def _profile(fn, tag: str, trace_dir: str | None):
    import torch
    from torch.profiler import ProfilerActivity, profile

    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    iters = 20
    w0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - w0) * 1000.0 / iters
    busy_us = sum(
        getattr(k, "self_device_time_total", 0) or getattr(k, "self_cuda_time_total", 0)
        for k in prof.key_averages()
    )
    busy_ms = busy_us / 1000.0 / iters
    idle = max(0.0, 1.0 - busy_ms / wall_ms) if wall_ms else 0.0
    print(f"\n[{tag}] profiler: wall={wall_ms:.2f}ms gpu_busy={busy_ms:.2f}ms "
          f"idle_frac={idle:.2f}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    if trace_dir:
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, f"moss_td_encoder_{tag}.json")
        prof.export_chrome_trace(path)
        print(f"[{tag}] chrome trace -> {path}")


def _fmt(times):
    return f"{statistics.mean(times):.2f} +/- {statistics.pstdev(times):.2f} ms"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--trace-dir", default=None,
                    help="if set, export chrome traces here (batch=1 only)")
    ap.add_argument("--clear-inductor-cache", action="store_true",
                    help="rm the on-disk inductor cache first -> measures COLD compile")
    ap.add_argument("--dynamic", choices=["true", "false", "auto"], default="true",
                    help="torch.compile dynamic= (auto -> None). false = static per-shape")
    args = ap.parse_args()
    dynamic = {"true": True, "false": False, "auto": None}[args.dynamic]

    if args.clear_inductor_cache:
        for d in glob.glob("/tmp/torchinductor_*"):
            shutil.rmtree(d, ignore_errors=True)
        print("cleared /tmp/torchinductor_* (cold compile)")

    import torch

    _init_tp()
    enc, audio_cfg = _build_encoder(args.model_path)
    n_mel = int(audio_cfg.num_mel_bins)
    print(f"torch {torch.__version__} | encoder: d_model={audio_cfg.d_model} "
          f"layers={audio_cfg.encoder_layers} heads={audio_cfg.encoder_attention_heads} "
          f"n_mel={n_mel}")

    for batch in args.batches:
        feats, pos_ids = _make_inputs(batch, n_mel)
        print(f"\n{'='*64}\nbatch={batch}  input={tuple(feats.shape)}\n{'='*64}")

        eager = _timed(lambda: enc(feats, pos_ids, None), args.warmup, args.iters)
        print(f"eager    : {_fmt(eager)}")

        try:
            compiled = torch.compile(enc, dynamic=dynamic)
            t0 = time.perf_counter()
            compiled(feats, pos_ids, None)
            torch.cuda.synchronize()
            print(f"compile wall (first call, this shape): "
                  f"{(time.perf_counter()-t0)*1000:.0f} ms")
            comp = _timed(lambda: compiled(feats, pos_ids, None), args.warmup, args.iters)
            print(f"compiled : {_fmt(comp)}")
            print(f"speedup  : {statistics.mean(eager)/statistics.mean(comp):.3f}x  "
                  f"(>1 = compiled faster)")
            if args.trace_dir and batch == 1:
                _profile(lambda: enc(feats, pos_ids, None), "eager", args.trace_dir)
                _profile(lambda: compiled(feats, pos_ids, None), "compiled",
                         args.trace_dir)
        except Exception as e:  # noqa: BLE001 -- a torch/inductor bug on a shape
            print(f"compiled : FAILED -> {type(e).__name__}: {str(e)[:160]}")

    # dynamo coverage (does the encoder compile in one graph?)
    print(f"\n{'='*64}\ndynamo.explain (batch=1)\n{'='*64}")
    feats, pos_ids = _make_inputs(1, n_mel)
    import torch._dynamo as dynamo

    exp = dynamo.explain(enc)(feats, pos_ids, None)
    print(f"graphs={exp.graph_count}  graph_breaks={exp.graph_break_count}  "
          f"ops={exp.op_count}")

    # recompile probe: does one compiled object survive a batch-dim change?
    if len(args.batches) > 1:
        print(f"\n{'='*64}\nrecompile probe: one compiled obj across batches "
              f"{args.batches}\n{'='*64}")
        obj = torch.compile(enc, dynamic=dynamic)
        for batch in args.batches:
            f, p = _make_inputs(batch, n_mel)
            try:
                obj(f, p, None)
                torch.cuda.synchronize()
                print(f"  batch={batch}: ok")
            except Exception as e:  # noqa: BLE001
                print(f"  batch={batch}: {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
