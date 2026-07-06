# `torch.compile` on the MOSS-Transcribe-Diarize Whisper encoder: measured impact

> 中文版: [moss_td_encoder_compile_analysis.zh.md](./moss_td_encoder_compile_analysis.zh.md)

**Scope:** A100-40GB · bf16 · torch 2.11.0+cu130 · movies800 (~16 s clips) · greedy. Reproduce with `scripts/profile_moss_encoder.py` (see end).

## Summary

`torch.compile(dynamic=False)` on the Whisper encoder is a **real, correctness-preserving win**: **+20% on the isolated encoder** (N=50) and **+3.8% end-to-end throughput** (c=8, non-overlapping across 3 runs, CER unchanged). It's a legitimate optimization — but with caveats, and **CUDA graph (separately owned) is a cheaper lever for the same class of gain**, so for the encoder we lean CUDA graph.

Caveats for enabling compile:
- Needs a compile **per distinct `n_chunks` shape** (~20 s warm-cache / ~87 s cold each), warmed up at startup — and large-chunk buckets can **OOM the warmup on 40 GB**.
- `dynamic=True` hits a torch 2.11 `InductorError` at larger chunk counts → use **`dynamic=False`**.
- At **c=16 on 40 GB the card is OOM-limited** (completion 56–82/96), so the clean throughput comparison is at c=8.

## End-to-end throughput (c=8, greedy, fresh server per run, 96 samples)

| | run1 / run2 / run3 | worst | mean | vs eager | CER |
|---|---|---|---|---|---|
| eager | 10.06 / 10.19 / 10.00 | 10.00 | 10.08 | — | 5.12 |
| **compile (dynamic=False)** | 10.29 / 10.53 / 10.60 | 10.29 | **10.47** | **+3.8%** | 5.14 |

Ranges **don't overlap** (comp worst 10.29 > eager best 10.19) → real, not noise; CER unchanged → outputs preserved. Fresh server per run is required — a persistent server's 2nd+ eval degrades (see the note at the end). Single-request latency @c=1 moves only ~1% (the encoder is ~2.4% of one request's latency); the **throughput** gain is larger because speeding prefill frees GPU for decode batching.

## Three arguments (each one number)

1. **The encoder is a thin slice of e2e — ~2.4%.** Measured at concurrency=1 on movies800: encode **11.1 ms** vs total request latency **465 ms** (median 436 ms, ~16 s audio). Any encoder-only optimization is bounded by this.

2. **Two-thirds of encoder GPU time is already-optimal GEMM + flash-attention — compile can't touch it.** By Self-CUDA time: `addmm` (cuBLAS ampere tensor-core) **50%** and flash-attention **17%** are emitted **byte-identical** by compile (GEMM 98.7 vs 98.7 ms, flash 33.4 vs 33.2 ms). Only the remaining ~33% (LayerNorm/GELU/residual-add + permute copies) is fuseable.

3. **The isolated encoder speeds up ~20%, and it carries to +3.8% throughput.** Isolated encode (CUDA-event, N=50): eager **11.3 ± 0.06 ms** → **10.3 ms (`dynamic=True`, 1.10×)** / **9.3 ms (`dynamic=False`, 1.20×)** — statistically real (fuses the ~33% elementwise, GPU-busy 9.86→9.15 ms, trims launches, trace events 78k→59k). That 20% encoder win shows up as **+3.8% e2e throughput at c=8** (table above) — larger than the ~1% single-request-latency share, because prefill speedup frees GPU for decode batching. Cost: a compile per `n_chunks` shape (~20 s warm-cache).

## Cost / benefit

| | detail |
|---|---|
| **Benefit** | isolated encoder **−20%** (`dynamic=False`); **+3.8% e2e throughput** (c=8, clean, correctness-preserved). |
| **Cost 1 — compile per shape** | ~87 s cold / ~20 s warm-cache, **per distinct `n_chunks` shape**. Cached on disk (`/tmp/torchinductor_*`) across restarts, so *not* per-boot — but each unseen shape stalls its first request unless warmed at startup, and `n_chunks` grows with audio length. |
| **Cost 2 — warmup can OOM on 40 GB** | warming large-chunk buckets (16/32) at startup OOMs on 40 GB after the KV pool is reserved; on movies800 (1-chunk audio) warming just bucket `(1,)` is enough and cheap. Deployments with long audio need bucket coverage + padding. |
| **Cost 3 — version fragility** | `dynamic=True` raises `InductorError` in `tiling_utils.get_pw_red_splits` (a torch 2.11 dynamic-shape bug) at larger chunk counts. `dynamic=False` avoids it and is faster — so use `dynamic=False`. |

**Recommendation.** The encoder **is** worth optimizing (real +3.8% throughput, correctness-preserved). `torch.compile(dynamic=False)` delivers it with a per-shape startup warmup. But **CUDA graph is a cheaper lever for the same class of gain** (near-instant capture, no ~87 s compile, no version crash), so if you optimize the encoder, prefer CUDA graph; `torch.compile` is a valid alternative.

## Levers that move the encoder

| lever | measured | notes |
|---|---|---|
| **`torch.compile(dynamic=False)`** | −20% encoder, **+3.8% e2e throughput (c=8)** | Real, correctness-preserved (this doc). Cost: per-shape startup warmup + version fragility. |
| **CUDA graph** | +15% @batch=1, +0% @batch=16 | Cheaper lever for the same launch-overhead win; a **separate work item**, out of scope here. Shown to attribute how much of compile's batch=1 win is launch overhead vs fusion. |
| **Batching** | eager 11.3 ms @b1 → 7.5 ms/item @b16 (~1.5×) | Biggest per-item win, but the mm-embed dispatch calls the encoder **once per request**, so it doesn't engage on the serving path today. |

## Data appendix

### Isolated encoder latency (CUDA-event, N=50, mean ± std)

| batch (n_chunks) | eager | `compile(dynamic=True)` | `compile(dynamic=False)` | CUDA graph (eager) |
|---|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3 (1.10×) | **9.3 (1.20×)** | **9.6 (1.17×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError** | **100.8 (1.20×)** | **119.9 (1.00×)** |

Compile wall: **~87 s cold / ~20 s warm-cache** (first call, per shape). `dynamic=False` compiles every shape cleanly; `dynamic=True` recompiles per shape and crashes at batch=16. CUDA graph (`--cuda-graph`) helps only at batch=1 (launch-bound); at batch=16 the encoder is compute-saturated so removing launch overhead does nothing — while compile's batch=16 win is fusion/copy-elimination, not launches.

### End-to-end (`compile(dynamic=False)`, fresh server per run, greedy, 96 samples)

| metric | eager | compiled | Δ |
|---|---|---|---|
| server-side encode | ~11 ms | 9.4 ms | −15% (win reaches the server) |
| **throughput c=8** (worst/mean of 3) | 10.0 / 10.08 qps | **10.29 / 10.47 qps** | **+2.9% worst / +3.8% mean** (ranges non-overlapping) |
| latency c=1 mean | 0.446 s | 0.441 s | −1% (encoder is ~2.4% of a single request) |

**c=16 is not a clean comparison on 40 GB** — partial OOM (completion 56–82/96) makes throughput noise-dominated (nominal eager 13.1 / comp 12.5, but comp's low mean is one 56/96 OOM run, not a regression). The clean, non-OOM comparison is c=8 above.

**Note — the server degrades after the first eval.** On a *persistent* server, the 2nd and later eval runs produce garbage output (a 96-sample greedy eval gave CER 5.0 as the first run, then 67.9 on identical reps). All numbers here use a **fresh server per run** (as does the CUDA-graph work). This looks like a state leak across eval boundaries (mm/radix cache or scheduler) and is worth a separate issue — it's independent of `torch.compile`.

### Kernel breakdown (torch.profiler, batch=1, 20 iters, Self-CUDA)

| kernel | eager | compiled |
|---|---|---|
| `addmm` GEMM (cuBLAS ampere) | 50% / 98.7 ms | **98.7 ms (identical)** |
| flash attention | 17% / 33.4 ms | **33.2 ms (identical)** |
| LayerNorm / GELU / add | ~20% | fused → `triton_*_fused_*` |
| copy / clone / contiguous (permute) | ~10% / 19.1 ms | **≈ gone** (folded into `*_view` triton) |
| GPU-busy / iter | **9.86 ms** | **9.15 ms** (−7%) |

`torch._dynamo.explain`: **graphs=1, graph_breaks=0, ops=344** (encoder self-attention is `scaled_dot_product_attention`, fully traceable — not RadixAttention).

### Reference implementation (gated, off by default)

The actual "add compile" change is **~12 lines** in `sglang_omni/models/moss_transcribe_diarize/sglang_model.py`. The compile decision sits right where the encoder is built (`__init__`), the call site uses `self.encoder_runner(...)`, and `warmup_encoder_compile()` just pre-triggers the per-shape compilation at startup:

```python
# __init__, right after the encoder is created:
self.whisper_encoder = WhisperEncoder(config.audio_config, quant_config)
if os.getenv("MOSS_ENCODER_COMPILE") == "1":
    self.encoder_runner = torch.compile(self.whisper_encoder, dynamic=False)
else:
    self.encoder_runner = self.whisper_encoder

# call site:
whisper_features = self.encoder_runner(input_features, encoder_position_ids, forward_batch)

# startup (wire into the stage factory if you flip the flag):
def warmup_encoder_compile(self, buckets=(1, 2, 4, 8, 16, 32)):
    if os.getenv("MOSS_ENCODER_COMPILE") != "1":
        return
    cfg, p = self.config.audio_config, next(self.whisper_encoder.parameters())
    frames = int(cfg.max_source_positions) * 2
    pos = torch.arange((frames - 1) // 2 + 1, device=p.device, dtype=torch.long)
    for n in buckets:
        feats = torch.zeros(n, int(cfg.num_mel_bins), frames, device=p.device, dtype=p.dtype)
        self.encoder_runner(feats, pos, None)
```

Notes: `torch.compile` is lazy (it compiles on the first forward per shape), so wrapping in `__init__` before weights load is safe — `load_weights` targets the untouched `self.whisper_encoder` and the compiled wrapper shares the same tensors. **Warmup is mandatory** — without it the first request of each new audio length stalls ~20–87 s. `dynamic=False` avoids the torch 2.11 inductor crash. All no-ops unless the flag is set, so it's a ready-to-flip option, not a default.

### Chrome traces
`docs/traces/moss_td_encoder_{eager,compiled}.json.gz` — the compiled one is `dynamic=False` (the recommended config). Drag into https://ui.perfetto.dev (no decompress needed). Eager ~78k events vs compiled ~59k (fused kernels); the `ampere_bf16_gemm` and `flash_fwd_kernel` rows are unchanged between the two.

### Reproduce
```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/profile_moss_encoder.py \
    --batches 1 16 --iters 50 --warmup 10 --trace-dir docs/traces
# --dynamic false   -> static per-shape compile: ~20% faster, batch=16 compiles cleanly
# --dynamic true    -> default; ~10% faster, batch=16 raises InductorError
# --cuda-graph      -> also time a manual CUDA-graph capture (eager): +15% @b1, +0% @b16
# --clear-inductor-cache -> the cold-compile number
```
The script instantiates the encoder alone (TP=1 init, random weights — timing is weight-independent), runs eager vs `torch.compile(dynamic=True)`, and prints the latency table, `dynamo.explain`, profiler breakdown + traces, and the recompile probe. It isolates the encoder (excludes VQ-adaptor / time-merge / scheduler) — the exact `is_encoder` SDPA path the server runs, so numbers transfer (server-side encode measured 11.1 ms vs the script's 11.3 ms).

## Caveat

A100-40GB only. On H100/H200/B200 the GEMM+flash get faster, so the fixed launch/dispatch overhead becomes a **larger** fraction of the encoder — so compile's launch-trimming (and the separately-tracked CUDA-graph lever) could matter more there. This conclusion is scoped to A100 + torch 2.11 + this long-audio workload; short-audio ASR also raises the encoder's e2e share.
