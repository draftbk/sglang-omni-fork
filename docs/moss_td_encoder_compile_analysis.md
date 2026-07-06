# Decision: don't enable `torch.compile` on the MOSS-Transcribe-Diarize Whisper encoder

> 中文版: [moss_td_encoder_compile_analysis.zh.md](./moss_td_encoder_compile_analysis.zh.md)

**Scope:** A100-40GB · bf16 · torch 2.11.0+cu130 · movies800 (~16 s clips). Reproduce with `scripts/profile_moss_encoder.py` (see end).

## Decision

**Do not enable encoder `torch.compile`.** It is not that compile fails — it genuinely speeds the *isolated* encoder up (**~10% `dynamic=True`, ~20% `dynamic=False`**, measured, N=50). We're declining it because:

1. **The win is ~1% end-to-end — measured, both latency and throughput.** The encoder is only **~2.4% of a request** (encode 11 ms vs 465 ms total @c=1). Server before/after with `dynamic=False` (the encoder win *does* reach the server — encode 11→9.4 ms): c=1 latency 0.446→0.441 s (−1%), c=16 throughput 15.8→16.1 qps (+1.6%) — both within run-to-run noise.
2. **It isn't free.** Cost is an ~87 s cold / ~20 s warm-cache compile **per distinct `n_chunks` shape** (audio length varies → many shapes), plus compiled dispatch/guard overhead and warmup machinery.
3. **The bottleneck is elsewhere.** 67% of encoder GPU time is already-optimal cuBLAS GEMM + flash-attention; the real throughput/latency bottleneck is **LLM decode**.

Trading real per-shape compile cost + complexity for a ~0.5% e2e gain is not worth it. (`dynamic=True` additionally hits a torch 2.11 `InductorError` at batch=16, but `dynamic=False` avoids that — so the crash is **not** the reason; the economics are.)

## Three arguments (each one number)

1. **The encoder is a thin slice of e2e — ~2.4%.** Measured at concurrency=1 on movies800: encode **11.1 ms** vs total request latency **465 ms** (median 436 ms, ~16 s audio). Any encoder-only optimization is bounded by this.

2. **Two-thirds of encoder GPU time is already-optimal GEMM + flash-attention — compile can't touch it.** By Self-CUDA time: `addmm` (cuBLAS ampere tensor-core) **50%** and flash-attention **17%** are emitted **byte-identical** by compile (GEMM 98.7 vs 98.7 ms, flash 33.4 vs 33.2 ms). Only the remaining ~33% (LayerNorm/GELU/residual-add + permute copies) is fuseable.

3. **Compile's real encoder win is swamped by e2e share and per-shape compile cost.** Isolated encode (CUDA-event, N=50): eager **11.3 ± 0.06 ms** → **10.3 ms (`dynamic=True`, 1.10×)** / **9.3 ms (`dynamic=False`, 1.20×)** — statistically real (it fuses the ~33% elementwise, GPU-busy 9.86→9.15 ms, trims launches, trace events 78k→59k). But even 20% × 2.4% ≈ **0.5% e2e**, against an ~87 s cold / ~20 s warm-cache compile **per distinct `n_chunks` shape**.

## Cost / benefit — why not to enable

| | detail |
|---|---|
| **Benefit** | isolated encoder −10% (`dynamic=True`) to **−20% (`dynamic=False`)**; **e2e ~0.5%** (encoder is ~2.4% of e2e). |
| **Cost 1 — compile stall per shape** | ~87 s cold, ~20 s warm-cache, **per distinct `n_chunks` shape**. Cached on disk (`/tmp/torchinductor_*`) across restarts, so it is *not* per-boot — but the first time each shape is seen it stalls that request, and `n_chunks` grows with audio length (dozens of shapes for long audio). |
| **Cost 2 — the win is unmeasurable e2e** | with `dynamic=False` the −20% encoder win *does* reach server-side encode (11→9.4 ms), but at ~2.4% e2e share it is ~1% of the request — server c=1 latency and c=16 throughput both moved ~1%, inside run-to-run noise. (`dynamic=True` additionally has CPU guard/dispatch overhead that ate even the encode-level win.) |
| **Not a cost: the batch=16 crash** | `dynamic=True` raises `InductorError` in `tiling_utils.get_pw_red_splits` (torch 2.11 dynamic-shape bug), but **`dynamic=False` compiles every shape cleanly and is faster** — so the crash is a config artifact, not a real blocker. |

**Can startup warmup remove the stall?** Yes, mostly — with `dynamic=False` you'd compile each `n_chunks` shape at boot (dozens × ~20 s warm-cache) so no request stalls. That is a real, deployable path. It is just not *worth* the machinery for a ~0.5% e2e gain on a decode-bound workload.

## What actually moves the encoder (and why it's still not e2e)

| lever | measured | verdict |
|---|---|---|
| **Batching** | eager 11.3 ms @b1 → 7.5 ms/item @b16 (~1.5×) | Biggest per-item win, but the mm-embed dispatch calls the encoder **once per request** (always a single request's chunks), so it doesn't engage on the serving path. |
| **`torch.compile`** | −10% (`dynamic=True`) / −20% (`dynamic=False`) encoder, ~1% e2e | Not worth it (this doc) — small e2e share + per-shape compile. |
| **CUDA graph** | +15% @batch=1, +0% @batch=16 (measured; used here only to *attribute* compile's win, below) | A separate launch-overhead lever, **out of scope for this compile-focused doc** (tracked as its own work item). Included only to show how much of compile's batch=1 win is launch overhead vs fusion. |

## Data appendix

### Isolated encoder latency (CUDA-event, N=50, mean ± std)

| batch (n_chunks) | eager | `compile(dynamic=True)` | `compile(dynamic=False)` | CUDA graph (eager) |
|---|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3 (1.10×) | **9.3 (1.20×)** | **9.6 (1.17×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError** | **100.8 (1.20×)** | **119.9 (1.00×)** |

Compile wall: **~87 s cold / ~20 s warm-cache** (first call, per shape). `dynamic=False` compiles every shape cleanly; `dynamic=True` recompiles per shape and crashes at batch=16. CUDA graph (`--cuda-graph`) helps only at batch=1 (launch-bound); at batch=16 the encoder is compute-saturated so removing launch overhead does nothing — while compile's batch=16 win is fusion/copy-elimination, not launches.

### End-to-end server before/after (`compile(dynamic=False)`, encoder win reaches the server)

| | eager | compiled | Δ |
|---|---|---|---|
| server-side encode | ~11 ms | **9.4 ms** | −15% (win *does* reach server) |
| c=1 latency mean | 0.446 s | 0.441 s | −1% (noise) |
| c=1 latency p95 | 0.897 s | 0.975 s | +8% (noise) |
| c=16 throughput | 15.8 qps | 16.1 qps | +1.6% (within run-to-run noise) |
| c=16 latency mean | 0.913 s | 0.892 s | −2.3% |

The encoder-level −15–20% is real and now visibly reaches server-side encode — but e2e it is ~1% either way, consistent with the ~2.4% encoder share and inside run-to-run variance.

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

The actual "add compile" change is **~12 lines** in `sglang_omni/models/moss_transcribe_diarize/sglang_model.py`. The compile decision sits right where the encoder is built (`__init__`), the call site uses `self._enc(...)`, and `warmup_encoder_compile()` just pre-triggers the per-shape compilation at startup:

```python
# __init__, right after the encoder is created:
self.whisper_encoder = WhisperEncoder(config.audio_config, quant_config)
if os.getenv("MOSS_ENCODER_COMPILE") == "1":
    self._enc = torch.compile(self.whisper_encoder, dynamic=False)
else:
    self._enc = self.whisper_encoder

# call site:
whisper_features = self._enc(input_features, encoder_position_ids, forward_batch)

# startup (wire into the stage factory if you flip the flag):
def warmup_encoder_compile(self, buckets=(1, 2, 4, 8, 16, 32)):
    if os.getenv("MOSS_ENCODER_COMPILE") != "1":
        return
    cfg, p = self.config.audio_config, next(self.whisper_encoder.parameters())
    frames = int(cfg.max_source_positions) * 2
    pos = torch.arange((frames - 1) // 2 + 1, device=p.device, dtype=torch.long)
    for n in buckets:
        self._enc(torch.zeros(n, int(cfg.num_mel_bins), frames,
                              device=p.device, dtype=p.dtype), pos, None)
```

Notes: `torch.compile` is lazy (it compiles on the first forward per shape), so wrapping in `__init__` before weights load is safe — `load_weights` targets the untouched `self.whisper_encoder` and the compiled wrapper shares the same tensors. **Warmup is mandatory** — without it the first request of each new audio length stalls ~20–87 s. `dynamic=False` avoids the torch 2.11 inductor crash. All no-ops unless the flag is set, so it's a ready-to-flip option, not a default.

### Chrome traces
`docs/traces/moss_td_encoder_{eager,compiled}.json.gz` — the compiled one is `dynamic=False` (the recommended-if-you-must config). Drag into https://ui.perfetto.dev (no decompress needed). Eager ~78k events vs compiled ~59k (fused kernels); the `ampere_bf16_gemm` and `flash_fwd_kernel` rows are unchanged between the two.

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
