# Does `torch.compile` help the MOSS-Transcribe-Diarize Whisper encoder?

**Scope:** A100-40GB · bf16 · torch 2.11.0+cu130 · `torch.compile(dynamic=True)` · movies800 (~16 s clips). Reproduce with `scripts/profile_moss_encoder.py` (see end).

## Verdict

**No — don't enable it (by default).** `torch.compile` *does* make the isolated encoder faster — **~10% with `dynamic=True`, ~20% with `dynamic=False`** (real, not noise) — but the encoder is only **~2.4% of end-to-end**, so even the 20% is **~0.5% of e2e**, and it costs an ~87 s cold / ~20 s warm-cache compile **per input shape** (and `n_chunks` varies per request). The bottleneck is **LLM decode**, not the encoder. (Note: `dynamic=True` also raises a hard `InductorError` at batch=16 — a torch 2.11 bug — but that is avoidable with `dynamic=False`, so it is *not* the reason against.)

## Three arguments (each one number)

1. **The encoder is a thin slice of e2e — ~2.4%.** Measured at concurrency=1 on movies800: encode **11.1 ms** vs total request latency **465 ms** (median 436 ms, ~16 s audio). Any encoder-only optimization is bounded by this.

2. **Two-thirds of encoder GPU time is already-optimal GEMM + flash-attention — compile can't touch it.** By Self-CUDA time: `addmm` (cuBLAS ampere tensor-core) **50%** and flash-attention **17%** are emitted **byte-identical** by compile (GEMM 98.7 vs 98.7 ms, flash 33.4 vs 33.2 ms). Only the remaining ~33% (LayerNorm/GELU/residual-add + permute copies) is fuseable.

3. **Compile's real encoder win is swamped by e2e share and per-shape compile cost.** Isolated encode (CUDA-event, N=50): eager **11.3 ± 0.06 ms** → **10.3 ms (`dynamic=True`, 1.10×)** / **9.3 ms (`dynamic=False`, 1.20×)** — statistically real (it fuses the ~33% elementwise, GPU-busy 9.86→9.15 ms, trims launches, trace events 78k→59k). But even 20% × 2.4% ≈ **0.5% e2e**, against an ~87 s cold / ~20 s warm-cache compile **per distinct `n_chunks` shape**.

## Cost / benefit — why not to enable

| | detail |
|---|---|
| **Benefit** | isolated encoder −10% (`dynamic=True`) to **−20% (`dynamic=False`)**; **e2e ~0.5%** (encoder is ~2.4% of e2e). |
| **Cost 1 — compile stall per shape** | ~87 s cold, ~20 s warm-cache, **per distinct `n_chunks` shape**. Cached on disk (`/tmp/torchinductor_*`) across restarts, so it is *not* per-boot — but the first time each shape is seen it stalls that request, and `n_chunks` grows with audio length (dozens of shapes for long audio). |
| **Cost 2 — dispatch overhead** | compiled path adds CPU guard/dispatch overhead; in the full server path the encoder win did not survive to request latency. |
| **Not a cost: the batch=16 crash** | `dynamic=True` raises `InductorError` in `tiling_utils.get_pw_red_splits` (torch 2.11 dynamic-shape bug), but **`dynamic=False` compiles every shape cleanly and is faster** — so the crash is a config artifact, not a real blocker. |

**Can startup warmup remove the stall?** Yes, mostly — with `dynamic=False` you'd compile each `n_chunks` shape at boot (dozens × ~20 s warm-cache) so no request stalls. That is a real, deployable path. It is just not *worth* the machinery for a ~0.5% e2e gain on a decode-bound workload.

## What actually moves the encoder (and why it's still not e2e)

| lever | measured | verdict |
|---|---|---|
| **Batching** | eager 11.3 ms @b1 → 7.5 ms/item @b16 (~1.5×) | Biggest per-item win, but the mm-embed dispatch calls the encoder **once per request** (always a single request's chunks), so it doesn't engage on the serving path. |
| **CUDA graph** | would remove the launch/dispatch gap (~1 ms of the 11 ms) | The only thing targeting small-batch overhead; needs a captured graph per shape. ~1 ms of a ~2.4%-of-e2e encoder. |
| **`torch.compile`** | −10% (`dynamic=True`) / −20% (`dynamic=False`) encoder, ~0.5% e2e | Not worth it (this doc) — small e2e share + per-shape compile. |

## Data appendix

### Isolated encoder latency (CUDA-event, N=50, mean ± std)

| batch (n_chunks) | eager | `torch.compile(dynamic=True)` | `torch.compile(dynamic=False)` |
|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3 ± 0.01 ms (1.10×) | **9.3 ± 0.02 ms (1.20×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError** (torch 2.11 dyn-shape bug) | **100.8 ± 0.03 ms (1.20×)** |

Compile wall: **~87 s cold / ~20 s warm-cache** (first call, per shape). `dynamic=False` compiles every shape cleanly; `dynamic=True` recompiles per shape and crashes at batch=16.

### Kernel breakdown (torch.profiler, batch=1, 20 iters, Self-CUDA)

| kernel | eager | compiled |
|---|---|---|
| `addmm` GEMM (cuBLAS ampere) | 50% / 98.7 ms | **98.7 ms (identical)** |
| flash attention | 17% / 33.4 ms | **33.2 ms (identical)** |
| LayerNorm / GELU / add | ~20% | fused → `triton_*_fused_*` |
| copy / clone / contiguous (permute) | ~10% / 19.1 ms | **≈ gone** (folded into `*_view` triton) |
| GPU-busy / iter | **9.86 ms** | **9.15 ms** (−7%) |

`torch._dynamo.explain`: **graphs=1, graph_breaks=0, ops=344** (encoder self-attention is `scaled_dot_product_attention`, fully traceable — not RadixAttention).

### Chrome traces
`docs/traces/moss_td_encoder_{eager,compiled}.json.gz` (drag into https://ui.perfetto.dev — no decompress needed). Eager 78k events vs compiled 59k (fused kernels); the `ampere_bf16_gemm` and `flash_fwd_kernel` rows are unchanged between the two.

### Reproduce
```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/profile_moss_encoder.py \
    --batches 1 16 --iters 50 --warmup 10 --trace-dir docs/traces
# --dynamic false   -> static per-shape compile: ~20% faster, batch=16 compiles cleanly
# --dynamic true    -> default; ~10% faster, batch=16 raises InductorError
# --clear-inductor-cache -> the cold-compile number
```
The script instantiates the encoder alone (TP=1 init, random weights — timing is weight-independent), runs eager vs `torch.compile(dynamic=True)`, and prints the latency table, `dynamo.explain`, profiler breakdown + traces, and the recompile probe. It isolates the encoder (excludes VQ-adaptor / time-merge / scheduler) — the exact `is_encoder` SDPA path the server runs, so numbers transfer (server-side encode measured 11.1 ms vs the script's 11.3 ms).

## Caveat

A100-40GB only. On H100/H200/B200 the GEMM+flash get faster, so the fixed launch/dispatch overhead becomes a **larger** fraction of the encoder — CUDA-graph (and compile's launch-trimming) could matter more there. This conclusion is scoped to A100 + torch 2.11 + this long-audio workload; short-audio ASR also raises the encoder's e2e share.
