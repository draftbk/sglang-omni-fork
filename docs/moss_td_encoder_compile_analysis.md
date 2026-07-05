# MOSS-Transcribe-Diarize: why `torch.compile` doesn't speed up the Whisper encoder

**Model:** OpenMOSS-Team/MOSS-Transcribe-Diarize · **HW:** A100-40GB, bf16
**Method:** in-process `torch.profiler` (CPU+CUDA) + `torch._dynamo.explain`, eager vs compiled, measured on the real serving path (env-gated instrumentation, not committed).

## TL;DR

Applying `torch.compile` to the Whisper encoder is a **wash (~0 end-to-end), not a regression** — because there is almost nothing for it to speed up:

- **~67% of encoder GPU time is already-optimal GEMM (cuBLAS) + flash-attention.** Compile leaves these kernels byte-identical.
- Compile *does* fuse the remaining ~33% (LayerNorm / GELU / residual-add) and eliminates the permute/reshape **memory copies** — cutting GPU-busy ~9.85 → 9.14 ms (~7%).
- That ~7% (~0.7 ms) is **within run-to-run noise of the ~11 ms encode** and is offset by compile's own CPU dispatch/guard overhead at batch=1 → **no measurable warm speedup**.
- It is **not** a graph-coverage problem: the encoder compiles into **1 graph, 0 graph breaks**.
- The encoder is **~1% of end-to-end** (11 ms encode vs 1–2 s LLM decode), so even a hypothetical 2× encoder moves e2e < 1%.

## Measurements

### Warm encode latency (un-profiled — the reliable number)

| batch (n_chunks) | eager | compiled |
|---|---|---|
| 1 | **11.2 ms** | **11.2 ms** (compile ≈ 0) |
| 16 | 123 ms (7.7 ms/item, ~1.45× vs solo) | `InductorError` on the 2nd dynamic shape |

First-call compile cost: **~21–87 s** (paid once per input shape).

### Kernel breakdown (profiler, batch=1, 20 iters, by Self-CUDA time)

| kernel | eager | compiled | note |
|---|---|---|---|
| `addmm` (qkv/out/fc1/fc2 GEMM) | 49.9% / 98.34 ms | 53.8% / **98.36 ms** | identical `ampere_bf16_gemm` (cuBLAS) — untouched |
| flash attention | 16.9% / 33.3 ms | 18.2% / **33.2 ms** | identical `flash_fwd_kernel` — untouched |
| LayerNorm | 4.5% | fused → `triton_*_fused_native_layer_norm` | fused |
| residual add | 11.0% | fused → `triton_poi_fused_add_view` | fused |
| **copy / clone / contiguous** (permute+reshape) | **9.9% / 19.4 ms** | **≈ gone** (folded into `*_view` triton) | eliminated |
| GELU / mul | 5.8% | fused → `triton_poi_fused_gelu_view` | fused |
| **GPU-busy / iter** | **9.85 ms** | **9.14 ms** (−7%) | |

### `torch._dynamo.explain`

```
graphs = 1    graph_breaks = 0    ops = 344
```
The whole encoder is captured in a single graph — its self-attention uses PyTorch's `scaled_dot_product_attention` (flash), which is fully traceable.

### Launch / idle

Profiler idle-fraction: 0.42 (eager) → 0.20 (compiled) — compile does trim kernel count/launches, but profiler CPU overhead inflates this. Real idle ≈ (11.2 − 9.85) ≈ **1.35 ms (~12%)**. The compiled path adds noticeable **CPU dispatch/guard overhead** (worse with `dynamic=True`), which cancels the GPU saving when the GPU only has ~9 ms of work.

## Root cause

1. **Compute is already optimal.** GEMM → cuBLAS ampere tensor-core kernels; attention → flash. That is 67% of GPU time and compile emits the exact same kernels. Nothing to gain.
2. **The fuseable part is small.** The ~33% of elementwise/memory (LayerNorm/GELU/add + permute-copies) is what inductor fuses — real, but only ~0.7 ms at batch=1, lost in noise.
3. **Compile's own overhead cancels it.** At batch=1 the GPU does only ~9 ms of work; the compiled dispatch/guard path adds CPU overhead not hidden by such small GPU work.
4. **Variable batch is hostile to compile.** `n_chunks` varies per request → `dynamic=True` either recompiles per shape (~87 s each) or errors (`InductorError` at batch=16).

## What *would* move the encoder (and why it still doesn't matter for e2e)

| lever | ceiling | verdict |
|---|---|---|
| **Batching** (fill GPU / bigger GEMM) | 11.2 → 7.7 ms/item at batch 16 (~1.45×) | Real per-item win, but the multimodal embed dispatch invokes the encoder **once per request** (it always sees a single request's chunks), so cross-request batching does not engage on the serving path today. |
| **CUDA graph** | removes the ~1.35 ms launch/dispatch → ~10 ms | Targets the one thing compile can't at batch=1; needs static shapes (a captured graph per `n_chunks`), and it's ~1 ms of an 11 ms encode. |
| **`torch.compile`** | ~0 measured | Not worth it: no speedup + per-shape recompile/crash + 21–87 s warmup. |

## Conclusion

The Whisper encoder is a small, already-well-optimized slice (~1% of end-to-end). `torch.compile` correctly fuses the little that is fuseable, but the dominant GEMM+flash is untouchable and the net is within noise. For MOSS-Transcribe-Diarize the throughput/latency bottleneck is **LLM decode**, not the encoder — optimization effort (including the decode-side `torch.compile` path) belongs there.

### How to reproduce

Instrumentation was env-gated in `get_audio_feature` (not committed):
- `torch._dynamo.explain(encoder)(...)` → graph/break/op counts.
- `torch.profiler.profile([CPU, CUDA])` around 20 warm encoder calls → `key_averages()` table + chrome trace + GPU-busy vs wall (idle fraction).
- eager vs `torch.compile(encoder, dynamic=True)`; batch sizes via tiling the mel batch dim.
