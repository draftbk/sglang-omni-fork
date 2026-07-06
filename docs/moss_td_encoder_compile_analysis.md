# `torch.compile` support for the MOSS-Transcribe-Diarize Whisper encoder

> 中文版: [moss_td_encoder_compile_analysis.zh.md](./moss_td_encoder_compile_analysis.zh.md)

**Scope:** A100-40GB · bf16 · torch 2.11.0+cu130 · movies800 (~16 s clips) · greedy. Reproduce with `scripts/profile_moss_encoder.py` (see end).

## Summary

This adds an **opt-in, off-by-default** `torch.compile(dynamic=False)` path for the Whisper encoder, gated by `MOSS_ENCODER_COMPILE=1`, with per-shape warmup wired into startup. It is a real, measured throughput win:

- **+20% on the isolated encoder** (CUDA-event, N=50).
- **+3.8% end-to-end throughput @ c=8** and **+5.3% @ c=16** — non-overlapping across runs, at **matched output length** (see the fairness check below).
- **Accuracy cost is tiny but non-zero**: CER +~0.05 absolute (~1% relative). `torch.compile` is **not** bitwise-identical to eager, so greedy decoding diverges on rare tokens.

It is off by default because it has real costs (per-shape compile/warmup, version fragility) and because **CUDA graph — a separate work item — is a cheaper lever for the same launch-overhead gain**. This PR makes compile a supported, documented option for deployments that want it.

## How to enable

```
MOSS_ENCODER_COMPILE=1                 # wrap the encoder in torch.compile(dynamic=False)
MOSS_ENCODER_COMPILE_BUCKETS=1,2,4     # optional; n_chunks shapes to warm at startup (default: 1)
```

`warmup_encoder_compile()` is called from the stage factory after `init_device_graphs()`; it is a no-op unless the flag is set. Warmup is required — without it the first request of each new `n_chunks` shape stalls ~20–87 s. On 40 GB, warming large buckets can OOM (see costs), so the default warms only bucket `1` (movies800 is 1-chunk).

## End-to-end throughput

**c=8, greedy, fresh server per run, 96 samples** (default `mem_fraction_static`):

| | run1 / run2 / run3 | worst | mean | vs eager | CER |
|---|---|---|---|---|---|
| eager | 10.06 / 10.19 / 10.00 | 10.00 | 10.08 | — | 5.12 |
| **compile (dynamic=False)** | 10.29 / 10.53 / 10.60 | 10.29 | **10.47** | **+3.8%** | 5.14 |

**c=16, greedy, fresh server per run, 96 samples, `mem_fraction_static=0.5`** (first run per config discarded as cold disk-cache):

| | warm runs | mean | vs eager | CER |
|---|---|---|---|---|
| eager | 13.58 / 13.93 | 13.76 | — | 5.10–5.11 |
| **compile (dynamic=False)** | 14.60 / 14.37 / 14.51 | **14.49** | **+5.3%** | 5.15–5.16 |

Ranges don't overlap (comp worst 14.37 > eager best 13.93) → real, not noise.

### Fairness check (is the QPS comparison apples-to-apples?)

Same dataset/samples, greedy. Verified from per-sample logs:

| | total audio (s) | total output (chars) | mean out |
|---|---|---|---|
| eager | 1366.3 | 17144–17157 | 179 |
| compile | 1366.3 | 17161 | 179 |

Input is identical (same audio) and **output length matches within <0.1%** — compile's output is if anything marginally longer, so the throughput gain is **not** an artifact of shorter generation. The input-normalized metric agrees: `audio_throughput_s_per_s` 195.8 → 206.1 = +5.3%.

## Three arguments (each one number)

1. **The encoder is a thin slice of single-request latency — ~2.4%.** At concurrency=1: encode **11.1 ms** vs total request latency **465 ms** (~16 s audio). So c=1 latency barely moves; the win shows up in **throughput**, where speeding prefill frees GPU for decode batching.

2. **Two-thirds of encoder GPU time is already-optimal GEMM + flash-attention — compile can't touch it.** By Self-CUDA time: `addmm` (cuBLAS ampere tensor-core) **50%** and flash-attention **17%** are emitted **byte-identical** by compile (GEMM 98.7 vs 98.7 ms, flash 33.4 vs 33.2 ms). Only the remaining ~33% (LayerNorm/GELU/residual-add + permute copies) is fuseable.

3. **The isolated encoder speeds up ~20%, and it carries to +3.8–5.3% throughput.** Isolated encode (CUDA-event, N=50): eager **11.3 ± 0.06 ms** → **10.3 ms (`dynamic=True`, 1.10×)** / **9.3 ms (`dynamic=False`, 1.20×)** — fuses the ~33% elementwise, GPU-busy 9.86→9.15 ms, trace events 78k→59k. Cost: a compile per `n_chunks` shape (~20 s warm-cache).

## Cost / benefit

| | detail |
|---|---|
| **Benefit** | isolated encoder **−20%** (`dynamic=False`); **+3.8% throughput @ c=8, +5.3% @ c=16**. |
| **Cost 1 — accuracy is not bitwise-identical** | CER +~0.05 abs (~1% rel), consistent across runs. Fused LayerNorm/GELU change FP reduction order → rare greedy token flips. Negligible for most uses, but real. |
| **Cost 2 — compile per shape** | ~87 s cold / ~20 s warm-cache, **per distinct `n_chunks` shape**. Cached on disk (`/tmp/torchinductor_*`) across restarts, so *not* per-boot — but each unseen shape stalls its first request unless warmed at startup. |
| **Cost 3 — warmup can OOM on 40 GB** | warming large-chunk buckets (16/32) OOMs after the KV pool is reserved; on movies800 (1-chunk) warming bucket `1` is enough. Long-audio deployments need bucket coverage + padding. |
| **Cost 4 — version fragility** | `dynamic=True` raises `InductorError` in `tiling_utils.get_pw_red_splits` (torch 2.11 dynamic-shape bug) at larger chunk counts. `dynamic=False` avoids it and is faster — so use `dynamic=False`. |

**Recommendation.** Enable it where throughput matters and a ~1%-relative CER shift is acceptable, using `dynamic=False` + startup warmup. **CUDA graph (separate work item) is a cheaper lever for the same class of gain** (near-instant capture, no ~87 s compile, no version crash, bitwise-identical), so prefer it for the encoder; `torch.compile` is a valid, now-supported alternative.

## Note: the c=16 OOM was config, not hardware

The A100-40GB "can't do c=16" story turned out to be **KV-pool over-reservation**, not a real memory need — worth recording because it's independent of `torch.compile`:

- With the default `mem_fraction_static`, startup reserves a **29.4 GB KV pool (275,456 tokens)**, leaving only **7.67 GB** free.
- The workload's peak KV occupancy is **`token usage: 0.04–0.05` (~5%)** — the pool is 20× oversized for movies800 (~16 s clips → ~1 k tokens/request).
- At c=16, the thin 7.67 GB headroom can't absorb the encoder's transient O(L²) activations + CUDA-graph capture → OOM. **Not** because KV ran out (it's 5% used).
- Setting **`mem_fraction_static=0.5`** shrinks the pool to 17.7 GB / 165,913 tokens (still 20× the demand), frees **19.37 GB**, and c=16 completes **96/96**. Since only ~5% of the pool is ever used, compute/throughput are unaffected — this only converts wasted reserved VRAM into headroom.

A 2B-class model does not need 40 GB for this workload; the OOM was tunable. (This is a workload-specific tuning knob, not a shipped default — long-audio deployments that actually use the 72 k-token budget want a larger pool.)

## Data appendix

### Isolated encoder latency (CUDA-event, N=50, mean ± std)

| batch (n_chunks) | eager | `compile(dynamic=True)` | `compile(dynamic=False)` | CUDA graph (eager) |
|---|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3 (1.10×) | **9.3 (1.20×)** | **9.6 (1.17×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError** | **100.8 (1.20×)** | **119.9 (1.00×)** |

Compile wall: **~87 s cold / ~20 s warm-cache** (first call, per shape). CUDA graph helps only at batch=1 (launch-bound); at batch=16 the encoder is compute-saturated so removing launch overhead does nothing — while compile's batch=16 win is fusion/copy-elimination.

### Kernel breakdown (torch.profiler, batch=1, 20 iters, Self-CUDA)

| kernel | eager | compiled |
|---|---|---|
| `addmm` GEMM (cuBLAS ampere) | 50% / 98.7 ms | **98.7 ms (identical)** |
| flash attention | 17% / 33.4 ms | **33.2 ms (identical)** |
| LayerNorm / GELU / add | ~20% | fused → `triton_*_fused_*` |
| copy / clone / contiguous (permute) | ~10% / 19.1 ms | **≈ gone** (folded into `*_view` triton) |
| GPU-busy / iter | **9.86 ms** | **9.15 ms** (−7%) |

`torch._dynamo.explain`: **graphs=1, graph_breaks=0, ops=344** (the encoder's self-attention is `scaled_dot_product_attention`, fully traceable — not RadixAttention).

### Reference implementation (env-gated)

The "add compile" change lives in `sglang_omni/models/moss_transcribe_diarize/sglang_model.py` (compile decision at `__init__`, call site uses `self.encoder_runner(...)`) and one warmup call in `stages.py`:

```python
# __init__, right after the encoder is built:
self.whisper_encoder = WhisperEncoder(config.audio_config, quant_config)
if os.getenv("MOSS_ENCODER_COMPILE") == "1":
    self.encoder_runner = torch.compile(self.whisper_encoder, dynamic=False)
else:
    self.encoder_runner = self.whisper_encoder

# call site:
whisper_features = self.encoder_runner(input_features, encoder_position_ids, forward_batch)

# warmup (called from the stage factory after init_device_graphs):
def warmup_encoder_compile(self):
    if os.getenv("MOSS_ENCODER_COMPILE") != "1":
        return
    buckets = tuple(int(x) for x in os.getenv("MOSS_ENCODER_COMPILE_BUCKETS", "1").split(","))
    cfg, p = self.config.audio_config, next(self.whisper_encoder.parameters())
    frames = int(cfg.max_source_positions) * 2
    pos = torch.arange((frames - 1) // 2 + 1, device=p.device, dtype=torch.long)
    for n in buckets:
        feats = torch.zeros(n, int(cfg.num_mel_bins), frames, device=p.device, dtype=p.dtype)
        self.encoder_runner(feats, pos, None)
```

`torch.compile` is lazy (compiles on first forward per shape), so wrapping in `__init__` before weights load is safe — `load_weights` targets `self.whisper_encoder` and the compiled wrapper shares the same tensors. All no-ops unless the flag is set.

### Reproduce
```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/profile_moss_encoder.py \
    --batches 1 16 --iters 50 --warmup 10
# --dynamic false   -> static per-shape compile: ~20% faster, batch=16 compiles cleanly
# --dynamic true    -> default; ~10% faster, batch=16 raises InductorError
# --cuda-graph      -> also time a manual CUDA-graph capture (eager): +15% @b1, +0% @b16
# --clear-inductor-cache -> the cold-compile number
```
The script instantiates the encoder alone (TP=1 init, random weights — timing is weight-independent) — the exact `is_encoder` SDPA path the server runs, so numbers transfer (server-side encode 11.1 ms vs the script's 11.3 ms).

## Caveat

A100-40GB only. On H100/H200/B200 the GEMM+flash get faster, so fixed launch/dispatch overhead becomes a **larger** fraction of the encoder — compile's launch-trimming (and the separately-tracked CUDA-graph lever) could matter more there. Scoped to A100 + torch 2.11 + this long-audio workload; short-audio ASR raises the encoder's e2e share.
