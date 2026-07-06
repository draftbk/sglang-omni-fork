# `torch.compile` 在 MOSS-Transcribe-Diarize Whisper encoder 上的实测影响

> English version: [moss_td_encoder_compile_analysis.md](./moss_td_encoder_compile_analysis.md)

**测试范围(Scope):** A100-40GB · bf16 · torch 2.11.0+cu130 · movies800(约 16 秒音频)· greedy。复现见文末 `scripts/profile_moss_encoder.py`。

## 摘要

`torch.compile(dynamic=False)` 对 Whisper encoder 是一个**真实、且不损失正确性**的提升:**孤立 encoder +20%**(N=50)、**端到端吞吐 +3.8%**(c=8,3 次运行区间不重叠,CER 不变)。这是个合理的优化 —— 但有代价,而且 **CUDA graph(单独负责)是同类收益里更便宜的手段**,所以对 encoder 我们更倾向 CUDA graph。

启用 compile 的注意点:
- **每个不同的 `n_chunks` 形状**都要编一次(~20 s warm-cache / ~87 s 冷),需在启动时预热 —— 而在 40 GB 上,预热**大 chunk 桶会 OOM**。
- `dynamic=True` 在大 chunk 数时撞 torch 2.11 的 `InductorError` → 用 **`dynamic=False`**。
- **c=16 在 40 GB 上受 OOM 限制**(完成 56–82/96),所以干净的吞吐对比放在 c=8。

## 端到端吞吐(c=8,greedy,每次 run 一个全新 server,96 样本)

| | run1 / run2 / run3 | worst | mean | vs eager | CER |
|---|---|---|---|---|---|
| eager | 10.06 / 10.19 / 10.00 | 10.00 | 10.08 | — | 5.12 |
| **compile(dynamic=False)** | 10.29 / 10.53 / 10.60 | 10.29 | **10.47** | **+3.8%** | 5.14 |

两组区间**不重叠**(comp 最差 10.29 > eager 最好 10.19)→ 真实,非噪声;CER 不变 → 输出保持一致。必须用 fresh-server-per-run —— 常驻 server 的第 2 次起 eval 会退化(见文末)。c=1 单请求延迟只动 ~1%(encoder 只占单请求 ~2.4%);**吞吐**收益更大,因为加速 prefill 释放了 GPU 给 decode 批量。

## 三个论点(每个配一个数)

1. **encoder 在单请求延迟里只是薄薄一层——约 2.4%。** movies800、并发=1 实测:编码 **11.1 ms** vs 请求总延迟 **465 ms**(中位 436 ms,约 16 s 音频)。但吞吐收益不受这个占比封顶(见论点 3)。

2. **encoder 三分之二的 GPU 时间已经是最优的 GEMM + flash-attention——compile 动不了。** 按 Self-CUDA 时间:`addmm`(cuBLAS ampere tensor-core)占 **50%**、flash-attention 占 **17%**,compile 输出的是**字节级一致**的 kernel(GEMM 98.7 vs 98.7 ms,flash 33.4 vs 33.2 ms)。只有剩下 ~33%(LayerNorm/GELU/残差 add + permute 拷贝)是可融合的。

3. **孤立 encoder 提升 ~20%,并传导为 +3.8% 吞吐。** 孤立编码(CUDA-event,N=50):eager **11.3 ± 0.06 ms** → **10.3 ms(`dynamic=True`,1.10×)** / **9.3 ms(`dynamic=False`,1.20×)**——统计上真实(融合那 ~33% elementwise,GPU-busy 9.86→9.15 ms,减少 launch,trace 事件 78k→59k)。这 20% 的 encoder 提升在 **c=8 表现为 +3.8% 吞吐**(上表)——比 ~1% 的单请求延迟占比大,因为加速 prefill 释放了 GPU 给 decode 批量。代价:每个 `n_chunks` 形状一次编译(~20 s warm-cache)。

## 成本 / 收益

| | 细节 |
|---|---|
| **收益** | 孤立 encoder **−20%**(`dynamic=False`);**e2e 吞吐 +3.8%**(c=8,干净、不损失正确性)。 |
| **成本 1——每形状编译** | ~87 s 冷 / ~20 s warm-cache,**每个不同的 `n_chunks` 形状**。会缓存到磁盘(`/tmp/torchinductor_*`)、跨重启复用,所以**不是每次启动都付**——但每个没见过的形状第一次会卡住那个请求(除非启动时预热),而 `n_chunks` 随音频长度增长。 |
| **成本 2——预热在 40 GB 上会 OOM** | KV pool 占满后,启动时预热大 chunk 桶(16/32)会 OOM;movies800(1-chunk 音频)只预热 `(1,)` 就够、且便宜。长音频部署需要桶覆盖 + padding。 |
| **成本 3——版本脆弱** | `dynamic=True` 在大 chunk 数时抛 `InductorError`(`tiling_utils.get_pw_red_splits`,torch 2.11 动态形状 bug)。`dynamic=False` 规避且更快 —— 所以用 `dynamic=False`。 |

**建议。** encoder **值得**优化(真实 +3.8% 吞吐、不损失正确性)。`torch.compile(dynamic=False)` + 启动预热能拿到。但 **CUDA graph 是同类收益里更便宜的手段**(捕获近乎瞬时、无 ~87 s 编译、无版本崩溃),所以要优化 encoder 优先选 CUDA graph;`torch.compile` 是可用的备选。

## 能提 encoder 的手段

| 手段 | 实测 | 说明 |
|---|---|---|
| **`torch.compile(dynamic=False)`** | encoder −20%,**e2e 吞吐 +3.8%(c=8)** | 真实、不损失正确性(本文)。代价:每形状启动预热 + 版本脆弱。 |
| **CUDA graph** | +15% @batch=1,+0% @batch=16 | 同类 launch-overhead 收益里更便宜的手段;**独立工作项**,不在本文范围。列出用于拆解 compile 在 batch=1 的提升里有多少来自 launch、多少来自融合。 |
| **Batching(批量)** | eager 11.3 ms @b1 → 7.5 ms/item @b16(~1.5×) | 单条收益最大,但 mm-embed 派发**每请求单独调一次** encoder,所以在 serving 路径上不 engage。 |

## 数据附录

### 孤立 encoder 延迟(CUDA-event,N=50,均值 ± 标准差)

| batch(n_chunks) | eager | `compile(dynamic=True)` | `compile(dynamic=False)` | CUDA graph(eager) |
|---|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3(1.10×) | **9.3(1.20×)** | **9.6(1.17×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError** | **100.8(1.20×)** | **119.9(1.00×)** |

编译墙钟(首次调用,每形状):**~87 s 冷 / ~20 s warm-cache**。`dynamic=False` 每个形状都能干净编译;`dynamic=True` 每形状重编译、并在 batch=16 崩溃。CUDA graph(`--cuda-graph`)只在 batch=1 有用(launch-bound);batch=16 时 encoder 已被计算占满,消掉 launch 开销没用 —— 而 compile 在 batch=16 的收益来自融合/消拷贝,不是 launch。

### 端到端(`compile(dynamic=False)`,每次 run 全新 server,greedy,96 样本)

| 指标 | eager | compiled | Δ |
|---|---|---|---|
| server 端编码 | ~11 ms | 9.4 ms | −15%(提升传导到 server) |
| **吞吐 c=8**(3 次 worst/mean) | 10.0 / 10.08 qps | **10.29 / 10.47 qps** | **worst +2.9% / mean +3.8%**(区间不重叠) |
| 延迟 c=1 均值 | 0.446 s | 0.441 s | −1%(encoder 只占单请求 ~2.4%) |

**c=16 在 40 GB 上不是干净对比** —— 部分 OOM(完成 56–82/96)使吞吐被噪声主导(名义 eager 13.1 / comp 12.5,但 comp 的低均值来自一次 56/96 的 OOM run,不是回归)。干净、非 OOM 的对比就是上面的 c=8。

**注意 —— server 在第一次 eval 后会退化。** 在**常驻** server 上,第 2 次起的 eval 会产出垃圾输出(一次 96 样本 greedy eval:第一次 CER 5.0,相同 reps 第二次 67.9)。本文所有数字都用**每次 run 一个全新 server**(CUDA-graph 工作项也是这么做的)。这看起来是跨 eval 边界的状态泄漏(mm/radix cache 或 scheduler),值得单开一个 issue —— 与 `torch.compile` 无关。

### Kernel 拆分(torch.profiler,batch=1,20 iters,Self-CUDA)

| kernel | eager | compiled |
|---|---|---|
| `addmm` GEMM(cuBLAS ampere) | 50% / 98.7 ms | **98.7 ms(完全一致)** |
| flash attention | 17% / 33.4 ms | **33.2 ms(完全一致)** |
| LayerNorm / GELU / add | ~20% | 融合 → `triton_*_fused_*` |
| copy / clone / contiguous(permute) | ~10% / 19.1 ms | **≈ 消失**(折进 `*_view` triton) |
| GPU-busy / iter | **9.86 ms** | **9.15 ms**（−7%） |

`torch._dynamo.explain`:**graphs=1, graph_breaks=0, ops=344**(encoder 的 self-attention 是 `scaled_dot_product_attention`,完全可 trace——不是 RadixAttention)。

### 参考实现(gated,默认关)

真正"加 compile"的代码只有 **~12 行**,在 `sglang_omni/models/moss_transcribe_diarize/sglang_model.py`。compile 的决定就放在 encoder 创建处(`__init__`),调用点用 `self.encoder_runner(...)`,`warmup_encoder_compile()` 只负责在启动时按形状预触发编译:

```python
# __init__,紧跟在 encoder 创建之后:
self.whisper_encoder = WhisperEncoder(config.audio_config, quant_config)
if os.getenv("MOSS_ENCODER_COMPILE") == "1":
    self.encoder_runner = torch.compile(self.whisper_encoder, dynamic=False)
else:
    self.encoder_runner = self.whisper_encoder

# 调用点:
whisper_features = self.encoder_runner(input_features, encoder_position_ids, forward_batch)

# 启动时(要开这个 flag 就接到 stage 工厂里):
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

说明:`torch.compile` 是惰性的(首次前向、按形状才真正编译),所以在 `__init__`、权重加载前 wrap 是安全的 —— `load_weights` 作用于未动的 `self.whisper_encoder`,编译后的 wrapper 共享同一批 tensor。**warmup 是必须的** —— 不做的话每个新音频长度的第一个请求会卡 ~20–87 s。`dynamic=False` 规避 torch 2.11 的 inductor 崩溃。flag 未设时全是 no-op —— 是"随时可开"的选项,而非默认。

### Chrome traces
`docs/traces/moss_td_encoder_{eager,compiled}.json.gz` —— compiled 那个是 `dynamic=False`(推荐配置)。直接拖进 https://ui.perfetto.dev,无需解压。eager ~78k 事件 vs compiled ~59k(kernel 被融合);两者的 `ampere_bf16_gemm` 和 `flash_fwd_kernel` 行完全一样。

### 复现
```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/profile_moss_encoder.py \
    --batches 1 16 --iters 50 --warmup 10 --trace-dir docs/traces
# --dynamic false   -> 静态每形状编译:快约 20%,batch=16 干净编译
# --dynamic true    -> 默认;快约 10%,batch=16 抛 InductorError
# --cuda-graph      -> 额外测一次手动 CUDA graph 捕获(eager):+15% @b1,+0% @b16
# --clear-inductor-cache -> 得到冷编译数字
```
脚本单独实例化 encoder(TP=1 初始化,随机权重——计时与权重无关),跑 eager vs `torch.compile`,打印延迟表、`dynamo.explain`、profiler 拆分 + traces、以及 recompile 探针。它隔离测 encoder(不含 VQ-adaptor / time-merge / scheduler)——正是 server 跑的那条 `is_encoder` SDPA 路径,所以数字可迁移(server 端实测编码 11.1 ms vs 脚本 11.3 ms)。

## 注意事项(Caveat)

仅在 A100-40GB 上测过。在 H100/H200/B200 上 GEMM+flash 更快,固定的 launch/dispatch 开销在 encoder 里占比会**更大**——那时 compile 削减 launch 的作用(以及单独负责的 CUDA-graph 手段)可能更有意义。本结论限定在 A100 + torch 2.11 + 这种长音频负载;短音频 ASR 也会抬高 encoder 在 e2e 中的占比。
