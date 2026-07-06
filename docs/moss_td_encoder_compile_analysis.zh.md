# 决策:不在 MOSS-Transcribe-Diarize 的 Whisper encoder 上启用 `torch.compile`

> English version: [moss_td_encoder_compile_analysis.md](./moss_td_encoder_compile_analysis.md)

**测试范围(Scope):** A100-40GB · bf16 · torch 2.11.0+cu130 · movies800(约 16 秒音频)。复现见文末 `scripts/profile_moss_encoder.py`。

## 决策

**不启用 encoder `torch.compile`。** 不是因为 compile 失败——它确实让**孤立的 encoder** 变快了(**`dynamic=True` ~10%,`dynamic=False` ~20%**,实测,N=50)。不采用的原因是:

1. **端到端(e2e)只提升 ~1% —— 已实测,延迟和吞吐都测了。** encoder 只占单个请求的 **~2.4%**(编码 11 ms vs 请求总延迟 465 ms @并发=1)。用 `dynamic=False` 做 server 端 before/after(这个提升**确实**传导到了 server —— 编码 11→9.4 ms):c=1 延迟 0.446→0.441 s(−1%)、c=16 吞吐 15.8→16.1 qps(+1.6%)—— 两者都在 run-to-run 噪声内。
2. **它不是免费的。** 代价是**每个不同的 `n_chunks` 形状**都要 ~87 s(冷)/ ~20 s(warm-cache)编译一次(音频长度多变 → 形状很多),外加编译路径的 dispatch/guard 开销和预热机制。
3. **瓶颈不在这里。** encoder 67% 的 GPU 时间已经是最优的 cuBLAS GEMM + flash-attention;真正的吞吐/延迟瓶颈是 **LLM decode**。

用真实的"每形状编译成本 + 复杂度"去换 ~0.5% 的 e2e 收益,不划算。(`dynamic=True` 还会在 batch=16 撞上 torch 2.11 的 `InductorError`,但 `dynamic=False` 能避开——所以**崩溃不是理由,经济账才是**。)

## 三个论点(每个配一个数)

1. **encoder 在 e2e 里只是薄薄一层——约 2.4%。** movies800、并发=1 实测:编码 **11.1 ms** vs 请求总延迟 **465 ms**(中位 436 ms,约 16 s 音频)。任何只针对 encoder 的优化都被这个占比封顶。

2. **encoder 三分之二的 GPU 时间已经是最优的 GEMM + flash-attention——compile 动不了。** 按 Self-CUDA 时间:`addmm`(cuBLAS ampere tensor-core)占 **50%**、flash-attention 占 **17%**,compile 输出的是**字节级一致**的 kernel(GEMM 98.7 vs 98.7 ms,flash 33.4 vs 33.2 ms)。只有剩下 ~33%(LayerNorm/GELU/残差 add + permute 拷贝)是可融合的。

3. **compile 对 encoder 的真实收益被 e2e 占比和每形状编译成本淹没。** 孤立编码(CUDA-event,N=50):eager **11.3 ± 0.06 ms** → **10.3 ms(`dynamic=True`,1.10×)** / **9.3 ms(`dynamic=False`,1.20×)**——统计上真实(它融合了那 ~33% 的 elementwise,GPU-busy 9.86→9.15 ms,减少 launch,trace 事件 78k→59k)。但即便 20% × 2.4% ≈ **0.5% e2e**,却要付出**每个 `n_chunks` 形状** ~87 s 冷 / ~20 s warm-cache 的编译代价。

## 成本 / 收益——为什么不启用

| | 细节 |
|---|---|
| **收益** | 孤立 encoder −10%(`dynamic=True`)到 **−20%(`dynamic=False`)**;**e2e ~0.5%**(encoder 占 e2e ~2.4%)。 |
| **成本 1——每形状编译卡顿** | ~87 s 冷 / ~20 s warm-cache,**每个不同的 `n_chunks` 形状**都要一次。会缓存到磁盘(`/tmp/torchinductor_*`)、跨重启复用,所以**不是每次启动都付**——但每个形状第一次出现时会卡住那个请求,而 `n_chunks` 随音频长度增长(长音频有几十种形状)。 |
| **成本 2——提升在 e2e 上测不出来** | 用 `dynamic=False` 时 −20% 的 encoder 提升**确实**传导到了 server 端编码(11→9.4 ms),但占 e2e ~2.4%,摊下来只是请求的 ~1% —— server c=1 延迟和 c=16 吞吐都只动了 ~1%,落在 run-to-run 噪声内。(`dynamic=True` 还额外有 CPU guard/dispatch 开销,连编码层面的提升都被吃掉了。) |
| **不算成本:batch=16 崩溃** | `dynamic=True` 会在 `tiling_utils.get_pw_red_splits` 抛 `InductorError`(torch 2.11 动态形状 bug),但 **`dynamic=False` 每个形状都能干净编译、而且更快**——所以崩溃是配置层面的产物,不是真正的阻碍。 |

**启动预热能不能消掉卡顿?** 大体可以——用 `dynamic=False` 在启动时把每个 `n_chunks` 形状都编一遍(几十个 × ~20 s warm-cache),这样请求就不卡了。这是一条真实可部署的路径。只是为了 decode-bound 负载上 ~0.5% 的 e2e 收益,**不值得**搭这套机制。

## 什么才真正能提 encoder(以及为什么仍然对 e2e 无感)

| 手段 | 实测 | 结论 |
|---|---|---|
| **Batching(批量)** | eager 11.3 ms @b1 → 7.5 ms/item @b16(~1.5×) | 单条收益最大,但 mm-embed 派发**每请求单独调一次** encoder(永远只有单个请求的 chunk),所以在 serving 路径上根本不 engage。 |
| **`torch.compile`** | encoder −10%(`dynamic=True`)/ −20%(`dynamic=False`),e2e ~1% | 不值得(本文)——e2e 占比小 + 每形状编译。 |
| **CUDA graph** | +15% @batch=1,+0% @batch=16(实测;此处仅用于**拆解** compile 的提升来源,见下) | 一个独立的 launch-overhead 手段,**不在本文(compile)范围内**(由单独的工作项负责)。列出只是为了说明 compile 在 batch=1 的提升里有多少来自 launch 开销、多少来自融合。 |

## 数据附录

### 孤立 encoder 延迟(CUDA-event,N=50,均值 ± 标准差)

| batch(n_chunks) | eager | `compile(dynamic=True)` | `compile(dynamic=False)` | CUDA graph(eager) |
|---|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3(1.10×) | **9.3(1.20×)** | **9.6(1.17×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError** | **100.8(1.20×)** | **119.9(1.00×)** |

编译墙钟(首次调用,每形状):**~87 s 冷 / ~20 s warm-cache**。`dynamic=False` 每个形状都能干净编译;`dynamic=True` 每形状重编译、并在 batch=16 崩溃。CUDA graph(`--cuda-graph`)只在 batch=1 有用(launch-bound);batch=16 时 encoder 已被计算占满,消掉 launch 开销没用 —— 而 compile 在 batch=16 的收益来自融合/消拷贝,不是 launch。

### 端到端 server before/after(`compile(dynamic=False)`,提升确实传导到 server)

| | eager | compiled | Δ |
|---|---|---|---|
| server 端编码 | ~11 ms | **9.4 ms** | −15%(提升**确实**到 server) |
| c=1 延迟均值 | 0.446 s | 0.441 s | −1%(噪声) |
| c=1 延迟 p95 | 0.897 s | 0.975 s | +8%(噪声) |
| c=16 吞吐 | 15.8 qps | 16.1 qps | +1.6%(run-to-run 噪声内) |
| c=16 延迟均值 | 0.913 s | 0.892 s | −2.3% |

encoder 层面的 −15~20% 是真的、现在也确实体现在了 server 端编码上 —— 但 e2e 无论如何只有 ~1%,和 ~2.4% 的 encoder 占比一致,且落在 run-to-run 波动之内。

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

真正"加 compile"的代码只有 **~12 行**,在 `sglang_omni/models/moss_transcribe_diarize/sglang_model.py`。compile 的决定就放在 encoder 创建处(`__init__`),调用点用 `self._enc(...)`,`warmup_encoder_compile()` 只负责在启动时按形状预触发编译:

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
`docs/traces/moss_td_encoder_{eager,compiled}.json.gz` —— compiled 那个是 `dynamic=False`(推荐-如果非做不可 的配置)。直接拖进 https://ui.perfetto.dev,无需解压。eager ~78k 事件 vs compiled ~59k(kernel 被融合);两者的 `ampere_bf16_gemm` 和 `flash_fwd_kernel` 行完全一样。

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
