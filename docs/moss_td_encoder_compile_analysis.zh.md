# 决策:不在 MOSS-Transcribe-Diarize 的 Whisper encoder 上启用 `torch.compile`

> English version: [moss_td_encoder_compile_analysis.md](./moss_td_encoder_compile_analysis.md)

**测试范围(Scope):** A100-40GB · bf16 · torch 2.11.0+cu130 · movies800(约 16 秒音频)。复现见文末 `scripts/profile_moss_encoder.py`。

## 决策

**不启用 encoder `torch.compile`。** 不是因为 compile 失败——它确实让**孤立的 encoder** 变快了(**`dynamic=True` ~10%,`dynamic=False` ~20%**,实测,N=50)。不采用的原因是:

1. **端到端(e2e)只提升 ~0.5%。** encoder 只占单个请求的 **~2.4%**(编码 11 ms vs 请求总延迟 465 ms @并发=1),所以对它提 20% 摊到 e2e ≈ 0.5%——低于请求延迟噪声,而且在 server 路径里根本没显出来。
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
| **成本 2——dispatch 开销** | 编译路径增加 CPU guard/dispatch 开销;在完整 server 路径里,encoder 的提升没能传导到请求延迟。 |
| **不算成本:batch=16 崩溃** | `dynamic=True` 会在 `tiling_utils.get_pw_red_splits` 抛 `InductorError`(torch 2.11 动态形状 bug),但 **`dynamic=False` 每个形状都能干净编译、而且更快**——所以崩溃是配置层面的产物,不是真正的阻碍。 |

**启动预热能不能消掉卡顿?** 大体可以——用 `dynamic=False` 在启动时把每个 `n_chunks` 形状都编一遍(几十个 × ~20 s warm-cache),这样请求就不卡了。这是一条真实可部署的路径。只是为了 decode-bound 负载上 ~0.5% 的 e2e 收益,**不值得**搭这套机制。

## 什么才真正能提 encoder(以及为什么仍然对 e2e 无感)

| 手段 | 实测 | 结论 |
|---|---|---|
| **Batching(批量)** | eager 11.3 ms @b1 → 7.5 ms/item @b16(~1.5×) | 单条收益最大,但 mm-embed 派发**每请求单独调一次** encoder(永远只有单个请求的 chunk),所以在 serving 路径上根本不 engage。 |
| **CUDA graph** | 可消掉 launch/dispatch 间隙(11 ms 里约 1 ms) | 唯一针对小 batch 开销的手段;需要每个形状一张捕获的 graph。~1 ms 而已,且 encoder 只占 e2e ~2.4%。 |
| **`torch.compile`** | encoder −10%(`dynamic=True`)/ −20%(`dynamic=False`),e2e ~0.5% | 不值得(本文)——e2e 占比小 + 每形状编译。 |

## 数据附录

### 孤立 encoder 延迟(CUDA-event,N=50,均值 ± 标准差)

| batch(n_chunks) | eager | `torch.compile(dynamic=True)` | `torch.compile(dynamic=False)` |
|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3 ± 0.01 ms(1.10×) | **9.3 ± 0.02 ms(1.20×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError**(torch 2.11 动态形状 bug) | **100.8 ± 0.03 ms(1.20×)** |

编译墙钟(首次调用,每形状):**~87 s 冷 / ~20 s warm-cache**。`dynamic=False` 每个形状都能干净编译;`dynamic=True` 每形状重编译、并在 batch=16 崩溃。

### Kernel 拆分(torch.profiler,batch=1,20 iters,Self-CUDA)

| kernel | eager | compiled |
|---|---|---|
| `addmm` GEMM(cuBLAS ampere) | 50% / 98.7 ms | **98.7 ms(完全一致)** |
| flash attention | 17% / 33.4 ms | **33.2 ms(完全一致)** |
| LayerNorm / GELU / add | ~20% | 融合 → `triton_*_fused_*` |
| copy / clone / contiguous(permute) | ~10% / 19.1 ms | **≈ 消失**(折进 `*_view` triton) |
| GPU-busy / iter | **9.86 ms** | **9.15 ms**（−7%） |

`torch._dynamo.explain`:**graphs=1, graph_breaks=0, ops=344**(encoder 的 self-attention 是 `scaled_dot_product_attention`,完全可 trace——不是 RadixAttention)。

### Chrome traces
`docs/traces/moss_td_encoder_{eager,compiled}.json.gz`(直接拖进 https://ui.perfetto.dev,无需解压)。eager 78k 事件 vs compiled 59k(kernel 被融合);两者的 `ampere_bf16_gemm` 和 `flash_fwd_kernel` 行完全一样。

### 复现
```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/profile_moss_encoder.py \
    --batches 1 16 --iters 50 --warmup 10 --trace-dir docs/traces
# --dynamic false   -> 静态每形状编译:快约 20%,batch=16 干净编译
# --dynamic true    -> 默认;快约 10%,batch=16 抛 InductorError
# --clear-inductor-cache -> 得到冷编译数字
```
脚本单独实例化 encoder(TP=1 初始化,随机权重——计时与权重无关),跑 eager vs `torch.compile`,打印延迟表、`dynamo.explain`、profiler 拆分 + traces、以及 recompile 探针。它隔离测 encoder(不含 VQ-adaptor / time-merge / scheduler)——正是 server 跑的那条 `is_encoder` SDPA 路径,所以数字可迁移(server 端实测编码 11.1 ms vs 脚本 11.3 ms)。

## 注意事项(Caveat)

仅在 A100-40GB 上测过。在 H100/H200/B200 上 GEMM+flash 更快,固定的 launch/dispatch 开销在 encoder 里占比会**更大**——那时 CUDA-graph(以及 compile 削减 launch 的作用)可能更有意义。本结论限定在 A100 + torch 2.11 + 这种长音频负载;短音频 ASR 也会抬高 encoder 在 e2e 中的占比。
