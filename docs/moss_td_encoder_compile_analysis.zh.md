# MOSS-Transcribe-Diarize Whisper encoder 的 `torch.compile` 支持

> English version: [moss_td_encoder_compile_analysis.md](./moss_td_encoder_compile_analysis.md)

**测试范围(Scope):** A100-40GB · bf16 · torch 2.11.0+cu130 · movies800(约 16 秒音频)· greedy。复现见文末 `scripts/profile_moss_encoder.py`。

## 摘要

本 PR 为 Whisper encoder 加了一条**默认关闭、可选开启**的 `torch.compile(dynamic=False)` 路径,由 stage 的 `enable_torch_compile` server arg 控制(和 `fishaudio_s2_pro` 用的是同一个 flag),并在启动时按形状预热。这是一个真实、可测的吞吐提升:

- **孤立 encoder +20%**(CUDA-event,N=50)。
- **端到端吞吐 c=8 +3.8%、c=16 +5.3%** —— 多次运行区间不重叠,且**输出长度一致**(见下方公平性核对)。
- **精度代价小但非零**:CER +~0.05 绝对值(~1% 相对)。`torch.compile` 与 eager **不是逐位一致**的,所以 greedy 解码会在极少数 token 上分叉。

默认关闭的原因:它有真实代价(每形状编译/预热、版本脆弱),而且 **CUDA graph(单独的工作项)在同类 launch 开销收益上是更便宜的手段**。本 PR 把 compile 做成一个**受支持、有文档**的选项,给需要的部署用。

## 如何开启

在 stage 的 `factory_args`(pipeline config)里设 `enable_torch_compile`,和 `fishaudio_s2_pro` 编译它的 decoder 是同一套:

```yaml
factory_args:
  enable_torch_compile: true
  encoder_compile_buckets: [1, 2, 3, 4]   # 要预热的 n_chunks 形状(默认)
```

stage 工厂调 `compile_encoder(buckets)` —— 里面做 `set_torch_compile_config()`、`torch.compile(dynamic=False)`、并预热这些桶 —— 然后把 `server_args.enable_torch_compile = False`,免得 sglang 又去编译 LLM decoder。(`torch_compile_max_bs` 被校验强制要求、自动设为 `max_running_requests`,但 encoder 用不到它。)

**bucket 是 `n_chunks` 轴(音频长度),不是并发** —— encoder 每请求单独跑,batch 维是**一条**请求的 chunk 数(约 音频秒数/30),永远不是并发请求数。因为 `dynamic=False` 按**精确**形状编译,bucket 必须是**连续区间**:`1,2,3,4` 精确覆盖 ≤2 分钟音频,而 `1,2,4` 这种带空档的会漏掉 3-chunk 音频(它会在首次请求现场编译 ~20–87 s,之后缓存)。比顶桶更长的音频付这一次性编译。40 GB 上预热大桶会 OOM,想调高区间要配 `mem_fraction_static=0.5` 腾余量。

## 端到端吞吐

**c=8,greedy,每次 run 一个全新 server,96 样本**(默认 `mem_fraction_static`):

| | run1 / run2 / run3 | worst | mean | vs eager | CER |
|---|---|---|---|---|---|
| eager | 10.06 / 10.19 / 10.00 | 10.00 | 10.08 | — | 5.12 |
| **compile(dynamic=False)** | 10.29 / 10.53 / 10.60 | 10.29 | **10.47** | **+3.8%** | 5.14 |

**c=16,greedy,每次 run 一个全新 server,96 样本,`mem_fraction_static=0.5`**(每个配置第一次 run 作为冷磁盘缓存丢弃):

| | warm runs | mean | vs eager | CER |
|---|---|---|---|---|
| eager | 13.58 / 13.93 | 13.76 | — | 5.10–5.11 |
| **compile(dynamic=False)** | 14.60 / 14.37 / 14.51 | **14.49** | **+5.3%** | 5.15–5.16 |

区间不重叠(comp 最差 14.37 > eager 最好 13.93)→ 真实,非噪声。

### 公平性核对(QPS 对比是同口径吗?)

同数据集/样本,greedy。从 per-sample 日志核对:

| | 总音频(秒) | 总输出(字符) | 平均输出 |
|---|---|---|---|
| eager | 1366.3 | 17144–17157 | 179 |
| compile | 1366.3 | 17161 | 179 |

输入完全一致(同音频),**输出长度差异 <0.1%** —— compile 的输出甚至略长,所以吞吐提升**不是**"输出更短"的假象。输入归一化指标也一致:`audio_throughput_s_per_s` 195.8 → 206.1 = +5.3%。

## 三个论点(每个配一个数)

1. **encoder 在单请求延迟里只是薄薄一层——约 2.4%。** 并发=1:编码 **11.1 ms** vs 请求总延迟 **465 ms**(约 16 s 音频)。所以 c=1 延迟几乎不动;提升体现在**吞吐**上——加速 prefill 释放 GPU 给 decode 批量。

2. **encoder 三分之二的 GPU 时间已经是最优的 GEMM + flash-attention——compile 动不了。** 按 Self-CUDA 时间:`addmm`(cuBLAS ampere tensor-core)占 **50%**、flash-attention 占 **17%**,compile 输出**字节级一致**的 kernel(GEMM 98.7 vs 98.7 ms,flash 33.4 vs 33.2 ms)。只有剩下 ~33%(LayerNorm/GELU/残差 add + permute 拷贝)可融合。

3. **孤立 encoder 提升 ~20%,并传导为 +3.8–5.3% 吞吐。** 孤立编码(CUDA-event,N=50):eager **11.3 ± 0.06 ms** → **10.3 ms(`dynamic=True`,1.10×)** / **9.3 ms(`dynamic=False`,1.20×)**——融合那 ~33% elementwise,GPU-busy 9.86→9.15 ms,trace 事件 78k→59k。代价:每个 `n_chunks` 形状一次编译(~20 s warm-cache)。

## 成本 / 收益

| | 细节 |
|---|---|
| **收益** | 孤立 encoder **−20%**(`dynamic=False`);**吞吐 c=8 +3.8%、c=16 +5.3%**。 |
| **成本 1——精度非逐位一致** | CER +~0.05 绝对(~1% 相对),多次运行一致。融合的 LayerNorm/GELU 改变了 FP 归约顺序 → greedy 下极少数 token 翻转。多数场景可忽略,但真实存在。 |
| **成本 2——每形状编译** | ~87 s 冷 / ~20 s warm-cache,**每个不同 `n_chunks` 形状**一次。缓存到磁盘(`/tmp/torchinductor_*`)、跨重启复用,所以**不是每次启动都付**——但每个没见过的形状第一次会卡住那个请求(除非启动时预热)。 |
| **成本 3——预热在 40 GB 上会 OOM** | KV pool 占满后预热大 bucket(16/32)会 OOM;默认 `(1,2,3,4)` 安全。长音频部署调高区间需配 `mem_fraction_static=0.5` 腾余量。 |
| **成本 4——版本脆弱** | `dynamic=True` 在大 chunk 数时抛 `InductorError`(`tiling_utils.get_pw_red_splits`,torch 2.11 动态形状 bug)。`dynamic=False` 规避且更快——所以用 `dynamic=False`。 |

**建议。** 在乎吞吐、且能接受 ~1% 相对 CER 抖动的部署可以开,用 `dynamic=False` + 启动预热。但 **CUDA graph(单独工作项)是同类收益里更便宜的手段**(捕获近乎瞬时、无 ~87 s 编译、无版本崩溃、逐位一致),所以 encoder 优先选它;`torch.compile` 是一个可用、现已受支持的备选。

## 补充:c=16 的 OOM 是配置问题,不是硬件

A100-40GB "跑不了 c=16" 其实是 **KV pool 过度预留**,不是真的显存不够——值得记录,因为它和 `torch.compile` 无关:

- 默认 `mem_fraction_static` 下,启动预留 **29.4 GB KV pool(275,456 token)**,只剩 **7.67 GB**。
- 负载峰值 KV 占用是 **`token usage: 0.04–0.05`(~5%)**—— 对 movies800(~16 s → 单请求 ~1k token)来说,pool 超配了 20 倍。
- c=16 时,7.67 GB 的薄余量吃不下 encoder 瞬时 O(L²) 激活 + CUDA graph 捕获 → OOM。**不是** KV 用完(才用 5%)。
- 设 **`mem_fraction_static=0.5`** 把 pool 缩到 17.7 GB / 165,913 token(仍是需求的 20 倍),腾出 **19.37 GB**,c=16 完成 **96/96**。因为只用到 ~5% 的 pool,compute/吞吐不受影响——这只是把浪费的预留 VRAM 变成余量。

一个 2B 级模型对这个负载根本不需要 40 GB;OOM 是可调的。(这是一个工作负载相关的调节旋钮,不是 shipped 默认值——真正用到 72k token 预算的长音频部署需要更大的 pool。)

## 数据附录

### 孤立 encoder 延迟(CUDA-event,N=50,均值 ± 标准差)

| batch(n_chunks) | eager | `compile(dynamic=True)` | `compile(dynamic=False)` | CUDA graph(eager) |
|---|---|---|---|---|
| 1 | 11.3 ± 0.06 ms | 10.3(1.10×) | **9.3(1.20×)** | **9.6(1.17×)** |
| 16 | 120.5 ± 0.04 ms | **InductorError** | **100.8(1.20×)** | **119.9(1.00×)** |

编译墙钟:**~87 s 冷 / ~20 s warm-cache**(首次调用,每形状)。CUDA graph 只在 batch=1 有用(launch-bound);batch=16 时 encoder 被计算占满,消掉 launch 开销没用——而 compile 在 batch=16 的收益来自融合/消拷贝。

### Kernel 拆分(torch.profiler,batch=1,20 iters,Self-CUDA)

| kernel | eager | compiled |
|---|---|---|
| `addmm` GEMM(cuBLAS ampere) | 50% / 98.7 ms | **98.7 ms(一致)** |
| flash attention | 17% / 33.4 ms | **33.2 ms(一致)** |
| LayerNorm / GELU / add | ~20% | 融合 → `triton_*_fused_*` |
| copy / clone / contiguous(permute) | ~10% / 19.1 ms | **≈ 消失**(折进 `*_view` triton) |
| GPU-busy / iter | **9.86 ms** | **9.15 ms**(−7%) |

`torch._dynamo.explain`:**graphs=1, graph_breaks=0, ops=344**(encoder 的 self-attention 是 `scaled_dot_product_attention`,完全可 trace——不是 RadixAttention)。

### 参考实现

`sglang_model.py` 里是 `compile_encoder()`(调用点用 `self.encoder_runner(...)`,默认指向未编译的 encoder);`stages.py` 触发它,gate 在 `enable_torch_compile` 上,放在 `init_device_graphs()` 之前:

```python
# sglang_model.py —— __init__ 里 encoder_runner 默认指向未编译的 encoder:
self.encoder_runner = self.whisper_encoder

def compile_encoder(self, buckets: Tuple[int, ...] = (1, 2, 3, 4)) -> None:
    from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config
    set_torch_compile_config()
    self.encoder_runner = torch.compile(self.whisper_encoder, dynamic=False)
    cfg = self.config.audio_config
    p = next(self.whisper_encoder.parameters())
    frames = int(cfg.max_source_positions) * 2
    pos = torch.arange((frames - 1) // 2 + 1, device=p.device, dtype=torch.long)
    for n in buckets:                                   # 连续的 n_chunks 区间
        feats = torch.zeros(n, int(cfg.num_mel_bins), frames, device=p.device, dtype=p.dtype)
        self.encoder_runner(feats, pos, None)

# stages.py —— infra 建好后、init_device_graphs 之前:
if bool(server_args.enable_torch_compile):
    model_worker.model_runner.model.compile_encoder(encoder_compile_buckets)
    server_args.enable_torch_compile = False            # 别再去编译 LLM decoder
```

`torch.compile` 是惰性的(首次前向、按形状才编译)。在 `load_weights` 之后 wrap 也没问题 —— 编译后的 wrapper 共享 `self.whisper_encoder` 的 tensor。

### 复现
```
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python scripts/profile_moss_encoder.py \
    --batches 1 16 --iters 50 --warmup 10
# --dynamic false   -> 静态每形状编译:快约 20%,batch=16 干净编译
# --dynamic true    -> 默认;快约 10%,batch=16 抛 InductorError
# --cuda-graph      -> 额外测一次手动 CUDA graph 捕获(eager):+15% @b1,+0% @b16
# --clear-inductor-cache -> 得到冷编译数字
```
脚本单独实例化 encoder(TP=1 初始化,随机权重——计时与权重无关)——正是 server 跑的那条 `is_encoder` SDPA 路径,所以数字可迁移(server 端编码 11.1 ms vs 脚本 11.3 ms)。

## 注意事项(Caveat)

仅在 A100-40GB 上测过。在 H100/H200/B200 上 GEMM+flash 更快,固定的 launch/dispatch 开销在 encoder 里占比会**更大**——那时 compile 削减 launch 的作用(以及单独负责的 CUDA-graph 手段)可能更有意义。本结论限定在 A100 + torch 2.11 + 这种长音频负载;短音频 ASR 会抬高 encoder 在 e2e 中的占比。
