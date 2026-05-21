# nanoops —— nanochat 的教学 fork

> English version: [README.md](README.md)

本 fork 在 [karpathy/nanochat](https://github.com/karpathy/nanochat) 基础上，
做两件相互关联的事：

1. **`nanoops/` —— 补上 nanchat 教学链路里"PyTorch 算子内部"那一块缺失。**
   nanchat 把整条 LLM 训练流水线（tokenizer → 训练循环 → eval → chat UI）
   讲得很完整，但里面的 PyTorch 算子都是黑盒——`F.linear`、
   `F.scaled_dot_product_attention`、`F.cross_entropy` 等等都是直接拿来用。
   nanoops 把这些黑盒打开：nanchat 用到的每个 PyTorch 算子（`Mm` /
   `Linear` / `RMSNorm` / `Softmax` / `CrossEntropy` /
   `ScaledDotProductAttention` / `ApplyRotaryEmb` / 滑动窗口 attention …）
   都用自定义 `torch.autograd.Function` 重写过——显式 forward + backward、
   in-place / 内存敏感实现，并在
   [`nanoops/README_zh.md`](nanoops/README_zh.md) 的附录里附了完整
   数学推导（也有 [English 版](nanoops/README.md)）。从源码里能看到
   `softmax_backward` 怎么用 `addcmul_` 融合、ctx 取舍怎么算账、
   GQA 怎么靠 `repeat_interleave + unflatten/sum` 收尾、在线 softmax /
   chunked LSE 长什么样、embedding backward 怎么做分段求和——不是停留
   在白板上的推导。

2. **在消费级 GPU 上同时优化 nanchat 训练的两个维度：速度 和 模型大小。**
   nanchat 的目标硬件是 H100 + FA3；在 3090 上 PyTorch SDPA 遇到
   sliding window mask 时会退化到慢路径，nanoops 的手写算子绕开这条
   退化路径（**速度**维度——d20 base-train 从 22.7k 涨到 ~30.5k tok/s，
   +34% 吞吐）。另外 Python 层的 `SlidingWindowSDPA`（按 window 分块算
   attention，P 矩阵峰值砍 ~4×）+ MLP activation checkpoint 再省 ~3.7 GiB
   —— 这两个一起打开**模型大小**维度：`--depth=24`（nanchat 参考模型尺寸，
   ~1.5 B 参数，原本在 24 GiB 卡上任何 batch size 都 OOM）现在能以
   `--device-batch-size=1` 装下并跑起来。所以消费级硬件既能**训得更快**
   （小配置吞吐拉满）又能**训原本根本装不下的更大模型**。

### 实际效果

**`--depth=24` 是 nanchat 的参考模型尺寸——3090 这种消费级显卡（RTX
3090 / 4090 等 24 GiB 级别的卡）原本根本跑不起来。** 用 nanchat 原生代码
在 24 GiB 卡上训 d24，**任何 batch size 都 OOM**：1.5 B 参数 auto-config
加宽到 `n_embd=1536` × 24 层 + AdamW state + bf16 gradients + 每个 sliding
layer 的完整 `(L, L)` attention 概率矩阵——加起来就是装不下。nanchat 的
参考硬件是 8× H100 节点——**对在家学习或预算有限的人来说远超能力**。

本 fork 的全套优化（SlidingWindowSDPA 把 chunked attention 砍到带状、
不存完整 P + MLP activation checkpoint + `expandable_segments` allocator）
总共比 PyTorch baseline 路径省下 **~9 GiB 的 peak 显存**，让 d24 终于能
在同一张消费级显卡上以 `--device-batch-size=1` **真的装下并跑起来**。
**这个项目的意义就是把 nanchat 的默认训练拉进初学者硬件预算的范围**。

| 配置             | nanchat 原生 | nanoops 整套在 2× RTX 3090 上    |
| ---------------- | ------------ | -------------------------------- |
| `--depth=20`, B=4 | ~22.7k tok/s | **~30.5k tok/s** (+34%, ~31h)  |
| `--depth=24`, B=1 | OOM          | **~15.8k tok/s** (~61h)        |

**算成钱**：3090 spot 租赁价格大约 $0.18/卡/小时，2× 3090 一台机 ~$0.36/h
≈ $8.6/天 ≈ **$60/周**。一次完整的 `--depth=24` 训练 ~2.5 天，**算力成本
约 $22**；`--depth=20` 训练 ~31 h，**不到 $12**。原本目标硬件是 8× H100
节点，本 fork 让这个训练在一台双 3090 桌面机上可行。

**很适合初学者上手**。一次 d24 训练只花掉一周 GPU 预算的一小部分，
剩下的 ~$40 / ~4-5 天 GPU 时间正好用来"折腾"——读一下
`nanoops/functional.py` 里某个算子的实现、把某个 in-place trick 改掉、
往 `.backward()` 加个 print、跑个 20-iter 看 loss 曲线和 MFU 怎么变。
整套代码量小到可以拿调试器一步步走完，配套测试
（`tests/test_nanoops_e2e.py`, `tests/test_sdpa_parity.py` 等）会把每个
算子跟 PyTorch reference 对拍——**永远有 ground truth 可以参照**。

### 实测加速过程（d20 base_train, 2× RTX 3090）

| 配置                                  | tok/sec    | MFU       | Peak 显存    | vs baseline |
| ------------------------------------- | ---------- | --------- | ------------ | ----------- |
| PyTorch SDPA, B=2 (baseline)          | 22,725     | 46.2%     | 16.5 GiB     | —           |
| nanoops Lookup default, B=2           | 28,800     | 58.5%     | 19.7 GiB     | +27%        |
| + SlidingWindowSDPA, B=2              | 30,594     | 62.2%     | 17.6 GiB     | +35%        |
| + B=4 + expandable_segments           | 32,678     | 66.4%     | 22.7 GiB     | +44%        |
| **+ MLP_CHECKPOINT（当前默认）**      | **30,500** | **62.0%** | **19.0 GiB** | **+34%, 留余量给 d24** |

所有行 loss 曲线在 bf16 数值噪声范围内**完全一致**。完整 A/B 分析记录在
[`SlidingWindowSDPA` 的 docstring](nanoops/functional.py)。

### 怎么跑

```bash
# speedrun.sh 中 base_train 步骤的 drop-in 替代版——
# 默认 --depth=24 --device-batch-size=1（2× RTX 3090 上装得下的最大 nanchat 配置）。
# 三个优化默认全开：sliding-window SDPA + MLP activation checkpoint + expandable_segments allocator。
bash nanoops/train.sh

# 也可以覆盖默认值——比如 2× RTX 3090 上吞吐最大的 setup：
bash nanoops/train.sh --depth=20 --device-batch-size=4

# 自动设置的环境变量（train.sh 帮你 export）：
#   NANOOPS=1                                       启用 nanoops 集成
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   回收碎片化内存
#   NANOOPS_MLP_CHECKPOINT=1                        省 ~3.7 GiB peak
#
# Opt-in 实验开关：
#   NANOOPS_LOOKUP_SORTED=1       试一下"排序+分段求和"的 embedding backward
```

详见：
- [`nanoops/README_zh.md`](nanoops/README_zh.md)（中文）/ [`nanoops/README.md`](nanoops/README.md)（English）——按算子排的 TODO + 数学推导附录
- [`nanoops/integration.py`](nanoops/integration.py) —— 注入 nanchat 的 monkey-patch 怎么写（不动 upstream 模型代码）

### nanchat 上游

本 fork 完整保留了 nanchat 训练流水线（tokenization、pretraining、
finetuning、evaluation、inference、chat UI）。关于 nanchat 本身的介绍、
GPT-2 leaderboard、使用方法等，请参见 [README.md](README.md) 后半部
（保留了原版英文文档），或者直接看
[karpathy/nanochat](https://github.com/karpathy/nanochat) 上游仓库。
