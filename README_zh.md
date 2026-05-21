# nanoops —— nanochat 的教学 fork

> English version: [README.md](README.md)

本 fork 在 [karpathy/nanochat](https://github.com/karpathy/nanochat) 基础上，
做两件相互关联的事：

1. **`nanoops/` —— 通过手写算子学 PyTorch。**
   nanchat 用到的每个 PyTorch 算子（`Mm` / `Linear` / `RMSNorm` /
   `Softmax` / `CrossEntropy` / `ScaledDotProductAttention` /
   `ApplyRotaryEmb` / 滑动窗口 attention …）都用自定义
   `torch.autograd.Function` 重写过——显式 forward + backward、in-place
   / 内存敏感实现，并在
   [`nanoops/README_zh.md`](nanoops/README_zh.md) 的附录里附了完整
   数学推导（也有 [English 版](nanoops/README.md)）。从源码里能看到
   `softmax_backward` 怎么用 `addcmul_` 融合、ctx 取舍怎么算账、
   GQA 怎么靠 `repeat_interleave + unflatten/sum` 收尾、在线 softmax /
   chunked LSE 长什么样、embedding backward 怎么做分段求和——不是停留
   在白板上的推导。

2. **在消费级 GPU（如 RTX 3090）上优化 nanchat 训练速度。**
   nanchat 的目标硬件是 H100 + FA3；在 3090 上 PyTorch SDPA 遇到
   sliding window mask 时会退化到慢路径。nanoops 的手写算子绕开这条
   退化路径，再加上 Python 层的 `SlidingWindowSDPA`（按 window 分块算
   attention，计算量和 P 矩阵峰值都降到 ~1/4），叠加之后让 d20
   base-train 步骤能在 24 GiB 显存上以 `--device-batch-size=4` 跑起来
   （之前会 OOM）。

### 在 2× RTX 3090 上的实测加速（d20 base_train）

| 配置                                  | tok/sec    | MFU       | Peak 显存    | vs baseline |
| ------------------------------------- | ---------- | --------- | ------------ | ----------- |
| PyTorch SDPA, B=2 (baseline)          | 22,725     | 46.2%     | 16.5 GiB     | —           |
| nanoops Lookup default, B=2           | 28,800     | 58.5%     | 19.7 GiB     | +27%        |
| + SlidingWindowSDPA, B=2              | 30,594     | 62.2%     | 17.6 GiB     | +35%        |
| **+ B=4 + expandable_segments**       | **32,678** | **66.4%** | **22.7 GiB** | **+44%**    |

4 行配置的 loss 曲线在 bf16 数值噪声范围内**完全一致**。完整 A/B
分析记录在
[`SlidingWindowSDPA` 的 docstring](nanoops/functional.py)。

### 怎么跑

```bash
# speedrun.sh 中 base_train 步骤的 drop-in 替代版——
# 已默认开启 nanoops 集成 + 滑动窗口 SDPA + expandable_segments allocator。
bash nanoops/train.sh

# 自动设置的环境变量（train.sh 帮你 export）：
#   NANOOPS=1                                       启用 nanoops 集成
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   回收碎片化内存
#
# A/B 对比用的 opt-out 开关：
#   NANOOPS_NO_SLIDING_WINDOW=1   切回 naive SDPA 路径
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
