# nanoops Triton Kernels

面向 **Triton 初学者**——介绍在 **RTX 3090 架构下最优 fusion** 怎么做。
读完之后你能理解 `nanoops/triton_kernels.py` 里每个 kernel 的设计取舍：
block size 是怎么按 3090 真实预算选出来的、fusion 边界为什么落在那里、
什么留 register / 什么 save ctx / 什么 backward 重算——全部都是对着芯片
的 compute/bandwidth 比例算出来的，不是凭感觉。

文档从最底层的硬件数字（SM 数、shared mem、register file）出发，一层
层往上推到 fused kernel 设计。读完后回头看 `triton_kernels.py` 里任何
一个 kernel，你都能讲清楚每个选择背后的理由。

> English version: [TRITON.md](TRITON.md)

---

## 第 1 章 —— 目标硬件：RTX 3090 (Ampere SM_86 / GA102)

下面每一个 block size 选择、每一次"shared memory 装不装得下"判断、每一个
"为什么这是独立 kernel 不是一个大 fused kernel"的取舍——**全部都是对着
这些参数定的**。换其他 GPU（4090 / A100 / H100 / 消费级 Ada）跑大部分
kernel 还能用，但 tile size 多半已经不是最优了。

### 整 chip 总览

| 项                  | 数值                  |
| ------------------- | --------------------- |
| **SM 数**           | **82**                |
| 总 FP32 cores       | 82 × 128 = 10,496     |
| 总 Tensor cores     | 82 × 4 = 328          |
| 总 register file    | 82 × 64K = 5.4 M regs |
| 总 shared mem       | 82 × 100 KB = 8.2 MB  |
| **显存**            | **24 GB GDDR6X**      |
| HBM 带宽            | **936 GB/s**          |
| Compute capability  | **8.6**               |

### 每 SM 资源

| 资源                     | 数值              | 说明                                  |
| ------------------------ | ----------------- | ------------------------------------- |
| L1 / Shared memory      | 128 KB combined   | 零和分配：`shared_carveout + L1 = 128 KB`；sm_86 允许的 shared 值是 `{0, 8, 16, 32, 64, 100} KB`——选 shared 100 KB 时 L1 只剩 28 KB |
| Shared mem **每 SM 总量** | **100 KB**       | 最大 carveout = 共住 block 共享的整池 |
| Shared mem **每 block 上限** | **100 KB**     | 单个 block 能申请的硬上限——等于每 SM 池子总大小（所以单个 block 可以独占整池，但代价是 blocks/SM = 1） |
| Registers / SM          | 65,536 × 32-bit   | = 256 KB；per-thread 分配，SM 上所有活跃 thread 共享这个池 → **既限活跃 thread 数也限 blocks/SM**（通过 thread 换算） |
| Max threads / SM        | 1,536             | = 48 warps；卡 blocks/SM 为 `1536 / threads_per_block` |
| Max blocks / SM         | 16                | 不管资源够不够，硬件 block 共住上限 |
| Tensor cores            | 4 (3rd gen)       | bf16 / fp16 / tf32 / int8             |
| FP32 cores              | 128               |                                       |
| Warp size               | 32 threads        |                                       |

### 每 Block 限制

（NVIDIA 全名叫 "thread block"，但 CUDA API（`blockDim`、`blockIdx`、
`<<<grid, block>>>`）和本文其余地方都简称 "**block**"。两个是同一个概念。）

| 限制                | 数值       |
| ------------------- | ---------- |
| Max threads          | 1024 (32 warps) |
| Max shared memory    | **100 KB** |
| Max registers/thread | 255（超了 spill 到 local memory，慢）|

### Compile-time vs runtime：什么时候 frozen

理解本章关键的一条事实：**几乎所有影响 occupancy 的量都是编译期 frozen
的**，不是 launch 时定的。

| 量                          | 何时定           | 谁定                          |
| --------------------------- | ---------------- | ----------------------------- |
| `threads_per_block`         | **编译期**       | 用户（Triton 通过 `num_warps × 32`，CUDA C++ 通过 `<<<grid, block>>>`） |
| `reg/thread`                | **编译期**       | 编译器（Triton / NVCC）静态分析 kernel 后决定 |
| `shared_mem / block`        | **编译期**       | 编译器（静态算 `tl.load` buffer + 累加器 + `num_stages` pipeline 深度） |
| `grid_dim` (block 总数)      | runtime          | 用户 —— `kernel[grid](...)` |
| `blocks/SM`                 | runtime          | 硬件（GigaThread Engine）；精确公式见下文 |
| 每个 block 落哪个 SM         | runtime          | 硬件（GigaThread Engine） |

含义：

- 一个 Triton kernel 定义如果用 `triton.autotune` 扫 `num_warps` /
  `num_stages` / `BLOCK_*`，**会编译出 N 份独立 binary**——每份各有自己
  frozen 的 `reg/thread` 和 `shared/block`。
- `reg/thread` **不是用户直接挑的**——只能通过写更简单 / 复杂 / pipeline
  深的 kernel 间接影响。
- 编译完后，每个 kernel 的 occupancy **是确定的**——硬件不会 runtime 重新
  调资源分配。

GigaThread Engine 用的精确公式（以 warp 为粒度——**warp 才是真正的调度
单位，不是 thread**）：

```
warps_per_block = ⌈threads_per_block / 32⌉
regs_per_warp   = ⌈(32 · reg_per_thread) / 256⌉ · 256       # 256 对齐
regs_per_block  = warps_per_block · regs_per_warp

blocks/SM = min(
    16,                                                   # ① 硬件 block 上限
    ⌊48 / warps_per_block⌋,                               # ② 每 SM warp 上限（= 1536 thread）
    ⌊100 KB / shared_per_block⌋,                          # ③ Shared mem 池
    ⌊65,536 / regs_per_block⌋,                            # ④ Register file 池
)

resident_warps = blocks/SM · warps_per_block
occupancy      = resident_warps / 48
```

几个细节：
- **`⌈ / 32⌉`（warp 取整）** —— `threads_per_block` 不是 32 倍数时，最后
  一个 warp 仍占整 32-lane 资源（有 inactive lane）。
- **`⌈ / 256⌉ · 256`（register 分配粒度）** —— Ampere 上 register 按
  warp 分配，且每 warp 按 256 整数倍 round up。所以 32 thread 每个用 33
  register 不是 `32·33 = 1,056`，而是 `⌈1,056/256⌉·256 = 1,280` ——
  浪费 224 个 register slot。`reg_per_thread` 是 8 的倍数时正好对齐 256
  per-warp 边界，浪费为 0；编译器选了奇数 register 数才看得到。
- **`⌊ ⌋`（block count 整数）** —— SM 不能装 1.5 个 block。
- **② 本质是 warp 上限** —— 48-warp scheduler 是硬件硬约束；"1536 thread"
  只是 `48 × 32` 的换算。

### FMA 和 MMA —— 两种基础算子

下面的吞吐表都用 **FMA / MMA / FLOPs** 计数，先把这俩定义清楚。

**FMA —— Fused Multiply-Add（标量）**
- CUDA core 的基础 op：`d = a * b + c`
- "Fused" = 最后只做**一次 rounding**（而不是两次），数值更准
- 1 个 FMA = **2 FLOPs**（1 乘 + 1 加）
- 所有现代 CPU/GPU FPU 都 1 周期完成 FMA；分两步算 `a*b` 再 `+c` 速度
  一样但数值更糟

**MMA —— Matrix Multiply-Accumulate（矩阵）**
- Tensor core 的基础 op：`C = A @ B + C`，A/B/C 都是小矩阵
- Ampere 3rd-gen Tensor core 形状是 **16×16×16**（一次 MMA 做一个 16×16
  矩阵 × 16×16 矩阵，累加到 16×16 结果上）
- 这样 1 个 MMA = 16 × 16 × 16 = **4,096 个 multiply-add = 8,192 FLOPs**
- 通过一条 warp 级 `mma.sync` PTX 指令发射。这 4,096 个 multiply-add 在
  Tensor core 上 pipeline 跨多周期完成——**每 SM 持续吞吐 1,024 BF16
  FLOPs/cycle**（见下面峰值算力表），即每个 Tensor core 持续 256 BF16
  FLOPs/cycle，所以一条 16×16×16 MMA（8,192 FLOPs）在一个 Tensor core
  上摊销 **~32 周期**。

也就是说**一次 MMA 等于 4,096 次 FMA 的工作量**。这就是为什么架构要拆开
：少数几个 Tensor core 跑 MMA >> 多个 CUDA core 跑 FMA。

### RTX 3090 峰值算力

Boost clock 1695 MHz，82 个 SM。

| 精度 / 单元                    | 每 SM 吞吐                  | 整 chip 峰值        |
| ------------------------------ | --------------------------- | ------------------- |
| FP32 (CUDA cores)              | 128 cores × 2 FLOPs/cycle  | **35.6 TFLOPS**     |
| TF32 (Tensor cores)            | 512 FLOPs/cycle             | **71 TFLOPS**       |
| **FP16 / BF16 (Tensor cores)** | 1024 FLOPs/cycle            | **142 TFLOPS**      |
| INT8 (Tensor cores)            | 2048 ops/cycle              | 284 TOPS            |

**Memory 侧**：HBM 带宽 936 GB/s。bf16 算力 / 带宽 = 142 TFLOPS / 936
GB/s = **每 byte 152 FLOPs**。一个 op 的 arithmetic intensity（每 byte
FLOPs）**低于这个值就是带宽 bound，高于就是算力 bound**。

**Arithmetic intensity 怎么算**（matmul `C = A @ B`，shape
`(M, K) @ (K, N) → (M, N)`，全 bf16）：

- **FLOPs** = `2·M·N·K`——每个 output 元素需要 K 次 multiply-add，每个
  multiply-add = 2 FLOPs
- **Bytes** = `2·(M·K + K·N + M·N)`——读 A + 读 B + 写 C，bf16 每元素 2 字节
- **AI** = FLOPs / Bytes = `M·N·K / (M·K + K·N + M·N)`（2 互相消掉）

跟 152 FLOPs/byte 分水岭（= 142 TFLOPS / 936 GB/s）比，**高于**是算力
bound，**低于**是带宽 bound。

nanchat d24 训练时各 matmul (M=2048, K=1536, N 不同)：

| Matmul 位置                | Shape (M, K, N)       | AI (FLOPs/byte) |
| -------------------------- | --------------------- | --------------- |
| c_q / c_k / c_v / attn c_proj | (2048, 1536, 1536) | **558**         |
| MLP c_fc (D → 4D)          | (2048, 1536, 6144)    | **770**         |
| MLP c_proj (4D → D)        | (2048, 6144, 1536)    | **770**         |

全部远超 152 FLOPs/byte 分水岭 → **算力 bound**。

**RMSNorm 和其他 elementwise / reduction op 严重 bandwidth-bound** ——
比 break-even 差 100×。对一行 D 个元素：

- FLOPs ≈ `4·D`（`mean(x²)` ≈ 2D，再 `x · rms_inv · weight` = 2D）
- Bytes (bf16) ≈ `4·D`（**fused** kernel 读 x **一次** = 2D，写 y = 2D；
  weight 整 kernel 共享 `D`，摊薄忽略）。朴素 2-pass 实现要读 x 两遍 →
  6D bytes，AI 降到 0.67 FLOPs/byte。Triton/CUDA kernel 把整行 x 持
  register 跨两 pass，HBM 只读一次。
- **AI ≈ 1 FLOPs/byte** —— 比 152 break-even 差约 **100×**，重度带宽 bound

各 op AI 对比（训练 shape）：

| Op                          | AI (FLOPs/byte) | 性质        |
| --------------------------- | --------------- | ----------- |
| MLP / QKV matmul            | 558 – 770       | 算力 bound  |
| SDPA (B=1, L=2048)          | ~114 naive / ~1024 Flash（见下方 SDPA 段） | 带宽 → 算力 |
| **RMSNorm**                 | **~1**          | 带宽 bound  |
| Elementwise (add, relu)     | ~0.5            | 带宽 bound  |
| Memcpy H2D / D2H            | 0               | 带宽 bound  |

**这就是 fusion 设计的根源**：独立的 RMSNorm kernel 做 ~4·M·D 字节 HBM
流量，但算力收益接近 0。把 norm fuse 进邻接的 matmul kernel
（`NormMLPReluSquare`、`NormQKVProjection`），让归一化的中间值留在
register 里，**省下 4·M·D 字节 HBM 流量**（norm 输出写回 + matmul 输入
再读）——带宽 bound 的 op 上**纯赚**，没副作用。`fused_add_norm` 在
block 边界也是一样的道理。

**SDPA：为什么 Flash Attention 存在。** 同样的 AI 视角下，SDPA 有
**两个** AI 数字，取决于 `(L, L)` 的 P 矩阵是否物化到 HBM。

nanchat d24 (B=1, H=12, L=2048, D_head=128) forward attention
（两个 matmul：`Q@K^T` 和 `attn@V`，加 softmax）：

| 实现                                    | 总 FLOPs | 总 Bytes | AI                |
| --------------------------------------- | -------- | -------- | ----------------- |
| Naive SDPA（物化 P 到 HBM）             | 25.8 G   | ~226 MB  | **~114 FLOPs/byte** (偏带宽)  |
| Flash SDPA（P 留 register）             | 25.8 G   | ~25 MB   | **~1024 FLOPs/byte** (算力 bound) |

字节拆解：这个 scale 下 `Q + K + V + O ≈ 25 MB`（4 个 `B·H·L·D =
12·2048·128` bf16 tensor）。P 矩阵是 `B·H·L·L = 100 MB`，naive SDPA
**写 P 然后再从 HBM 读回**（~200 MB），所以 naive 合计 ~225 MB，而
Flash 只 ~25 MB。Flash 用 online softmax + tile streaming 让 P 留
register，把 SDPA 从带宽 bound (~114 < 152 break-even) **翻成算力
bound** (~1024 >> break-even)。

Flash Attention 比 naive SDPA 快 2-4× 的根源——bytes 少 ~9×，FLOPs 也
有变化但幅度小：

- **Forward FLOPs**：几乎相同（Flash 多 ~5%，是 online softmax 的 rescale
  bookkeeping）。
- **Backward FLOPs**：Flash **多 ~33%**——backward 时不存 P，而是 `Q@K^T`
  现算一遍重建 P。典型的 **FLOPs 换 memory** trade。

净效果：Flash backward 多算 ~30% 但 HBM 带宽省 ~10×，wall time 仍少 2-4×
（在带宽 bound 阶段，省带宽远比多算 FLOPs 重要）。Sliding window attention
把 bytes 和 FLOPs 都按比例缩小（band size `W` 代替全 `L`），但 Flash-vs-
naive 的比值不变。

### Tensor cores vs CUDA (FP32) cores

**完全不同的两套硬件单元**，做不同工作。理解切分方式决定你怎么算 kernel
吞吐。

| 单元              | 每 SM | 每周期做啥（持续吞吐）                       | 每 SM 总吞吐                |
| ----------------- | ----- | -------------------------------------------- | --------------------------- |
| **FP32 cores**    | 128   | 每核 1 FMA/cycle = 2 FLOPs/cycle             | **256 FP32 FLOPs/cycle**    |
| **Tensor cores**  | 4     | 每核持续 256 BF16 FLOPs/cycle（一条 16×16×16 MMA 摊销 ~32 周期） | **1024 BF16 FLOPs/cycle**   |

**4 个 Tensor cores 的 bf16 算力是 128 FP32 cores 的 fp32 算力的 4×**
（1024 / 256）。切 bf16/fp16 + 用 `tl.dot` 才能解锁那 4×。

**为什么每 SM 只有 4 个 Tensor cores：**

1. **硅面积代价大**。一个 Tensor core ≈ 几十个 FP32 core 的晶体管数。
   4 个已经吃掉 SM 很大一块面积。
2. **调度上 perfect 1:1**。Ampere SM 有 **4 个 warp scheduler**；
   每周期每个 scheduler 发 1 条 `mma.sync` 给 1 个 Tensor core。
   4 schedulers × 4 Tensor cores 完美对应——加第 5 个也没人发指令给它。
3. **数据带宽喂不饱更多**。peak 下（每 SM 1024 BF16 FLOPs/cycle = 512
   multiply-adds/cycle），即便算上 register 里 tile 复用，4 个 Tensor
   core 一起仍要从 shared memory 拉 ~1 KB/cycle 量级的 operand 数据
   ——已经吃掉每 SM SMEM port 带宽的不小一块。

### Warps、warp schedulers 和 cores 的关系

**warp** = 32 个线程，**GPU 调度的基本单位**。

```
                    SM（RTX 3090 上 82 个之一）
   ┌──────────────────────────────────────────────────────────┐
   │  4 warp schedulers（round-robin 调度）                    │
   │     │                                                     │
   │     ├──► 每周期发一条 warp 指令 ─►                        │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     ├─────►│ 128 FP32 cores（= 4 组 × 32 lanes） │       │
   │     │      │ scalar 操作: fma, sin, add, ld, st… │       │
   │     │      └─────────────────────────────────────┘       │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     └─────►│ 4 Tensor cores                      │       │
   │            │ `mma.sync` → 矩阵 multiply-accumulate│      │
   │            └─────────────────────────────────────┘       │
   │                                                           │
   │  Register file: 65,536 × 32-bit（全 warp 共享）           │
   │  Shared memory: 100 KB（每 thread block）                 │
   └──────────────────────────────────────────────────────────┘
```

**关键关系：**

- **1 个 warp 1 次发 1 条指令**。要么发给 FP32 cores（32 lane × 1 fp32
  op = 一周期 32 个 op，每 thread 1 个），要么发给 Tensor core（整 warp
  协同生成 1 个 matrix `mma`）。
- **`mma.sync` 指令是 warp 级**，不是 thread 级——warp 内 32 个 thread
  全部参与。各自的 per-thread register fragment 拼成 Tensor core 要读的
  (16, 16) 矩阵 tile。
- 每 SM 可以**最多容纳 48 个 warp**（= 1536 threads）。Scheduler 在它们
  之间挑：一个 warp 卡在 load 等内存时，调度器切到别的 warp 跑。这就是
  **latency hiding**——GPU 不需要 CPU 那种花哨的 out-of-order 执行。
- **Block size 影响 occupancy**。一个 block 256 threads = 8 warps。
  100 KB shared memory per block → 一个 SM 只装得下 1 block → 48 warp
  容量里只有 8 warp 活跃 → occupancy 低 → 能 hide latency 的 warp 少 →
  吞吐掉。把 tile 调小（少占 shared mem）让 2-3 个 block 共驻同 SM 能
  提升 occupancy。

**Triton 跟 CUDA C++ 在 warp 控制上的区别：**

- **CUDA C++**：手动调 `__syncwarp()`、管 `mma::sync`、选 fragment layout。
- **Triton**：把 warp 隐藏。用户只看到 `tl.program_id`（block 级）和
  `tl.arange`（编译时 vector）。Triton 编译器决定怎么把你写的 vector op
  映射到 32-thread warp、哪个 lane 读哪个元素。用户能调的只有：
  - `num_warps=4`（默认）—— 每 block 几个 warp。大 = 单 block 内并行高
    但能共驻的 block 少。
  - `num_stages=2` —— `tl.load` / `tl.dot` 的 pipelining 深度。

### 这些参数对我们 kernel 意味着什么

- **Shared memory 是最紧的瓶颈**。我们大部分 kernel 的 tile 尺寸（BLOCK_M,
  BLOCK_N, BLOCK_K）都是选成**合并工作集**（input tiles + accumulator）
  ≤ 60-80 KB per block，留点 headroom 给 runtime，**避免**一个 SM 只能跑
  一个 block 的低 occupancy。

- **3090 的 shared memory 大约是 A100 (164 KB) 的一半、H100 (228 KB) 的
  三分之一**。所以 Flash Attention 在 3090 上**必须用更小的 tile**——很多
  公开的 Flash kernel 数据是 H100 上跑出来的，那些 Q/K/V tile 直接搬到
  3090 上要么 spill 要么编译都过不了。

- **82 个 SM 是塞满 GPU 的最小 grid 大小**。launch 少于这个就有 SM 闲着。
  对于 per-row reduction 类 kernel（RMSNorm, softmax, fused_add_norm），
  我们沿 M (batch × seq) 维度拆 tile，因为训练时 M 通常 ≥ 2048——并行度
  绰绰有余。

- **没有 FA3 那套 TMA / async-copy 硬件**。Hopper 专属的技巧（TMA、
  distributed shared memory、`wgmma` warp specialization）在 Ampere 上
  用不了。我们的 Flash SDPA 是**经典 Triton 教程版**，不是 FA3 版。

- **把 elementwise fuse 进 `tl.dot` epilogue**。独立 elementwise
  kernel 是 **带宽 bound**（见上面 AI 表里 RMSNorm / add / relu，全都
  ≪ 152 break-even）——真正的开销是 HBM round-trip + kernel launch
  overhead，**不是 FLOPs**。`tl.dot` 主导的 kernel 里 FP32 cores 本来就
  闲着（即便单独跑 elementwise，也只有 BF16 Tensor peak 的 ~1/4，
  35.6 / 142 TFLOPS），所以把 elementwise 折进 matmul epilogue 是**白
  捡的 side-channel**：省掉 elementwise 自己的 HBM round-trip
  （matmul 的 HBM 开销不变）、不多一次 launch、也不抢 Tensor core 的
  FLOP/s。

---

## 第 2 章 —— FusedAddNorm：2-op fusion 的最小完整样本

本 repo 里最简单的 fused kernel。**纯学习用**——nanchat 生产 block
里 RMSNorm 已经直接 fold 进相邻的 matmul kernel（attn 走
`NormQKVProjection`，mlp 走 `NormMLPReluSquare`），所以根本不存在
独立的 `add → norm` op 边界让这个 kernel 上 hot path。但这个 kernel
用到的每一个 pattern 都是更大 fused kernel 的基石，所以它是学这些
patterns 最干净的样本。

### 这个 kernel 算什么

数学：
```
summed = x + residual
y      = summed · rsqrt(mean(summed²) + eps) · weight    # weight 可选
```

API 返回两个张量：
- `y` → 流给下一个 block 的 matmul 输入
- `summed` → 下一个 block 的 residual stream（caller 不用再 recompute
  `x + residual`）

整条 autograd 链路由 3 个 Triton kernel 撑起：

| Kernel | Grid | 角色 |
|---|---|---|
| `_fused_add_norm_fwd_kernel` | 1D over M | fwd：写 `y` + `summed` + `rms_inv` |
| `_fused_add_norm_inner_kernel` | 1D over M | bwd 阶段 1：预算 `inner[m]` |
| `_fused_add_norm_bwd_kernel` | 2D over (M, D) | bwd 阶段 2：写 `d_summed` + `dnw_partial` |

### 2.1 Forward kernel

单 pass 设计——一个 program 处理一个 `(BLOCK_M, BLOCK_D)` tile，
load 完所有数据，把 reduction 和 per-element 输出全算完，写回。

```python
@triton.jit
def _fused_add_norm_fwd_kernel(x_ptr, res_ptr, nw_ptr,
                               y_ptr, summed_ptr, rms_inv_ptr,
                               M, D, eps,
                               BLOCK_M: tl.constexpr,
                               BLOCK_D: tl.constexpr,
                               HAS_NW: tl.constexpr):
    pid_m = tl.program_id(0)
    rows  = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols  = tl.arange(0, BLOCK_D)
    mask  = (rows < M)[:, None] & (cols < D)[None, :]
    offs  = rows[:, None] * D + cols[None, :]

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    r = tl.load(res_ptr + offs, mask=mask, other=0.0)
    summed = x + r
    tl.store(summed_ptr + offs, summed, mask=mask)        # ← caller 的 residual stream + bwd 用

    summed_f32 = summed.to(tl.float32)                    # 从这里开始 fp32，对齐 F.rms_norm 精度
    sum_sq = tl.sum(summed_f32 * summed_f32, axis=1)
    rms_inv = 1.0 / tl.sqrt(sum_sq / D + eps)
    tl.store(rms_inv_ptr + rows, rms_inv, mask=rows < M)  # ← bwd 用

    y = summed_f32 * rms_inv[:, None]
    if HAS_NW:
        nw = tl.load(nw_ptr + cols, mask=cols < D, other=0.0).to(tl.float32)
        y = y * nw[None, :]
    tl.store(y_ptr + offs, y.to(y_ptr.dtype.element_ty), mask=mask)
```

注意的 patterns：
- **`BLOCK_D = next_power_of_2(D)`** 配 `col_mask` 处理非 2 幂 `D`
  （nanchat d24 `D=1536` → `BLOCK_D=2048`，512 lane 被 mask 掉）。
  Triton 的 `tl.arange` 要求长度是 2 的幂。
- **2D mask** = row mask × col mask，应用到所有 load/store。越界 load
  返回 `0`（**故意选 0**——`sum_sq` 的加法/乘法 identity）。
- **`HAS_NW: tl.constexpr`** 是编译期分支。无 affine weight 时直接
  跳过 `nw` 的 load + 乘法；`(HAS_NW=True, HAS_NW=False)` 两个值各
  编译出一份 binary。
- **store 时 `element_ty` cast** 让同一个 kernel 支持 caller 传任意
  dtype（bf16 / fp16 / fp32）无需修改。
- **`summed` 是 kernel 中途写 HBM 的，不是最后**。故意如此：caller
  的 residual stream 需要 bf16 版，但下游 fp32 compute pipeline 用
  `summed_f32` 留在 register 里继续算，不重新 load。

### 2.2 Backward：为什么是 2 个 kernel

最朴素的想法是写一个跟 fwd 同构（1D grid）的 bwd kernel。**不行**
——register 装不下：

1D-over-M 的 bwd 每 program 要同时 alive `summed`、`dy`、`y_norm`、
`g_eff`、`dx`、加上 per-channel 的 `dnw` 累加器。`D=1536` 时
~5 × 32 × 1536 = 245K fp32 元素 / 128 thread ≈ **1900 regs/thread**
——惨烈 spill。

修法：**bwd grid 同时切 D 维度**。每 program 只处理一个
`(BLOCK_M, BLOCK_D)` 的 `dx` 输出切片。单 program 的 tile 从 245K
缩到 10K 量级，register 压力下来了。

但切 D 引入新问题——bwd 数学里那个 per-row reduction：

```
inner[m] = mean_d(g_eff[m, *] · y_norm[m, *])
```

跨**整个** D 维。同一个 m_tile 的所有 program 都要这个标量。最朴
素的做法：每个 program 自己 loop 整 D 算一遍 `inner[m]`。`D=1536,
BLOCK_D=64` 下这是 **24× 冗余计算**。

L2 cache 吸收了大部分**字节流量**（相邻 d_tile program 共享同
m_tile 的 `summed`/`dy` row，~96 KB per m_tile 完全塞进 3090 的
6 MB L2），所以 wall time 代价不是 24×，但 FLOPs 是真的浪费。
学 Flash Attention 的「预算 reduction」pattern，拆成 2 个 kernel：

**Stage 1 —— `_fused_add_norm_inner_kernel`**（1D over M，每
program 处理 `BLOCK_M` 行，单 tile 涵盖整 D）
```python
inner[m] = mean_d(g_eff[m, *] * y_norm[m, *])
# 写到 (M,) fp32 buffer
```

**Stage 2 —— `_fused_add_norm_bwd_kernel`**（2D over M × D，每
program 处理一个 `(BLOCK_M, BLOCK_D)` 输出 tile）
```python
inner_m = tl.load(inner_ptr + rows)              # 每行一个标量，预备好了
dx[m, d] = rms_inv[m] * (g_eff[m, d] - y_norm[m, d] * inner_m[m]) + d_ext[m, d]
```

bwd kernel 原来跨整 D 的 pass 1 缩成「每行一个标量 load」。总 HBM
流量跟单 kernel 版本相同（`summed`/`dy` 在 inner 读一次、在 bwd 读
一次），但 compute 上不再有跨 program 的 per-row reduction 重复。

### 2.3 d_summed_external 折进 bwd kernel

autograd Function 返回两个输出（`y`、`summed`），所以 backward 收到
两份梯度（`dy`、`d_summed_external`）。前者是 norm 输出梯度，后者
来自 caller 在下游直接用了 `summed`。

朴素写法：bwd kernel 算 `d_summed_from_norm`，外面 Python 加一行
`d_summed_total = d_summed_from_norm + d_summed_external`。这是
**多一次 torch elementwise op**——单独的 kernel launch（~10-30 μs）
+ 4·M·D HBM round-trip（d24 shape 下 ~10-15 μs）。

直接把 `d_summed_external` 作为 bwd kernel 的额外输入，把这次加法
折进同一个 tile store：

```python
d_summed_tile = rms_inv * (g_eff - y_norm * inner) + d_ext   # ← 「+ d_ext」就是融
tl.store(d_summed_ptr + offs, d_summed_tile.to(...), mask=mask)
```

纯 register-level 加法，零额外 HBM 流量、零额外 kernel launch。

### 2.4 Sizing —— 套用第 1 章的预算公式

Forward kernel 用第 1 章「what this means for our kernels」给的公式：

```python
BLOCK_D = triton.next_power_of_2(D)                              # tl.arange 要求 pow-of-2
num_warps = 4                                                    # 目标值（Triton 默认）
BLOCK_M = max(1, min(
    triton.next_power_of_2(M // 64),                             # M-saturation：grid ≳ 64
    triton.next_power_of_2(4096 * num_warps // BLOCK_D),         # reg budget：tile ≤ 16K
))
tile = BLOCK_M * BLOCK_D
num_warps = max(4, min(16, triton.next_power_of_2(max(1, tile // 4096))))   # tile 超 budget 时升 nw
```

d24 shape (M=2048, D=1536) 走一遍：
- `BLOCK_D = next_pow_of_2(1536) = 2048`
- `M // 64 = 32 → next_pow_of_2 = 32`
- `4096 × 4 // 2048 = 8 → next_pow_of_2 = 8`
- `BLOCK_M = min(32, 8) = 8`
- `tile = 8 × 2048 = 16K → nw 保持 4`
- Grid：`cdiv(2048, 8) = 256` programs

Backward kernel 用**固定 config**（BLOCK_M=32, BLOCK_D=64, nw=4），
**不**走 autotune。原因不是性能（autotune 选的也差不多），而是
Triton autotune 的 dispatch 保留了一些不能 CUDA Graph stream
capture 的操作。写死 config 让 bwd 路径对 graph 友好。

inner kernel（1D over M，结构跟 fwd 同构）用跟 fwd 一样的公式。

### 2.5 数值精度

这个 kernel **bit-for-bit** 对齐 `F.rms_norm` 在 bf16 输入上的行为
——实测验证：PyTorch 的 `F.rms_norm` 内部把 bf16 升 fp32 做 reduction
和 elementwise scale，最后 cast 回去。我们一模一样：

| 操作 | 在哪 | 为什么 |
|---|---|---|
| HBM load `x`、`r` | bf16（caller dtype） | 省带宽 |
| `summed = x + r`、residual-stream store | register 里 bf16 | caller 要 bf16 |
| `sum_sq` reduction、`rsqrt`、y 乘法 | **register 里 fp32** | 匹配 F.rms_norm 精度 |
| 最终 `y` store | cast 回 bf16 | caller 要 bf16 |

bf16 形态的 `summed` **只在 store 那一刻短暂需要**——下游 fp32
pipeline 一直用 `summed_f32` 在 register 里，直到最后 y store 再 cast。

**净 register 压力跟「全 fp32 internal」实现一样**（tile=16384, nw=4
实测 n_regs=255）。bf16 路径是为了 **HBM dtype 兼容**，不是 register
省钱。这里有个反直觉的点：「低精度省 register」的直觉在 reduction
kernel 上是错的——一旦某处需要 fp32 精度，编译器会把 fp32 版本一直
保留在剩下的 kernel 里。

### 2.6 预期收益 —— HBM 和 launch 账本

实测之前先算账。一句话总结：**前向真省 HBM round-trip；后向省的是
launch 和中间 buffer，HBM 字节数没省**。

#### Forward

Naive native（两步走：`summed = x + r`，然后 `y = F.rms_norm(summed)`）：
```
torch.add (x + r → summed):
  read x        M·D
  read r        M·D
  write summed  M·D
F.rms_norm (summed → y):
  read summed   M·D                     ← fusion 省掉的就是这次
  write y       M·D
─────────────────────────────────────────
合计:           5·M·D
```

Fused（一个 kernel 全做完）：
```
read x          M·D
read r          M·D
write summed    M·D    (caller residual stream 还是要)
write y         M·D
─────────────────────────────────────────
合计:           4·M·D
```

净 forward 收益：
- **HBM：省 1·M·D 字节** —— `summed` 从 `x + r` 直接接到 norm
  reduction，全程留 register，不进 HBM。
- **Kernel launches：省 1 次**（两个 op 合成一个 Triton kernel）。
- **中间 buffer：省 1 个**（torch.add 不需要单独分配 summed 输出）。

对 d24（M=2048, D=1536, bf16）：1·M·D = 6.3 MB / 936 GB/s ≈ 6.7 μs
HBM 时间，再加上避免的 ~10-30 μs launch overhead。

#### Backward

Naive native（单 `F.rms_norm.backward` kernel + Python 外加 external 梯度）：
```
F.rms_norm.backward:
  read summed              M·D
  read dy                  M·D
  write d_summed_from_norm M·D                              ─┐
                                                             │ subtotal 3·M·D
Python d_summed = d_summed_from_norm + d_summed_external:
  read d_summed_from_norm  M·D
  read d_summed_external   M·D
  write d_summed_total     M·D                              ─┘ subtotal 3·M·D
─────────────────────────────────────────
合计:                      6·M·D
```

Fused（2-kernel split，d_summed_external 折进 bwd kernel）：
```
_fused_add_norm_inner_kernel:
  read summed/y            M·D
  read dy                  M·D
  write inner_buf          ~0  (M floats)                   ─┐ subtotal 2·M·D
                                                             │
_fused_add_norm_bwd_kernel:                                  │
  read summed/y            M·D   ← 重复读（L2 大概率 hit）
  read dy                  M·D   ← 重复读（L2 大概率 hit）
  read d_summed_external   M·D
  read inner_buf           ~0
  write d_summed           M·D                              ─┘ subtotal 4·M·D
─────────────────────────────────────────
合计:                      6·M·D
```

HBM 总字节数一样。那 bwd fusion 到底省了什么？

- **省 1 个 intermediate buffer**：native 要给
  `d_summed_from_norm`（M·D 字节）分配一个中转 buffer，挂在 norm
  bwd kernel 和 Python `+` 之间。fused 跳过这个分配。
- **省 1 次 kernel launch**：native 用一个 torch elementwise add
  kernel 做 `d_summed_from_norm + d_summed_external` 的合并（~10-30
  μs launch）。fused 把它折进 bwd kernel 的 `dx` store——`+ d_ext`
  在 register 里 store 前完成。
- **省 1 个 autograd 图节点**：native 走 `AccumulateGrad` 步骤；
  fused 直接交出已合并梯度。

但 fused 这边也加了 inner 预算 kernel 一次 launch，**正好抵消了
省下来的那次 add launch**——**净 launch 数跟 native 一样**。真正
的赢点是 intermediate buffer + Python `+` op + cache locality：

- 两个 fused kernel 间的 L2 复用：第二个 kernel 重读第一个 kernel
  刚读过的 `summed` 和 `dy`，3090 的 6 MB L2 吸收掉这些「重复」。
- vs native 的中间 buffer 真的流过 HBM，fused 路径让梯度各部分更
  靠近 register / cache。

#### Takeaway

**Forward fusion 省字节 + launch + buffer**——干净的三合一。
**Backward fusion 只省 launch + buffer**（字节数同）。这种不对称
到处都出现：Flash Attention 的 forward 加速比 backward 大、kernel
fusion 项目通常宣传「forward 快 N 倍」但 backward 接近持平，都是
这个原因。教训：**只要能在 register 里复用上一步的值、而不是从
HBM 来回搬，就是真实的带宽节省；其它（launch overhead、buffer
allocation）都是次级的打磨**。

### 2.7 性能现实

kernel 本身跟 native 持平甚至赢一点：

| 测量方式 | fused | native | 对比 |
|---|---|---|---|
| Kernel-only timing（CUDA event 包 kernel launch） | ~88 μs | ~91 μs | fused 持平 / 略胜 |
| **CUDA Graph fwd replay** | **~76 μs** | ~90 μs | **fused 快 15%** |
| 普通 eager `fused_add_norm(...)` call | ~184 μs | ~91 μs | fused 慢 2× |

eager 模式下慢 2× **不是 kernel 慢**——是 `autograd.Function` +
Triton dispatch 的 per-call 固定开销（~100 μs：3× `torch.empty_like`、
`save_for_backward`、ctx 属性设置、kernel launch arg 打包）。这些
overhead 在 compile / graph 捕获的大 pipeline 里会消失。

核心教训：**kernel 这么短的时候，Python 框架开销很容易超过 kernel
本身的 GPU 工作量**。对单个 op `torch.compile(fused_add_norm)` 反而
**更慢**——compile dispatcher 自己加开销。fusion 真正赚到只有两种
情况：
1. op 被包在 `torch.compile(model)` 整 model 里，dispatcher 开销被摊掉
2. 整 train step 用 `torch.cuda.CUDAGraph` capture，per-call Python
   工作只在 capture 时跑一次

这跟 production 级 transformer kernel **都越做越大**的原因是一致的
（Flash Attention 整段 `Q@K^T → softmax → @V` 全在一个 kernel 里，
不是分小步走）：kernel launch 和 Python dispatcher 的开销是每 op 的
常数，更长的 kernel 摊薄得多，哪怕 per-element 吞吐不比一串小 kernel
更高，wall time 上还是赢。

这也是为什么 nanchat 生产路径跳过这个 kernel：`NormMLPReluSquare`
和 `NormQKVProjection` 把 norm 直接 fold 进 matmul kernel，根本不
存在让这个 `add+norm` fusion 挂上去的 op 边界。这个 kernel 是来
演示 patterns 的；真正让 patterns 体现价值的，是更大的生产 kernel。

---

（第 3 章及后续 —— 待写）
