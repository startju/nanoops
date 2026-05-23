# nanoops Triton Kernels

`nanoops/triton_kernels.py` 里 fused CUDA kernel 的深度解读。面向**读过
代码、想理解设计取舍**的人（block size 怎么选的、fusion 边界在哪、ctx
该 save 什么），**不是 Triton 入门教程**。

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
- 通过一条 `mma.sync` PTX 指令发射，吞吐 1/cycle（内部 pipeline 多周期
  完成但每周期可以发新的一条）

也就是说**一次 MMA 等于 4,096 次 FMA 的工作量**。这就是为什么架构要拆开
：少数几个 Tensor core 跑 MMA >> 多个 CUDA core 跑 FMA。

### RTX 3090 峰值算力

Boost clock 1695 MHz，82 个 SM。

| 精度 / 单元                    | 每 SM 吞吐                  | 整 chip 峰值        |
| ------------------------------ | --------------------------- | ------------------- |
| FP32 (CUDA cores)              | 128 cores × 2 FLOPs/cycle  | **35.6 TFLOPS**     |
| TF32 (Tensor cores)            | 512 FLOPs/cycle             | **71 TFLOPS**       |
| **FP16 / BF16 (Tensor cores)** | 1024 FLOPs/cycle            | **142 TFLOPS**      |
| FP16 / BF16 (Tensor) + 2:4 稀疏 | 2048 FLOPs/cycle           | 284 TFLOPS          |
| INT8 (Tensor cores)            | 2048 ops/cycle              | 284 TOPS            |
| INT8 (Tensor) + 2:4 稀疏        | 4096 ops/cycle              | 568 TOPS            |

（**2:4 稀疏** = 一种硬件加速的 weight 格式：**每 4 个连续元素里最多 2
个非零**（另 2 个**必须**是 0）。Tensor core 跳过零乘法，吞吐翻倍。
Ampere 所有 Tensor-core dtype（FP16 / BF16 / TF32 / INT8）都支持，
PTX 指令 `mma.sp.sync`。需要 weight 预先 prune 到这个 pattern——通常用
于 inference weight，nanoops 不用。）

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
- Bytes (bf16) ≈ `4·D`（读 x = 2D，写 y = 2D；weight 整 kernel 共享 `D`，
  摊薄忽略）
- **AI ≈ 1 FLOPs/byte** —— 比 152 break-even 差约 **100×**，重度带宽 bound

各 op AI 对比（训练 shape）：

| Op                          | AI (FLOPs/byte) | 性质        |
| --------------------------- | --------------- | ----------- |
| MLP / QKV matmul            | 558 – 770       | 算力 bound  |
| SDPA `Q@K^T` (B=1, L=2048)  | ~50 – 100       | 中间        |
| **RMSNorm**                 | **~1**          | 带宽 bound  |
| Elementwise (add, relu)     | ~0.5            | 带宽 bound  |
| Memcpy H2D / D2H            | 0               | 带宽 bound  |

**这就是 fusion 设计的根源**：独立的 RMSNorm kernel 做 ~4·M·D 字节 HBM
流量，但算力收益接近 0。把 norm fuse 进邻接的 matmul kernel
（`NormMLPReluSquare`、`NormQKVProjection`），让归一化的中间值留在
register 里，**省下 2·M·D 的 HBM round-trip**——带宽 bound 的 op 上**纯
赚**，没副作用。`fused_add_norm` 在 block 边界也是一样的道理。

**SDPA：为什么 Flash Attention 存在。** 同样的 AI 视角下，SDPA 有
**两个** AI 数字，取决于 `(L, L)` 的 P 矩阵是否物化到 HBM。

nanchat d24 (B=1, H=12, L=2048, D_head=128) 完整 attention
(Q@K^T + softmax + @V，两步合计)：

| 实现                                    | 总 FLOPs | 总 Bytes | AI                |
| --------------------------------------- | -------- | -------- | ----------------- |
| Naive SDPA（物化 P 到 HBM）             | 25.8 G   | ~226 MB  | **~114 FLOPs/byte** (偏带宽)  |
| Flash SDPA（P 留 register）             | 25.8 G   | ~25 MB   | **~1024 FLOPs/byte** (算力 bound) |

P 矩阵在这个 scale 下是 `B·H·L·L = 100 MB`，naive 字节数被它主导。
Flash 用 online softmax + tile streaming 让 P 留 register，HBM 流量
缩到只剩 `Q + K + V + O` (~25 MB)，把 SDPA 从带宽 bound (~114 < 152
break-even) **翻成算力 bound** (~1024 >> break-even)。

**8x AI 提升**就是 Flash Attention 比 naive SDPA 快 2-4× 的根源——
**FLOPs 没变，HBM 流量少 9×**。Sliding window attention 把两个数字按
比例缩小（band size W 代替 L），但 Flash-vs-naive 的比值不变。

### Tensor cores vs CUDA (FP32) cores

**完全不同的两套硬件单元**，做不同工作。理解切分方式决定你怎么算 kernel
吞吐。

| 单元              | 每 SM | 每周期做啥                                | 每 SM 总吞吐                |
| ----------------- | ----- | ----------------------------------------- | --------------------------- |
| **FP32 cores**    | 128   | 1 个 scalar fp32 multiply-add             | 256 fp32 FMA/cycle          |
| **Tensor cores**  | 4     | 1 个小矩阵 `mma`（如 16×16×16 bf16）—— 一条指令 256 个 multiply-add | **1024 bf16 FMA/cycle**     |

**4 个 Tensor cores 的 bf16 算力大约是 128 FP32 cores 的 fp32 算力的 4×**。
切 bf16/fp16 + 用 `tl.dot` 才能解锁那 4×。

**为什么每 SM 只有 4 个 Tensor cores：**

1. **硅面积代价大**。一个 Tensor core ≈ 几十个 FP32 core 的晶体管数。
   4 个已经吃掉 SM 很大一块面积。
2. **调度上 perfect 1:1**。Ampere SM 有 **4 个 warp scheduler**；
   每周期每个 scheduler 发 1 条 `mma.sync` 给 1 个 Tensor core。
   4 schedulers × 4 Tensor cores 完美对应——加第 5 个也没人发指令给它。
3. **数据带宽喂不饱更多**。每个 Tensor core 一条指令吃 (16, 16) tile 的
   fp16 = 512 B。4 个一起 ~2 KB/cycle 出 shared memory——已经接近
   shared memory port 带宽上限。

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

### Occupancy：register / thread / block 三个 budget 怎么共同决定

Occupancy = `SM 上活跃 warp 数 / SM 最大 warp 数`（3090 上 = 48）。
**越高一般越好**——更多 warp 候选 → scheduler 切换时更容易找到能跑的 warp →
更好的 latency hiding。三个 per-SM cap（register / thread / block）加上
per-block shared mem cap 都是 `blocks/SM` 的**同时上限**——实际值是所有
限制的**最小值**。

举例 kernel A：128 thread/block，32 regs/thread，4 KB shared mem/block：

| 限制                              | 单 block 用量    | 允许 blocks/SM      |
| ---------------------------------- | --------------- | ------------------- |
| Register (65,536 总)              | 128 × 32 = 4 K  | 16                  |
| Thread (1,536 总)                 | 128             | **12**              |
| Shared mem (100 KB 总)            | 4 KB            | 25                  |
| Per-SM block 硬上限                | n/a             | 16                  |
| **取 min → 实际 blocks/SM**        |                 | **12**              |

12 blocks × (128/32) = **48 个活跃 warp** → **100% occupancy**。

更重的 kernel B：512 thread/block，64 regs/thread，48 KB shared mem/block：

| 限制                              | 单 block 用量    | 允许 blocks/SM      |
| ---------------------------------- | --------------- | ------------------- |
| Register                          | 512 × 64 = 32 K | 2                   |
| Thread                            | 512             | 3                   |
| **Shared mem**                    | **48 KB**       | **2**               |
| Per-SM block 硬上限                | n/a             | 16                  |
| **min**                           |                 | **2**               |

2 blocks × (512/32) = **32 个活跃 warp** → 32/48 = **67% occupancy**。

#### Register pressure —— 真正卡 occupancy 的情况

单 thread 架构上限是 **255 个 register**，意味着 `reg/thread × active
thread ≤ 65,536`。所以 `255 × 1536 = 391,680` **物理不可能**——你没法让
1536 个 thread 每个都用 255 个 register。硬件用 register 数量**直接限制
活跃 thread 数**。

不同 reg/thread 对应 occupancy（3090，先不考虑 shared mem / block cap）：

| reg/thread | 活跃 thread/SM    | 活跃 warp | Occupancy |
| ---------- | ----------------- | --------- | --------- |
| 32         | 1,536（硬上限）    | 48        | **100%**  |
| 42         | 1,536（硬上限）    | 48        | **100%** ← sweet spot (65,536/42=1,560) |
| 43         | 1,524             | 47        | 98%       |
| 64         | 1,024             | 32        | 67%       |
| 128        | 512               | 16        | 33%       |
| 255（最大）| **256**           | **8**     | **17%**   |

42 reg/thread 是分水岭：再多 occupancy 就开始掉，因为 register file 比
1536 thread 上限先用完。

#### 为啥编译器有时还是会选高 reg 数

超过 255 register 不行——多出来的就**spill 到 local memory**（HBM 上的私有
区域）。一次 spill 访问 ≈ 300 cycle，vs register 读 1 cycle。**Spill 比
低 occupancy 糟得多**，所以编译器（NVCC、Triton）会**优先降 occupancy 避免
spill**。

复杂的 SDPA backward kernel 可能真的需要 100+ register/thread →
33% occupancy → 但 register 换来的更大 tile 弥补了少 warp 的损失。
nanoops 的 matmul kernel 一般落在 50-100 reg/thread 区间，30-70% occupancy。

Triton **没有 API 直接设 reg/thread**——编译器根据 kernel body、`BLOCK_*`
大小、`num_warps`、`num_stages` 自己决定。CUDA 路径可以用
`nvcc -maxrregcount=N` 强制 cap，但很少这么做——编译器的
spill-vs-occupancy 取舍通常比手调准。

Triton 提供三个旋钮可以调：

- **BLOCK_M / BLOCK_N / BLOCK_K** ↑ → 单 block 的 thread / register /
  shared mem 用量 ↑ → blocks/SM ↓ → occupancy 可能掉。
- **`num_warps=N`**（`triton.Config` 里）↑ → 单 block thread 数 ↑ → 同上。
- **`num_stages=N`** ↑ → 多一份 pipeline buffer 在 shared mem 里 →
  shared mem 用量 ↑ → blocks/SM ↓。

**没有放之四海的最优 occupancy**，看 kernel 性质：

- **算力 bound**（matmul-heavy）：**大 tile + 中等 occupancy** (50-75%)
  通常最快——少而大的 block 让算术单元持续饱和、调度切换 overhead 少。
- **带宽 bound**（norm、elementwise）：**高 occupancy** (80-100%) 更
  重要——warp switching hide HBM 延迟是主要杠杆。

Triton 的 `triton.autotune` 会跑多组 `(BLOCK, num_warps, num_stages)`
config 实测选最快——比手算靠谱，因为最优 config 还跟 cache 行为相关。

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

- **Tensor cores 决定 FLOP/s 上限**。**不**用 `tl.dot` 的 op（普通
  elementwise）跑在 FP32 cores 上，只有 peak FLOP/s 的 ~1/10。所以
  能 fuse 进 `tl.dot` epilogue 把 elementwise launch 消化掉的，**几乎
  总是值得做**。

### 与其他 GPU 对比（移植参考）

| GPU         | SMs | Shared/SM | 显存            | 带宽         | 对我们 kernel 的影响                |
| ----------- | --- | --------- | --------------- | ------------ | ----------------------------------- |
| **RTX 3090**| 82  | 100 KB    | 24 GB GDDR6X    | 936 GB/s     | （我们的目标硬件）                  |
| RTX 4090    | 128 | 100 KB    | 24 GB GDDR6X    | 1008 GB/s    | 同样 tile size 应该直接能用         |
| A100 80GB   | 108 | 164 KB    | 80 GB HBM2e     | 1935 GB/s    | tile 可以放大 ~1.5×                 |
| H100 SXM    | 132 | 228 KB    | 80 GB HBM3      | 3000+ GB/s   | 大 tile + 解锁 FA3 / TMA 路径       |

---

（第 2 章及后续 —— 单 kernel 深度解读 —— 待写）
