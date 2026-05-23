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

### 每 SM 资源

| 资源                     | 数值              | 说明                                  |
| ------------------------ | ----------------- | ------------------------------------- |
| L1 / Shared memory      | 128 KB combined   | 可配置切分 L1 vs shared              |
| Shared mem **每 block 上限** | **100 KB**        | 我们 kernel 必须遵守的硬上限         |
| Registers / SM          | 65,536 × 32-bit   | = 256 KB                              |
| Max threads / SM        | 1,536             | = 48 warps                            |
| Max blocks / SM         | 16                |                                       |
| Tensor cores            | 4 (3rd gen)       | bf16 / fp16 / tf32 / int8             |
| FP32 cores              | 128               |                                       |
| Warp size               | 32 threads        |                                       |

### 每 Thread Block 限制

| 限制                | 数值       |
| ------------------- | ---------- |
| Max threads          | 1024 (32 warps) |
| Max shared memory    | **100 KB** |
| Max registers/thread | 255（超了 spill 到 local memory，慢）|

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
| FP16 (Tensor) + 2:4 稀疏       | 2048 FLOPs/cycle            | 284 TFLOPS          |
| INT8 (Tensor cores)            | 2048 ops/cycle              | 284 TOPS            |

（**2:4 稀疏** = 一种硬件加速的 weight 格式：**每 4 个连续元素里最多 2
个非零**（另 2 个**必须**是 0）。Tensor core 跳过零乘法，吞吐翻倍。需要
weight 预先 prune 到这个 pattern——通常用于 inference weight，nanoops
不用。）

**Memory 侧**：HBM 带宽 936 GB/s。bf16 算力 / 带宽 = 142 TFLOPS / 936
GB/s = **每 byte 152 FLOPs**。一个 op 的 arithmetic intensity（每 byte
FLOPs）**低于这个值就是带宽 bound，高于就是算力 bound**。

nanchat d24 matmul (M=2048, K=1536, N=1536)：arithmetic intensity =
2*M*K*N / (M*K + K*N + M*N) ≈ **766 FLOPs/byte** → 妥妥算力 bound。
普通 elementwise op (relu, add) ≈ 1-2 FLOPs/byte → **带宽 bound**。
SDPA 的 `Q@K^T` 介于中间，看 seq length。

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
