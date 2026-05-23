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
