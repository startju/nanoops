# nanoops Triton Kernels

Deep dive into the fused CUDA kernels in `nanoops/triton_kernels.py`. Aimed
at someone who's read the kernels and wants to understand the design
choices (block sizes, fusion boundaries, what to save in ctx, etc.) —
not a Triton tutorial.

> 中文版：[TRITON_zh.md](TRITON_zh.md)

---

## Chapter 1 — Target hardware: RTX 3090 (Ampere SM_86 / GA102)

Every block-size choice, every "fits in shared memory" check, every
"why is this a separate kernel and not one giant fused kernel" trade-off
below is keyed to these numbers. If you port to a different GPU
(RTX 4090 / A100 / H100 / consumer Ada), most kernels will run, but the
tile sizes are likely no longer optimal.

### Per-SM resources

| Resource                | Value           | Notes                                     |
| ----------------------- | --------------- | ----------------------------------------- |
| L1 / shared memory      | 128 KB combined | configurable split between L1 and shared  |
| Shared mem **per block**| **100 KB max**  | hard cap our kernels must respect         |
| Registers per SM        | 65,536 × 32-bit | = 256 KB                                  |
| Max threads / SM        | 1,536           | = 48 warps                                |
| Max blocks / SM         | 16              |                                           |
| Tensor cores            | 4 (3rd gen)     | bf16 / fp16 / tf32 / int8                 |
| FP32 cores              | 128             |                                           |
| Warp size               | 32 threads      |                                           |

### Per-thread-block limits

| Limit              | Value      |
| ------------------ | ---------- |
| Max threads        | 1024 (32 warps) |
| Max shared memory  | **100 KB** |
| Max registers/thread | 255 (more → spill to local memory, slow) |

### Whole-chip totals

| Item                | Value                |
| ------------------- | -------------------- |
| **SMs**             | **82**               |
| Total FP32 cores    | 82 × 128 = 10,496    |
| Total Tensor cores  | 82 × 4 = 328         |
| Total register file | 82 × 64K = 5.4 M regs|
| Total shared mem    | 82 × 100 KB = 8.2 MB |
| **Device memory**   | **24 GB GDDR6X**     |
| HBM bandwidth       | **936 GB/s**         |
| Compute capability  | **8.6**              |

### FMA vs MMA — the two primitive ops

Throughput tables below count work in **FMA / MMA / FLOPs**. Worth
nailing down what each means first.

**FMA — Fused Multiply-Add (scalar)**
- The basic op of a CUDA core: `d = a * b + c`
- "Fused" = one rounding step at the end (not two), better numerics
- 1 FMA = **2 FLOPs** (one multiply + one add)
- Every modern CPU/GPU FPU does FMA in 1 cycle; doing `a*b` and `+c`
  separately is the same speed but worse numerics

**MMA — Matrix Multiply-Accumulate (matrix)**
- The basic op of a Tensor core: `C = A @ B + C` where A, B, C are
  small matrices
- Ampere 3rd-gen Tensor core: shape is **16×16×16** (one MMA does the
  product of a 16×16 matrix and a 16×16 matrix, accumulating into a
  16×16 result)
- 1 such MMA = 16 × 16 × 16 = **4,096 multiply-adds = 8,192 FLOPs**
- Issued as a single `mma.sync` PTX instruction; takes ~1 cycle
  (pipelined across multiple cycles internally but throughput is 1/cycle)

So one MMA does **the same work as 4,096 FMAs**. That's the asymmetry
that justifies the architectural split: a few Tensor cores running
MMAs >> many CUDA cores running FMAs.

### RTX 3090 peak compute

Boost clock 1695 MHz, 82 SMs.

| Precision / unit                 | Per-SM throughput          | Whole-chip peak     |
| -------------------------------- | -------------------------- | ------------------- |
| FP32 (CUDA cores)                | 128 cores × 2 FLOPs/cycle  | **35.6 TFLOPS**     |
| TF32 (Tensor cores)              | 512 FLOPs/cycle            | **71 TFLOPS**       |
| **FP16 / BF16 (Tensor cores)**   | 1024 FLOPs/cycle           | **142 TFLOPS**      |
| FP16 (Tensor) with 2:4 sparsity  | 2048 FLOPs/cycle           | 284 TFLOPS          |
| INT8 (Tensor cores)              | 2048 ops/cycle             | 284 TOPS            |

(**2:4 sparsity** = a hardware-accelerated weight format where every
4 consecutive elements have at most 2 non-zero (the other 2 are
exactly zero). The Tensor core skips the zero multiplies, doubling
throughput. Requires the weight tensor to be pre-pruned to this
pattern — typically applied to inference weights, not used in nanoops.)

**Memory side**: 936 GB/s HBM bandwidth. Compute-to-bandwidth ratio at
bf16 = 142 TFLOPS / 936 GB/s = **152 FLOPs per byte**. Any op below
that ratio is bandwidth-bound; above it is compute-bound.

**How arithmetic intensity is computed** (for a matmul `C = A @ B`,
shapes `(M, K) @ (K, N) → (M, N)`, all bf16):

- **FLOPs** = `2·M·N·K` — each output element needs K multiply-adds (each
  multiply-add = 2 FLOPs)
- **Bytes** = `2·(M·K + K·N + M·N)` — read A + read B + write C, 2 bytes
  per bf16 element
- **AI** = FLOPs / Bytes = `M·N·K / (M·K + K·N + M·N)` (the 2's cancel)

The break-even at 152 FLOPs/byte (= 142 TFLOPS / 936 GB/s) tells you
whether the matmul is bottlenecked by GPU compute (above) or HBM
bandwidth (below).

For nanchat-d24 matmuls at training shapes (M=2048, K=1536; N varies):

| Matmul site                | Shape (M, K, N)       | AI (FLOPs/byte) |
| -------------------------- | --------------------- | --------------- |
| c_q / c_k / c_v / attn c_proj | (2048, 1536, 1536) | **558**         |
| MLP c_fc (D → 4D)          | (2048, 1536, 6144)    | **770**         |
| MLP c_proj (4D → D)        | (2048, 6144, 1536)    | **770**         |

All comfortably above the 152 FLOPs/byte break-even → **compute-bound**.

**RMSNorm and other elementwise/reduction ops are heavily
bandwidth-bound** — about 100× below break-even. For one row of D
elements:

- FLOPs ≈ `4·D` (mean(x²) = ~2D, then `x · rms_inv · weight` = 2D)
- Bytes (bf16) ≈ `4·D` (read x = 2D, write y = 2D; weight read is `D`
  per kernel, amortized away)
- **AI ≈ 1 FLOPs/byte** — vs 152 break-even, this is bandwidth-bound
  by ~100×

Summary table (training-shape scale):

| Op                          | AI (FLOPs/byte) | Bound      |
| --------------------------- | --------------- | ---------- |
| MLP / QKV matmul            | 558 – 770       | compute    |
| SDPA `Q@K^T` (B=1, L=2048)  | ~50 – 100       | mixed      |
| **RMSNorm**                 | **~1**          | bandwidth  |
| Elementwise (add, relu)     | ~0.5            | bandwidth  |
| Memcpy H2D / D2H            | 0               | bandwidth  |

**Why this motivates the fusion stack:** standalone RMSNorm kernel
does ~4·M·D bytes of HBM traffic for ~0 compute return. Fusing norm
into the adjacent matmul kernel (`NormMLPReluSquare`,
`NormQKVProjection`) keeps the normalized intermediate in registers,
saving the 2·M·D round-trip — pure bandwidth win on a bandwidth-bound
op, no downside. Same idea for fused_add_norm at block boundaries.

**SDPA: why Flash Attention exists.** Same AI lens shows the SDPA
case has *two* numbers depending on whether the `(L, L)` P matrix
gets materialized.

For nanchat d24 (B=1, H=12, L=2048, D_head=128), full attention
(Q@K^T + softmax + @V, both passes):

| Implementation                          | Total FLOPs | Total bytes | AI                |
| --------------------------------------- | ----------- | ----------- | ----------------- |
| Naive SDPA (materialize P to HBM)       | 25.8 G      | ~226 MB     | **~114 FLOPs/byte** (bandwidth-ish) |
| Flash SDPA (P stays in registers)       | 25.8 G      | ~25 MB      | **~1024 FLOPs/byte** (compute-bound) |

The P matrix is `B·H·L·L = 100 MB` at this scale and dominates the
naive byte count. Flash's online softmax + tile streaming keeps P in
registers so the HBM traffic shrinks to just `Q + K + V + O` (~25 MB),
flipping SDPA from bandwidth-bound (~114 < 152 break-even) to
compute-bound (~1024 >> break-even).

This **8x AI gain** is exactly why Flash Attention is 2-4× faster than
naive SDPA — same FLOPs, ~9× less HBM traffic. Sliding-window
attention shrinks both numbers proportionally (band of size W instead
of L), but the same Flash-vs-naive ratio still applies.

### Tensor cores vs CUDA (FP32) cores

These are **different physical units**, optimized for different work
patterns. Understanding the split changes how you reason about kernel
throughput.

| Unit              | Per SM | What each does per cycle                | SM throughput               |
| ----------------- | ------ | --------------------------------------- | --------------------------- |
| **FP32 cores**    | 128    | 1 scalar fp32 multiply-add              | 256 fp32 FMA/cycle          |
| **Tensor cores**  | 4      | 1 small matrix `mma` (e.g. 16×16×16 bf16) — 256 multiply-adds per instruction | **1024 bf16 FMA/cycle**     |

So 4 Tensor cores deliver ~4× the bf16 throughput that 128 FP32 cores
deliver in fp32. Going bf16/fp16 and using `tl.dot` unlocks that 4×.

**Why only 4 Tensor cores per SM:**

1. **Silicon area.** One Tensor core is ~tens of FP32-cores worth of
   transistors. Four already eats a big slice of the SM die.
2. **Scheduling matches.** An Ampere SM has **4 warp schedulers**;
   each cycle, each scheduler issues one `mma.sync` to one Tensor
   core. 4 schedulers × 4 Tensor cores = perfect 1:1 — adding a fifth
   Tensor core would just sit idle.
3. **Data bandwidth.** Each Tensor core consumes a (16, 16) tile of
   fp16 per instruction (512 B). Four together demand ~2 KB/cycle out
   of shared memory — that's already close to the shared-memory port
   bandwidth ceiling.

### Warps, warp schedulers, and which cores actually run

A **warp** = 32 threads. The fundamental scheduling unit.

```
                    SM (one of 82 on RTX 3090)
   ┌──────────────────────────────────────────────────────────┐
   │  4 warp schedulers (round-robin)                          │
   │     │                                                     │
   │     ├──► dispatch warp instruction every cycle ─►         │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     ├─────►│ 128 FP32 cores (= 4 lanes of 32)    │       │
   │     │      │ scalar ops: fma, sin, add, ld, st…  │       │
   │     │      └─────────────────────────────────────┘       │
   │     │                                                     │
   │     │      ┌─────────────────────────────────────┐       │
   │     └─────►│ 4 Tensor cores                      │       │
   │            │ `mma.sync` → matrix multiply-acc    │       │
   │            └─────────────────────────────────────┘       │
   │                                                           │
   │  Register file: 65,536 × 32-bit (shared by all warps)    │
   │  Shared memory: 100 KB (per thread block)                 │
   └──────────────────────────────────────────────────────────┘
```

**Key relationships:**

- **1 warp issues 1 instruction** at a time. Either to FP32 cores
  (32 lanes × 1 fp32 op = 32 ops in one cycle, one per thread) or to
  a Tensor core (the whole warp cooperates to produce one matrix `mma`).
- A **`mma.sync` instruction is warp-level**, not thread-level — all
  32 threads in the warp participate. Their per-thread register
  fragments concatenate into the (16, 16) matrix tiles the Tensor
  core reads.
- Up to **48 warps can be resident** per SM (= 1536 threads). The
  scheduler picks among them: when one warp stalls on a load, others
  fill the gap. This is **latency hiding** — the reason GPUs don't
  need fancy out-of-order execution like CPUs.
- **Block size matters for occupancy.** A block with 256 threads =
  8 warps. 100 KB shared memory per block → only 1 block fits per SM
  → 8 warps active out of 48 possible → low occupancy → fewer warps
  to hide latency with → throughput drops. Tighter tile sizes (less
  shared mem) let 2-3 blocks co-reside, raising occupancy.

**Triton vs CUDA C++ for warp control:**

- CUDA C++: you explicitly call `__syncwarp()`, manage `mma::sync`,
  pick fragment layouts.
- Triton: hides warps. You only see `tl.program_id` (block-level)
  and `tl.arange` (compile-time vector). Triton's compiler decides
  how to map your vector ops onto 32-thread warps and which lane
  reads which element. You only control:
  - `num_warps=4` (default) — how many warps per block (4 warps =
    128 threads). Bigger = more parallelism per block but fewer
    blocks can co-reside.
  - `num_stages=2` — pipelining depth for `tl.load`/`tl.dot`.

### Occupancy: how register / thread / block budgets combine

Occupancy = `resident warps on SM / max warps on SM` (48 for 3090).
Higher is generally better — more warp variety → more chances to hide
latency by switching when a warp stalls on memory. The three per-SM
caps (registers, threads, blocks) and the per-block shared-memory cap
all act as **simultaneous upper bounds** on `blocks/SM` — the actual
value is the minimum across all of them.

For a hypothetical kernel: 128 threads/block, 32 regs/thread, 4 KB
shared memory/block:

| Limit                              | Per-block usage | Blocks/SM allowed   |
| ---------------------------------- | --------------- | ------------------- |
| Registers (65,536 total)           | 128 × 32 = 4 K  | 16                  |
| Threads (1,536 total)              | 128             | **12**              |
| Shared memory (100 KB total)       | 4 KB            | 25                  |
| Per-SM block cap                   | n/a             | 16                  |
| **`min` → actual blocks/SM**       |                 | **12**              |

12 blocks × (128/32) = **48 warps resident** → **100% occupancy**.

For a heavier kernel: 512 threads/block, 64 regs/thread, 48 KB
shared/block:

| Limit                              | Per-block usage | Blocks/SM allowed   |
| ---------------------------------- | --------------- | ------------------- |
| Registers                          | 512 × 64 = 32 K | 2                   |
| Threads                            | 512             | 3                   |
| **Shared memory**                  | **48 KB**       | **2**               |
| Per-SM block cap                   | n/a             | 16                  |
| **`min`**                          |                 | **2**               |

2 blocks × (512/32) = **32 warps resident** → 32/48 = **67% occupancy**.

#### Register pressure — when the limit really bites

A per-thread architectural max of 255 registers means `reg/thread ×
active threads ≤ 65,536`. So `255 × 1536 = 391,680` is impossible —
you can't actually run 1,536 threads each using 255 registers. The
hardware uses register count to **directly cap active thread count**.

Sweep over reg/thread (3090, ignoring shared-mem/block-cap):

| reg/thread | Active threads/SM | Active warps | Occupancy |
| ---------- | ----------------- | ------------ | --------- |
| 32         | 1,536 (hw cap)    | 48           | **100%**  |
| 42         | 1,536 (hw cap)    | 48           | **100%** ← sweet spot (65,536/42=1,560) |
| 43         | 1,524             | 47           | 98%       |
| 64         | 1,024             | 32           | 67%       |
| 128        | 512               | 16           | 33%       |
| 255 (max)  | **256**           | **8**        | **17%**   |

42 regs/thread is the cliff: any more and occupancy starts dropping
because the register file runs out before the 1,536-thread cap does.

#### Why compilers sometimes pick high reg counts anyway

Going over 255 registers isn't allowed — they get **spilled to local
memory** (a private region in HBM). One spill access ≈ 300 cycles, vs
1 cycle for a register read. **Spilling is much worse than lower
occupancy**, so compilers (NVCC, Triton) trade occupancy down to avoid
spilling whenever they can.

A complex SDPA-backward kernel might genuinely need 100+ registers
per thread → 33% occupancy → and the larger tile size that the
register budget bought back more than makes up for fewer warps. nanoops'
matmul kernels typically land in the 50-100 reg/thread range,
30-70% occupancy.

There's no Triton API to set reg/thread directly; it's chosen by the
compiler based on the kernel body, `BLOCK_*` sizes, `num_warps`, and
`num_stages`. To force a cap, the CUDA path uses
`nvcc -maxrregcount=N`, but it's rarely the right move — the compiler's
spill-vs-occupancy trade-off is usually better than hand-tuning.

Three knobs Triton exposes to tune this:

- **BLOCK_M / BLOCK_N / BLOCK_K** ↑ → per-block thread / register /
  shared-memory usage ↑ → blocks/SM ↓ → occupancy can drop.
- **`num_warps=N`** (per `triton.Config`) ↑ → per-block thread count
  ↑ → same effect.
- **`num_stages=N`** ↑ → more pipelining buffers in shared memory →
  shared-memory usage ↑ → blocks/SM ↓.

There's no universally optimal occupancy — it depends on the kernel:

- **Compute-bound** (matmul-heavy): larger tiles + moderate occupancy
  (50-75%) typically wins, since fewer-bigger blocks keep the
  arithmetic units saturated with less scheduling churn.
- **Bandwidth-bound** (norm, elementwise): high occupancy (80-100%)
  matters more, since latency hiding via warp switching is the
  dominant lever.

Triton's `triton.autotune` tries multiple `(BLOCK, num_warps,
num_stages)` configs and picks the fastest empirically — typically the
right answer rather than manual derivation, since the actual best
config depends on cache effects too.

### What this means for our kernels

- **Shared memory budget is the binding constraint.** Most of our
  kernel tile sizes (BLOCK_M, BLOCK_N, BLOCK_K) were chosen so the
  combined working set (input tiles + accumulator) fits in ≤ 60–80 KB
  per block, leaving headroom for the runtime and not forcing a
  one-block-per-SM occupancy collapse.

- **3090's shared memory is ~half of A100 (164 KB) and ~third of
  H100 (228 KB).** This is why Flash Attention on 3090 needs smaller
  tile sizes than the canonical Flash kernels published with H100
  numbers. Larger Q/K/V tiles spill or refuse to compile.

- **82 SMs is the rough minimum grid size to saturate the GPU.**
  Anything that launches fewer programs leaves SMs idle. For
  per-row-reduction kernels (RMSNorm, softmax, fused_add_norm),
  we tile along the M (batch × seq) dim because M is typically
  ≥ 2048 in training — comfortable parallelism.

- **No FA3-style TMA / async-copy hardware.** Hopper-specific
  techniques (TMA, distributed shared memory, warp-specialization
  via `wgmma`) aren't usable on Ampere. Our Flash SDPA kernel is the
  classic Triton tutorial pattern, not the FA3 pattern.

- **Tensor cores are the FLOP/s budget.** Anything not using
  `tl.dot` (i.e. plain elementwise) runs on the FP32 cores and gets
  ~1/10 of the peak FLOP/s. Fusion that lets us replace a separate
  elementwise launch with an epilogue inside a `tl.dot`-using kernel
  is almost always a win.

### Comparison with other GPUs (for porting context)

| GPU         | SMs | Shared/SM | Mem            | Bandwidth   | Notable for our kernels             |
| ----------- | --- | --------- | -------------- | ----------- | ----------------------------------- |
| **RTX 3090**| 82  | 100 KB    | 24 GB GDDR6X   | 936 GB/s    | (our target)                        |
| RTX 4090    | 128 | 100 KB    | 24 GB GDDR6X   | 1008 GB/s   | Same tile sizes likely fine         |
| A100 80GB   | 108 | 164 KB    | 80 GB HBM2e    | 1935 GB/s   | Can roughly 1.5× the tile sizes     |
| H100 SXM    | 132 | 228 KB    | 80 GB HBM3     | 3000+ GB/s  | Bigger tiles + FA3 / TMA path opens |

---

(Chapters 2+ — per-kernel deep dive — TBD)
