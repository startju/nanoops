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

For nanchat-d24 matmuls (M=2048, K=1536, N=1536): arithmetic intensity
= 2*M*K*N / (M*K + K*N + M*N) ≈ 766 FLOPs/byte → comfortably
compute-bound. Plain elementwise ops (relu, add) ≈ 1-2 FLOPs/byte →
bandwidth-bound. SDPA's `Q@K^T` is somewhere in between depending on
seq length.

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
