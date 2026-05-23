# nanoops Triton Kernels — Architecture Notes

Deep dive into the fused CUDA kernels in `nanoops/triton_kernels.py`. Aimed
at someone who's read the kernels and wants to understand the design
choices (block sizes, fusion boundaries, what to save in ctx, etc.) —
not a Triton tutorial.

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
