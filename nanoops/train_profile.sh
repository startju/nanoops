#!/bin/bash
# Launch base_train under nsys profile — trace a short window of training
# iters so the .nsys-rep file stays manageable.
#
# What this does compared to nanoops/train.sh:
#   - Wraps the python/torchrun launcher in `nsys profile` with
#     --capture-range=cudaProfilerApi, so trace events are only emitted
#     between cudaProfilerStart and cudaProfilerStop.
#   - Sets NANOOPS_NSYS_START_STEP / NANOOPS_NSYS_END_STEP env vars,
#     which scripts/base_train.py reads to trigger the
#     cudaProfilerStart/Stop calls inside its training loop. Default
#     window is steps 5-7 (skips compile + iter-0 warmup; covers 3 fully
#     compiled iters).
#   - Forces --eval-every=-1 --core-metric-every=-1 so no eval forward
#     passes burn trace window or wall time.
#   - Caps --num-iterations to PROFILE_NUM_ITERS (default 10) so the
#     run actually exits after the trace window.
#
# Usage:
#   bash nanoops/train_profile.sh                     # defaults below
#   PROFILE_START=8 PROFILE_END=11 bash nanoops/train_profile.sh
#   PROFILE_NUM_ITERS=15 bash nanoops/train_profile.sh
#   NPROC=1 bash nanoops/train_profile.sh             # single-GPU profile
#   PROFILE_OUTPUT=/tmp/myrun bash nanoops/train_profile.sh
#
# After the run:
#   scp <server>:/tmp/nanoops_profile.nsys-rep .
#   nsight-sys nanoops_profile.nsys-rep              # local GUI

set -e
source .venv/bin/activate

# nsys profile window: bracket which steps are captured. Pick the window
# AFTER compile+warmup so what's in the trace is steady-state training.
PROFILE_START=${PROFILE_START:-5}
PROFILE_END=${PROFILE_END:-8}
PROFILE_NUM_ITERS=${PROFILE_NUM_ITERS:-10}
PROFILE_OUTPUT=${PROFILE_OUTPUT:-/tmp/nanoops_profile}

# Env vars consumed by scripts/base_train.py to drive cudaProfilerStart/Stop
export NANOOPS_NSYS_START_STEP=$PROFILE_START
export NANOOPS_NSYS_END_STEP=$PROFILE_END

# Reuse the same nanoops integration env vars as the regular train.sh —
# NVTX + cudaProfiler hooks live inside the SAME code path, so the patch
# stack we profile is exactly what production training uses.
export NANOOPS=1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
export NANOOPS_MLP_CHECKPOINT="${NANOOPS_MLP_CHECKPOINT:-1}"
export NANOOPS_OFFLOAD_OPTIM="${NANOOPS_OFFLOAD_OPTIM:-1}"
export NANOOPS_L_ATTN_CHECKPOINT="${NANOOPS_L_ATTN_CHECKPOINT:-1}"

NPROC=${NPROC:-2}
WANDB_RUN=${WANDB_RUN:-dummy}

NSYS_FLAGS=(
    --trace=cuda,nvtx,cudnn,cublas
    --capture-range=cudaProfilerApi
    --capture-range-end=stop
    --output=$PROFILE_OUTPUT
    --force-overwrite=true
)

# Same NPROC dispatch logic as nanoops/train.sh: NPROC=1 uses plain python
# (no torchrun → no RANK env → MuonAdamW path); NPROC>=2 uses torchrun
# (→ DistMuonAdamW path).
COMMON_ARGS=(
    --depth=24
    --target-param-data-ratio=8
    --device-batch-size=1
    --val-device-batch-size=16
    --run=$WANDB_RUN
    --num-iterations=$PROFILE_NUM_ITERS
    --eval-every=-1
    --core-metric-every=-1
)

if [ "$NPROC" = "1" ]; then
    nsys profile "${NSYS_FLAGS[@]}" \
        python -u -m scripts.base_train "${COMMON_ARGS[@]}" "$@"
else
    nsys profile "${NSYS_FLAGS[@]}" \
        torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
            "${COMMON_ARGS[@]}" "$@"
fi

# Tips for inspecting the trace:
#   - Local GUI: `nsight-sys $PROFILE_OUTPUT.nsys-rep`
#   - CLI stats: `nsys stats $PROFILE_OUTPUT.nsys-rep` (CUDA API + kernel time tables)
#   - Filter by NVTX: in GUI, "Timeline → Filter" → search "iter_5" / "fwd+bwd" / "optim_step"
#   - For nanoops you'll most want to look at:
#     - SDPA kernels under "fwd+bwd" → time per layer + chunk granularity
#     - cudaMemcpyAsync under "optim_step" → PCIe H2D/D2H for state offload
#     - NCCL kernels (dual-GPU): all_reduce/all_gather, look for idle gaps
#     - cudaMalloc/cudaFree spikes → allocator hot paths
