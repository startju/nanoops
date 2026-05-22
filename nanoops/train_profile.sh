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
PROFILE_START=${PROFILE_START:-2}
PROFILE_END=${PROFILE_END:-4}
PROFILE_NUM_ITERS=${PROFILE_NUM_ITERS:-5}
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

# ─── Post-run: auto-print summary stats so you don't have to grep nsys CLI ───
NSYS_REP=${PROFILE_OUTPUT}.nsys-rep
if [ -f "$NSYS_REP" ]; then
    SIZE=$(du -h "$NSYS_REP" | cut -f1)
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " Trace ready: $NSYS_REP ($SIZE)"
    echo "════════════════════════════════════════════════════════════"

    echo ""
    echo "── NVTX Range Summary ──"
    nsys stats --report nvtx_sum --force-export=true "$NSYS_REP" 2>/dev/null \
        | grep -A 100 "NVTX Range Summary" | head -20

    echo ""
    echo "── Top 10 CUDA Kernels ──"
    nsys stats --report cuda_gpu_kern_sum --format table "$NSYS_REP" 2>/dev/null \
        | grep -A 13 "CUDA GPU Kernel Summary" | head -15

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo " View locally:"
    echo "   scp <server>:$NSYS_REP ."
    echo "   nsight-sys $(basename $NSYS_REP)"
    echo " More CLI reports:"
    echo "   nsys stats --report cuda_gpu_sum     $NSYS_REP   # all GPU activity"
    echo "   nsys stats --report cuda_api_sum     $NSYS_REP   # CUDA API calls"
    echo "   nsys stats --report nvtx_pushpop_sum $NSYS_REP   # per-NVTX breakdown"
    echo "════════════════════════════════════════════════════════════"
fi
