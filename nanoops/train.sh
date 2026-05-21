#!/bin/bash
# Launch base_train with nanoops swapped into nanchat's F namespace.
#
# What this does: sets NANOOPS=1 (the env var that scripts/base_train.py
# reads to call nanoops.integration.patch_nanchat() at startup), then
# launches a d24 base-training run sized for 2× RTX 3090.
#
# Usage:
#   bash nanoops/train.sh                       # defaults below
#   bash nanoops/train.sh --num-iterations=10   # pass extra args through
#   NPROC=4 bash nanoops/train.sh               # override GPU count
#   WANDB_RUN=myrun bash nanoops/train.sh       # enable wandb logging
#
# Compared to runs/speedrun.sh this is just the base_train step in isolation
# (no tokenizer / dataset / SFT / eval) — meant for iterating on nanoops
# itself, not for end-to-end training.

set -e
source .venv/bin/activate

export NANOOPS=1
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
# expandable_segments unfragments PyTorch's caching allocator. With this
# off, the 1-2 GiB of "reserved but unallocated" memory at B=4 prevents
# the last needed allocation and OOMs the run, even though SlidingWindowSDPA
# saves ~2 GiB of P-matrix peak. Setting it lets B=4 fit at ~23.4/24 GiB.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# MLP activation checkpoint ON by default — at B=4 it saves ~3.7 GiB of
# MLP intermediate activations (relu output + relu² output + the
# c_fc/c_proj Mm input ctxs) for a +7% wall-time cost (one extra MLP
# forward in backward). Cost: 0.62 s/GiB freed, strictly better than
# ATTN checkpoint's 0.96 s/GiB. The freed headroom is what lets larger
# --depth runs fit on a 24 GiB card. Opt out by unsetting before bash.
export NANOOPS_MLP_CHECKPOINT="${NANOOPS_MLP_CHECKPOINT:-1}"

NPROC=${NPROC:-2}
WANDB_RUN=${WANDB_RUN:-dummy}

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=24 \
    --target-param-data-ratio=8 \
    --device-batch-size=1 \
    --run=$WANDB_RUN \
    "$@"
# --depth=24 / device-batch-size=1 on 2× RTX 3090 (24 GiB each):
# d24 auto-widens to D=1536, n_layer=24, ~1.5B params, ~1.67× heavier than
# d20. Even with all three optimizations active (sliding window +
# expandable_segments + MLP_CHECKPOINT) only B=1 fits — B=2 OOMs by ~20 MiB,
# B=4 OOMs by ~1 GiB.
#
# Measured numbers at d24 + B=1:
#   tok/sec  ~15,800
#   MFU       ~53% (B=1 micro-batches don't saturate GEMMs)
#   dt        ~66.5 s/iter (256 grad-accum steps × 270 ms)
#   peak mem  ~22.3 GiB (1.7 GiB headroom)
#   ETA       ~61 h for full 3320-iter run
#
# Drop to --depth=20 --device-batch-size=4 for the throughput sweet spot
# (~30.5k tok/s, ~31h ETA, much better MFU).
