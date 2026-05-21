#!/bin/bash
# Launch base_train with nanoops swapped into nanchat's F namespace.
#
# What this does: sets NANOOPS=1 (the env var that scripts/base_train.py
# reads to call nanoops.integration.patch_nanchat() at startup), then
# launches the same d20 base-training command speedrun.sh uses.
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
    --depth=20 \
    --target-param-data-ratio=8 \
    --device-batch-size=4 \
    --run=$WANDB_RUN \
    "$@"
# device-batch-size=4 (matches speedrun) is now possible because:
#   1. SlidingWindowSDPA (default ON) cuts ~2 GiB of P-matrix peak
#      across the 15 sliding layers
#   2. expandable_segments=True (set above) recovers the 1-2 GiB lost
#      to allocator fragmentation
# Together these let B=4 fit at ~23.4/24 GiB on 2× RTX 3090, giving
# ~33,000 tok/s, MFU 67% (vs B=2's 30,600 tok/s and the original
# 22,700 tok/s baseline — total ~45% improvement on the stack).
# Drop to --device-batch-size=2 if you OOM on tighter memory.
