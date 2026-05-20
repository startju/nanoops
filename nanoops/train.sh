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

NPROC=${NPROC:-2}
WANDB_RUN=${WANDB_RUN:-dummy}

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- \
    --depth=20 \
    --target-param-data-ratio=8 \
    --device-batch-size=4 \
    --run=$WANDB_RUN \
    "$@"
