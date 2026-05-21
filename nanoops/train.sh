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
# device-batch-size=4 (matches speedrun) is possible on 24 GiB cards
# because three optimizations stack:
#   1. SlidingWindowSDPA (default ON): cuts ~2 GiB of P-matrix peak
#      across the 15 sliding layers
#   2. expandable_segments=True (set above): recovers ~1-2 GiB lost to
#      allocator fragmentation
#   3. NANOOPS_MLP_CHECKPOINT=1 (set above): cuts ~3.7 GiB of MLP
#      activations for +7% wall time (0.62 s/GiB freed)
# Measured on 2× RTX 3090, d20:
#   tok/sec ~30,500, MFU ~62%, peak ~19 GiB, dt ~34s, ETA ~31h.
# (Without MLP_CHECKPOINT it'd be ~32,700 tok/s + 22.7 GiB peak / 29h ETA,
# but the freed memory is what lets deeper runs / larger seq-len fit.)
#
# Depth scaling notes (on 2× RTX 3090, 24 GiB each):
#   --depth=20: B=4 fits comfortably (this default)
#   --depth=22: untested, likely B=2
#   --depth=24: only B=1 fits (~22.3 GiB peak, MFU drops to ~53%, 2× slower)
#               — auto-config widens to D=1536 and 1.67× params, too much
#               for 24 GiB cards even with all the optimizations on.
# Drop --device-batch-size for tighter memory.
