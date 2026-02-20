#!/usr/bin/env bash
set -euo pipefail

# --- Configuration defaults ---
GPU_IDS=(0)

GEN_LENGTH=24
BLOCK_LENGTH=24
STEPS=24

# Allow overriding GPU IDs via CLI args.
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi
GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
echo "Using GPUs: $GPU_LIST"

CUDA_VISIBLE_DEVICES=$GPU_LIST \
python main.py \
  model=dtreerpo \
  strategy=llada_native \
  eval=sudoku \
  strategy.gen_length=$GEN_LENGTH \
  strategy.block_length=$BLOCK_LENGTH \
  strategy.steps=$STEPS \
  strategy.temperature=0.0 \
  strategy.cfg_scale=0.0 \
  strategy.mask_id=126336 \
  strategy.remasking=low_confidence
