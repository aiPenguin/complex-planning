#!/usr/bin/env bash
set -euo pipefail

# PACE-Flow-Dyn strategy matching Dream experiment length/steps
python main.py model=llada strategy=pace_flow_dyn eval=sudoku \
  strategy.steps=24 \
  strategy.max_new_tokens=24 \
  strategy.block_length=24 \
  strategy.cfg_scale=0 
