#!/usr/bin/env bash
set -euo pipefail

# LLaDA native strategy matching Dream experiment length/steps
python main.py model=llada strategy=llada_native eval=sudoku \
  strategy.steps=24 \
  strategy.gen_length=24 \
  strategy.block_length=24 \
  strategy.temperature=0 \
  strategy.cfg_scale=0 
