#!/usr/bin/env bash
set -euo pipefail

# PACE-Flow-Dyn strategy matching Dream experiment length/steps
python main.py model=dream strategy=pace_flow_dyn eval=sudoku \
  strategy.steps=24 \
  strategy.max_new_tokens=24 \
  strategy.temperature=0 \
  strategy.top_p=1 \
  strategy.alg=entropy \
  strategy.alg_temp=0 
