#!/usr/bin/env bash
set -euo pipefail

# Dream native hyperparameters (as provided)
python main.py model=dream strategy=dream_native eval=sudoku \
  strategy.steps=24 \
  strategy.max_new_tokens=24 \
  strategy.temperature=0 \
  strategy.top_p=1 \
  strategy.alg=entropy \
  strategy.alg_temp=0 
