# complex-planning

Hydra-based evaluation harness for diffusion-style planning models (LLaDA and Dream).

**What This Repository Contains**
- Model wrappers for LLaDA and Dream that expose a common `generate` API.
- Unmasking strategies that implement the diffusion-style decoding loops.
- Evaluators for Sudoku, Countdown (CD), and trip-planning tasks.
- Hydra configs to mix and match model, strategy, and evaluator settings.

**Project Layout**
- `main.py`: Entry point that wires model, strategy, and evaluator via Hydra.
- `configs/`: Hydra configs for model, strategy, and eval selections.
- `src/models/`: Model wrappers with `generate()` implementations.
- `src/strategies/`: Unmasking strategies used during generation.
- `src/eval/`: Task evaluators and dataset-specific scoring.
- `src/utils/`: Shared helpers and borrowed Dream generation utilities.
- `data/`: Evaluation datasets (JSON/JSONL).

**Quickstart**
1. Run the default configuration (LLaDA + native strategy + Sudoku eval):

```bash
python main.py
```

2. Override configuration pieces via Hydra:

```bash
python main.py model=dream strategy=dream_native eval=trip
```

Hydra writes outputs under `output/YYYY-MM-DD/HH-MM-SS` by default (see `configs/config.yaml`).

**Notes**
- The Hugging Face model configs use `trust_remote_code: true`; review the upstream repositories before running.
