from dataclasses import dataclass
from pathlib import Path
from typing import Any, List
import textwrap

from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm


@dataclass
class ModelGenerator:
    """Thin adapter so evaluators only rely on a `generate()` method."""
    model: object
    batch_size: int | None = None
    run_label: str | None = None
    sample_max_chars: int = 600

    def _format_sample(self, text: str) -> str:
        preview = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(preview) > self.sample_max_chars:
            preview = preview[: self.sample_max_chars] + "...(truncated)"
        return preview

    def generate(self, prompts: List[str]) -> List[str]:
        if prompts:
            label = self.run_label or "generate"
            total = len(prompts)
            if self.batch_size is None:
                print(f"\n[{label}] inputs={total} batch_size=None")
            else:
                num_batches = (total + self.batch_size - 1) // self.batch_size
                print(
                    f"\n[{label}] inputs={total} batch_size={self.batch_size} batches={num_batches}"
                )
            sample = self._format_sample(prompts[0])
            print("[sample input]\n" + textwrap.indent(sample, "  "))

        if self.batch_size is None:
            return self.model.generate(prompts)

        outputs: List[str] = []
        total = len(prompts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        batch_iter = range(0, len(prompts), self.batch_size)
        for start in tqdm(batch_iter, total=num_batches, desc="generate", unit="batch"):
            batch = prompts[start : start + self.batch_size]
            outputs.extend(self.model.generate(batch))
        return outputs


class Engine:
    """Hydra-wired orchestrator that builds model, strategy, and evaluator."""
    def __init__(
        self,
        model_cfg: DictConfig,
        strategy_cfg: DictConfig,
        eval_cfg: DictConfig,
        batch_size: int | None = None,
    ) -> None:
        self.model_cfg = model_cfg
        self.strategy_cfg = strategy_cfg
        self.eval_cfg = eval_cfg
        self.batch_size = batch_size
        self.strategy = self._init_strategy()
        self.model = self._init_model(self.strategy)
        self.output_dir = self._resolve_output_dir()
        self.generator = ModelGenerator(self.model, batch_size=self.batch_size)
        self.evaluator = self._init_evaluator()
        self.generator.run_label = self.evaluator.__class__.__name__
        setattr(self.evaluator, "output_dir", str(self.output_dir))

    def _init_model(self, strategy: object) -> object:
        """Instantiate the model with the configured strategy injected."""
        # Keep the strategy instance intact; Hydra otherwise converts it to DictConfig.
        return instantiate(self.model_cfg, strategy=strategy, _convert_="object")

    def _init_strategy(self) -> object:
        """Instantiate the unmasking strategy used by the model."""
        return instantiate(self.strategy_cfg)

    def _init_evaluator(self) -> object:
        """Instantiate the evaluation task implementation."""
        return instantiate(self.eval_cfg)

    def generate(self, prompts: List[str]) -> List[str]:
        """Proxy generation through the configured model."""
        return self.generator.generate(prompts)

    def run(self):
        """Run the evaluator in either `evaluate()` or callable form."""
        try:
            self._print_run_summary()
            if hasattr(self.evaluator, "evaluate"):
                return self.evaluator.evaluate(self.generator)
            if callable(self.evaluator):
                return self.evaluator(self.generator)
            raise TypeError("Evaluator must be callable or define evaluate().")
        except Exception as exc:
            print("\n[error] Evaluation failed.")
            print("Hint: use HYDRA_FULL_ERROR=1 to see the full stack trace.")
            raise exc

    def _print_run_summary(self) -> None:
        model_name = getattr(self.model_cfg, "model_name", None)
        model_target = getattr(self.model_cfg, "_target_", type(self.model).__name__)
        device = getattr(self.model, "device", None)
        dtype = getattr(getattr(self.model, "model", None), "dtype", None)

        strategy = self.strategy
        strategy_name = strategy.__class__.__name__
        strategy_fields = {}
        for key in (
            "steps",
            "gen_length",
            "max_new_tokens",
            "block_length",
            "temperature",
            "cfg_scale",
            "num_particles",
            "top_p",
            "top_k",
            "alg",
            "remasking",
        ):
            if hasattr(strategy, key):
                strategy_fields[key] = getattr(strategy, key)

        evaluator = self.evaluator
        eval_name = evaluator.__class__.__name__
        eval_fields = {}
        for key in ("n_few_shots", "max_items", "variants", "n_values"):
            if hasattr(evaluator, key):
                eval_fields[key] = getattr(evaluator, key)

        print("\n=== Run Summary ===")
        print(f"model:    {model_target}")
        if model_name:
            print(f"model_id: {model_name}")
        if device is not None:
            print(f"device:   {device}")
        if dtype is not None:
            print(f"dtype:    {dtype}")
        print(f"strategy: {strategy_name} {strategy_fields}")
        print(f"eval:     {eval_name} {eval_fields}")
        print(f"batch:    {self.batch_size}")
        print(f"output:   {self.output_dir}")

    def _resolve_output_dir(self) -> Path:
        try:
            runtime_output = HydraConfig.get().runtime.output_dir
            output_dir = Path(runtime_output)
            output_dir.mkdir(parents=True, exist_ok=True)
            return output_dir
        except Exception:
            fallback = Path("output") / "runs" / "unknown_run"
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
