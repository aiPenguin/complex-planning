from dataclasses import dataclass
from typing import List

from hydra.utils import instantiate
from omegaconf import DictConfig


@dataclass
class ModelGenerator:
    """Thin adapter so evaluators only rely on a `generate()` method."""
    model: object
    batch_size: int | None = None

    def generate(self, prompts: List[str]) -> List[str]:
        if self.batch_size is None:
            return self.model.generate(prompts)

        outputs: List[str] = []
        for start in range(0, len(prompts), self.batch_size):
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
        self.generator = ModelGenerator(self.model, batch_size=self.batch_size)
        self.evaluator = self._init_evaluator()

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
        if hasattr(self.evaluator, "evaluate"):
            return self.evaluator.evaluate(self.generator)
        if callable(self.evaluator):
            return self.evaluator(self.generator)
        raise TypeError("Evaluator must be callable or define evaluate().")
