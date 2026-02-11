import re
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))


OmegaConf.register_new_resolver("slug", _slug, replace=True)

from src.engine import Engine


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Instantiate configured components and run the evaluator."""
    print("\n=== Run Config ===")
    print(f"model:    {cfg.model._target_} ({cfg.model.model_name})")
    print(f"strategy: {cfg.strategy._target_}")
    print(f"eval:     {cfg.eval._target_}")
    print(f"batch:    {cfg.batch_size}\n")
    engine = Engine(cfg.model, cfg.strategy, cfg.eval, batch_size=cfg.batch_size)
    engine.run()


if __name__ == "__main__":
    main()
