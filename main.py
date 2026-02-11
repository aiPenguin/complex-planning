import hydra
from omegaconf import DictConfig

from src.engine import Engine


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Instantiate configured components and run the evaluator."""
    engine = Engine(cfg.model, cfg.strategy, cfg.eval)
    engine.run()


if __name__ == "__main__":
    main()
