from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import json
import textwrap

from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from tqdm import tqdm
import torch

from src.utils.pace_utils import (
    consume_strategy_particle_log,
    consume_strategy_generation_candidates,
    append_particle_log,
    finalize_particle_logs,
    append_generation_candidates,
    finalize_generation_candidates,
)

@dataclass
class ModelGenerator:
    """Thin adapter so evaluators only rely on a `generate()` method."""
    model: object
    batch_size: int | None = None
    run_label: str | None = None
    sample_max_chars: int = 600
    _particle_logs: List[torch.Tensor] = field(default_factory=list, init=False, repr=False)
    _candidate_logs: List[dict] = field(default_factory=list, init=False, repr=False)
    _candidate_segments: List[dict] = field(default_factory=list, init=False, repr=False)

    def _format_sample(self, text: str) -> str:
        preview = text.replace("\r\n", "\n").replace("\r", "\n")
        if len(preview) <= self.sample_max_chars:
            return preview
        head_len = self.sample_max_chars // 2
        tail_len = self.sample_max_chars - head_len
        head = preview[:head_len]
        tail = preview[-tail_len:]
        return f"{head}\n...(truncated)...\n{tail}"

    def generate(self, prompts: List[str]) -> List[str]:
        if prompts:
            label = self.run_label or "generate"
            total = len(prompts)
            if self.batch_size is None:
                tqdm.write(f"\n[{label}] inputs={total} batch_size=None")
            else:
                num_batches = (total + self.batch_size - 1) // self.batch_size
                tqdm.write(
                    f"\n[{label}] inputs={total} batch_size={self.batch_size} batches={num_batches}"
                )
            sample = self._format_sample(prompts[0])
            tqdm.write("[sample input]")
            tqdm.write(textwrap.indent(sample, "  "))

        if prompts:
            self._candidate_segments.append(
                {"label": self.run_label or "generate", "count": len(prompts)}
            )

        if self.batch_size is None:
            outputs = self.model.generate(prompts)
            self._record_strategy_logs()
            return outputs

        outputs: List[str] = []
        total = len(prompts)
        num_batches = (total + self.batch_size - 1) // self.batch_size
        batch_iter = range(0, len(prompts), self.batch_size)
        for start in tqdm(batch_iter, total=num_batches, desc="generate", unit="batch"):
            batch = prompts[start : start + self.batch_size]
            outputs.extend(self.model.generate(batch))
            self._record_strategy_logs()
        return outputs

    def _record_strategy_logs(self) -> None:
        log = consume_strategy_particle_log(self.model)
        append_particle_log(self._particle_logs, log)
        candidates = consume_strategy_generation_candidates(self.model)
        append_generation_candidates(self._candidate_logs, candidates)

    def consume_particle_log(self) -> torch.Tensor | None:
        return finalize_particle_logs(self._particle_logs)

    def consume_generation_candidates(self) -> dict | None:
        payload = finalize_generation_candidates(self._candidate_logs)
        if payload is None:
            return None
        payload["segments"] = list(self._candidate_segments)
        self._candidate_segments.clear()
        return payload

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
        self.generator = ModelGenerator(
            self.model,
            batch_size=self.batch_size,
        )
        self.evaluator = self._init_evaluator()
        self.generator.run_label = self.evaluator.__class__.__name__
        setattr(self.evaluator, "output_dir", str(self.output_dir))

    def _init_model(self, strategy: object) -> object:
        """Instantiate the model with the configured strategy injected."""
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
        self._print_run_summary()
        result = self.evaluator.evaluate(self.generator)
        self._save_particle_log()
        return result

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

    def _save_particle_log(self) -> None:
        log = self.generator.consume_particle_log()
        if log is not None:
            filename = getattr(self.strategy, "log_particles_filename", "pace_particles.pt")
            path = Path(self.output_dir) / filename
            torch.save(log, path)
            print(f"[particle_log] saved: {path}")
        candidates = self.generator.consume_generation_candidates()
        if candidates is None:
            return
        cand_name = getattr(self.strategy, "log_candidates_filename", "pace_candidates.pt")
        cand_path = Path(self.output_dir) / cand_name
        torch.save(candidates, cand_path)
        print(f"[particle_log] saved: {cand_path}")

        tokenizer = getattr(self.model, "tokenizer", None)
        prompt_len = candidates.get("prompt_len")
        primary_tokens = candidates.get("primary")
        if tokenizer is None or primary_tokens is None or prompt_len is None:
            return
        gen_tokens = primary_tokens[:, prompt_len:]
        decoded_primary = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        decoded_secondary = None
        secondary_tokens = candidates.get("secondary")
        if isinstance(secondary_tokens, torch.Tensor):
            decoded_secondary = tokenizer.batch_decode(
                secondary_tokens[:, prompt_len:], skip_special_tokens=True
            )
        decoded_particles = None
        particles = candidates.get("particles")
        if isinstance(particles, torch.Tensor):
            flat = particles[:, :, prompt_len:].reshape(-1, particles.shape[-1] - prompt_len)
            decoded_flat = tokenizer.batch_decode(flat, skip_special_tokens=True)
            decoded_particles = [
                decoded_flat[i : i + particles.shape[1]]
                for i in range(0, len(decoded_flat), particles.shape[1])
            ]

        decoded_payload = {
            "primary": decoded_primary,
            "secondary": decoded_secondary,
            "primary_source": candidates.get("primary_source"),
            "secondary_source": candidates.get("secondary_source"),
            "particles": decoded_particles,
            "segments": candidates.get("segments"),
        }
        decoded_path = cand_path.with_suffix(".json")
        decoded_path.write_text(json.dumps(decoded_payload, ensure_ascii=False, indent=2))
        print(f"[particle_log] saved: {decoded_path}")
