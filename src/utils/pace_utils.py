"""
Shared helpers for PACE-Flow-Dyn integration in model wrappers.
"""
from __future__ import annotations

from typing import Tuple

import torch


def is_pace_strategy(strategy: object) -> bool:
    """Lightweight capability check for PACE-Flow-Dyn strategies."""
    return bool(getattr(strategy, "supports_pace_flow_dyn", False))


def use_particle_expansion(strategy: object) -> bool:
    """Whether the strategy expects particle expansion and finalize()."""
    return (
        is_pace_strategy(strategy)
        and getattr(strategy, "num_particles", 1) > 1
        and hasattr(strategy, "finalize")
    )


def init_pace_state(
    strategy: object,
    *,
    batch_size: int,
    seq_len: int,
    prompt_len: int,
    mask_id: int,
    device: torch.device,
    total_steps: int,
) -> None:
    """Initialize per-run state for a PACE strategy."""
    strategy.reset_state(
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_len=prompt_len,
        mask_id=mask_id,
        device=device,
        total_steps=total_steps,
    )


def expand_for_particles(
    strategy: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    """Repeat inputs along the batch dimension for particle sampling."""
    repeat = strategy.num_particles
    input_ids = input_ids.repeat_interleave(repeat, dim=0)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat_interleave(repeat, dim=0)
    return input_ids, attention_mask


def finalize_particles(strategy: object, sequences: torch.Tensor) -> torch.Tensor:
    """Reduce particle-expanded sequences to a single sequence per batch."""
    return strategy.finalize(sequences)


def consume_strategy_particle_log(model: object) -> torch.Tensor | None:
    """Fetch and clear any per-step particle logs if the strategy exposes them."""
    strategy = getattr(model, "strategy", None)
    if strategy is None:
        return None
    if hasattr(strategy, "consume_particle_log"):
        return strategy.consume_particle_log()
    return None


def append_particle_log(
    particle_logs: list[torch.Tensor],
    log: torch.Tensor | None,
) -> None:
    """Normalize a log tensor to CPU and append it to a batch list."""
    if log is None:
        return
    log = log.detach()
    if log.device.type != "cpu":
        log = log.cpu()
    particle_logs.append(log)


def finalize_particle_logs(particle_logs: list[torch.Tensor]) -> torch.Tensor | None:
    """Concatenate batch logs into a single tensor and clear the list."""
    if not particle_logs:
        return None
    if len(particle_logs) == 1:
        log = particle_logs[0]
    else:
        base_shape = particle_logs[0].shape[1:]
        for idx, part in enumerate(particle_logs[1:], start=1):
            if part.shape[1:] != base_shape:
                raise ValueError(
                    "Inconsistent particle log shapes across batches: "
                    f"index=0 shape={particle_logs[0].shape}, "
                    f"index={idx} shape={part.shape}."
                )
        log = torch.cat(particle_logs, dim=0)
    particle_logs.clear()
    return log
