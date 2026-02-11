"""
Shared helpers for PACE-Flow-Dyn integration in model wrappers.
"""
from __future__ import annotations

from typing import Tuple

import torch


def is_pace_strategy(strategy: object) -> bool:
    return bool(getattr(strategy, "supports_pace_flow_dyn", False))


def use_particle_expansion(strategy: object) -> bool:
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
) -> None:
    strategy.reset_state(
        batch_size=batch_size,
        seq_len=seq_len,
        prompt_len=prompt_len,
        mask_id=mask_id,
        device=device,
    )


def expand_for_particles(
    strategy: object,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> Tuple[torch.Tensor, torch.Tensor | None]:
    repeat = strategy.num_particles
    input_ids = input_ids.repeat_interleave(repeat, dim=0)
    if attention_mask is not None:
        attention_mask = attention_mask.repeat_interleave(repeat, dim=0)
    return input_ids, attention_mask


def finalize_particles(strategy: object, sequences: torch.Tensor) -> torch.Tensor:
    return strategy.finalize(sequences)
