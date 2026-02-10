from typing import Protocol

import torch


class UnmaskingStrategy(Protocol):
    steps: int
    gen_length: int
    block_length: int
    cfg_scale: float
    mask_id: int

    def unmask(
        self,
        *,
        x: torch.Tensor,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        block_end: int,
        num_transfer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        ...
