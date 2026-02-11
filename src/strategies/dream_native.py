"""
Native Dream diffusion unmasking strategy.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from src.utils.dream_utils import sample_tokens


@dataclass
class DreamNativeStrategy:
    max_new_tokens: int = 512
    steps: int = 512
    temperature: float = 0.2
    top_p: float | None = 0.95
    top_k: int | None = None
    alg: str = "entropy"
    alg_temp: float | None = 0.0
    eps: float = 1e-3

    def unmask(
        self,
        *,
        x: torch.Tensor,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        step: int,
        steps: int,
        mask_token_id: int,
    ) -> torch.Tensor:
        if not mask_index.any().item():
            return x

        mask_logits = logits[mask_index]
        is_last = step >= steps - 1

        if self.alg == "origin":
            p_transfer = 1 - s / t if not is_last else 1
            x0 = torch.full_like(x[mask_index], mask_token_id)
            transfer_index_t_s = torch.rand(*x0.shape, device=x.device) < p_transfer
            if transfer_index_t_s.any().item():
                _, x0[transfer_index_t_s] = sample_tokens(
                    mask_logits[transfer_index_t_s],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
            x[mask_index] = x0
            return x

        if self.alg == "maskgit_plus":
            confidence, x0 = sample_tokens(
                mask_logits, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k
            )
        elif self.alg == "topk_margin":
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                margin_confidence=True,
            )
        elif self.alg == "entropy":
            confidence, x0 = sample_tokens(
                mask_logits,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                neg_entropy=True,
            )
        else:
            raise RuntimeError(f"Unknown alg: {self.alg}")

        num_mask_token = mask_index.sum() / mask_index.shape[0]
        number_transfer_tokens = int(num_mask_token * (1 - s / t)) if not is_last else int(num_mask_token)
        full_confidence = torch.full_like(x, -torch.inf, device=x.device, dtype=logits.dtype)
        full_confidence[mask_index] = confidence
        if number_transfer_tokens <= 0:
            return x

        if self.alg_temp is None or self.alg_temp == 0:
            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
        else:
            full_confidence = full_confidence / self.alg_temp
            full_confidence = F.softmax(full_confidence, dim=-1)
            transfer_index = torch.multinomial(full_confidence, num_samples=number_transfer_tokens)

        x_ = torch.full_like(x, mask_token_id)
        x_[mask_index] = x0
        row_indices = torch.arange(x.size(0), device=x.device).unsqueeze(1).expand_as(transfer_index)
        x[row_indices, transfer_index] = x_[row_indices, transfer_index]
        return x


__all__ = ["DreamNativeStrategy"]
