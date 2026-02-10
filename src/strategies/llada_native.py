"""
Native LLaDA unmasking strategy extracted from the original generate loop.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from src.strategies.base import UnmaskingStrategy


@dataclass
class LLaDANativeStrategy(UnmaskingStrategy):
    steps: int = 128
    gen_length: int = 128
    block_length: int = 128
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336
    eos_token_id: int = 126081
    eot_token_id: int = 126348
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False

    def _add_gumbel_noise(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** self.temperature
        return logits.exp() / gumbel_noise

    def unmask(
        self,
        *,
        x: torch.Tensor,
        logits: torch.Tensor,
        mask_index: torch.Tensor,
        block_end: int,
        num_transfer_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if self.logits_eos_inf:
            logits[:, :, self.eos_token_id] = -torch.inf

        logits_with_noise = self._add_gumbel_noise(logits)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if self.confidence_eos_eot_inf:
            logits_with_noise[:, :, self.eos_token_id] = -torch.inf
            logits_with_noise[:, :, self.eot_token_id] = -torch.inf

        if self.remasking == "low_confidence":
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )
        elif self.remasking == "random":
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x.device)
        else:
            raise NotImplementedError(self.remasking)

        x0_p[:, block_end:] = -np.inf

        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -np.inf)

        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x.device)
        for j in range(confidence.shape[0]):
            k = int(num_transfer_tokens[j].item())
            if k <= 0:
                continue
            _, select_index = torch.topk(confidence[j], k=k)
            transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]

        return x
