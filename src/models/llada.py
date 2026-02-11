"""
LLaDA model wrapper.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from transformers import AutoModel, AutoTokenizer

from src.strategies.base import UnmaskingStrategy
from src.utils.pace_utils import (
    is_pace_strategy,
    init_pace_state,
    expand_for_particles,
    finalize_particles,
    use_particle_expansion,
)


class LLaDA:
    """Wrapper that exposes LLaDA diffusion decoding via `generate()`."""
    def __init__(
        self,
        model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: str = "cuda",
        torch_dtype: str | torch.dtype | None = "bfloat16",
        trust_remote_code: bool = True,
        padding_side: str = "left",
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        strategy: UnmaskingStrategy | None = None,
    ) -> None:
        if strategy is None:
            raise ValueError("LLaDA requires an unmasking strategy instance.")
        self._validate_strategy(strategy)
        self.strategy = strategy
        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=self._resolve_dtype(torch_dtype),
        ).to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if padding_side:
            self.tokenizer.padding_side = padding_side

        if self.tokenizer.pad_token_id == self.strategy.mask_id:
            raise ValueError("Tokenizer pad token id must not match mask_id.")

        self.use_chat_template = use_chat_template
        self.add_generation_prompt = add_generation_prompt

    @staticmethod
    def _validate_strategy(strategy: UnmaskingStrategy) -> None:
        missing = [
            name
            for name in ("steps", "gen_length", "block_length", "cfg_scale", "mask_id")
            if not hasattr(strategy, name)
        ]
        if missing:
            raise ValueError(f"Strategy missing required attributes: {', '.join(missing)}")
        if not hasattr(strategy, "unmask"):
            raise ValueError("Strategy must implement unmask().")

    def _resolve_dtype(self, dtype: str | torch.dtype | None) -> torch.dtype | None:
        if dtype is None or isinstance(dtype, torch.dtype):
            return dtype
        dtype_key = str(dtype).lower()
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype_key not in mapping:
            raise ValueError(f"Unsupported torch_dtype: {dtype}")
        return mapping[dtype_key]

    def _format_prompts(self, prompts: Sequence[str]) -> List[str]:
        """Apply the tokenizer chat template when enabled."""
        if not self.use_chat_template:
            return list(prompts)
        messages = [{"role": "user", "content": prompt} for prompt in prompts]
        return [
            self.tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=self.add_generation_prompt,
                tokenize=False,
            )
            for message in messages
        ]

    @staticmethod
    def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        """Evenly split masked tokens across diffusion steps."""
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = (
            torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
            + base
        )
        for i in range(mask_num.size(0)):
            num_transfer_tokens[i, : remainder[i]] += 1
        return num_transfer_tokens

    def generate(self, prompts: Sequence[str] | str) -> List[str]:
        if isinstance(prompts, str):
            prompt_list = [prompts]
        else:
            prompt_list = list(prompts)

        if not prompt_list:
            return []

        formatted = self._format_prompts(prompt_list)
        encoded = self.tokenizer(
            formatted,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        output_ids = self._generate_ids(input_ids, attention_mask)
        decoded = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return decoded

    @torch.no_grad()
    def _generate_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the block-wise diffusion loop and return full token ids."""
        prompt = input_ids
        device = prompt.device
        steps = self.strategy.steps
        gen_length = getattr(self.strategy, "max_new_tokens", None) or self.strategy.gen_length
        block_length = getattr(self.strategy, "block_length", gen_length)
        mask_id = self.strategy.mask_id
        cfg_scale = self.strategy.cfg_scale

        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
            device=device,
        )
        x[:, : prompt.shape[1]] = prompt.clone()

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (prompt.shape[0], gen_length),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=-1,
            )

        prompt_index = x != mask_id

        is_pace = is_pace_strategy(self.strategy)
        use_particles = use_particle_expansion(self.strategy)
        if is_pace:
            init_pace_state(
                self.strategy,
                batch_size=prompt.shape[0],
                seq_len=x.shape[1],
                prompt_len=prompt.shape[1],
                mask_id=mask_id,
                device=device,
                total_steps=steps,
            )
        if use_particles:
            x, attention_mask = expand_for_particles(self.strategy, x, attention_mask)
            prompt_index = prompt_index.repeat_interleave(
                self.strategy.num_particles, dim=0
            )

        if gen_length % block_length != 0:
            raise ValueError("gen_length must be divisible by block_length.")
        num_blocks = gen_length // block_length

        if steps % num_blocks != 0:
            raise ValueError("steps must be divisible by the number of blocks.")
        steps_per_block = steps // num_blocks

        early_stop = False
        for num_block in range(num_blocks):
            start = prompt.shape[1] + num_block * block_length
            end = prompt.shape[1] + (num_block + 1) * block_length
            block_mask_index = x[:, start:end] == mask_id
            num_transfer_tokens = self._get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                if cfg_scale > 0.0:
                    # Classifier-free guidance via conditional/unconditional logits.
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    attention_mask_ = (
                        torch.cat([attention_mask, attention_mask], dim=0)
                        if attention_mask is not None
                        else None
                    )
                    logits = self.model(x_, attention_mask=attention_mask_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.model(x, attention_mask=attention_mask).logits

                if is_pace:
                    x = self.strategy.unmask(
                        x=x,
                        logits=logits,
                        mask_index=mask_index,
                        block_end=end,
                        num_transfer_tokens=num_transfer_tokens[:, i],
                        step=i,
                        steps=steps_per_block,
                    )
                    if self.strategy.should_stop():
                        early_stop = True
                        break
                else:
                    x = self.strategy.unmask(
                        x=x,
                        logits=logits,
                        mask_index=mask_index,
                        block_end=end,
                        num_transfer_tokens=num_transfer_tokens[:, i],
                    )
            if early_stop:
                break

        if is_pace and hasattr(self.strategy, "finalize"):
            return finalize_particles(self.strategy, x)
        return x

    def __call__(self, prompts: Sequence[str] | str) -> List[str]:
        return self.generate(prompts)


__all__ = ["LLaDA"]
