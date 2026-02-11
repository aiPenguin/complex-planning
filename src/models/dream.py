"""
Dream model wrapper.
"""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from src.utils.dream_utils import DreamGenerationMixin, DreamModelOutput
from src.utils.pace_utils import (
    is_pace_strategy,
    init_pace_state,
    expand_for_particles,
    finalize_particles,
    use_particle_expansion,
)


class Dream(DreamGenerationMixin):
    """Wrapper that exposes Dream diffusion decoding via `generate()`."""
    def __init__(
        self,
        model_name: str = "Dream-org/Dream-v0-Instruct-7B",
        device: str = "cuda",
        torch_dtype: str | torch.dtype | None = "bfloat16",
        trust_remote_code: bool = True,
        padding_side: str = "left",
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        strategy: object | None = None,
    ) -> None:
        if strategy is None:
            raise ValueError("Dream requires an unmasking strategy instance.")
        self._validate_strategy(strategy)
        self.strategy = strategy

        self.device = torch.device(device)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            torch_dtype=self._resolve_dtype(torch_dtype),
        ).to(self.device)
        self.model.eval()
        self.config = self.model.config
        self.generation_config = self.model.generation_config

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if padding_side:
            self.tokenizer.padding_side = padding_side

        self.use_chat_template = use_chat_template
        self.add_generation_prompt = add_generation_prompt

    @staticmethod
    def _validate_strategy(strategy: object) -> None:
        missing = [
            name
            for name in (
                "max_new_tokens",
                "steps",
                "temperature",
                "top_p",
                "top_k",
                "alg",
                "alg_temp",
                "eps",
            )
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

    def __call__(self, input_ids, attention_mask=None, tok_idx=None):
        return self.model(input_ids, attention_mask=attention_mask, tok_idx=tok_idx)

    def generate(self, prompts: Sequence[str] | str) -> List[str]:
        """Run diffusion generation and decode the completion portion."""
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

        output = self.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=getattr(self.strategy, "max_new_tokens", None)
            or getattr(self.strategy, "gen_length", None),
            steps=self.strategy.steps,
            temperature=self.strategy.temperature,
            top_p=self.strategy.top_p,
            top_k=self.strategy.top_k,
            alg=self.strategy.alg,
            alg_temp=self.strategy.alg_temp,
            eps=self.strategy.eps,
            return_dict_in_generate=True,
            output_history=False,
        )
        output_ids = output.sequences if hasattr(output, "sequences") else output
        decoded = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return decoded

    @torch.no_grad()
    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor | None,
        generation_config,
        generation_tokens_hook_func,
        generation_logits_hook_func,
    ):
        """Dream-style diffusion loop, mirroring HF generation hooks."""
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps

        is_pace = is_pace_strategy(self.strategy)
        use_particles = use_particle_expansion(self.strategy)
        base_batch = input_ids.shape[0]
        prompt_len = input_ids.shape[1]
        if use_particles:
            input_ids, attention_mask = expand_for_particles(
                self.strategy, input_ids, attention_mask
            )

        histories = [] if (return_dict_in_generate and output_history) else None
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if is_pace:
            init_pace_state(
                self.strategy,
                batch_size=base_batch,
                seq_len=max_length,
                prompt_len=prompt_len,
                mask_id=mask_token_id,
                device=x.device,
                total_steps=steps,
            )

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
        x = generation_tokens_hook_func(None, x, None)

        for i in range(steps):
            mask_index = x == mask_token_id
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            logits = generation_logits_hook_func(i, x, logits)

            x = self.strategy.unmask(
                x=x,
                logits=logits,
                mask_index=mask_index,
                t=timesteps[i],
                s=timesteps[i + 1],
                step=i,
                steps=steps,
                mask_token_id=mask_token_id,
            )

            x = generation_tokens_hook_func(i, x, logits)
            if histories is not None:
                histories.append(x.clone())

            if is_pace and self.strategy.should_stop():
                break

        if is_pace and hasattr(self.strategy, "finalize"):
            x = finalize_particles(self.strategy, x)

        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        return x


__all__ = ["Dream"]
