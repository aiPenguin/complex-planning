"""
DTreeRPO model wrapper for the Hydra pipeline.
"""
from __future__ import annotations

from typing import List, Sequence
import contextlib
import importlib
import sys
import types

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def _ensure_torch_distributed_tensor_compat() -> None:
    """Ensure `torch.distributed.tensor` attribute exists for PEFT compatibility."""
    dist = getattr(torch, "distributed", None)
    if dist is None or hasattr(dist, "tensor"):
        return

    # Preferred: real module available in this torch build.
    try:
        tensor_mod = importlib.import_module("torch.distributed.tensor")
        dist.tensor = tensor_mod
        return
    except Exception:
        pass

    # Older/newer internal path in some builds.
    try:
        tensor_mod = importlib.import_module("torch.distributed._tensor")
        dist.tensor = tensor_mod
        sys.modules.setdefault("torch.distributed.tensor", tensor_mod)
        return
    except Exception:
        pass

    # Last resort: stub to satisfy attribute access/import checks.
    tensor_stub = types.ModuleType("torch.distributed.tensor")
    dist.tensor = tensor_stub
    sys.modules.setdefault("torch.distributed.tensor", tensor_stub)


class DTreeRPO:
    """Wrapper that exposes DTreeRPO diffusion decoding via `generate()`."""
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str | torch.dtype | None = "bfloat16",
        trust_remote_code: bool = True,
        padding_side: str = "left",
        use_chat_template: bool = True,
        add_generation_prompt: bool = True,
        adapter_path: str | None = None,
        strategy: object | None = None,
    ) -> None:
        if strategy is None:
            raise ValueError("DTreeRPO requires an unmasking strategy instance.")
        self._validate_strategy(strategy)
        self.strategy = strategy

        _ensure_torch_distributed_tensor_compat()
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if padding_side:
            self.tokenizer.padding_side = padding_side

        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise ImportError(
                    "peft is required to load adapter_path. Install peft or omit adapter_path."
                ) from exc
            self.model = PeftModel.from_pretrained(self.model, adapter_path).to(self.device)
            self.model.eval()

        self.use_chat_template = use_chat_template
        self.add_generation_prompt = add_generation_prompt

    @staticmethod
    def _validate_strategy(strategy: object) -> None:
        required = ("mask_id", "cfg_scale", "temperature", "remasking")
        missing = [name for name in required if not hasattr(strategy, name)]
        if missing:
            raise ValueError(f"Strategy missing required attributes: {', '.join(missing)}")

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

    def _resolve_gen_length(self) -> int:
        max_new = getattr(self.strategy, "max_new_tokens", None)
        if max_new is not None:
            return int(max_new)
        gen_length = getattr(self.strategy, "gen_length", None)
        if gen_length is not None:
            return int(gen_length)
        raise ValueError("Strategy must define max_new_tokens or gen_length.")

    def _resolve_block_length(self, gen_length: int) -> int:
        block_length = getattr(self.strategy, "block_length", None)
        if block_length is None:
            return gen_length
        return int(block_length)

    def _resolve_steps(self, gen_length: int) -> int:
        steps = getattr(self.strategy, "diffusion_steps", None)
        if steps is None:
            steps = getattr(self.strategy, "steps", None)
        if steps is None:
            steps = max(1, gen_length // 2)
        return int(steps)

    @staticmethod
    def _add_gumbel_noise(
        logits: torch.Tensor,
        temperature: float,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if temperature <= 0.0:
            return logits
        logits = logits.to(dtype)
        noise = torch.rand_like(logits, dtype=dtype)
        gumbel_noise = (-torch.log(torch.clamp(noise, min=1e-9))) ** temperature
        return logits.exp() / gumbel_noise

    @staticmethod
    def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
        if steps <= 0:
            bsz = mask_index.size(0)
            return torch.zeros(bsz, 0, device=mask_index.device, dtype=torch.int64)
        mask_num = mask_index.sum(dim=1, keepdim=True)
        base = mask_num // steps
        remainder = mask_num % steps
        num_transfer_tokens = base.expand(-1, steps).clone()
        if (remainder > 0).any():
            idx = torch.arange(steps, device=mask_index.device)
            front_mask = idx.unsqueeze(0) < remainder
            num_transfer_tokens[front_mask] += 1
        return num_transfer_tokens.to(torch.int64)

    def _autocast_context(self):
        if self.device.type == "cuda":
            return torch.cuda.amp.autocast(enabled=True)
        return contextlib.nullcontext()

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

        output_ids = self._generate_ids(input_ids, attention_mask=attention_mask)
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
        gen_length = self._resolve_gen_length()
        block_length = self._resolve_block_length(gen_length)
        total_steps = self._resolve_steps(gen_length)

        if gen_length % block_length != 0:
            raise ValueError("gen_length must be divisible by block_length.")

        batch_size, prompt_length = input_ids.shape
        num_blocks = gen_length // block_length
        steps_per_block = max(1, total_steps // num_blocks)

        mask_id = int(getattr(self.strategy, "mask_id"))
        cfg_scale = float(getattr(self.strategy, "cfg_scale", 0.0))
        temperature = float(getattr(self.strategy, "temperature", 0.0))
        remasking = getattr(self.strategy, "remasking", "low_confidence")
        dtype = self.model.dtype

        x = torch.full(
            (batch_size, prompt_length + gen_length),
            mask_id,
            dtype=torch.long,
            device=self.device,
        )
        x[:, :prompt_length] = input_ids.clone()

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, gen_length),
                        dtype=attention_mask.dtype,
                        device=self.device,
                    ),
                ],
                dim=-1,
            )

        prompt_index = x != mask_id

        for block_idx in range(num_blocks):
            start_idx = prompt_length + block_idx * block_length
            end_idx = prompt_length + (block_idx + 1) * block_length
            block_mask_index_now = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = self._get_num_transfer_tokens(
                block_mask_index_now, steps_per_block
            )

            for step_i in range(steps_per_block):
                mask_index_full = x == mask_id
                with self._autocast_context():
                    if cfg_scale > 0.0:
                        un_x = x.clone()
                        un_x[prompt_index] = mask_id
                        x_ = torch.cat([x, un_x], dim=0)
                        if attention_mask is not None:
                            attention_mask_ = torch.cat(
                                [attention_mask, attention_mask], dim=0
                            )
                        else:
                            attention_mask_ = None
                        logits = self.model(x_, attention_mask=attention_mask_).logits
                        logits, un_logits = torch.chunk(logits, 2, dim=0)
                        logits = un_logits + (cfg_scale + 1.0) * (logits - un_logits)
                    else:
                        logits = self.model(x, attention_mask=attention_mask).logits

                logits_with_noise = self._add_gumbel_noise(logits, temperature, dtype)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == "low_confidence":
                    p = F.softmax(logits.to(dtype), dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == "random":
                    x0_p = torch.rand_like(x0, dtype=torch.float32)
                else:
                    raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

                x0_p[:, :start_idx] = -float("inf")
                x0_p[:, end_idx:] = -float("inf")

                x0 = torch.where(mask_index_full, x0, x)
                confidence = torch.where(mask_index_full, x0_p, -torch.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=self.device)
                per_step_budget = num_transfer_tokens[:, step_i]

                for j in range(batch_size):
                    k_plan = int(per_step_budget[j].item())
                    if k_plan <= 0:
                        continue
                    block_mask_index_step = x[j, start_idx:end_idx] == mask_id
                    k_avail = int(block_mask_index_step.sum().item())
                    if k_avail <= 0:
                        continue
                    k = min(k_plan, k_avail)
                    block_conf = confidence[j, start_idx:end_idx]
                    _, candidate_indices_in_block = torch.topk(block_conf, k=k)
                    select_indices_global = candidate_indices_in_block + start_idx
                    transfer_index[j, select_indices_global] = True

                x[transfer_index] = x0[transfer_index]

        return x

    def __call__(self, prompts: Sequence[str] | str) -> List[str]:
        return self.generate(prompts)


__all__ = ["DTreeRPO"]
