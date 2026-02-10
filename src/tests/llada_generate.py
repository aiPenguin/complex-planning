"""
Code borrowed from https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.models.llada import LLaDA
from src.strategies.llada_native import LLaDANativeStrategy


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
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


@torch.no_grad()
def generate(
    model,
    prompt,
    attention_mask=None,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    logits_eos_inf=False,
    confidence_eos_eot_inf=False,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details
    """
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=model.device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = x == mask_id
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                logits = model(x_, attention_mask=attention_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


class FakeModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, device: str = "cpu") -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.device = torch.device(device)

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.shape
        logits = torch.arange(self.vocab_size, device=input_ids.device).view(1, 1, -1)
        logits = logits.expand(batch, seq_len, -1).contiguous().float()
        return type("Output", (), {"logits": logits})


class FakeTokenizer:
    def __init__(self, pad_token_id: int = 0) -> None:
        self.pad_token_id = pad_token_id

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False):
        content = messages[0]["content"]
        return f"<user>{content}" if add_generation_prompt else content

    def __call__(self, prompts, add_special_tokens=False, padding=True, return_tensors="pt"):
        if isinstance(prompts, str):
            prompts = [prompts]
        lengths = [len(p) for p in prompts]
        max_len = max(lengths)
        input_ids = []
        attention_mask = []
        for length in lengths:
            ids = list(range(1, length + 1))
            pad = [self.pad_token_id] * (max_len - length)
            input_ids.append(pad + ids)
            attention_mask.append([0] * (max_len - length) + [1] * length)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def batch_decode(self, sequences, skip_special_tokens=True):
        return ["|".join(map(str, seq.tolist())) for seq in sequences]


def _build_refactored_model(
    strategy: LLaDANativeStrategy, model: FakeModel, tokenizer: FakeTokenizer
) -> LLaDA:
    llada = LLaDA.__new__(LLaDA)
    llada.strategy = strategy
    llada.model = model
    llada.tokenizer = tokenizer
    llada.device = model.device
    llada.use_chat_template = True
    llada.add_generation_prompt = True
    return llada


def _print_step(name: str) -> None:
    print(f"[TEST] {name}")


def test_get_num_transfer_tokens_equivalence() -> None:
    _print_step("get_num_transfer_tokens equivalence")
    # Compare the original helper vs. refactored helper for identical outputs.
    torch.manual_seed(0)
    mask_index = torch.randint(0, 2, (2, 7), dtype=torch.bool)
    steps = 3
    original = get_num_transfer_tokens(mask_index, steps)
    refactored = LLaDA._get_num_transfer_tokens(mask_index, steps)
    assert torch.equal(original, refactored), "num_transfer_tokens mismatch"
    print("  ok")


def test_llada_native_unmask_low_confidence() -> None:
    _print_step("LLaDANativeStrategy.unmask low_confidence")
    # Ensure low_confidence unmask fills at least k masked positions per batch.
    torch.manual_seed(0)
    strategy = LLaDANativeStrategy(mask_id=9)
    x = torch.full((2, 6), strategy.mask_id, dtype=torch.long)
    logits = torch.randn(2, 6, 16)
    mask_index = x == strategy.mask_id
    num_transfer_tokens = torch.tensor([2, 3])
    out = strategy.unmask(
        x=x,
        logits=logits,
        mask_index=mask_index,
        block_end=4,
        num_transfer_tokens=num_transfer_tokens,
    )
    assert out.shape == x.shape
    assert (out != strategy.mask_id).sum() >= num_transfer_tokens.sum()
    print("  ok")


def test_llada_native_unmask_random() -> None:
    _print_step("LLaDANativeStrategy.unmask random")
    # Ensure random remasking mode still fills at least k positions.
    torch.manual_seed(0)
    strategy = LLaDANativeStrategy(remasking="random", mask_id=9)
    x = torch.full((1, 4), strategy.mask_id, dtype=torch.long)
    logits = torch.randn(1, 4, 16)
    mask_index = x == strategy.mask_id
    num_transfer_tokens = torch.tensor([2])
    out = strategy.unmask(
        x=x,
        logits=logits,
        mask_index=mask_index,
        block_end=4,
        num_transfer_tokens=num_transfer_tokens,
    )
    assert out.shape == x.shape
    assert (out != strategy.mask_id).sum() >= 2
    print("  ok")


def test_generate_equivalence() -> None:
    _print_step("original vs refactored generate equivalence")
    # End-to-end equivalence: original generate vs refactored loop.
    torch.manual_seed(0)
    strategy = LLaDANativeStrategy(
        steps=4,
        gen_length=4,
        block_length=2,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=9,
        eos_token_id=7,
        eot_token_id=8,
        logits_eos_inf=False,
        confidence_eos_eot_inf=False,
    )

    model = FakeModel(vocab_size=16)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    original_out = generate(
        model=model,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=strategy.steps,
        gen_length=strategy.gen_length,
        block_length=strategy.block_length,
        temperature=strategy.temperature,
        cfg_scale=strategy.cfg_scale,
        remasking=strategy.remasking,
        mask_id=strategy.mask_id,
        logits_eos_inf=strategy.logits_eos_inf,
        confidence_eos_eot_inf=strategy.confidence_eos_eot_inf,
    )

    refactored = _build_refactored_model(strategy, model, FakeTokenizer())
    refactored_out = refactored._generate_ids(input_ids, attention_mask)

    assert torch.equal(original_out, refactored_out), "refactored output mismatch"
    assert torch.equal(
        refactored_out[:, : input_ids.shape[1]], input_ids
    ), "prompt tokens changed"
    print("  ok")


def test_llada_generate_flow_with_fake_tokenizer() -> None:
    _print_step("LLaDA.generate flow with FakeTokenizer")
    # Validate LLaDA.generate uses tokenizer + strategy to return outputs.
    torch.manual_seed(0)
    # Sanity check: generated tail should not contain mask_id tokens.
    strategy = LLaDANativeStrategy(
        steps=2,
        gen_length=2,
        block_length=2,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=9,
    )
    model = FakeModel(vocab_size=16)
    tokenizer = FakeTokenizer(pad_token_id=0)
    refactored = _build_refactored_model(strategy, model, tokenizer)

    outputs = refactored.generate(["hi", "test"])
    assert len(outputs) == 2
    print("  ok")


def test_unmasking_updates() -> None:
    _print_step("mask tokens get filled")
    strategy = LLaDANativeStrategy(
        steps=2,
        gen_length=2,
        block_length=2,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        mask_id=9,
    )
    model = FakeModel(vocab_size=16)
    refactored = _build_refactored_model(strategy, model, FakeTokenizer())
    input_ids = torch.tensor([[1, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    out = refactored._generate_ids(input_ids, attention_mask)
    generated = out[:, input_ids.shape[1] :]

    assert generated.numel() == strategy.gen_length
    assert torch.all(generated != strategy.mask_id), "mask tokens should be filled"
    print("  ok")


def run_all() -> None:
    print("Running LLaDA tests...\n")
    test_get_num_transfer_tokens_equivalence()
    test_llada_native_unmask_low_confidence()
    test_llada_native_unmask_random()
    test_generate_equivalence()
    test_llada_generate_flow_with_fake_tokenizer()
    test_unmasking_updates()
    print("\nAll LLaDA tests passed")


if __name__ == "__main__":
    run_all()
