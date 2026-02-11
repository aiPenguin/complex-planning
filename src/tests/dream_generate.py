"""
Dream generation tests derived from the official usage example.
"""
from __future__ import annotations

import torch

from src.models.dream import Dream
from src.strategies.dream_native import DreamNativeStrategy
from src.utils.dream_utils import DreamGenerationConfig, DreamGenerationMixin


class FakeModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 16,
        device: str = "cpu",
        mask_token_id: int = 1,
        pad_token_id: int = 0,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.device = torch.device(device)
        self.config = type(
            "Config",
            (),
            {
                "mask_token_id": mask_token_id,
                "pad_token_id": pad_token_id,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
                "max_position_embeddings": 128,
                "get_text_config": lambda self, decoder=True: self,
                "to_dict": lambda self: {
                    "mask_token_id": mask_token_id,
                    "pad_token_id": pad_token_id,
                    "bos_token_id": bos_token_id,
                    "eos_token_id": eos_token_id,
                    "max_position_embeddings": 128,
                },
            },
        )()
        self.generation_config = DreamGenerationConfig(
            mask_token_id=mask_token_id,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

    def forward(self, input_ids, attention_mask=None, tok_idx=None):
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


class OriginalDream(DreamGenerationMixin):
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.device = model.device
        self.config = model.config
        self.generation_config = model.generation_config

    def __call__(self, input_ids, attention_mask=None, tok_idx=None):
        return self.model(input_ids, attention_mask=attention_mask, tok_idx=tok_idx)


def _build_refactored_model(
    strategy: DreamNativeStrategy, model: FakeModel, tokenizer: FakeTokenizer
) -> Dream:
    dream = Dream.__new__(Dream)
    dream.strategy = strategy
    dream.model = model
    dream.device = model.device
    dream.tokenizer = tokenizer
    dream.use_chat_template = True
    dream.add_generation_prompt = True
    dream.config = model.config
    dream.generation_config = model.generation_config
    return dream


def _print_step(name: str) -> None:
    print(f"[TEST] {name}")


def test_diffusion_generate_equivalence() -> None:
    _print_step("original vs refactored diffusion_generate")
    strategy = DreamNativeStrategy(
        max_new_tokens=4,
        steps=4,
        temperature=0.2,
        top_p=0.95,
        top_k=None,
        alg="entropy",
        alg_temp=0.0,
        eps=1e-3,
    )
    model = FakeModel()
    input_ids = torch.tensor([[4, 5, 6], [7, 8, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    generation_config = DreamGenerationConfig(
        max_new_tokens=strategy.max_new_tokens,
        steps=strategy.steps,
        temperature=strategy.temperature,
        top_p=strategy.top_p,
        top_k=strategy.top_k,
        alg=strategy.alg,
        alg_temp=strategy.alg_temp,
        eps=strategy.eps,
        mask_token_id=model.config.mask_token_id,
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
    )

    original = OriginalDream(model)
    refactored = _build_refactored_model(strategy, model, FakeTokenizer())

    torch.manual_seed(0)
    original_out = original.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_history=False,
    )

    torch.manual_seed(0)
    refactored_out = refactored.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_history=False,
    )

    assert torch.equal(original_out.sequences, refactored_out.sequences), "output mismatch"
    print("  ok")


def test_generate_flow_with_fake_tokenizer() -> None:
    _print_step("Dream.generate flow with FakeTokenizer")
    strategy = DreamNativeStrategy(max_new_tokens=2, steps=2, temperature=0.0, top_p=None, alg="entropy")
    model = FakeModel()
    tokenizer = FakeTokenizer(pad_token_id=model.config.pad_token_id)
    refactored = _build_refactored_model(strategy, model, tokenizer)

    outputs = refactored.generate(["hi", "test"])
    assert len(outputs) == 2
    print("  ok")


def test_unmask_origin_last_step_fills_all() -> None:
    _print_step("DreamNativeStrategy.unmask origin fills on last step")
    strategy = DreamNativeStrategy(temperature=0.0, top_p=None, alg="origin")
    model = FakeModel()
    x = torch.full((1, 4), model.config.mask_token_id, dtype=torch.long)
    logits = model(x).logits
    mask_index = x == model.config.mask_token_id
    out = strategy.unmask(
        x=x,
        logits=logits,
        mask_index=mask_index,
        t=torch.tensor(1.0),
        s=torch.tensor(0.5),
        step=strategy.steps - 1,
        steps=strategy.steps,
        mask_token_id=model.config.mask_token_id,
    )
    assert torch.all(out != model.config.mask_token_id), "mask tokens should be filled"
    print("  ok")


def test_unmask_maskgit_plus_fills_all() -> None:
    _print_step("DreamNativeStrategy.unmask maskgit_plus fills all")
    strategy = DreamNativeStrategy(temperature=0.0, top_p=None, alg="maskgit_plus")
    model = FakeModel()
    x = torch.full((1, 4), model.config.mask_token_id, dtype=torch.long)
    logits = model(x).logits
    mask_index = x == model.config.mask_token_id
    out = strategy.unmask(
        x=x,
        logits=logits,
        mask_index=mask_index,
        t=torch.tensor(1.0),
        s=torch.tensor(0.0),
        step=0,
        steps=strategy.steps,
        mask_token_id=model.config.mask_token_id,
    )
    assert torch.all(out != model.config.mask_token_id), "mask tokens should be filled"
    print("  ok")


def test_unmask_topk_margin_partial() -> None:
    _print_step("DreamNativeStrategy.unmask topk_margin partial transfer")
    strategy = DreamNativeStrategy(temperature=0.0, top_p=None, alg="topk_margin")
    model = FakeModel()
    x = torch.full((1, 4), model.config.mask_token_id, dtype=torch.long)
    logits = model(x).logits
    mask_index = x == model.config.mask_token_id
    out = strategy.unmask(
        x=x,
        logits=logits,
        mask_index=mask_index,
        t=torch.tensor(1.0),
        s=torch.tensor(0.5),
        step=0,
        steps=strategy.steps,
        mask_token_id=model.config.mask_token_id,
    )
    filled = (out != model.config.mask_token_id).sum().item()
    assert filled == 2, f"expected 2 filled tokens, got {filled}"
    print("  ok")


def test_unmask_entropy_uses_confidence() -> None:
    _print_step("DreamNativeStrategy.unmask entropy fills some tokens")
    strategy = DreamNativeStrategy(temperature=0.0, top_p=None, alg="entropy")
    model = FakeModel()
    x = torch.full((1, 3), model.config.mask_token_id, dtype=torch.long)
    logits = model(x).logits
    mask_index = x == model.config.mask_token_id
    out = strategy.unmask(
        x=x,
        logits=logits,
        mask_index=mask_index,
        t=torch.tensor(1.0),
        s=torch.tensor(0.5),
        step=0,
        steps=strategy.steps,
        mask_token_id=model.config.mask_token_id,
    )
    assert (out != model.config.mask_token_id).sum().item() > 0
    print("  ok")


def test_unmask_no_mask_no_change() -> None:
    _print_step("DreamNativeStrategy.unmask no mask short-circuit")
    strategy = DreamNativeStrategy(temperature=0.0, top_p=None, alg="entropy")
    model = FakeModel()
    x = torch.tensor([[4, 5, 6]], dtype=torch.long)
    logits = model(x).logits
    mask_index = x == model.config.mask_token_id
    out = strategy.unmask(
        x=x.clone(),
        logits=logits,
        mask_index=mask_index,
        t=torch.tensor(1.0),
        s=torch.tensor(0.5),
        step=0,
        steps=strategy.steps,
        mask_token_id=model.config.mask_token_id,
    )
    assert torch.equal(out, x), "should return input when no masks"
    print("  ok")


def test_prompt_tokens_preserved() -> None:
    _print_step("Dream diffusion preserves prompt tokens")
    strategy = DreamNativeStrategy(max_new_tokens=3, steps=3, temperature=0.0, top_p=None, alg="entropy")
    model = FakeModel()
    input_ids = torch.tensor([[4, 5], [6, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    refactored = _build_refactored_model(strategy, model, FakeTokenizer())
    output = refactored.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=strategy.max_new_tokens,
        steps=strategy.steps,
        temperature=strategy.temperature,
        top_p=strategy.top_p,
        top_k=strategy.top_k,
        alg=strategy.alg,
        alg_temp=strategy.alg_temp,
        eps=strategy.eps,
        return_dict_in_generate=True,
        output_history=False,
    )
    assert torch.equal(output.sequences[:, : input_ids.shape[1]], input_ids)
    print("  ok")


def test_generated_length_matches_max_new_tokens() -> None:
    _print_step("Dream diffusion respects max_new_tokens")
    strategy = DreamNativeStrategy(max_new_tokens=4, steps=2, temperature=0.0, top_p=None, alg="entropy")
    model = FakeModel()
    input_ids = torch.tensor([[4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    refactored = _build_refactored_model(strategy, model, FakeTokenizer())
    output = refactored.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=strategy.max_new_tokens,
        steps=strategy.steps,
        temperature=strategy.temperature,
        top_p=strategy.top_p,
        top_k=strategy.top_k,
        alg=strategy.alg,
        alg_temp=strategy.alg_temp,
        eps=strategy.eps,
        return_dict_in_generate=True,
        output_history=False,
    )
    expected = input_ids.shape[1] + strategy.max_new_tokens
    assert output.sequences.shape[1] == expected, "unexpected output length"
    print("  ok")


def run_all() -> None:
    print("Running Dream tests...\n")
    test_diffusion_generate_equivalence()
    test_generate_flow_with_fake_tokenizer()
    test_unmask_origin_last_step_fills_all()
    test_unmask_maskgit_plus_fills_all()
    test_unmask_topk_margin_partial()
    test_unmask_entropy_uses_confidence()
    test_unmask_no_mask_no_change()
    test_prompt_tokens_preserved()
    test_generated_length_matches_max_new_tokens()
    print("\nAll Dream tests passed")


if __name__ == "__main__":
    run_all()
