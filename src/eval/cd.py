"""
Code borrowed and refactored from https://github.com/DreamLM/Dream/tree/main
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

from src.utils.eval_utils import read_jsonl, write_jsonl


class CDEvaluator:
    def __init__(
        self,
        data_dir: str = "data",
        variants: Iterable[int] | int = (3, 4, 5),
        n_few_shots: int = 8,
        prediction_path: str | None = None,
    ) -> None:
        if isinstance(variants, int):
            self.variants = [variants]
        else:
            self.variants = list(variants)
        self.data_dir = data_dir
        self.n_few_shots = n_few_shots
        self.prediction_path = prediction_path

    @staticmethod
    def _check_eq(left_str: str, right_str: str) -> bool:
        left_matches = re.match(r"(\d+)([+\-*/])(\d+)", left_str)
        if left_matches:
            return eval(left_str) == float(right_str)
        return False

    def _metric(self, inputs: List[str], preds: List[str]) -> float:
        cor = 0
        for query, pred in zip(inputs, preds):
            subequations = pred.split(",")
            match = True
            query_numbers = Counter(query.split(",")[:-1])
            for subeq in subequations:
                try:
                    left, right = subeq.split("=")
                    match &= self._check_eq(left, right)
                    left_side_numbers = re.findall(r"(\d+)(?=[+\-/*=])", subeq)
                    query_numbers.subtract(left_side_numbers)
                    query_numbers.update({right: 1})
                except Exception:
                    match = False
                if not match:
                    break

            answer = query.split(",")[-1]
            pred_ans = pred.split("=")[-1]

            query_numbers.subtract({query.split(",")[-1]: 1})
            numbers_match = all(value == 0 for value in query_numbers.values())
            cor += match and (answer == pred_ans) and numbers_match

        return cor / len(preds)

    def _build_template(self, data: list[dict], variant: int) -> str:
        prefix = (
            f"Given {variant + 1} numbers, use +-*/ to operate over the first {variant} numbers to achieve the last number.\n\n"
        )
        shots = "\n\n".join(
            [f"Input: {i['input']}\nOutput: {i['output']}" for i in data[: self.n_few_shots]]
        )
        return f"{prefix}{shots}\n\nInput: {{input}}\nOutput: "

    def _run_variant(self, generator, variant: int) -> None:
        data = read_jsonl(f"{self.data_dir}/cd{variant}_test.jsonl")
        template = self._build_template(data, variant)
        data = data[self.n_few_shots :]

        inputs = [template.format(input=i["input"]) for i in data]
        print("Example input: ", inputs[0])
        generations = generator.generate(inputs)
        generations = [g.split("<|end_of_text|>")[0].split("\n")[0] for g in generations]
        print("Acc: ", self._metric([i["input"] for i in data], generations))
        if self.prediction_path is not None:
            write_jsonl(
                [
                    {"input": i["input"], "gold": i["output"], "prediction": j}
                    for i, j in zip(data, generations)
                ],
                f"{self.prediction_path}_{variant}",
            )

    def evaluate(self, generator) -> None:
        for variant in self.variants:
            self._run_variant(generator, variant)

    def __call__(self, generator) -> None:
        self.evaluate(generator)
