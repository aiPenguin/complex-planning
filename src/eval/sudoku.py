"""
Code borrowed and refactored from https://github.com/DreamLM/Dream/tree/main
"""
from __future__ import annotations

from typing import Iterable, List

import numpy as np

from src.utils.eval_utils import read_jsonl, write_jsonl


class SudokuEvaluator:
    """Evaluates 4x4 Sudoku completions via exact validity checks."""
    def __init__(
        self,
        data_dir: str = "data",
        n_values: Iterable[int] | int = (10,),
        n_few_shots: int = 8,
        max_items: int | None = None,
        prediction_path: str | None = None,
    ) -> None:
        if isinstance(n_values, int):
            self.n_values = [n_values]
        else:
            self.n_values = list(n_values)
        self.data_dir = data_dir
        self.n_few_shots = n_few_shots
        self.max_items = max_items
        self.prediction_path = prediction_path

    @staticmethod
    def is_valid_sudoku(input_grid: str, prediction: str) -> bool:
        """Validate that the prediction solves the 4x4 Sudoku with given clues."""
        prediction = prediction[: len(input_grid)]
        input_array = np.array([list(map(int, row)) for row in input_grid.strip().split("\n")])
        try:
            grid = np.array([list(map(int, row)) for row in prediction.strip().split("\n")])
            if grid.shape != (4, 4):
                return False
        except Exception:
            return False

        non_zero_mask = input_array != 0
        if not np.all(input_array[non_zero_mask] == grid[non_zero_mask]):
            return False

        expected_set = {1, 2, 3, 4}
        for row in grid:
            if set(row) != expected_set:
                return False

        for col in range(4):
            if set(grid[row][col] for row in range(4)) != expected_set:
                return False

        for start_row in (0, 2):
            for start_col in (0, 2):
                subgrid = {
                    grid[r][c]
                    for r in range(start_row, start_row + 2)
                    for c in range(start_col, start_col + 2)
                }
                if subgrid != expected_set:
                    return False

        return True

    def _build_prompt(self, examples: List[dict], item: dict) -> str:
        """Assemble a few-shot prompt from examples and a target item."""
        template = (
            "Fill the positions where the values are 0 in a 4x4 grid with digits 1-4 so "
            "that each column, each row, and each of the four 2x2 subgrids that compose "
            "the grid contains all of the digits from 1 to 4.\n\n"
        )
        template += "\n\n".join(
            [f"Input:\n{i['input']}\nOutput:\n{i['output']}" for i in examples]
        )
        template += "\n\nInput:\n{input}\nOutput:\n "
        return template.format(input=item["input"])

    def _clean_generation(self, text: str) -> str:
        """Strip special tokens and whitespace from model output."""
        cleaned = text.split("<|endoftext|>")[0]
        cleaned = cleaned.split("\n\n")[0]
        return cleaned.replace(" ", "")

    def evaluate(self, generator) -> None:
        """Run generation and print accuracy for each configured dataset split."""
        for n in self.n_values:
            data = read_jsonl(f"{self.data_dir}/sudoku_4x4_{n}.jsonl")
            examples = data[: self.n_few_shots]
            data = data[self.n_few_shots :]
            if self.max_items is not None:
                data = data[: self.max_items]

            inputs = [self._build_prompt(examples, item) for item in data]

            generations = generator.generate(inputs)
            generations = [self._clean_generation(g) for g in generations]

            acc = (
                sum(self.is_valid_sudoku(item["input"], gen) for item, gen in zip(data, generations))
                / len(data)
            )
            print(f"[Sudoku] Accuracy: {acc:.4f}")

            if self.prediction_path is not None:
                write_jsonl(
                    [
                        {
                            "input": item["input"],
                            "gold": item["output"],
                            "prediction": gen,
                        }
                        for item, gen in zip(data, generations)
                    ],
                    f"{self.prediction_path}_{n}",
                )

    def __call__(self, generator) -> None:
        self.evaluate(generator)


if __name__ == "__main__":
    evaluator = SudokuEvaluator()
    print(evaluator.is_valid_sudoku("1000\n0002\n4003\n0000", "1234\n3412\n4123\n2341"))
    print(evaluator.is_valid_sudoku("0020\n0034\n0400\n1000", "2413\n3142\n4321\n1234"))
