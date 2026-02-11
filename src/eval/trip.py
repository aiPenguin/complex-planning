"""
Code borrowed and refactored from https://github.com/DreamLM/Dream/tree/main
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.utils.eval_utils import write_jsonl, save_run_artifacts


class TripEvaluator:
    """Evaluates trip-planning outputs via exact-match itinerary checks."""
    def __init__(
        self,
        data_dir: str = "data",
        max_items: int | None = None,
        prediction_path: str | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.max_items = max_items
        self.prediction_path = prediction_path

    @staticmethod
    def _parse_response(response: str) -> list[tuple[str, int]]:
        """Parse model output into (city, days) tuples; returns [] on failure."""
        pattern_visit = r"\d+-\d+"
        pattern_flight = r".*Day (\d+).*from (\w+) to (\w+)"
        pattern_days = r"European cities for (\d+) days"

        days, flights, flight_days = [], [], []
        total_days = None
        for piece in response.split("\n"):
            days_match = re.findall(pattern_days, piece)
            if days_match:
                total_days = int(days_match[0])

            visit_match = re.findall(pattern_visit, piece)
            if visit_match:
                days.append(visit_match[0])
                end_day = int(visit_match[0].split("-")[1])
                if end_day == total_days:
                    break
            flight_match = re.findall(pattern_flight, piece)
            if flight_match:
                flights.append(flight_match[0])

        visit_cities, parsed_plan = [], []
        for flight_day, begin_city, end_city in flights:
            flight_days.append(int(flight_day))
            if not visit_cities:
                visit_cities.append(begin_city)
                visit_cities.append(end_city)
            else:
                visit_cities.append(end_city)

        if not days or not flights or not visit_cities:
            return []
        last_day = int(days[-1].split("-")[1])
        flight_days = [1] + flight_days + [last_day]
        for i, visit_city in enumerate(visit_cities):
            city_stay = flight_days[i + 1] - flight_days[i] + 1
            parsed_plan.append((visit_city, city_stay))

        return parsed_plan

    @staticmethod
    def _compute_example_score(cities: str, durations: str, parsed_plan: list[Any]) -> float:
        """Compute strict exact-match score for a single example."""
        stays = [x for x in cities.split("**") if x]
        days = [int(x) for x in durations.split("**") if x]
        num_stays = min(len(stays), len(parsed_plan))
        num_match = 0
        for i in range(num_stays):
            if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
                num_match += 1
            else:
                break
        return 0.0 if num_match / len(stays) < 1.0 else 1.0

    def _compute_score(
        self, cities: list[str], durations: list[str], responses: list[str]
    ) -> float:
        parsed_plans = [self._parse_response(response) for response in responses]
        hard_scores = [
            self._compute_example_score(city, duration, parsed_plan)
            for city, duration, parsed_plan in zip(cities, durations, parsed_plans)
        ]
        print([i for i, j in enumerate(hard_scores) if j == 1.0])
        hard_acc = sum(hard_scores) / len(hard_scores)
        return hard_acc

    def _metric(self, data: dict, preds: list[str]) -> float:
        """Aggregate accuracy and print summary statistics."""
        cities, durations, responses = [], [], []
        sample_count = 0
        for item, pred in zip(data.values(), preds):
            cities.append(item["cities"])
            durations.append(item["durations"])
            responses.append(pred)
            sample_count += 1

        hard_acc = self._compute_score(cities, durations, responses)
        print(f"EM Accuracy of {sample_count} samples: {hard_acc}")
        return hard_acc

    def evaluate(self, generator) -> None:
        """Run generation and evaluate the resulting trip plans."""
        with open(f"{self.data_dir}/trip_planning.json") as f:
            data = json.load(f)
        if self.max_items is not None:
            data = dict(list(data.items())[: self.max_items])

        inputs = []
        for item in data.values():
            splits = item["prompt_5shot"].split("TASK:")
            inputs.append("TASK:".join(splits[:3] + [splits[-1]]))

        generations = generator.generate(inputs)
        generations = [g.split("<|endoftext|>")[0].split("\n\nTASK")[0] for g in generations]
        hard_acc = self._metric(data, generations)

        save_run_artifacts(
            output_dir=getattr(self, "output_dir", None),
            name="trip",
            sample_prompt=inputs[0],
            predictions=generations,
            analysis={
                "accuracy": hard_acc,
                "num_items": len(inputs),
            },
        )
        if self.prediction_path is not None:
            write_jsonl(
                [
                    {"input": item["prompt_0shot"], "gold": item["golden_plan"], "prediction": gen}
                    for item, gen in zip(data.values(), generations)
                ],
                self.prediction_path,
            )

    def __call__(self, generator) -> None:
        self.evaluate(generator)
