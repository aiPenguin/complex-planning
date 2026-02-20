#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.eval_utils import read_jsonl


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clean_sudoku(text: str) -> str:
    cleaned = text.split("<|endoftext|>")[0]
    cleaned = cleaned.split("\n\n")[0]
    return cleaned.replace(" ", "")


def _is_valid_sudoku(input_grid: str, prediction: str) -> bool:
    prediction = prediction[: len(input_grid)]
    try:
        input_array = [
            list(map(int, row)) for row in input_grid.strip().split("\n")
        ]
        grid = [list(map(int, row)) for row in prediction.strip().split("\n")]
        if len(grid) != 4 or any(len(row) != 4 for row in grid):
            return False
    except Exception:
        return False

    non_zero_mask = [
        [val != 0 for val in row] for row in input_array
    ]
    for r in range(4):
        for c in range(4):
            if non_zero_mask[r][c] and input_array[r][c] != grid[r][c]:
                return False

    expected = {1, 2, 3, 4}
    for row in grid:
        if set(row) != expected:
            return False
    for col in range(4):
        if set(grid[row][col] for row in range(4)) != expected:
            return False
    for start_row in (0, 2):
        for start_col in (0, 2):
            subgrid = {
                grid[r][c]
                for r in range(start_row, start_row + 2)
                for c in range(start_col, start_col + 2)
            }
            if subgrid != expected:
                return False
    return True


def _clean_cd(text: str) -> str:
    return text.split("<|end_of_text|>")[0].split("\n")[0]


def _check_eq(left_str: str, right_str: str) -> bool:
    left_matches = re.match(r"(\d+)([+\-*/])(\d+)", left_str)
    if left_matches:
        return eval(left_str) == float(right_str)
    return False


def _cd_is_correct(query: str, pred: str) -> bool:
    subequations = pred.split(",")
    match = True
    query_numbers = Counter(query.split(",")[:-1])
    for subeq in subequations:
        try:
            left, right = subeq.split("=")
            match &= _check_eq(left, right)
            left_side_numbers = re.findall(r"(\d+)(?=[+\\-/*=])", subeq)
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
    return bool(match and (answer == pred_ans) and numbers_match)


def _clean_trip(text: str) -> str:
    return text.split("<|endoftext|>")[0].split("\n\nTASK")[0]


def _parse_trip_response(response: str) -> list[tuple[str, int]]:
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


def _trip_is_correct(item: dict, pred: str) -> bool:
    parsed_plan = _parse_trip_response(pred)
    stays = [x for x in item["cities"].split("**") if x]
    days = [int(x) for x in item["durations"].split("**") if x]
    num_stays = min(len(stays), len(parsed_plan))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    return bool(num_match / len(stays) >= 1.0)


def _accuracy(
    data: list,
    preds: list[str],
    is_correct: Callable[[dict, str], bool],
) -> float:
    if not data:
        return 0.0
    correct = [is_correct(item, pred) for item, pred in zip(data, preds)]
    return sum(correct) / len(correct)


def _any_particle_accuracy(
    data: list,
    particles: list[list[str]] | None,
    is_correct: Callable[[dict, str], bool],
) -> float:
    if not data or particles is None:
        return 0.0
    flags = []
    for item, preds in zip(data, particles):
        ok = False
        for pred in preds:
            if is_correct(item, pred):
                ok = True
                break
        flags.append(ok)
    return sum(flags) / len(flags)


def _resolve_segment_data(label: str, output_dir: Path, data_dir: Path) -> tuple[str, list]:
    analysis_path = output_dir / f"{label}_analysis.json"
    analysis = _load_json(analysis_path) if analysis_path.exists() else {}

    if label.startswith("sudoku_4x4_"):
        n = int(label.split("_")[-1])
        data = read_jsonl(data_dir / f"sudoku_4x4_{n}.jsonl")
        n_few_shots = int(analysis.get("n_few_shots", 0))
        data = data[n_few_shots :]
        if analysis.get("num_items") is not None:
            data = data[: int(analysis["num_items"])]
        return "sudoku", data

    if label.startswith("cd") and label[2:].isdigit():
        variant = int(label[2:])
        data = read_jsonl(data_dir / f"cd{variant}_test.jsonl")
        n_few_shots = int(analysis.get("n_few_shots", 0))
        data = data[n_few_shots :]
        if analysis.get("num_items") is not None:
            data = data[: int(analysis["num_items"])]
        return "cd", data

    if label == "trip":
        with (data_dir / "trip_planning.json").open("r", encoding="utf-8") as f:
            data_dict = json.load(f)
        if analysis.get("num_items") is not None:
            data_dict = dict(list(data_dict.items())[: int(analysis["num_items"])])
        data = list(data_dict.values())
        return "trip", data

    raise ValueError(f"Unknown segment label: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute accuracies for primary/secondary/mode/best/any-particle."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run output directory containing pace_candidates.json.",
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=None,
        help="Optional decoded candidates JSON (default: <output-dir>/pace_candidates.json).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset directory (default: data).",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    candidates_path = args.candidates or (output_dir / "pace_candidates.json")
    if not candidates_path.exists():
        raise FileNotFoundError(f"Missing candidates file: {candidates_path}")

    candidates = _load_json(candidates_path)
    segments = candidates.get("segments") or [{"label": "default", "count": len(candidates["primary"])}]

    primary = candidates.get("primary") or []
    secondary = candidates.get("secondary") or []
    primary_source = candidates.get("primary_source") or []
    secondary_source = candidates.get("secondary_source") or []
    particles = candidates.get("particles")

    offset = 0
    for seg in segments:
        label = seg.get("label") or "default"
        count = int(seg.get("count", 0))
        seg_primary = primary[offset : offset + count]
        seg_secondary = secondary[offset : offset + count] if secondary else None
        seg_primary_source = primary_source[offset : offset + count]
        seg_secondary_source = secondary_source[offset : offset + count]
        seg_particles = (
            particles[offset : offset + count] if particles is not None else None
        )
        offset += count

        task, data = _resolve_segment_data(label, output_dir, args.data_dir)
        if task == "sudoku":
            cleaner = _clean_sudoku
            is_correct = lambda item, pred: _is_valid_sudoku(item["input"], pred)
        elif task == "cd":
            cleaner = _clean_cd
            is_correct = lambda item, pred: _cd_is_correct(item["input"], pred)
        else:
            cleaner = _clean_trip
            is_correct = lambda item, pred: _trip_is_correct(item, pred)

        seg_primary = [cleaner(p) for p in seg_primary]
        seg_secondary = [cleaner(p) for p in seg_secondary] if seg_secondary else None
        if seg_particles is not None:
            seg_particles = [[cleaner(p) for p in preds] for preds in seg_particles]

        mode_preds = []
        best_preds = []
        for i in range(len(seg_primary)):
            src = seg_primary_source[i] if i < len(seg_primary_source) else None
            if src == "mode":
                mode_preds.append(seg_primary[i])
                best_preds.append(seg_secondary[i] if seg_secondary else seg_primary[i])
            else:
                best_preds.append(seg_primary[i])
                mode_preds.append(seg_secondary[i] if seg_secondary else seg_primary[i])

        primary_acc = _accuracy(data, seg_primary, is_correct)
        secondary_acc = _accuracy(data, seg_secondary, is_correct) if seg_secondary else 0.0
        mode_acc = _accuracy(data, mode_preds, is_correct)
        best_acc = _accuracy(data, best_preds, is_correct)
        any_acc = _any_particle_accuracy(data, seg_particles, is_correct)

        print(f"[{label}] primary={primary_acc:.4f} secondary={secondary_acc:.4f} "
              f"mode={mode_acc:.4f} best={best_acc:.4f} any_particle={any_acc:.4f}")


if __name__ == "__main__":
    main()
