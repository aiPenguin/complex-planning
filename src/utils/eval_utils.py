import json
from pathlib import Path


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    with open(file_path, 'w') as file:
        for item in data:
            file.write(json.dumps(item) + '\n')

def write_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=2)


def save_run_artifacts(
    *,
    output_dir: str | Path | None,
    name: str,
    sample_prompt: str,
    predictions: list[str],
    analysis: dict,
) -> None:
    if output_dir is None:
        return
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        {
            "sample_prompt": sample_prompt,
            "predictions": predictions,
        },
        out_dir / f"{name}_results.json",
    )
    write_json(analysis, out_dir / f"{name}_analysis.json")
