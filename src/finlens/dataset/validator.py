"""
Validate synthetic dataset and split into train/val.

Checks every example against the ExtractionResult schema,
filters bad data, reports quality stats, and splits.

Usage:
    uv run python -m src.finlens.dataset.validator
"""

import json
import random
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from src.finlens.schemas import ExtractionResult

console = Console()

MIN_INPUT_LENGTH = 100  # chars — anything shorter is too simple
VAL_RATIO = 0.15


def validate_example(example: dict) -> tuple[bool, str]:
    """Validate a single training example. Returns (is_valid, error_message)."""
    # Check required keys
    if "input_text" not in example:
        return False, "missing input_text"
    if "expected_output" not in example:
        return False, "missing expected_output"

    # Check input text length
    if len(example["input_text"]) < MIN_INPUT_LENGTH:
        return False, f"input_text too short ({len(example['input_text'])} chars)"

    # Validate output against schema
    try:
        ExtractionResult.model_validate(example["expected_output"])
    except ValidationError as e:
        return False, f"schema error: {e.errors()[0]['msg']}"

    # Check extraction has actual content
    output = example["expected_output"]
    total_items = (
        len(output.get("risk_factors", []))
        + len(output.get("material_events", []))
        + len(output.get("financial_obligations", []))
    )
    if total_items == 0:
        return False, "no extractions (empty risk_factors, events, and obligations)"

    return True, ""


def validate_dataset(
    input_path: str = "data/dataset.jsonl",
    train_path: str = "data/train.jsonl",
    val_path: str = "data/val.jsonl",
) -> None:
    """Validate all examples, report stats, split into train/val."""
    input_file = Path(input_path)
    if not input_file.exists():
        console.print(f"[red]✗ File not found: {input_path}[/]")
        return

    # Load and validate
    valid = []
    invalid = []
    seen_texts = set()

    with open(input_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
            except json.JSONDecodeError:
                invalid.append((line_num, "invalid JSON"))
                continue

            # Duplicate check
            text_hash = hash(example.get("input_text", "")[:200])
            if text_hash in seen_texts:
                invalid.append((line_num, "duplicate"))
                continue
            seen_texts.add(text_hash)

            is_valid, error = validate_example(example)
            if is_valid:
                valid.append(example)
            else:
                invalid.append((line_num, error))

    # Report
    total = len(valid) + len(invalid)
    console.print(f"\n[bold]Dataset Validation Report[/]\n")

    stats = Table(title="Stats")
    stats.add_column("Metric", style="cyan")
    stats.add_column("Value", style="green")
    stats.add_row("Total examples", str(total))
    stats.add_row("Valid", f"{len(valid)} ({len(valid)/total:.0%})")
    stats.add_row("Invalid", f"{len(invalid)} ({len(invalid)/total:.0%})")
    console.print(stats)

    if invalid:
        console.print(f"\n[yellow]Invalid examples:[/]")
        for line_num, error in invalid[:10]:  # show first 10
            console.print(f"  Line {line_num}: {error}")
        if len(invalid) > 10:
            console.print(f"  ... and {len(invalid) - 10} more")

    if len(valid) < 50:
        console.print(f"\n[red]✗ Only {len(valid)} valid examples — need at least 50[/]")
        return

    # Quality stats
    avg_risks = sum(len(e["expected_output"]["risk_factors"]) for e in valid) / len(valid)
    avg_events = sum(len(e["expected_output"]["material_events"]) for e in valid) / len(valid)
    avg_obligations = sum(len(e["expected_output"]["financial_obligations"]) for e in valid) / len(valid)
    avg_text_len = sum(len(e["input_text"]) for e in valid) / len(valid)

    quality = Table(title="Quality")
    quality.add_column("Metric", style="cyan")
    quality.add_column("Value", style="green")
    quality.add_row("Avg input length", f"{avg_text_len:.0f} chars")
    quality.add_row("Avg risk factors", f"{avg_risks:.1f}")
    quality.add_row("Avg material events", f"{avg_events:.1f}")
    quality.add_row("Avg financial obligations", f"{avg_obligations:.1f}")
    console.print()
    console.print(quality)

    # Split
    random.seed(42)
    random.shuffle(valid)
    split_idx = int(len(valid) * (1 - VAL_RATIO))
    train = valid[:split_idx]
    val = valid[split_idx:]

    # Write
    for path, data in [(train_path, train), (val_path, val)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for example in data:
                f.write(json.dumps(example) + "\n")

    console.print(f"\n[bold green]Done![/]")
    console.print(f"  Train: {len(train)} examples → {train_path}")
    console.print(f"  Val:   {len(val)} examples → {val_path}\n")


if __name__ == "__main__":
    validate_dataset()