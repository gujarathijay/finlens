"""
Evaluation pipeline for FinLens extraction.

Scores model predictions against ground truth on:
- JSON parse rate
- Schema compliance
- Extraction accuracy (risk factors, events, obligations)
- Field accuracy (severity, category, impact)

Usage:
    # Evaluate from predictions file:
    uv run python -m src.finlens.evaluation.eval --predictions data/predictions.jsonl

    # Evaluate from val set (ground truth vs ground truth — baseline test):
    uv run python -m src.finlens.evaluation.eval --val-baseline data/val.jsonl
"""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from src.finlens.schemas import ExtractionResult

console = Console()


def parse_json_safe(text: str) -> dict | None:
    """Try to parse JSON from model output. Handles markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def validate_schema(data: dict) -> tuple[bool, str]:
    """Validate against ExtractionResult schema."""
    try:
        ExtractionResult.model_validate(data)
        return True, ""
    except ValidationError as e:
        return False, e.errors()[0]["msg"]


def score_extraction(predicted: dict, expected: dict) -> dict:
    """Score a single prediction against ground truth."""
    scores = {}

    # Count matching risk factors (by category)
    pred_risk_cats = {r.get("category") for r in predicted.get("risk_factors", [])}
    exp_risk_cats = {r.get("category") for r in expected.get("risk_factors", [])}
    if exp_risk_cats:
        scores["risk_category_recall"] = len(pred_risk_cats & exp_risk_cats) / len(exp_risk_cats)
    else:
        scores["risk_category_recall"] = 1.0

    # Count matching severity levels
    pred_severities = [r.get("severity") for r in predicted.get("risk_factors", [])]
    exp_severities = [r.get("severity") for r in expected.get("risk_factors", [])]
    if exp_severities:
        matches = sum(1 for p, e in zip(pred_severities, exp_severities) if p == e)
        scores["severity_accuracy"] = matches / len(exp_severities)
    else:
        scores["severity_accuracy"] = 1.0

    # Count matching events (by impact)
    pred_impacts = [e.get("impact") for e in predicted.get("material_events", [])]
    exp_impacts = [e.get("impact") for e in expected.get("material_events", [])]
    if exp_impacts:
        matches = sum(1 for p, e in zip(pred_impacts, exp_impacts) if p == e)
        scores["event_impact_accuracy"] = matches / len(exp_impacts)
    else:
        scores["event_impact_accuracy"] = 1.0

    # Count extraction completeness
    pred_total = (
        len(predicted.get("risk_factors", []))
        + len(predicted.get("material_events", []))
        + len(predicted.get("financial_obligations", []))
    )
    exp_total = (
        len(expected.get("risk_factors", []))
        + len(expected.get("material_events", []))
        + len(expected.get("financial_obligations", []))
    )
    if exp_total > 0:
        scores["extraction_completeness"] = min(pred_total / exp_total, 1.0)
    else:
        scores["extraction_completeness"] = 1.0

    return scores


def evaluate_predictions(predictions_path: str) -> None:
    """Evaluate a predictions file. Each line: {"input_text", "expected_output", "predicted_output"}."""
    path = Path(predictions_path)
    if not path.exists():
        console.print(f"[red]✗ Not found: {predictions_path}[/]")
        return

    total = 0
    json_pass = 0
    schema_pass = 0
    all_scores = []

    with open(path) as f:
        for line in f:
            total += 1
            example = json.loads(line.strip())
            predicted_text = example.get("predicted_output", "")
            expected = example.get("expected_output", {})

            # JSON parse check
            parsed = parse_json_safe(predicted_text) if isinstance(predicted_text, str) else predicted_text
            if parsed is None:
                continue
            json_pass += 1

            # Schema check
            valid, _ = validate_schema(parsed)
            if valid:
                schema_pass += 1

            # Extraction accuracy
            scores = score_extraction(parsed, expected)
            all_scores.append(scores)

    _print_report(total, json_pass, schema_pass, all_scores)


def evaluate_baseline(val_path: str) -> None:
    """Baseline eval — ground truth vs ground truth. Should score 100%."""
    path = Path(val_path)
    if not path.exists():
        console.print(f"[red]✗ Not found: {val_path}[/]")
        return

    total = 0
    json_pass = 0
    schema_pass = 0
    all_scores = []

    with open(path) as f:
        for line in f:
            total += 1
            example = json.loads(line.strip())
            expected = example["expected_output"]

            json_pass += 1

            valid, _ = validate_schema(expected)
            if valid:
                schema_pass += 1

            scores = score_extraction(expected, expected)
            all_scores.append(scores)

    console.print("\n[bold yellow]Baseline eval (ground truth vs ground truth):[/]")
    _print_report(total, json_pass, schema_pass, all_scores)


def _print_report(total: int, json_pass: int, schema_pass: int, all_scores: list[dict]) -> None:
    """Print evaluation report."""
    console.print(f"\n[bold]Evaluation Report[/]\n")

    # Pass rates
    rates = Table(title="Pass Rates")
    rates.add_column("Metric", style="cyan")
    rates.add_column("Score", style="green")
    rates.add_row("JSON parse rate", f"{json_pass}/{total} ({100*json_pass/total:.0f}%)")
    rates.add_row("Schema compliance", f"{schema_pass}/{total} ({100*schema_pass/total:.0f}%)")
    console.print(rates)

    if not all_scores:
        console.print("[yellow]No valid predictions to score[/]")
        return

    # Accuracy scores
    accuracy = Table(title="Extraction Accuracy")
    accuracy.add_column("Metric", style="cyan")
    accuracy.add_column("Score", style="green")

    metrics = ["risk_category_recall", "severity_accuracy", "event_impact_accuracy", "extraction_completeness"]
    for metric in metrics:
        values = [s[metric] for s in all_scores if metric in s]
        if values:
            avg = sum(values) / len(values)
            accuracy.add_row(metric, f"{100*avg:.1f}%")

    console.print()
    console.print(accuracy)
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FinLens predictions")
    parser.add_argument("--predictions", type=str, help="Path to predictions JSONL")
    parser.add_argument("--val-baseline", type=str, help="Run baseline eval on val set")
    args = parser.parse_args()

    if args.val_baseline:
        evaluate_baseline(args.val_baseline)
    elif args.predictions:
        evaluate_predictions(args.predictions)
    else:
        console.print("[yellow]Provide --predictions or --val-baseline[/]")