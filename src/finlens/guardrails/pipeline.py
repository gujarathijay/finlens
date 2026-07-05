"""
Guardrails pipeline — runs all checks in sequence.

Usage:
    from src.finlens.guardrails.pipeline import run_guardrails

    result = run_guardrails(input_text="...", output_text='{"company_name": ...}')
    if result.passed:
        use(result.parsed_output)
    else:
        flag_for_review(result.failures)
"""

import json
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table

from src.finlens.guardrails.checks import (
    check_completeness,
    check_hallucination,
    check_input_length,
    check_json_parse,
    check_pii,
    check_schema,
)

console = Console()


@dataclass
class GuardrailResult:
    """Result of running all guardrail checks."""

    passed: bool
    parsed_output: dict | None
    checks: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def print_report(self) -> None:
        table = Table(title="Guardrail Results")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Detail", style="white")

        for check in self.checks:
            status = "✓" if check["passed"] else "✗"
            style = "green" if check["passed"] else "red"
            table.add_row(check["name"], f"[{style}]{status}[/]", check["detail"])

        console.print(table)

        if self.passed:
            console.print("\n[bold green]All checks passed[/]")
        else:
            console.print(f"\n[bold red]Failed: {', '.join(self.failures)}[/]")


def run_guardrails(input_text: str, output_text: str) -> GuardrailResult:
    """Run all guardrail checks on a model output."""
    result = GuardrailResult(passed=True, parsed_output=None)

    # Check 1: Input length
    passed, detail = check_input_length(input_text)
    result.checks.append({"name": "Input Length", "passed": passed, "detail": detail})
    if not passed:
        result.passed = False
        result.failures.append("input_length")
        return result  # stop early — bad input

    # Check 2: PII in output
    passed, detail = check_pii(output_text)
    result.checks.append({"name": "PII Detection", "passed": passed, "detail": detail})
    if not passed:
        result.passed = False
        result.failures.append("pii")

    # Check 3: JSON parse
    passed, detail, parsed = check_json_parse(output_text)
    result.checks.append({"name": "JSON Parse", "passed": passed, "detail": detail})
    if not passed:
        result.passed = False
        result.failures.append("json_parse")
        return result  # can't continue without parsed JSON

    result.parsed_output = parsed

    # Check 4: Schema compliance
    passed, detail = check_schema(parsed)
    result.checks.append({"name": "Schema", "passed": passed, "detail": detail})
    if not passed:
        result.passed = False
        result.failures.append("schema")

    # Check 5: Completeness
    passed, detail = check_completeness(parsed)
    result.checks.append({"name": "Completeness", "passed": passed, "detail": detail})
    if not passed:
        result.passed = False
        result.failures.append("completeness")

    # Check 6: Hallucination
    passed, detail = check_hallucination(parsed, input_text)
    result.checks.append({"name": "Hallucination", "passed": passed, "detail": detail})
    if not passed:
        result.passed = False
        result.failures.append("hallucination")

    return result


# ── Quick test ──
if __name__ == "__main__":
    # Test with a good example from val set
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

    val_path = Path("data/val.jsonl")
    if not val_path.exists():
        console.print("[red]data/val.jsonl not found[/]")
    else:
        with open(val_path) as f:
            example = json.loads(f.readline())

        console.print("[bold]Testing guardrails with val example:[/]\n")
        result = run_guardrails(
            input_text=example["input_text"],
            output_text=json.dumps(example["expected_output"]),
        )
        result.print_report()
