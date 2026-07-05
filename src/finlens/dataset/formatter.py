"""
Format dataset into chat-ready JSONL for fine-tuning.

Converts (input_text, expected_output) pairs into Llama chat format:
  [system message, user message, assistant response]

Usage:
    uv run python -m src.finlens.dataset.formatter
"""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.finlens.prompts import format_chat_messages

console = Console()


def format_file(input_path: str, output_path: str) -> int:
    """Format a JSONL file into chat-ready format. Returns count."""
    input_file = Path(input_path)
    if not input_file.exists():
        console.print(f"[red]✗ Not found: {input_path}[/]")
        return 0

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            example = json.loads(line.strip())
            messages = format_chat_messages(
                filing_text=example["input_text"],
                response_json=example["expected_output"],
            )
            fout.write(json.dumps({"messages": messages}) + "\n")
            count += 1

    return count


def format_dataset(
    train_in: str = "data/train.jsonl",
    val_in: str = "data/val.jsonl",
    train_out: str = "data/train_chat.jsonl",
    val_out: str = "data/val_chat.jsonl",
) -> None:
    """Format both train and val sets."""
    console.print("\n[bold]Formatting dataset for chat fine-tuning[/]\n")

    train_count = format_file(train_in, train_out)
    val_count = format_file(val_in, val_out)

    table = Table(title="Formatted Dataset")
    table.add_column("Split", style="cyan")
    table.add_column("Examples", style="green")
    table.add_column("Output", style="white")
    table.add_row("Train", str(train_count), train_out)
    table.add_row("Val", str(val_count), val_out)
    console.print(table)

    # Show a preview of the first example
    with open(train_out) as f:
        first = json.loads(f.readline())

    console.print("\n[bold]Preview (first example):[/]\n")
    for msg in first["messages"]:
        role = msg["role"]
        content = msg["content"][:150] + "..." if len(msg["content"]) > 150 else msg["content"]
        console.print(f"  [cyan]{role}:[/] {content}\n")

    console.print("[bold green]Done![/] Ready for fine-tuning.\n")


if __name__ == "__main__":
    format_dataset()