"""
Generate synthetic SEC filing training data using Claude.

Each example: realistic SEC 10-K text → structured ExtractionResult JSON.
Generates diverse examples across industries, risk types, and writing styles.

Usage:
    uv run python -m src.finlens.dataset.generator --count 250
"""

import argparse
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()

from src.finlens.schemas import ExtractionResult, get_extraction_schema

console = Console()

# Diversity seeds — we cycle through these to get varied examples
INDUSTRIES = [
    "technology", "banking", "pharmaceuticals", "automotive",
    "retail", "energy", "telecommunications", "healthcare",
    "aerospace", "insurance", "real estate", "manufacturing",
]

SCENARIOS = [
    "regulatory risks and compliance challenges",
    "litigation and legal proceedings",
    "debt maturities and financial obligations",
    "cybersecurity incidents and data breaches",
    "supply chain disruptions",
    "environmental and climate-related risks",
    "competitive market pressures",
    "acquisition or merger activity",
    "leadership changes and restructuring",
    "foreign currency and macroeconomic risks",
    "product liability and safety recalls",
    "intellectual property disputes",
]

PROMPT_TEMPLATE = """Generate a realistic SEC 10-K filing excerpt and its structured extraction.

INDUSTRY: {industry}
SCENARIO FOCUS: {scenario}
COMPANY: Invent a realistic fictional company name for this industry.

STEP 1: Write a realistic 10-K filing passage (150-300 words). It should:
- Read like actual SEC filing language (formal, legalistic)
- Cover 2-4 risk factors, 1-2 material events, and 1-2 financial obligations
- Include specific dollar amounts, dates, and legal references where appropriate
- Focus on the scenario but include other realistic details too

STEP 2: Extract structured data from the passage matching this exact JSON schema:

{schema}

IMPORTANT:
- "evidence" fields should be short phrases from the passage
- Use only these risk categories: regulatory, market, operational, financial, legal, cybersecurity, environmental, competitive
- Use only these severity values: high, medium, low
- Use only these impact values: positive, negative, neutral
- Dates should be YYYY-MM-DD or YYYY-MM format
- Dollar amounts like "$2.5B", "$140M", "$3.2 billion"

Respond with ONLY a JSON object with two keys:
- "input_text": the SEC filing passage you wrote
- "expected_output": the ExtractionResult JSON

No markdown, no explanation, just the JSON object."""


def generate_batch(
    client: anthropic.Anthropic,
    industry: str,
    scenario: str,
    schema: dict,
) -> dict | None:
    """Generate one training example."""
    prompt = PROMPT_TEMPLATE.format(
        industry=industry,
        scenario=scenario,
        schema=json.dumps(schema, indent=2),
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=2000,
            temperature=0.9,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]

        parsed = json.loads(text)

        # Validate the output matches our schema
        ExtractionResult.model_validate(parsed["expected_output"])

        return parsed

    except (json.JSONDecodeError, KeyError, Exception) as e:
        console.print(f"  [yellow]⚠ Skipped (parse error: {e})[/]")
        return None


def generate_dataset(count: int = 250, output_path: str = "data/dataset.jsonl") -> None:
    """Generate the full synthetic dataset."""
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    schema = get_extraction_schema()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    attempts = 0
    max_attempts = int(count * 1.3)  # allow some failures

    console.print(f"\n[bold]Generating {count} training examples[/]\n")

    for i in track(range(max_attempts), description="Generating..."):
        if len(examples) >= count:
            break

        # Cycle through industries and scenarios for diversity
        industry = INDUSTRIES[i % len(INDUSTRIES)]
        scenario = SCENARIOS[i % len(SCENARIOS)]

        result = generate_batch(client, industry, scenario, schema)
        attempts += 1

        if result:
            examples.append(result)
            if len(examples) % 25 == 0:
                console.print(f"  [green]✓ {len(examples)}/{count} generated[/]")

        # Rate limiting — stay well under API limits
        time.sleep(0.5)

    # Write JSONL
    with open(output, "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

    console.print(f"\n[bold green]Done![/] {len(examples)} examples → {output}")
    console.print(f"Attempts: {attempts}, Success rate: {len(examples)/attempts:.0%}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic SEC filing dataset")
    parser.add_argument("--count", type=int, default=250, help="Number of examples to generate")
    parser.add_argument("--output", type=str, default="data/dataset.jsonl", help="Output path")
    args = parser.parse_args()

    generate_dataset(count=args.count, output_path=args.output)