"""
Prompt templates for FinLens extraction.

Contains the system prompt and user prompt template.
Used in: training data formatting, inference, and evaluation.
One definition, used everywhere — keeps prompts consistent.
"""

import json

from src.finlens.schemas import get_extraction_schema

SYSTEM_PROMPT = """You are a financial document analyst specializing in SEC filing extraction.

Your task: Given a section of a SEC 10-K filing, extract structured data into JSON.

You must extract:
1. Risk factors — with category, severity (high/medium/low), and supporting evidence
2. Material events — with dates and impact (positive/negative/neutral)
3. Financial obligations — with amounts, deadlines, and category

Respond with ONLY valid JSON matching this schema:

{schema}

Rules:
- Extract ALL relevant items, not just the first one
- Use exact enum values: severity must be "high", "medium", or "low"
- Dates in YYYY-MM-DD or YYYY-MM format
- Dollar amounts like "$2.5B", "$140M"
- Evidence should be short phrases from the source text
- If a field is unknown, use null
- No explanation, no markdown — just the JSON object"""


def get_system_prompt() -> str:
    """Build the system prompt with the current JSON schema."""
    schema = get_extraction_schema()
    return SYSTEM_PROMPT.format(schema=json.dumps(schema, indent=2))


def get_user_prompt(filing_text: str) -> str:
    """Wrap filing text in the user prompt template."""
    return f"Extract structured data from the following SEC filing section:\n\n{filing_text}"


def format_chat_messages(filing_text: str, response_json: dict | None = None) -> list[dict]:
    """
    Build chat messages. Merges system prompt into user message
    for models that don't support system role (like Gemma).
    """
    system = get_system_prompt()
    user = get_user_prompt(filing_text)

    # Combine system + user into one user message
    combined_user = f"{system}\n\n---\n\n{user}"

    messages = [
        {"role": "user", "content": combined_user},
    ]

    if response_json is not None:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(response_json, indent=2),
            }
        )

    return messages

    return messages
