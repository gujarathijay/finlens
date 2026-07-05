"""
Individual guardrail checks.

Each check takes an input/output and returns (passed: bool, detail: str).
Checks are independent — you can add, remove, or reorder them.
"""

import json
import re

from pydantic import ValidationError

from src.finlens.schemas import ExtractionResult


# ── Check 1: JSON Parse ──

def check_json_parse(output_text: str) -> tuple[bool, str, dict | None]:
    """Can we parse the output as JSON?"""
    text = output_text.strip()

    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(text)
        return True, "valid JSON", parsed
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {e.msg} at position {e.pos}", None


# ── Check 2: Schema Compliance ──

def check_schema(parsed: dict) -> tuple[bool, str]:
    """Does the parsed JSON match our ExtractionResult schema?"""
    try:
        ExtractionResult.model_validate(parsed)
        return True, "schema valid"
    except ValidationError as e:
        errors = "; ".join(err["msg"] for err in e.errors()[:3])
        return True if len(e.errors()) == 0 else False, f"schema errors: {errors}"


# ── Check 3: Hallucination Detection ──

def check_hallucination(parsed: dict, source_text: str) -> tuple[bool, str]:
    """
    Check if evidence fields reference text actually in the source.
    Flags extractions where evidence doesn't appear in the input.
    """
    source_lower = source_text.lower()
    flagged = []

    for i, risk in enumerate(parsed.get("risk_factors", [])):
        evidence = risk.get("evidence", "")
        if evidence and len(evidence) > 10:
            # Check if key phrases from evidence appear in source
            words = evidence.lower().split()
            key_phrases = [" ".join(words[j:j+3]) for j in range(0, len(words)-2)]
            matches = sum(1 for phrase in key_phrases if phrase in source_lower)
            if key_phrases and matches / len(key_phrases) < 0.3:
                flagged.append(f"risk_factors[{i}].evidence")

    if flagged:
        return False, f"possible hallucination in: {', '.join(flagged)}"
    return True, "evidence grounded in source"


# ── Check 4: PII Detection ──

PII_PATTERNS = {
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
}

def check_pii(output_text: str) -> tuple[bool, str]:
    """Detect PII patterns in model output."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, output_text):
            found.append(pii_type)

    if found:
        return False, f"PII detected: {', '.join(found)}"
    return True, "no PII detected"


# ── Check 5: Completeness ──

def check_completeness(parsed: dict) -> tuple[bool, str]:
    """Ensure the extraction isn't empty."""
    total = (
        len(parsed.get("risk_factors", []))
        + len(parsed.get("material_events", []))
        + len(parsed.get("financial_obligations", []))
    )
    if total == 0:
        return False, "extraction is empty — no items found"
    if not parsed.get("company_name"):
        return False, "missing company_name"
    if not parsed.get("summary"):
        return False, "missing summary"
    return True, f"{total} items extracted"


# ── Check 6: Input Length ──

def check_input_length(input_text: str, max_length: int = 8000) -> tuple[bool, str]:
    """Reject inputs that are too long or too short."""
    if len(input_text) < 50:
        return False, f"input too short ({len(input_text)} chars)"
    if len(input_text) > max_length:
        return False, f"input too long ({len(input_text)} chars, max {max_length})"
    return True, f"input length ok ({len(input_text)} chars)"