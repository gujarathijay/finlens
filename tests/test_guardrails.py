"""Tests for guardrail checks."""

from src.finlens.guardrails.checks import (
    check_completeness,
    check_input_length,
    check_json_parse,
    check_pii,
    check_schema,
)


def test_valid_json():
    passed, detail, parsed = check_json_parse('{"key": "value"}')
    assert passed is True
    assert parsed == {"key": "value"}


def test_invalid_json():
    passed, detail, parsed = check_json_parse("not json")
    assert passed is False
    assert parsed is None


def test_json_with_markdown_fences():
    text = '```json\n{"key": "value"}\n```'
    passed, detail, parsed = check_json_parse(text)
    assert passed is True


def test_pii_detects_email():
    passed, detail = check_pii("Contact us at john@example.com for details")
    assert passed is False
    assert "email" in detail


def test_pii_detects_ssn():
    passed, detail = check_pii("SSN: 123-45-6789")
    assert passed is False
    assert "ssn" in detail


def test_pii_clean():
    passed, detail = check_pii("The company reported $5M in revenue")
    assert passed is True


def test_input_too_short():
    passed, detail = check_input_length("short")
    assert passed is False


def test_input_too_long():
    passed, detail = check_input_length("x" * 10000)
    assert passed is False


def test_input_ok():
    passed, detail = check_input_length("x" * 500)
    assert passed is True


def test_completeness_empty():
    passed, detail = check_completeness(
        {"risk_factors": [], "material_events": [], "financial_obligations": []}
    )
    assert passed is False


def test_completeness_ok():
    passed, detail = check_completeness(
        {
            "company_name": "Test Corp",
            "summary": "A test summary",
            "risk_factors": [{"factor": "test"}],
            "material_events": [],
            "financial_obligations": [],
        }
    )
    assert passed is True


def test_schema_valid():
    valid_data = {
        "company_name": "Test Corp",
        "filing_type": "10-K",
        "fiscal_year": "2024",
        "risk_factors": [
            {
                "factor": "Risk",
                "category": "regulatory",
                "severity": "high",
                "evidence": "evidence text",
            }
        ],
        "material_events": [],
        "financial_obligations": [],
        "summary": "Test summary",
    }
    passed, detail = check_schema(valid_data)
    assert passed is True


def test_schema_invalid_enum():
    invalid_data = {
        "company_name": "Test Corp",
        "filing_type": "10-K",
        "fiscal_year": "2024",
        "risk_factors": [
            {
                "factor": "Risk",
                "category": "invalid_category",
                "severity": "high",
                "evidence": "text",
            }
        ],
        "material_events": [],
        "financial_obligations": [],
        "summary": "Test summary",
    }
    passed, detail = check_schema(invalid_data)
    assert passed is False
