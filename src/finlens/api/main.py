"""
FinLens FastAPI backend.

Endpoints:
    POST /extract    — extract structured data from SEC filing text
    GET  /health     — health check
    GET  /history    — recent extractions

Usage:
    uv run uvicorn src.finlens.api.main:app --host 0.0.0.0 --port 8080 --reload
"""

import json
import time

from fastapi import FastAPI, HTTPException
from sqlalchemy import select

from src.finlens.api.database import Extraction, async_session, init_db
from src.finlens.api.models import ExtractionRequest, ExtractionResponse, HealthResponse
from src.finlens.guardrails.pipeline import run_guardrails
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="FinLens API",
    description="SEC filing structured extraction",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ──

@app.on_event("startup")
async def startup():
    await init_db()


# ── Mock inference (replaced with vLLM in Phase 10) ──

async def mock_inference(filing_text: str) -> str:
    """
    Temporary mock — returns a plausible extraction.
    Will be replaced with actual vLLM call when model is deployed.
    """
    mock_output = {
        "company_name": "Mock Corp",
        "filing_type": "10-K",
        "fiscal_year": "2024",
        "risk_factors": [
            {
                "factor": "Regulatory compliance risk",
                "category": "regulatory",
                "severity": "high",
                "evidence": "subject to various regulatory requirements",
            }
        ],
        "material_events": [
            {
                "event": "Acquisition completed",
                "date": "2024-06-15",
                "impact": "positive",
                "details": "Acquired subsidiary for strategic expansion",
            }
        ],
        "financial_obligations": [
            {
                "obligation": "Long-term debt maturity",
                "amount": "$500M",
                "deadline": "2026-12-31",
                "category": "debt",
            }
        ],
        "summary": "Mock Corp faces regulatory risks while completing strategic acquisitions funded by long-term debt.",
    }
    return json.dumps(mock_output)


# ── Endpoints ──

@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """Extract structured data from SEC filing text."""
    start = time.time()

    # Run inference (mock for now)
    output_text = await mock_inference(request.filing_text)

    latency_ms = (time.time() - start) * 1000

    # Run guardrails
    guardrail_result = run_guardrails(
        input_text=request.filing_text,
        output_text=output_text,
    )

    # Determine status
    if guardrail_result.passed:
        status = "success"
    elif "hallucination" in guardrail_result.failures:
        status = "flagged"  # needs human review
    else:
        status = "failed"

    # Store in database
    parsed = guardrail_result.parsed_output or {}
    record = Extraction(
        input_text=request.filing_text,
        input_length=len(request.filing_text),
        output_json=output_text,
        company_name=parsed.get("company_name"),
        num_risks=len(parsed.get("risk_factors", [])),
        num_events=len(parsed.get("material_events", [])),
        num_obligations=len(parsed.get("financial_obligations", [])),
        guardrails_passed=guardrail_result.passed,
        guardrail_failures=", ".join(guardrail_result.failures) if guardrail_result.failures else None,
        latency_ms=latency_ms,
        status=status,
    )

    async with async_session() as session:
        session.add(record)
        await session.commit()
        await session.refresh(record)

    return ExtractionResponse(
        status=status,
        extraction=guardrail_result.parsed_output,
        guardrails_passed=guardrail_result.passed,
        guardrail_failures=guardrail_result.failures,
        latency_ms=round(latency_ms, 1),
        request_id=record.id,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    return HealthResponse(
        status="ok",
        model_loaded=False,  # will be True when vLLM is connected
        database="connected",
    )


@app.get("/history")
async def history(limit: int = 10):
    """Get recent extractions."""
    async with async_session() as session:
        query = select(Extraction).order_by(Extraction.id.desc()).limit(limit)
        result = await session.execute(query)
        records = result.scalars().all()

    return [
        {
            "id": r.id,
            "created_at": str(r.created_at),
            "company_name": r.company_name,
            "status": r.status,
            "num_risks": r.num_risks,
            "num_events": r.num_events,
            "num_obligations": r.num_obligations,
            "guardrails_passed": r.guardrails_passed,
            "latency_ms": r.latency_ms,
        }
        for r in records
    ]