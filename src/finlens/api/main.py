"""
FinLens FastAPI backend.

Supports three inference modes (set via INFERENCE_MODE env var):
    mock  — fake data, no model needed (default)
    local — runs model on CPU, slow but real results
    vllm  — calls vLLM GPU server, fast, for production

Usage:
    # Mock mode (default):
    uv run uvicorn src.finlens.api.main:app --host 0.0.0.0 --port 8080 --reload

    # Local mode (real model on CPU):
    INFERENCE_MODE=local uv run uvicorn src.finlens.api.main:app --port 8080

    # vLLM mode (GPU server):
    INFERENCE_MODE=vllm VLLM_URL=http://vllm:8000 \
        uv run uvicorn src.finlens.api.main:app --port 8080
"""

import json
import os
import time

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from src.finlens.api.database import Extraction, async_session, init_db
from src.finlens.api.models import ExtractionRequest, ExtractionResponse, HealthResponse
from src.finlens.guardrails.pipeline import run_guardrails
from src.finlens.monitoring.metrics import metrics_app, track_request
from src.finlens.monitoring.tracing import tracer
from src.finlens.prompts import format_chat_messages

app = FastAPI(
    title="FinLens API",
    description="SEC filing structured extraction",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/metrics", metrics_app)

# ── Config ──

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "mock")
VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
BASE_MODEL = os.environ.get("BASE_MODEL", "google/gemma-2-2b-it")
LORA_PATH = os.environ.get("LORA_PATH", "")  # your HuggingFace repo path

# Local model (loaded once)
_local_model = None
_local_tokenizer = None


# ── Startup ──


@app.on_event("startup")
async def startup():
    await init_db()
    print(f"Inference mode: {INFERENCE_MODE}")
    if INFERENCE_MODE == "local":
        _load_local_model()


def _load_local_model():
    """Load fine-tuned model on CPU. Called once on startup."""
    global _local_model, _local_tokenizer

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading base model: {BASE_MODEL}")
    _local_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if _local_tokenizer.pad_token is None:
        _local_tokenizer.pad_token = _local_tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    if LORA_PATH:
        print(f"Loading LoRA adapter: {LORA_PATH}")
        _local_model = PeftModel.from_pretrained(model, LORA_PATH)
    else:
        print("No LORA_PATH set — using base model without adapter")
        _local_model = model

    _local_model.eval()
    print("Model loaded and ready.")


# ── Inference ──


async def run_inference(filing_text: str) -> str:
    """Run inference in the configured mode."""
    messages = format_chat_messages(filing_text)

    if INFERENCE_MODE == "vllm":
        return await _vllm_inference(messages)
    elif INFERENCE_MODE == "local":
        return _local_inference(messages)
    else:
        return _mock_inference()


async def _vllm_inference(messages: list[dict]) -> str:
    """Call vLLM OpenAI-compatible API."""
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": "finlens",
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.1,
            },
        )
    return response.json()["choices"][0]["message"]["content"]


def _local_inference(messages: list[dict]) -> str:
    """Run model locally on CPU."""
    import torch

    prompt = _local_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _local_tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = _local_model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
        )

    # Decode only the new tokens (skip the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return _local_tokenizer.decode(new_tokens, skip_special_tokens=True)


def _mock_inference() -> str:
    """Return fake data for development."""
    return json.dumps(
        {
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
            "summary": (
                "Mock Corp faces regulatory risks while completing "
                "strategic acquisitions funded by long-term debt."
            ),
        }
    )


# ── Endpoints ──


@app.post("/extract", response_model=ExtractionResponse)
async def extract(request: ExtractionRequest):
    """Extract structured data from SEC filing text."""
    with tracer.start_as_current_span("extract_request") as span:
        start = time.time()

        with tracer.start_as_current_span("inference"):
            output_text = await run_inference(request.filing_text)

        latency_ms = (time.time() - start) * 1000

        with tracer.start_as_current_span("guardrails"):
            guardrail_result = run_guardrails(
                input_text=request.filing_text,
                output_text=output_text,
            )

        if guardrail_result.passed:
            status = "success"
        elif "hallucination" in guardrail_result.failures:
            status = "flagged"
        else:
            status = "failed"

        span.set_attribute("status", status)
        span.set_attribute("latency_ms", latency_ms)

        with tracer.start_as_current_span("database"):
            parsed = guardrail_result.parsed_output or {}
            num_items = (
                len(parsed.get("risk_factors", []))
                + len(parsed.get("material_events", []))
                + len(parsed.get("financial_obligations", []))
            )

            record = Extraction(
                input_text=request.filing_text,
                input_length=len(request.filing_text),
                output_json=output_text,
                company_name=parsed.get("company_name"),
                num_risks=len(parsed.get("risk_factors", [])),
                num_events=len(parsed.get("material_events", [])),
                num_obligations=len(parsed.get("financial_obligations", [])),
                guardrails_passed=guardrail_result.passed,
                guardrail_failures=(
                    ", ".join(guardrail_result.failures) if guardrail_result.failures else None
                ),
                latency_ms=latency_ms,
                status=status,
            )

            async with async_session() as session:
                session.add(record)
                await session.commit()
                await session.refresh(record)

        track_request(status, latency_ms, num_items, guardrail_result.failures)

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
        model_loaded=_local_model is not None or INFERENCE_MODE == "vllm",
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
