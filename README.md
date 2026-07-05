# FinLens — SEC Filing Intelligence Pipeline

> Production LLMOps system: Fine-tuned Gemma 2B for structured extraction from SEC 10-K filings, with guardrails, monitoring, and Kubernetes deployment.

## What It Does

FinLens takes raw SEC filing text and extracts structured JSON — risk factors with severity ratings, material events with dates, and financial obligations with amounts. It replaces hours of manual analyst review with sub-second automated extraction.

**Input:**
```
"The Company received a $340 million fine from the SEC for 
derivatives trading violations in September 2024. Our long-term 
debt of $12.8 billion matures in 2025..."
```

**Output:**
```json
{
  "risk_factors": [
    {"factor": "SEC investigation into derivatives trading", "severity": "high", "category": "legal"}
  ],
  "material_events": [
    {"event": "SEC fine for trading violations", "date": "2024-09", "impact": "negative"}
  ],
  "financial_obligations": [
    {"obligation": "Long-term debt maturity", "amount": "$12.8B", "deadline": "2025"}
  ]
}
```

## Architecture

```
┌──────────────────────────────────────────────────┐
│  TRAINING PIPELINE                                │
│                                                   │
│  Synthetic SEC Data (Claude API, 250 examples)    │
│       → Pydantic Schema Validation                │
│       → Chat Template Formatting                  │
│       → QLoRA Fine-tuning (Gemma 2B, T4 GPU)     │
│       → W&B Experiment Tracking                   │
│       → LoRA Adapter → HuggingFace Hub            │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  INFERENCE STACK                                  │
│                                                   │
│  vLLM Server (GPU, OpenAI-compatible API)         │
│       ↕                                           │
│  FastAPI Backend                                  │
│    ├── 6-check Guardrail Pipeline                 │
│    │   (JSON, schema, PII, hallucination,         │
│    │    completeness, input validation)            │
│    ├── SQLite Persistence (audit log)              │
│    ├── Prometheus Metrics (/metrics)               │
│    └── OpenTelemetry Distributed Tracing           │
│       ↕                                           │
│  React + TypeScript Dashboard                     │
└──────────────────────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│  DEPLOYMENT                                       │
│                                                   │
│  Docker (multi-stage builds)                      │
│  Kubernetes (API replicas, GPU scheduling,        │
│              health checks, resource limits)       │
│  CI/CD: GitHub Actions → Lint → Test → Eval Gate  │
└──────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | Gemma 2B, QLoRA (rank 16), PEFT |
| Training | SFTTrainer, bitsandbytes 4-bit, W&B |
| Data | Synthetic (Claude API), Pydantic validation |
| Inference | vLLM (PagedAttention, continuous batching) |
| Backend | FastAPI, async SQLAlchemy, SQLite |
| Frontend | React, TypeScript, Vite |
| Guardrails | JSON/schema/PII/hallucination/completeness checks |
| Monitoring | Prometheus metrics, OpenTelemetry tracing |
| Deployment | Docker multi-stage, Kubernetes, GPU scheduling |
| CI/CD | GitHub Actions, eval gate (90% threshold) |

## Training Results

- **Model:** google/gemma-2-2b-it + QLoRA adapter
- **Dataset:** 243 validated examples (206 train / 37 val)
- **Training Loss:** 0.025
- **Validation Loss:** 0.025
- **Token Accuracy:** 99.9%
- **Guardrail Pass Rate:** 97%
- **W&B Dashboard:** [View training run](https://wandb.ai/jaygujarathi1016-gft/finlens)

## Quick Start

### Prerequisites
- Python 3.12+
- [uv](https://astral.sh/uv) package manager
- Docker (optional, for containerized deployment)

### Install and Run

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/YOUR_USERNAME/finlens.git
cd finlens
uv sync --extra dev

# Run API (mock mode — no GPU needed)
uv run uvicorn src.finlens.api.main:app --port 8080

# Run frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open http://localhost:5173 — paste SEC filing text, click Extract.

### Run with Real Model

```bash
# Local CPU inference (slow, ~30-60s per request)
INFERENCE_MODE=local LORA_PATH=YOUR_USERNAME/finlens-lora \
  uv run uvicorn src.finlens.api.main:app --port 8080

# vLLM GPU inference (fast, requires CUDA GPU)
INFERENCE_MODE=vllm VLLM_URL=http://localhost:8000 \
  uv run uvicorn src.finlens.api.main:app --port 8080
```

### Docker

```bash
docker compose up --build
# API: localhost:8080 | Frontend: localhost:3000
```

## Project Structure

```
finlens/
├── config/settings.py              # Centralized configuration
├── src/finlens/
│   ├── schemas.py                   # Pydantic extraction schema
│   ├── prompts.py                   # System/user prompt templates
│   ├── dataset/
│   │   ├── generator.py             # Synthetic data generation (Claude)
│   │   ├── validator.py             # Schema validation + train/val split
│   │   └── formatter.py            # Chat template formatting
│   ├── training/
│   │   └── train.py                 # QLoRA fine-tuning script
│   ├── evaluation/
│   │   └── eval.py                  # Extraction accuracy scoring
│   ├── guardrails/
│   │   ├── checks.py                # 6 individual guardrail checks
│   │   └── pipeline.py             # Sequential check runner
│   ├── api/
│   │   ├── main.py                  # FastAPI (3 inference modes)
│   │   ├── models.py               # Request/response models
│   │   └── database.py             # SQLAlchemy + SQLite
│   └── monitoring/
│       ├── metrics.py               # Prometheus counters/histograms
│       └── tracing.py              # OpenTelemetry spans
├── frontend/                        # React + TypeScript dashboard
├── docker/                          # Multi-stage Dockerfiles
├── k8s/                            # Kubernetes manifests (GPU scheduling)
├── .github/workflows/ci.yml        # CI/CD with eval gate
└── tests/                          # Guardrail unit tests
```

## Deployment Plan

### Infrastructure Requirements

| Component | Resource | Estimated Cost |
|-----------|----------|---------------|
| API (2 replicas) | 0.5 CPU, 1GB RAM each | ~$30/month |
| Frontend | 0.2 CPU, 256MB RAM | ~$5/month |
| vLLM (GPU) | 1x A10 24GB GPU | ~$400/month |
| Database | SQLite (or Postgres for scale) | ~$10/month |
| Monitoring | Prometheus + Grafana | ~$20/month |
| **Total** | | **~$465/month** |

### Production Deployment Steps

1. Push model to HuggingFace Hub ✅
2. Build and push Docker images to container registry
3. Create Kubernetes cluster with GPU node pool
4. Apply K8s manifests (namespace → vllm → api → frontend)
5. Configure Prometheus scraping for /metrics endpoint
6. Set up Grafana dashboards (latency, error rate, throughput)
7. Enable GitHub Actions CI/CD pipeline
8. Run eval pipeline against production model
9. Configure alerts (latency > 5s, error rate > 5%)

### Scaling Strategy

- **Horizontal:** Increase API replicas (2 → 5) for more throughput
- **Vertical:** Upgrade GPU (A10 → A100) for faster inference
- **Batching:** vLLM continuous batching handles concurrent requests automatically
- **Caching:** Cache repeated extractions in SQLite by input hash

## License

MIT