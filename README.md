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
    {"factor": "SEC investigation", "severity": "high", "category": "legal"}
  ],
  "material_events": [
    {"event": "SEC fine", "date": "2024-09", "impact": "negative"}
  ],
  "financial_obligations": [
    {"obligation": "Debt maturity", "amount": "$12.8B", "deadline": "2025"}
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
│    ├── SQLite Persistence (audit log)             │
│    ├── Prometheus Metrics (/metrics)              │
│    └── OpenTelemetry Distributed Tracing          │
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
- Node.js 18+ (for frontend)
- Docker (optional)

### Install

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/YOUR_USERNAME/finlens.git
cd finlens
uv sync --extra dev

# Install frontend
cd frontend && npm install && cd ..
```

### Run Locally — Mock Mode (No GPU)

```bash
# Terminal 1 — API
uv run uvicorn src.finlens.api.main:app --port 8080 --reload

# Terminal 2 — Frontend
cd frontend && npm run dev
```

Open http://localhost:5173 — paste SEC text, click Extract.

### Run Locally — Real Model on CPU

```bash
# Install model dependencies
uv add transformers peft accelerate torch

# Terminal 1 — API with real model (first run downloads ~5GB)
HF_TOKEN=your-token \
INFERENCE_MODE=local \
LORA_PATH=YOUR_USERNAME/finlens-lora \
  uv run uvicorn src.finlens.api.main:app --port 8080

# Terminal 2 — Frontend
cd frontend && npm run dev
```

First request takes ~60s on CPU. Subsequent requests ~30s.

### Run with Docker

```bash
# Build and start both containers
docker compose up --build

# API: http://localhost:8080
# Frontend: http://localhost:3000
# Metrics: http://localhost:8080/metrics/
# API docs: http://localhost:8080/docs

# Stop
docker compose down
```

### Run with vLLM (Production, requires CUDA GPU)

```bash
# On a GPU server — start vLLM
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-2b-it \
    --enable-lora \
    --lora-modules finlens=YOUR_USERNAME/finlens-lora \
    --port 8000

# Start API pointing to vLLM
INFERENCE_MODE=vllm \
VLLM_URL=http://localhost:8000 \
  uv run uvicorn src.finlens.api.main:app --port 8080
```

## Kubernetes Deployment

### Prerequisites
- kubectl configured with a cluster
- Cluster with GPU node pool (for vLLM)
- Docker images pushed to a container registry

### Deploy Step by Step

```bash
# 1. Create namespace
kubectl apply -f k8s/namespace.yml

# 2. Create secrets
kubectl create secret generic finlens-secrets \
  --from-literal=hf-token=YOUR_HF_TOKEN \
  -n finlens

# 3. Deploy vLLM (GPU pod)
kubectl apply -f k8s/vllm-deployment.yml

# 4. Wait for vLLM to be ready
kubectl wait --for=condition=ready pod -l app=vllm -n finlens --timeout=300s

# 5. Deploy API
kubectl apply -f k8s/api-deployment.yml

# 6. Deploy frontend
kubectl apply -f k8s/frontend-deployment.yml
```

### Verify

```bash
# Check all pods are running
kubectl get pods -n finlens

# Expected:
# NAME                        READY   STATUS    RESTARTS
# vllm-xxx                    1/1     Running   0
# api-xxx                     1/1     Running   0
# api-yyy                     1/1     Running   0
# frontend-xxx                1/1     Running   0

# Check services
kubectl get svc -n finlens

# Expected:
# NAME       TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)
# vllm       ClusterIP      10.x.x.x        <none>          8000/TCP
# api        ClusterIP      10.x.x.x        <none>          8080/TCP
# frontend   LoadBalancer   10.x.x.x        34.x.x.x        80/TCP

# Get frontend external IP
kubectl get svc frontend -n finlens -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### Monitor

```bash
# View logs
kubectl logs -f deployment/api -n finlens
kubectl logs -f deployment/vllm -n finlens

# Check GPU allocation
kubectl describe node | grep -A 5 "nvidia.com/gpu"

# Scale API replicas
kubectl scale deployment api --replicas=5 -n finlens

# Rolling update (after pushing new image)
kubectl rollout restart deployment/api -n finlens
kubectl rollout status deployment/api -n finlens
```

### Troubleshoot

```bash
# Pod not starting?
kubectl describe pod <pod-name> -n finlens

# OOM or GPU issues?
kubectl top pods -n finlens

# vLLM not loading model?
kubectl logs deployment/vllm -n finlens --tail=50

# Delete everything and start fresh
kubectl delete namespace finlens
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
├── k8s/                            # Kubernetes manifests
├── .github/workflows/ci.yml        # CI/CD with eval gate
├── tests/                          # Guardrail unit tests
└── data/                           # Training data (gitignored)
```

## Deployment Plan

### Infrastructure

| Component | Resource | Estimated Cost |
|-----------|----------|---------------|
| API (2 replicas) | 0.5 CPU, 1GB RAM each | ~$30/month |
| Frontend | 0.2 CPU, 256MB RAM | ~$5/month |
| vLLM (GPU) | 1x A10 24GB GPU | ~$400/month |
| Database | SQLite / Postgres | ~$10/month |
| Monitoring | Prometheus + Grafana | ~$20/month |
| **Total** | | **~$465/month** |

### Scaling Strategy

- **Horizontal:** Increase API replicas (2 → 5) for more throughput
- **Vertical:** Upgrade GPU (A10 → A100) for faster inference
- **Batching:** vLLM continuous batching handles concurrency automatically
- **Caching:** Cache repeated extractions by input hash

## License

MIT