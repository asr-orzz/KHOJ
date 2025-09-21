Got it — here’s a **backend-only** README for **KHOJ+** with all frontend bits removed.

---

# 🚀 KHOJ+ — Hyper-Personalized Search & Recommendations 

A production-grade, Bharat-first platform for **autosuggest, search, and product recommendations**. KHOJ+ combines **sequence generation**, **hybrid retrieval**, and **trust-aware re-ranking** with **vernacular + price-intent understanding** to lift conversion and reduce returns.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-informational.svg)
![Redis](https://img.shields.io/badge/Redis-supported-red.svg)

---

## 📋 Table of Contents

* [Overview](#overview)
* [Key Features](#key-features)
* [Architecture](#architecture)
* [Technology Stack](#technology-stack)
* [Installation](#installation)
* [Quick Start](#quick-start)
* [API](#api)
* [Model Training](#model-training)
* [Configuration](#configuration)
* [Development](#development)
* [Testing](#testing)
* [Deployment](#deployment)
* [Contributing](#contributing)
* [License](#license)
* [Support](#support)

---

## 🎯 Overview

**KHOJ+** backend exposes REST APIs to power:

* **Neural Autosuggest** — context-aware, vernacular/code-mix friendly.
* **Hybrid Search** — semantic dual-encoder + BM25 retrieval, trust-aware LTR re-ranking.
* **Recommendations** — SRP/PDP slots tuned to **price intent** and **deliverability**.
* **Real-time Learning** — session signals, bandit-based continuous optimization.
* **Low Latency** — Redis caching and ONNX inference for sub-120ms autosuggest P95.

---

## ✨ Key Features

### 🤖 AI/ML

* **Sequence-based autosuggest** with short-sequence memory + diversity-aware beam search.
* **Vernacular & Price-Intent Understanding** (e.g., “sadi under 300”, “chasma”).
* **Hybrid Retrieval** (ANN + BM25) with calibrated score merging.
* **Trust-Aware Re-ranking (LTR)**: boosts 4★+, penalizes chronic high-return SKUs, ensures seller/assortment diversity, considers **pincode ETA**.
* **Bandit Optimizer** to continuously tune suggestion diversity and intent chips by cohort.

### 🛠️ Platform

* **FastAPI** with OpenAPI docs & pydantic validation.
* **Redis** for feature/result caches and session state.
* **Feature Flags** with A/B and multi-armed bandits.
* **Telemetry**: health endpoints, structured logs.

---

## 🏗️ Architecture

```
┌─────────────────────┐       ┌──────────────────────┐
│  Client / Caller    │◄──────►  FastAPI (KHOJ+ API) │
└─────────────────────┘       └──────────────────────┘
           │                          │
           │                 ┌───────────────────┐
           │                 │  KHOJ+ Models     │  (PyTorch + ONNX)
           │                 └───────────────────┘
           │                          │
           │                  ┌────────────────┐
           └──────────────────►   Redis Cache  │
                              └────────────────┘
                                      │
                              ┌────────────────┐
                              │   SQL DB       │
                              └────────────────┘
```

**Code layout**

* `api/` — FastAPI app, routers, schemas, experiments.
* `core/` — `autosuggest.py`, `retrieval.py`, `rerank.py`, `personalization.py`, utilities.
* `config/` — env & feature flags, hyper-params.
* `training/` — scripts for embeddings, autosuggest, LTR.
* `tests/` — unit/integration suites.

---

## 🛠️ Technology Stack

* **Backend:** Python 3.9+, FastAPI, Uvicorn, Pydantic, SQLAlchemy (SQLite/PostgreSQL)
* **ML:** PyTorch 2.x, ONNX Runtime, Sentence-Transformers, scikit-learn, NLTK, UMAP + HDBSCAN
* **Infra:** Redis, Docker, Make, pre-commit, logging/metrics of your choice

---

## 🚀 Installation

### Prerequisites

* Python 3.9+
* Redis (recommended)
* PostgreSQL (optional; SQLite default)
* Docker (optional)

### Setup

```bash
git clone https://github.com/yourorg/khoj-plus.git
cd khoj+
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then edit values
```

---

## 🏃 Quick Start

**Run API**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Open API docs**

* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

---

## 📚 API

### Endpoints

| Endpoint                     | Method | Description                                | Body                                                                                    |
| ---------------------------- | ------ | ------------------------------------------ | --------------------------------------------------------------------------------------- |
| `/api/v1/autosuggest`        | POST   | Context-aware suggestions + inferred chips | `{"query":"sadi u","max":8,"session_id":"...","pincode":"560001"}`                      |
| `/api/v1/search`             | POST   | Hybrid retrieval + trust-aware re-rank     | `{"query":"saree under 299","max":24,"chips":["under_299","4plus"],"pincode":"560001"}` |
| `/api/v1/recommend`          | POST   | SRP/PDP recommendations                    | `{"anchor":"SKU123","slot":"better_under","pincode":"560001"}`                          |
| `/api/v1/experiments/assign` | POST   | A/B or bandit cohort assignment            | `{"user_id":"...","surface":"autosuggest"}`                                             |
| `/health`                    | GET    | Liveness/readiness                         | —                                                                                       |

**Python example**

```python
import requests

r = requests.post("http://localhost:8000/api/v1/autosuggest",
                  json={"query": "sadi un", "max": 6, "session_id": "abc", "pincode": "560001"})
print(r.json())

r = requests.post("http://localhost:8000/api/v1/search",
                  json={"query": "saree under 299", "max": 24,
                        "chips": ["under_299","fast_delivery"], "pincode": "560001"})
print(r.json())
```

---

## 🧠 Model Training

### 1) Prepare Data

```bash
python training/prepare_queries.py         # cleans + mines vernacular/price intent
python training/build_embeddings.py        # trains dual encoders / sentence embeddings
```

### 2) Train Autosuggest

```bash
python training/train_autosuggest.py       # sequence model + beam search tuning
python training/export_onnx.py             # export to ONNX for low-latency inference
```

### 3) Train Re-ranker (LTR)

```bash
python training/build_ltr_dataset.py
python training/train_ltr.py               # LightGBM/XGBoost or Torch-based ranker
```

### 4) Validate & Pack

```bash
python training/offline_eval.py            # CTR@k, MRR, coverage; guardrails checks
python training/pack_artifacts.py          # saves tokenizers, models, clusters, configs
```

---

## ⚙️ Configuration

Create `.env` from example:

```bash
# Server
API_HOST=0.0.0.0
API_PORT=8000

# Storage
DATABASE_URL=sqlite:///khoj.db     # or postgres://user:pass@host/db
REDIS_URL=redis://localhost:6379

# Inference
AUTOSUGGEST_P95_MS=120
ONNX_THREADS=2

# Feature flags
ENABLE_BANDITS=true
ENABLE_PERSONALIZATION=true
ENABLE_TIME_DECAY=true
ENABLE_PRICE_INTENT=true

# ML Hyper-params (override defaults if needed)
BEAM_WIDTH=5
MAX_SUGGESTIONS=8
```

---

## 🧪 Development

```bash
pip install -r requirements-dev.txt
pre-commit install
black . && isort .
mypy .
```

Run tests:

```bash
pytest -q
pytest --cov=.
```

---

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV API_HOST=0.0.0.0 API_PORT=8000
EXPOSE 8000
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000"]
```

Build & run:

```bash
docker build -t khoj-backend .
docker run -p 8000:8000 --env-file .env khoj-backend
```

### Production Notes

* Use **PostgreSQL** for persistence, **Redis** with persistence enabled.
* Enable **gunicorn + uvicorn workers** for concurrency.
* Pin model artifacts via checksums; mount as read-only volume.
* Export models to **ONNX** for predictable latency.

---

## 🤝 Contributing

1. Fork & branch:

```bash
git checkout -b feature/your-feature
```

2. Add tests; run `pytest`.
3. Format & type-check (`black`, `isort`, `mypy`).
4. Open a PR with a clear description and benchmarks if relevant.

---

## 📄 License

MIT — see `LICENSE`.

---

## 📞 Support

* Issues: open a GitHub issue in this repo
* Discussions: start a thread in the repo’s Discussions tab

---

**Made with ❤️ by the KHOJ+ Team**
