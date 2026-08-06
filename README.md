# arXiv Paper Curator & RAG System

A production-grade **Agentic Retrieval-Augmented Generation (RAG)** pipeline for automated arXiv research paper ingestion, hybrid vector + keyword retrieval, and interactive question-answering.

---

## ⚡ Quick Start

Run the complete multi-service stack (FastAPI, Airflow, OpenSearch, PostgreSQL, Redis) with Docker Compose:

```bash
# 1. Clone & prepare environment variables
cp arxiv-paper-curator/.env.example arxiv-paper-curator/.env

# 2. Start all services
docker compose up --build -d
```

### Access Ports & Services

| Service | Endpoint / UI | Details |
| :--- | :--- | :--- |
| **FastAPI Backend** | `http://localhost:8080/docs` | OpenAPI documentation & interactive endpoints |
| **Airflow Webserver** | `http://localhost:8090` | Ingestion DAG management (`admin` / `admin`) |
| **OpenSearch** | `http://localhost:9200` | Hybrid vector & BM25 search engine |
| **PostgreSQL** | `localhost:5433` | Relational metadata store (`rag_db`) |
| **Redis** | `localhost:6379` | Query caching layer |

---

## ✨ Features

- **Automated Ingestion Pipeline**: Scheduled Airflow DAGs fetch, parse, and chunk arXiv papers (`cs.AI`, `cs.LG`, `cs.CL`).
- **Hybrid Retrieval**: Combines dense vector embeddings with BM25 keyword scoring using OpenSearch.
- **Agentic RAG**: Multi-step query rewriting, relevance evaluation, and streaming responses powered by LangGraph.
- **Observability & Caching**: Query response caching via Redis and telemetry tracing via Langfuse.
- **Interactive Interfaces**: Gradio Chat UI for web-based exploration and optional Telegram Bot integration.

---

## 🏗️ Architecture & Tech Stack

```
arXiv API ──> Airflow DAG ──> PDF Parsing & Chunking
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
          PostgreSQL (Metadata)          OpenSearch (Vectors & BM25)
                 │                                 │
                 └────────────────┬────────────────┘
                                  ▼
                     FastAPI Backend + LangGraph
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
           Gradio Web UI                     Telegram Bot
```

- **Backend Framework**: Python 3.12, FastAPI, AsyncPG, SQLAlchemy
- **Search & Storage**: OpenSearch 2.11, PostgreSQL 15, Redis 7
- **Workflow Orchestration**: Apache Airflow 2.10
- **AI & RAG Agents**: LangChain, LangGraph, OpenAI / Groq LLMs

---

## 🚀 Local Development Setup

If running without Docker for development:

### Prerequisites
- **Python 3.12+** & [`uv`](https://github.com/astral-sh/uv) (recommended)
- Running instances of PostgreSQL, OpenSearch, and Redis (or run via Docker Compose)

### Installation

```bash
cd arxiv-paper-curator

# Install dependencies
uv sync

# Run database migrations
uv run alembic upgrade head

# Launch local FastAPI server
uv run uvicorn src.main:app --reload --port 8000

# Launch Gradio UI
uv run python gradio_launcher.py
```

---

## 📌 Main API Endpoints

- `POST /ask`: Basic RAG response with retrieved context.
- `POST /agentic-ask`: Multi-step LangGraph agent decision pipeline with query expansion.
- `POST /hybrid-search`: Perform hybrid vector + lexical searches over indexed papers.
- `GET /ping`: Healthcheck endpoint.

---

## 📄 License

MIT
