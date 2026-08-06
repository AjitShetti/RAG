# arXiv Paper Curator

Core Python package and API service for the **arXiv Paper Curator & RAG System**.

## 🛠️ Components

- **`src/`**: FastAPI application, LangGraph agent workflows, OpenSearch hybrid search, and database models.
- **`airflow/dags/`**: Apache Airflow DAGs for fetching, parsing, and indexing arXiv papers.
- **`gradio_launcher.py`**: Web chat user interface powered by Gradio.

## 🚀 Running Locally

```bash
# Install dependencies
uv sync

# Run FastAPI app
uv run uvicorn src.main:app --reload --port 8000
```