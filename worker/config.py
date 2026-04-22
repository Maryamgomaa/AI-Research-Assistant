import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load project .env before Config defaults read os.environ (uvicorn does not load .env by itself).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Config:
    # ── Qdrant ──────────────────────────────────────────────────────────
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "research_papers")
    QDRANT_VECTOR_SIZE: int = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
    # Cloud-hosted Qdrant (set QDRANT_URL to full https:// URL to use cloud instead of host:port)
    QDRANT_URL: str = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # ── Embedding model ──────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
    EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # ── Ollama / LLM ─────────────────────────────────────────────────────
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    # ── Groq API ─────────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    # ── Worker ───────────────────────────────────────────────────────────
    WORKER_PORT: int = int(os.getenv("WORKER_PORT", "8000"))
    MAX_CONCURRENT_INGESTIONS: int = int(os.getenv("MAX_CONCURRENT_INGESTIONS", "3"))

    # ── Chunking ─────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "700"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # ── PDF cache ────────────────────────────────────────────────────────
    PDF_CACHE_DIR: str = os.getenv("PDF_CACHE_DIR", "/tmp/arxiv_pdfs")

    # ── Logging ──────────────────────────────────────────────────────────
    LOG_PATH: str = os.getenv("LOG_PATH", "logs/ingestion.log")

    # ── arXiv ────────────────────────────────────────────────────────────
    ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "10"))
    ARXIV_BASE_URL: str = "https://export.arxiv.org/api/query"

    # ── Jetson Nano hardware info ──────────────────────────────────────────
    DEVICE_INFO: str = os.getenv("DEVICE_INFO", "Jetson Nano 4GB")

    # ── n8n (for internal callbacks) ──────────────────────────────────────
    N8N_WEBHOOK_BASE: str = os.getenv("N8N_WEBHOOK_BASE", "http://localhost:5678")
    # If set, chat webhooks POST here first; n8n should return JSON with output/text (then Groq fallback).
    N8N_CHATBOT_WEBHOOK: str = os.getenv("N8N_CHATBOT_WEBHOOK", "")
    N8N_RESEARCH_WEBHOOK: str = os.getenv("N8N_RESEARCH_WEBHOOK", "")

    # ── Server-to-server webhooks (n8n / Telegram bot) ────────────────────
    # Required to call these endpoints (503 if unset). Use the same value in n8n HTTP headers.
    WEBHOOK_QUERY_SECRET: str = os.getenv("WEBHOOK_QUERY_SECRET", "")
    WEBHOOK_REPORT_SECRET: str = os.getenv("WEBHOOK_REPORT_SECRET", "")
    WEBHOOK_INGEST_SECRET: str = os.getenv("WEBHOOK_INGEST_SECRET", "")
    # Required for n8n/Telegram on /webhook/research and /webhook/chatbot unless caller uses Bearer JWT.
    WEBHOOK_LLM_SECRET: str = os.getenv("WEBHOOK_LLM_SECRET", "")
    # Dev-only: allow unauthenticated browser calls to LLM webhooks (not for production).
    ALLOW_ANONYMOUS_LLM_WEBHOOK: bool = os.getenv(
        "ALLOW_ANONYMOUS_LLM_WEBHOOK", ""
    ).strip().lower() in ("1", "true", "yes")

    # Optional: notify n8n before UI-driven ingest/report (orchestration preflight).
    N8N_UI_PIPELINE_WEBHOOK: str = os.getenv("N8N_UI_PIPELINE_WEBHOOK", "").strip()
    # Optional: Telegram PDF hits this n8n webhook first; workflow should forward to /webhook/ingest_pdf.
    N8N_PDF_INGEST_WEBHOOK: str = os.getenv("N8N_PDF_INGEST_WEBHOOK", "").strip()

    # ── Optional generic web search API (SerpAPI) ─────────────────────────
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")
