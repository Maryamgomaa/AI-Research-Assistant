# AI Research Assistant
### Arabic/English Research Discovery and Reporting Platform

This repository contains a complete research assistant pipeline for:
- discovering recent reliable papers from public sources,
- ingesting PDFs into a vector search store,
- generating structured per-paper reports,
- creating an overall topic summary,
- answering research questions with grounded citations.

## Project Structure
```
.
├── worker/
│   ├── __init__.py
│   ├── main.py              ← FastAPI app and endpoints
│   ├── config.py            ← config values from .env
│   ├── arxiv_parser.py      ← arXiv search and metadata extraction
│   ├── pdf_extractor.py     ← PyMuPDF PDF extraction
│   ├── chunker.py           ← sentence-aware chunking
│   ├── embedder.py          ← multilingual embedding pipeline
│   ├── qdrant_client.py     ← Qdrant collection, upsert, search
│   ├── llm_client.py        ← report and summary generation
│   ├── prompt_templates.py  ← LLM prompt templates
│   └── web_search.py        ← optional SerpAPI public source discovery
├── n8n/
│   └── workflows.json       ← import into n8n for orchestration
├── docker-compose.yml
├── Dockerfile.worker
├── requirements.txt
├── telegram_bot.py          ← optional Telegram → /webhook/chatbot bridge
├── .env.example
├── setup_jetson.sh
└── README.md
```

## Architecture Overview
1. User submits a topic through n8n.
2. The system expands the topic into search queries.
3. arXiv papers are discovered and metadata is extracted.
4. PDFs are validated and downloaded by the worker.
5. Text is extracted, chunked, embedded, and stored in Qdrant.
6. After ingestion, a report is generated for each paper.
7. A combined topic summary is created across the paper reports.
8. RAG QA is available for follow-up research questions.

## Multi-agent architecture (assessment / documentation)

The API follows a **coordinator–specialist** pattern: each route maps to a logical agent (session coordinator, research analyst, platform assistant, discovery librarian, per-paper analyst, topic synthesizer, RAG specialist). Specialist behaviour is the same Groq/Qdrant/arXiv code as before; the `multi_agent` package adds **named roles**, **handoff traces** on responses, and a manifest endpoint.

- **Code:** `worker/static/multi_agent/` (`definitions.py` = roles and `implements_in` pointers, `orchestrator.py` = handoff traces).
- **Primary app:** `GET http://localhost:8000/multi-agent/architecture` (when running `uvicorn worker.main:app`) returns JSON: agent list, responsibilities, coordination hints, and interaction flows.
- **Traces:** Successful and error responses from `POST /webhook/chatbot`, `POST /webhook/research`, `POST /discover`, and `POST /report` include a `multi_agent` object with `architecture` and ordered `handoffs` (coordinator ↔ specialist steps).
- **LLM stack:** Groq for generation; `langchain-groq` + `langchain-core` wrap the Groq chat API (falls back to direct HTTP if those packages are missing).
- **Telegram (optional):** With `TELEGRAM_BOT_TOKEN` set, run `python telegram_bot.py` from the project root to forward Telegram messages to `POST /webhook/chatbot` (start the API first).

## Requirements
- Python 3.11+
- Docker & Docker Compose
- **Groq:** `GROQ_API_KEY` in `.env` (required for chat webhooks, reports, and RAG answers).
- Optional: Ollama at `http://localhost:11434` only if you use auxiliary tooling that calls it.
- Qdrant running at `localhost:6333`
- Optional: `SERPAPI_API_KEY` for broader public web discovery

## Running the Project

1. **Set `GROQ_API_KEY` in `.env`** (required for the web UI chat, Ask AI box, reports, and RAG answers).

2. **Start Qdrant and n8n** (requires Docker):
   ```bash
   docker-compose up -d
   ```

3. **Start the API**:
   ```bash
   uvicorn worker.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Open the web UI**:
   Open `http://localhost:8000` in your browser.

## Web Interface
The web app allows you to:
- Discover papers by entering a research topic
- Select and ingest papers into the vector database
- View generated reports and topic summaries
- Ask questions in the main workspace (Groq + optional Qdrant context) and use the floating assistant

## Setup
1. Copy the environment template:
```bash
cp .env.example .env
```
2. Edit `.env` to match your deployment.
3. Start the stack:
```bash
docker-compose up -d
```
4. Start the worker service:
```bash
uvicorn worker.main:app --host 0.0.0.0 --port 8000
```

> Docker Compose starts Qdrant and n8n. Configure `GROQ_API_KEY` for AI features; Ollama is optional unless other tooling uses it.

## Jetson Nano Setup
On Jetson Nano, use the helper script:
```bash
bash setup_jetson.sh
```
Then run:
```bash
uvicorn worker.main:app --host 0.0.0.0 --port 8000
```

## End-to-end automation (order)

1. Copy `.env.example` → `.env`; set `GROQ_API_KEY`, **`WEBHOOK_*_SECRET`** for ingest/report/query/PDF automation, and **`WEBHOOK_LLM_SECRET`** (must match W3/W4 `X-Webhook-Secret` on `/webhook/research` calls). Sign in on the Web UI for Ask AI / chatbot, or set **`ALLOW_ANONYMOUS_LLM_WEBHOOK=1`** only for local dev.
2. `docker-compose up -d` (Qdrant + n8n). Run the API on the **host** with `uvicorn worker.main:app --port 8000` so `host.docker.internal:8000` from n8n reaches it (or change URLs in n8n to your LAN IP).
3. Import workflows from `n8n/workflows.json` and **activate** W3 (ask) and W4 (telegram-bridge) if you use them.
4. Optional Telegram: set `TELEGRAM_BOT_TOKEN`, same `WEBHOOK_INGEST_SECRET` as the API, then `python telegram_bot.py`. For **Telegram → n8n → backend**, set `N8N_TELEGRAM_GATEWAY_URL` to the production URL of the W4 webhook (e.g. `http://localhost:5678/webhook/telegram-bridge` after activation).

## n8n Workflow
Import `n8n/workflows.json` into n8n. The workflow includes:
- `W1 — Paper Discovery`: topic discovery, arXiv querying, PDF validation, ingestion
- `W2 — Ingestion Status`: ingestion callback and report trigger
- `W3 — RAG QA`: POST `/webhook/ask` → backend `POST /webhook/research` (Groq + Qdrant RAG)
- `W4 — Telegram → Backend`: optional bridge when `N8N_TELEGRAM_GATEWAY_URL` points at `/webhook/telegram-bridge`

## Web Interface
After starting the services, visit `http://localhost:8000` for the web interface.

The web app allows you to:
- Discover papers by entering a research topic
- Select and ingest papers into the vector database
- View generated reports and topic summaries
- Ask questions using RAG (Retrieval-Augmented Generation)

## Notes
- The worker package under `worker/` contains the actual ingestion and reporting logic.
- The `n8n` workflow automates topic search, ingestion, and report generation.
- To use public-source discovery beyond arXiv, configure `SERPAPI_API_KEY` and request `sources: ["arxiv", "web"]`.
- The report path returns both per-paper summaries and a combined topic-level summary.
