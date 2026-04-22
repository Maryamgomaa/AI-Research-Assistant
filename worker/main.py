"""Jetson Nano Ingestion & Research API

Routes:
  - / → web frontend
  - /discover, /ingest, /query, /report → authenticated research APIs
  - /webhook/chatbot, /webhook/research → LLM (+ RAG); require X-Webhook-Secret or Bearer JWT
  - /webhook/ingest, /webhook/report, /webhook/query, /webhook/ingest_pdf → n8n/Telegram (X-Webhook-Secret)
  - /multi-agent/architecture, /health
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import httpx
from fastapi import FastAPI, HTTPException, Request, Depends, File, UploadFile, Form, Header
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .arxiv_parser import ArxivParser
from .chunker import TextChunker
from .config import Config
from .embedder import EmbeddingPipeline
from .llm_client import GroqClient, RAGAnsweringEngine, ResearchReportGenerator
from .pdf_extractor import PDFExtractor
from .qdrant_client import QdrantManager
from .web_search import SerpapiSearcher
from .models import User, ChatHistory, create_tables
from .auth import (
    get_current_user,
    get_password_hash,
    verify_password,
    create_access_token,
    get_db,
    verify_llm_webhook_access,
)
from .routing_guard import enforce_or_log_secondary
from .static.multi_agent import (
    AgentRunTrace,
    agents_manifest,
    run_assistant_turn,
    run_research_turn,
    trace_discover,
    trace_report_pipeline,
)

# Create database tables on startup
create_tables()

log_path = Path(Config.LOG_PATH)
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger("ingestion_worker")


def _extract_forwarded_webhook_output(data: dict) -> str:
    if not isinstance(data, dict):
        return str(data).strip()
    out = (
        data.get("output")
        or data.get("message")
        or data.get("text")
        or data.get("reply")
    )
    if out:
        return str(out).strip()
    ch = data.get("choices")
    if ch and isinstance(ch, list) and ch[0].get("message", {}).get("content"):
        return ch[0]["message"]["content"].strip()
    return ""


async def _post_json(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def _require_webhook_secret(expected: str, provided: Optional[str], endpoint: str) -> None:
    exp = (expected or "").strip()
    if not exp:
        raise HTTPException(
            status_code=503,
            detail=f"{endpoint} disabled: set the matching WEBHOOK_*_SECRET in .env",
        )
    if (provided or "").strip() != exp:
        raise HTTPException(status_code=401, detail="Invalid X-Webhook-Secret")


async def _n8n_ui_pipeline_notify(event: str, payload: dict) -> None:
    """Optional orchestration preflight: notify n8n before UI-driven ingest/report."""
    url = (Config.N8N_UI_PIPELINE_WEBHOOK or "").strip()
    if not url:
        return
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(url, json={"event": event, "payload": payload})
        log.info("[ROUTING] n8n_ui_pipeline event=%s", event)
    except Exception as exc:
        log.warning("[ROUTING] n8n_ui_pipeline notify failed (%s): %s", event, exc)


class IngestionWorker:
    def __init__(self):
        self.extractor = PDFExtractor()
        self.chunker = TextChunker(
            chunk_size=Config.CHUNK_SIZE,
            overlap=Config.CHUNK_OVERLAP,
        )
        self.embedder = EmbeddingPipeline()
        self.qdrant = QdrantManager()
        self.pdf_cache_dir = Path(Config.PDF_CACHE_DIR)
        self.pdf_cache_dir.mkdir(parents=True, exist_ok=True)

    async def ingest_paper(self, paper_meta: dict) -> dict:
        arxiv_id = paper_meta.get("arxiv_id", "unknown")
        log.info(f"[INGEST] Starting ingestion for arXiv:{arxiv_id}")

        try:
            pdf_path = await self._download_pdf(paper_meta["pdf_url"], arxiv_id)
            raw_text = self.extractor.extract(pdf_path)
            if not raw_text or len(raw_text) < 200:
                raise ValueError(f"Extracted text too short ({len(raw_text)} chars)")

            chunks = self.chunker.chunk(raw_text)
            if not chunks:
                raise ValueError("No valid text chunks were extracted from the PDF")

            log.info(f"[INGEST] {len(chunks)} chunks from {arxiv_id}")
            vectors = await self.embedder.embed_batch(chunks)
            point_ids = await self.qdrant.upsert_chunks(
                collection=Config.QDRANT_COLLECTION,
                chunks=chunks,
                vectors=vectors,
                metadata=paper_meta,
            )

            result = {
                "status": "success",
                "arxiv_id": arxiv_id,
                "chunks_stored": len(point_ids),
                "pdf_path": str(pdf_path),
                "original_topic": paper_meta.get("original_topic", ""),
            }
            await self._notify_ingestion_complete(paper_meta, result)
            log.info(f"[INGEST] ✓ Stored {len(point_ids)} vectors for {arxiv_id}")
            return result

        except Exception as exc:
            log.error(f"[INGEST] ✗ Failed for {arxiv_id}: {exc}", exc_info=True)
            result = {
                "status": "error",
                "arxiv_id": arxiv_id,
                "error": str(exc),
                "original_topic": paper_meta.get("original_topic", ""),
            }
            await self._notify_ingestion_complete(paper_meta, result)
            return result

    async def batch_ingest(self, papers: list[dict]) -> list[dict]:
        semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT_INGESTIONS)

        async def guarded(paper):
            async with semaphore:
                return await self.ingest_paper(paper)

        results = await asyncio.gather(*[guarded(p) for p in papers])
        ok = sum(1 for r in results if r["status"] == "success")
        log.info(f"[BATCH] {ok}/{len(papers)} papers ingested successfully")
        return list(results)

    async def _download_pdf(self, pdf_url: str, arxiv_id: str) -> Path:
        safe_id = arxiv_id.replace("/", "_")
        pdf_path = self.pdf_cache_dir / f"{safe_id}.pdf"
        if pdf_path.exists():
            log.info(f"[DOWNLOAD] Cache hit: {pdf_path}")
            return pdf_path

        log.info(f"[DOWNLOAD] Fetching {pdf_url}")
        async with httpx.AsyncClient(
            timeout=60,
            follow_redirects=True,
            headers={"User-Agent": "AI-Research-Assistant/1.0 (educational)"},
        ) as client:
            resp = await client.get(pdf_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and len(resp.content) < 10_000:
                raise ValueError(f"Unexpected content-type '{content_type}' for {pdf_url}")

            pdf_path.write_bytes(resp.content)
            checksum = hashlib.md5(resp.content).hexdigest()
            log.info(f"[DOWNLOAD] Saved {pdf_path} ({len(resp.content)//1024}KB, md5={checksum})")

        return pdf_path

    async def _notify_ingestion_complete(self, paper_meta: dict, result: dict) -> None:
        webhook_base = Config.N8N_WEBHOOK_BASE.strip()
        if not webhook_base:
            return

        webhook_url = webhook_base.rstrip("/") + "/webhook/ingestion-complete"
        payload = {
            "arxiv_id": result.get("arxiv_id", ""),
            "status": result.get("status", "error"),
            "chunks_stored": result.get("chunks_stored", 0),
            "pdf_url": paper_meta.get("pdf_url", ""),
            "original_topic": paper_meta.get("original_topic", ""),
            "error": result.get("error", ""),
        }

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                await client.post(webhook_url, json=payload)
                log.info(f"[CALLBACK] Notified n8n at {webhook_url}")
        except Exception as exc:
            log.warning(f"[CALLBACK] Failed to notify n8n: {exc}")


class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str

class ChatHistoryResponse(BaseModel):
    id: int
    question: str
    answer: str
    sources: str
    created_at: datetime


app = FastAPI(title="Jetson Nano Research Worker", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="worker/static"), name="static")
templates = Jinja2Templates(directory="worker/templates")

_worker: Optional[IngestionWorker] = None

# Auth routes
@app.post("/register")
def register(user: RegisterRequest, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        hashed_password = get_password_hash(user.password)
        new_user = User(username=user.username, email=user.email, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        access_token = create_access_token(data={"sub": new_user.username})
        return {
            "message": "User created successfully and logged in",
            "access_token": access_token,
            "token_type": "bearer",
            "username": new_user.username,
            "user_id": new_user.id,
            "email": new_user.email,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Registration failed")
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/login")
def login(user: LoginRequest, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.username == user.username).first()
        if not db_user or not verify_password(user.password, db_user.hashed_password):
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        access_token = create_access_token(data={"sub": user.username})
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "username": db_user.username,
            "user_id": db_user.id,
            "email": db_user.email,
            "message": f"Successfully logged in as {db_user.username}"
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Login failed")
        raise HTTPException(status_code=500, detail="Unable to login at this time")

@app.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email}

@app.get("/chat_history")
def get_chat_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    histories = db.query(ChatHistory).filter(ChatHistory.user_id == current_user.id).all()
    return [ChatHistoryResponse(
        id=h.id,
        question=h.question,
        answer=h.answer,
        sources=h.sources,
        created_at=h.created_at
    ) for h in histories]

@app.post("/upload_paper")
@app.post("/upload_pdf")
async def upload_paper(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    enforce_or_log_secondary(request.url.path)
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    safe_name = Path(file.filename or "upload.pdf").name.replace("..", "_")
    file_path = upload_dir / safe_name
    content = await file.read()
    file_path.write_bytes(content)

    paper_meta = {
        "title": file.filename or safe_name,
        "authors": [current_user.username],
        "pdf_url": str(file_path.resolve()),
        "arxiv_id": f"user_{current_user.id}_{safe_name}",
        "original_topic": "user_upload",
    }

    results = await get_worker().batch_ingest([paper_meta])
    rid = paper_meta.get("arxiv_id", "")
    return {"results": results, "paper_id": rid, "paperId": rid}


def get_worker() -> IngestionWorker:
    global _worker
    if _worker is None:
        _worker = IngestionWorker()
    return _worker


async def run_report_pipeline(req: ReportRequest) -> dict:
    """Shared report generation (used by authenticated /report and /webhook/report)."""
    arxiv_ids = req.arxiv_ids or ([req.arxiv_id] if req.arxiv_id else [])

    if not req.topic or req.topic.strip() == "":
        req.topic = "general research"

    worker = get_worker()
    report_generator = ResearchReportGenerator()

    if not arxiv_ids:
        topic_vector = worker.embedder.embed_query(req.topic)
        hits = await worker.qdrant.search(
            collection=req.collection,
            query_vector=topic_vector,
            top_k=10,
        )
        arxiv_ids = list({h.get("arxiv_id") for h in hits if h.get("arxiv_id")})
        if not arxiv_ids:
            raise HTTPException(status_code=400, detail="No papers found for the given topic")

    topic_vector = worker.embedder.embed_query(req.topic)
    reports = []

    for arxiv_id in arxiv_ids[:5]:
        hits = await worker.qdrant.search(
            collection=req.collection,
            query_vector=topic_vector,
            top_k=req.top_k,
            arxiv_id_filter=arxiv_id,
        )

        if not hits:
            reports.append({"arxiv_id": arxiv_id, "error": "no chunks found in Qdrant"})
            continue

        paper_meta = {
            "arxiv_id": hits[0].get("arxiv_id", ""),
            "title": hits[0].get("title", ""),
            "authors": hits[0].get("authors", []),
            "abstract": hits[0].get("abstract", ""),
            "published": hits[0].get("published", ""),
            "pdf_url": hits[0].get("pdf_url", ""),
            "abs_url": hits[0].get("abs_url", ""),
            "primary_category": hits[0].get("primary_category", ""),
        }

        text_excerpt = "\n\n".join(hit["text"] for hit in hits)
        report = await report_generator.generate_report(paper_meta, text_excerpt, req.topic)
        reports.append(report)

    topic_summary = await report_generator.generate_topic_summary(req.topic, reports)
    _rep_trace = AgentRunTrace()
    trace_report_pipeline(_rep_trace, len(reports), req.topic or "")
    return {
        "topic": req.topic,
        "arxiv_ids": arxiv_ids,
        "reports": reports,
        "topic_summary": topic_summary,
        "multi_agent": _rep_trace.to_payload(),
    }


class IngestRequest(BaseModel):
    papers: list[dict]


class DiscoverRequest(BaseModel):
    topic: Optional[str] = ""
    max_results: int = Config.ARXIV_MAX_RESULTS
    sources: list[str] = Field(default_factory=lambda: ["arxiv"])
    categories: Optional[list[str]] = None
    citation_threshold: Optional[int] = None
    sort_by: str = "submittedDate"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    collection: str = Config.QDRANT_COLLECTION


class ReportRequest(BaseModel):
    topic: Optional[str] = "general research"
    arxiv_id: Optional[str] = None
    arxiv_ids: Optional[list[str]] = None
    collection: str = Config.QDRANT_COLLECTION
    top_k: int = 5


@app.post("/ingest")
async def ingest_endpoint(req: IngestRequest, current_user: User = Depends(get_current_user)):
    enforce_or_log_secondary("/ingest")
    if not req.papers:
        raise HTTPException(status_code=400, detail="No papers provided")
    await _n8n_ui_pipeline_notify(
        "ui_ingest",
        {"count": len(req.papers), "user": current_user.username},
    )
    results = await get_worker().batch_ingest(req.papers)
    return {"results": results}


@app.post("/webhook/ingest")
async def webhook_ingest(
    req: IngestRequest,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    """n8n W1: ingest papers without JWT. Same secret as /webhook/report (X-Webhook-Secret)."""
    _require_webhook_secret(Config.WEBHOOK_REPORT_SECRET, x_webhook_secret, "/webhook/ingest")
    if not req.papers:
        raise HTTPException(status_code=400, detail="No papers provided")
    results = await get_worker().batch_ingest(req.papers)
    return {"results": results}


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/discover")
async def discover_endpoint(req: DiscoverRequest, current_user: User = Depends(get_current_user)):
    if not req.topic or req.topic.strip() == "":
        req.topic = "general research"

    parser = ArxivParser()
    papers = []
    normalized_sources = [s.lower() for s in (req.sources or []) if isinstance(s, str)]
    if not normalized_sources:
        normalized_sources = ["arxiv"]

    if "arxiv" in normalized_sources:
        papers += [p.to_dict() for p in await parser.fetch_papers(
            req.topic,
            max_results=req.max_results,
            categories=req.categories,
            sort_by=req.sort_by,
        )]

    if any(source in normalized_sources for source in ["web", "google_scholar", "semantic_scholar", "pubmed", "openreview"]):
        searcher = SerpapiSearcher()
        papers += await searcher.search(
            req.topic,
            num_results=req.max_results,
            sources=normalized_sources,
            citation_threshold=req.citation_threshold,
            sort_by=req.sort_by,
        )

    _ma_trace = AgentRunTrace()
    trace_discover(_ma_trace, req.topic)
    return {
        "topic": req.topic,
        "sources": normalized_sources,
        "papers": papers,
        "multi_agent": _ma_trace.to_payload(),
    }


@app.post("/query")
async def query_endpoint(req: QueryRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    enforce_or_log_secondary("/query")
    worker = get_worker()
    query_vector = worker.embedder.embed_query(req.question)
    hits = await worker.qdrant.search(
        collection=req.collection,
        query_vector=query_vector,
        top_k=req.top_k,
    )

    llm = GroqClient()
    if hits:
        rag_engine = RAGAnsweringEngine(llm=llm)
        answer = await rag_engine.answer(req.question, hits)
    else:
        answer_text = await llm.generate(
            req.question,
            system=(
                "You are a helpful assistant (Groq + LangChain). "
                "No document context is available for this question—answer from general knowledge. "
                "Be accurate and concise."
            ),
        )
        answer = {
            "question": req.question,
            "answer": answer_text,
            "language": "en",
            "sources": [],
            "context_chunks_used": 0,
        }
    
    # Save to chat history
    sources_str = str([{"title": h.get("title", ""), "url": h.get("abs_url", "")} for h in hits])
    chat_entry = ChatHistory(
        user_id=current_user.id,
        question=req.question,
        answer=answer.get("answer", "") if isinstance(answer, dict) else str(answer),
        sources=sources_str
    )
    db.add(chat_entry)
    db.commit()
    
    return {"hits": hits, "answer": answer}


@app.post("/report")
async def report_endpoint(req: ReportRequest, current_user: User = Depends(get_current_user)):
    enforce_or_log_secondary("/report")
    await _n8n_ui_pipeline_notify(
        "ui_report",
        {
            "arxiv_ids": req.arxiv_ids or ([req.arxiv_id] if req.arxiv_id else []),
            "user": current_user.username,
        },
    )
    return await run_report_pipeline(req)


@app.post("/webhook/report")
async def webhook_report(
    req: ReportRequest,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    """n8n / automation: same as /report without user JWT. Requires X-Webhook-Secret."""
    _require_webhook_secret(Config.WEBHOOK_REPORT_SECRET, x_webhook_secret, "/webhook/report")
    return await run_report_pipeline(req)


class WebhookQueryBody(BaseModel):
    question: str
    top_k: int = 5
    collection: str = Config.QDRANT_COLLECTION


@app.post("/webhook/query")
async def webhook_query(
    req: WebhookQueryBody,
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    """Authenticated-style RAG + answer for n8n (no JWT). Requires X-Webhook-Secret."""
    enforce_or_log_secondary("/webhook/query")
    _require_webhook_secret(Config.WEBHOOK_QUERY_SECRET, x_webhook_secret, "/webhook/query")
    worker = get_worker()
    query_vector = worker.embedder.embed_query(req.question)
    hits = await worker.qdrant.search(
        collection=req.collection,
        query_vector=query_vector,
        top_k=req.top_k,
    )
    llm = GroqClient()
    if hits:
        rag_engine = RAGAnsweringEngine(llm=llm)
        answer = await rag_engine.answer(req.question, hits)
    else:
        answer_text = await llm.generate(
            req.question,
            system=(
                "You are a helpful assistant (Groq + LangChain). "
                "No document context is available—answer from general knowledge."
            ),
        )
        answer = {
            "question": req.question,
            "answer": answer_text,
            "language": "en",
            "sources": [],
            "context_chunks_used": 0,
        }
    return {"hits": hits, "answer": answer}


@app.post("/webhook/ingest_pdf")
async def webhook_ingest_pdf(
    file: UploadFile = File(...),
    chat_id: Optional[str] = Form(None),
    x_webhook_secret: Optional[str] = Header(None, alias="X-Webhook-Secret"),
):
    """Telegram bot / n8n: upload a PDF into Qdrant without user login. Requires X-Webhook-Secret."""
    enforce_or_log_secondary("/webhook/ingest_pdf")
    _require_webhook_secret(Config.WEBHOOK_INGEST_SECRET, x_webhook_secret, "/webhook/ingest_pdf")
    fname = (file.filename or "").strip()
    if not fname.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    upload_dir = Path("uploads") / "telegram"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe = Path(fname).name.replace("..", "_")
    dest = upload_dir / safe
    dest.write_bytes(await file.read())
    suffix = (chat_id or "anon").replace("/", "_")[:120]
    paper_meta = {
        "title": fname,
        "authors": ["telegram"],
        "pdf_url": str(dest.resolve()),
        "arxiv_id": f"telegram_{suffix}_{safe}",
        "original_topic": "telegram_upload",
    }
    results = await get_worker().batch_ingest([paper_meta])
    return {"results": results, "saved_as": str(dest)}


@app.get("/health")
def health():
    return {"status": "ok", "device": Config.DEVICE_INFO}


async def _groq_chatbot_reply(req: ChatbotRequest, system_prompt: str) -> str:
    llm = GroqClient()
    messages = []
    if req.history:
        for msg in req.history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
    messages.append({"role": "user", "content": req.message})
    return await llm.chat(messages, system=system_prompt)


async def _forward_or_groq_chatbot(req: ChatbotRequest, system_prompt: str) -> str:
    forward = (Config.N8N_CHATBOT_WEBHOOK or "").strip()
    payload = {
        "message": req.message,
        "history": req.history,
        "mode": req.mode,
        "systemPrompt": req.systemPrompt,
    }
    if forward:
        try:
            raw = await _post_json(forward, payload)
            text = _extract_forwarded_webhook_output(raw)
            if text:
                return text
            log.warning("[WEBHOOK] n8n chatbot returned no usable output; falling back to Groq")
        except Exception as exc:
            log.warning(f"[WEBHOOK] n8n chatbot forward failed ({exc}); falling back to Groq")
    return await _groq_chatbot_reply(req, system_prompt)


async def _groq_research_reply(req: ResearchRequest, system_prompt: str, rag_meta: dict) -> str:
    llm = GroqClient()
    worker = get_worker()
    rag_context = ""
    try:
        query_vector = worker.embedder.embed_query(req.question)
        hits = await worker.qdrant.search(
            collection=Config.QDRANT_COLLECTION,
            query_vector=query_vector,
            top_k=5,
        )
        if hits:
            rag_context = "\n\n".join(
                [
                    f"Paper: {hit.get('title', '')} - {hit.get('text', '')[:500]}"
                    for hit in hits[:5]
                ]
            )
    except Exception as e:
        log.warning(f"Failed to search Qdrant: {e}")

    rag_meta["used"] = bool(rag_context)
    messages = []
    if req.history:
        for msg in req.history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
            })
    if rag_context:
        messages.append({
            "role": "user",
            "content": (
                f"{req.question}\n\n---\n"
                "Optional excerpts from ingested papers (use only if relevant; "
                "otherwise answer from general knowledge):\n"
                f"{rag_context}"
            ),
        })
    else:
        messages.append({"role": "user", "content": req.question})
    return await llm.chat(messages, system=system_prompt)


async def _forward_or_groq_research(
    req: ResearchRequest, system_prompt: str, rag_meta: dict
) -> str:
    forward = (Config.N8N_RESEARCH_WEBHOOK or "").strip()
    payload = {
        "question": req.question,
        "history": req.history,
        "mode": req.mode,
        "systemPrompt": req.systemPrompt,
    }
    if forward:
        try:
            raw = await _post_json(forward, payload)
            text = _extract_forwarded_webhook_output(raw)
            if text:
                if "rag_used" in raw:
                    rag_meta["used"] = bool(raw.get("rag_used"))
                return text
            log.warning("[WEBHOOK] n8n research returned no usable output; falling back to Groq")
        except Exception as exc:
            log.warning(f"[WEBHOOK] n8n research forward failed ({exc}); falling back to Groq")
    return await _groq_research_reply(req, system_prompt, rag_meta)


class ChatbotRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = None
    mode: Optional[str] = "assistant"
    systemPrompt: Optional[str] = None


class ResearchRequest(BaseModel):
    question: str
    history: Optional[list[dict]] = None
    mode: Optional[str] = "research"
    systemPrompt: Optional[str] = None


@app.post("/webhook/chatbot")
async def webhook_chatbot(
    req: ChatbotRequest,
    _authorized: bool = Depends(verify_llm_webhook_access),
):
    """Handle chatbot questions about how to search, prompts, n8n, etc."""
    system_prompt = req.systemPrompt or """You are Athena Research Assistant (Groq + LangChain).
You can discuss any topic the user asks.
When relevant, give practical help on: arXiv search, Arabic/English NLP research prompts, this platform’s features, and wiring n8n (webhooks) for automation.
Be concise unless asked for detail."""

    trace = AgentRunTrace()
    try:

        async def _exec():
            return await _forward_or_groq_chatbot(req, system_prompt)

        out, ma = await run_assistant_turn(
            trace, (req.message or "")[:200], _exec
        )
        return {
            "message": req.message,
            "output": out,
            "status": "success",
            "multi_agent": ma,
        }
    except Exception as e:
        log.error(f"Chatbot webhook error: {e}", exc_info=True)
        return {
            "message": req.message,
            "output": f"Sorry, I couldn't process your request: {str(e)}",
            "status": "error",
            "multi_agent": trace.to_payload(),
        }


@app.post("/webhook/research")
async def webhook_research(
    req: ResearchRequest,
    _authorized: bool = Depends(verify_llm_webhook_access),
):
    """Handle research questions about papers, analysis, NLP topics."""
    system_prompt = req.systemPrompt or """You are Athena Research AI (Groq + LangChain).
You answer questions on any subject. You are especially strong on: academic research, NLP, Arabic NLP, ML, and paper analysis.
When optional excerpts from the user’s ingested papers are provided, use them when they help; otherwise rely on general knowledge.
Be precise and structured when appropriate."""

    trace = AgentRunTrace()
    rag_meta: dict = {"used": False}

    try:

        async def _exec():
            return await _forward_or_groq_research(req, system_prompt, rag_meta)

        out, ma = await run_research_turn(
            trace, (req.question or "")[:200], _exec
        )
        return {
            "question": req.question,
            "output": out,
            "status": "success",
            "rag_used": rag_meta["used"],
            "multi_agent": ma,
        }
    except Exception as e:
        log.error(f"Research webhook error: {e}", exc_info=True)
        return {
            "question": req.question,
            "output": f"Sorry, I couldn't process your research question: {str(e)}",
            "status": "error",
            "multi_agent": trace.to_payload(),
        }


@app.get("/multi-agent/architecture")
async def multi_agent_architecture():
    """Full agent roster, roles, coordination graph hints, and code pointers."""
    return agents_manifest()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=Config.WORKER_PORT)
