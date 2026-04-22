"""
Telegram ↔ (optional n8n) ↔ FastAPI worker.

Flow A — Direct (default):
  Telegram → POST {WORKER_BASE}{TELEGRAM_LLM_PATH} (e.g. /webhook/research) → Groq+RAG → Telegram

Flow B — Via n8n:
  Telegram → POST {N8N_TELEGRAM_GATEWAY_URL} → n8n workflow → backend → n8n returns JSON → Telegram
  Create workflow W4 in n8n (see n8n/workflows.json) or set N8N_TELEGRAM_GATEWAY_URL to your webhook URL.

PDFs:
  User sends a document → bot downloads from Telegram → POST /webhook/ingest_pdf (header X-Webhook-Secret).

Requires: TELEGRAM_BOT_TOKEN, GROQ_API_KEY on the API, WEBHOOK_LLM_SECRET (same as API) for /webhook/research,
WEBHOOK_INGEST_SECRET for direct PDF ingest; optional N8N_PDF_INGEST_WEBHOOK for n8n-first PDF orchestration.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Optional

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

TOKEN = (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()
WORKER_BASE = (os.getenv("WORKER_BASE") or "http://127.0.0.1:8000").rstrip("/")
# Optional: full URL to n8n webhook (must start with http:// or https://). Invalid/empty → direct to WORKER_BASE.
_raw_gateway = (os.getenv("N8N_TELEGRAM_GATEWAY_URL") or "").strip()
if _raw_gateway.startswith(("http://", "https://")):
    N8N_TELEGRAM_GATEWAY_URL = _raw_gateway
else:
    N8N_TELEGRAM_GATEWAY_URL = ""
# Path on the worker when not using n8n gateway (must be unauthenticated webhook)
TELEGRAM_LLM_PATH = (os.getenv("TELEGRAM_LLM_PATH") or "/webhook/research").strip()
if not TELEGRAM_LLM_PATH.startswith("/"):
    TELEGRAM_LLM_PATH = "/" + TELEGRAM_LLM_PATH
WEBHOOK_INGEST_SECRET = (os.getenv("WEBHOOK_INGEST_SECRET") or "").strip()
WEBHOOK_LLM_SECRET = (os.getenv("WEBHOOK_LLM_SECRET") or "").strip()
N8N_PDF_INGEST_WEBHOOK = (os.getenv("N8N_PDF_INGEST_WEBHOOK") or "").strip()


async def telegram_api(client: httpx.AsyncClient, method: str, **params: Any) -> dict:
    r = await client.get(f"https://api.telegram.org/bot{TOKEN}/{method}", params=params)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("description", "Telegram API error"))
    return data["result"]


async def send_message(client: httpx.AsyncClient, chat_id: int, text: str) -> None:
    if not TOKEN:
        return
    await client.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        json={"chat_id": chat_id, "text": text[:4096]},
    )


async def download_telegram_file(client: httpx.AsyncClient, file_id: str) -> bytes:
    meta = await telegram_api(client, "getFile", file_id=file_id)
    path = meta.get("file_path") or ""
    if not path:
        raise RuntimeError("No file_path from getFile")
    url = f"https://api.telegram.org/file/bot{TOKEN}/{path}"
    r = await client.get(url)
    r.raise_for_status()
    return r.content


async def ask_backend_llm(
    client: httpx.AsyncClient, text: str, chat_id: int, history: list[dict]
) -> str:
    """Route text to n8n gateway or directly to FastAPI /webhook/research or /webhook/chatbot."""
    if N8N_TELEGRAM_GATEWAY_URL:
        payload = {
            "message": text,
            "chat_id": chat_id,
            "history": history,
        }
        r = await client.post(N8N_TELEGRAM_GATEWAY_URL, json=payload, timeout=120.0)
        r.raise_for_status()
        data = r.json()
        out = (data.get("output") or data.get("answer") or "").strip()
        if not out:
            raise RuntimeError("n8n gateway returned no output: " + str(data)[:500])
        return out

    path = TELEGRAM_LLM_PATH
    if path in ("/webhook/chatbot", "/webhook/chat"):
        body: dict = {"message": text, "history": history}
    else:
        body = {"question": text, "history": history}
    headers: dict[str, str] = {}
    if WEBHOOK_LLM_SECRET:
        headers["X-Webhook-Secret"] = WEBHOOK_LLM_SECRET
    r = await client.post(
        f"{WORKER_BASE}{path}", json=body, headers=headers, timeout=120.0
    )
    r.raise_for_status()
    data = r.json()
    out = (data.get("output") or "").strip()
    if not out:
        raise RuntimeError("Empty output from worker — set GROQ_API_KEY on the API server.")
    return out


def _format_ingest_response(js: dict) -> str:
    results = js.get("results") or []
    if results and isinstance(results[0], dict):
        st = results[0].get("status", "")
        chunks = results[0].get("chunks_stored", 0)
        return f"Ingest {st}: {chunks} chunk(s) stored. You can ask questions about this PDF."
    out = (js.get("output") or js.get("message") or "").strip()
    if out:
        return out
    return f"Ingest finished: {js!s}"


async def ingest_pdf_n8n_then_backend(
    client: httpx.AsyncClient, chat_id: int, filename: str, data: bytes
) -> str:
    """Contract: n8n orchestration first when N8N_PDF_INGEST_WEBHOOK is set."""
    if not N8N_PDF_INGEST_WEBHOOK:
        return await ingest_pdf_backend(client, chat_id, filename, data)
    files = {"file": (filename, data, "application/pdf")}
    r = await client.post(
        N8N_PDF_INGEST_WEBHOOK,
        data={"chat_id": str(chat_id)},
        files=files,
        timeout=300.0,
    )
    r.raise_for_status()
    try:
        js = r.json()
    except Exception:
        return "PDF sent to n8n; check workflow response format."
    return _format_ingest_response(js) if isinstance(js, dict) else str(js)


async def ingest_pdf_backend(client: httpx.AsyncClient, chat_id: int, filename: str, data: bytes) -> str:
    if not WEBHOOK_INGEST_SECRET:
        return (
            "PDF ingest is disabled: set WEBHOOK_INGEST_SECRET in .env on the API and in this bot's environment."
        )
    files = {"file": (filename, data, "application/pdf")}
    form = {"chat_id": str(chat_id)}
    r = await client.post(
        f"{WORKER_BASE}/webhook/ingest_pdf",
        data=form,
        files=files,
        headers={"X-Webhook-Secret": WEBHOOK_INGEST_SECRET},
        timeout=300.0,
    )
    r.raise_for_status()
    js = r.json()
    return _format_ingest_response(js)


async def poll_loop() -> None:
    if not TOKEN:
        print("Set TELEGRAM_BOT_TOKEN in .env (from @BotFather).", file=sys.stderr)
        sys.exit(1)

    offset = 0
    # Per-chat short history for multi-turn (optional)
    histories: dict[int, list[dict]] = {}

    async with httpx.AsyncClient(timeout=120) as tg:
        while True:
            r = await tg.get(
                f"https://api.telegram.org/bot{TOKEN}/getUpdates",
                params={"offset": offset, "timeout": 50},
            )
            r.raise_for_status()
            for upd in r.json().get("result", []):
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat = msg.get("chat", {}).get("id")
                if chat is None:
                    continue
                cid = int(chat)
                hist = histories.setdefault(cid, [])

                doc = msg.get("document")
                mime = (doc or {}).get("mime_type") or ""
                fn = (doc or {}).get("file_name") or ""
                is_pdf = doc and (mime == "application/pdf" or fn.lower().endswith(".pdf"))
                if is_pdf:
                    fname = fn or "document.pdf"
                    try:
                        pdf_bytes = await download_telegram_file(tg, doc["file_id"])
                        async with httpx.AsyncClient(timeout=300) as api:
                            reply = await ingest_pdf_n8n_then_backend(
                                api, cid, fname, pdf_bytes
                            )
                    except Exception as exc:
                        reply = f"Could not ingest PDF: {exc}"
                    await send_message(tg, cid, reply)
                    continue

                text = (msg.get("text") or "").strip()
                if not text:
                    await send_message(tg, cid, "Send text questions or a PDF file.")
                    continue

                try:
                    async with httpx.AsyncClient(timeout=120) as api:
                        out = await ask_backend_llm(api, text, cid, hist[-8:])
                except Exception as exc:
                    out = f"Error: {exc}"
                hist.append({"role": "user", "content": text})
                hist.append({"role": "assistant", "content": out})
                await send_message(tg, cid, out)

            await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(poll_loop())
