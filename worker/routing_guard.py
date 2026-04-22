"""
Routing contract enforcement (logging + classification).

PRIMARY paths (single source of truth for automation docs):
  CHAT   → /webhook/research
  INGEST → /webhook/ingest   (after n8n orchestration in production flows)
  REPORT → /webhook/report   (n8n-driven automation)

SECONDARY paths are valid and retained; they log [ROUTING] for observability.
Optional strict mode: ENFORCE_PRIMARY_ROUTES rejects secondary POSTs with 409.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Optional

from fastapi import HTTPException

log = logging.getLogger("routing_guard")


class RouteDomain(str, Enum):
    CHAT = "chat"
    INGEST = "ingest"
    REPORT = "report"
    DISCOVER = "discover"
    AUTH = "auth"
    OTHER = "other"


PRIMARY_PATH = {
    RouteDomain.CHAT: "/webhook/research",
    RouteDomain.INGEST: "/webhook/ingest",
    RouteDomain.REPORT: "/webhook/report",
}

# Maps secondary route → (domain, primary path for messaging)
SECONDARY_ROUTES: dict[str, tuple[RouteDomain, str]] = {
    "/query": (RouteDomain.CHAT, PRIMARY_PATH[RouteDomain.CHAT]),
    "/webhook/query": (RouteDomain.CHAT, PRIMARY_PATH[RouteDomain.CHAT]),
    "/webhook/chatbot": (RouteDomain.CHAT, PRIMARY_PATH[RouteDomain.CHAT]),
    "/ingest": (RouteDomain.INGEST, PRIMARY_PATH[RouteDomain.INGEST]),
    "/webhook/ingest_pdf": (RouteDomain.INGEST, PRIMARY_PATH[RouteDomain.INGEST]),
    "/upload_paper": (RouteDomain.INGEST, PRIMARY_PATH[RouteDomain.INGEST]),
    "/upload_pdf": (RouteDomain.INGEST, PRIMARY_PATH[RouteDomain.INGEST]),
    "/report": (RouteDomain.REPORT, PRIMARY_PATH[RouteDomain.REPORT]),
}


def _strict_primary() -> bool:
    return os.getenv("ENFORCE_PRIMARY_ROUTES", "").strip().lower() in ("1", "true", "yes")


def enforce_or_log_secondary(path: str) -> None:
    """Log secondary usage; optionally reject POST handlers that call this first."""
    if path not in SECONDARY_ROUTES:
        return
    domain, primary = SECONDARY_ROUTES[path]
    log.info(
        "[ROUTING] secondary_path=%s domain=%s primary_reference=%s",
        path,
        domain.value,
        primary,
    )
    if _strict_primary():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "secondary_route_disabled",
                "message": f"Use primary path {primary} for domain {domain.value}",
                "primary": primary,
            },
        )


def classify_path(path: str) -> RouteDomain:
    if path.startswith("/webhook/research") or path == "/webhook/research":
        return RouteDomain.CHAT
    if path in ("/webhook/ingest",):
        return RouteDomain.INGEST
    if path in ("/webhook/report",):
        return RouteDomain.REPORT
    if path in ("/query", "/webhook/query", "/webhook/chatbot"):
        return RouteDomain.CHAT
    if path in ("/ingest", "/webhook/ingest_pdf", "/upload_paper", "/upload_pdf"):
        return RouteDomain.INGEST
    if path in ("/report",):
        return RouteDomain.REPORT
    if path == "/discover":
        return RouteDomain.DISCOVER
    if path in ("/login", "/register", "/me"):
        return RouteDomain.AUTH
    return RouteDomain.OTHER
