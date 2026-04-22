"""
Multi-agent system — agent identities, roles, and where they are implemented.

This module is the single source of truth for agent metadata exposed via
GET /multi-agent/architecture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class AgentProfile:
    """One logical agent in the Athena Research multi-agent layout."""

    id: str
    display_name: str
    responsibilities: str
    implements_in: List[str] = field(default_factory=list)
    coordinates_with: List[str] = field(default_factory=list)


AGENTS: Tuple[AgentProfile, ...] = (
    AgentProfile(
        id="coordinator",
        display_name="Session Coordinator",
        responsibilities=(
            "Accepts user-facing API requests, records the active workflow, and delegates "
            "work to specialist agents. Does not call an LLM itself."
        ),
        implements_in=[
            "multi_agent/orchestrator.py",
            "worker/main.py (FastAPI route handlers; uvicorn worker.main:app)",
        ],
        coordinates_with=[
            "research_analyst",
            "platform_assistant",
            "discovery_librarian",
            "paper_analyst",
            "topic_synthesizer",
            "rag_specialist",
        ],
    ),
    AgentProfile(
        id="research_analyst",
        display_name="Research Analyst",
        responsibilities=(
            "Answers open-ended research questions using the user’s session context "
            "(discovery results, pasted notes, reference PDF names, on-screen reports). "
            "Uses Groq (or forwarded n8n workflow) as the reasoning backend."
        ),
        implements_in=[
            "worker/main.py:webhook_research",
            "multi_agent/orchestrator.py:run_research_turn",
        ],
        coordinates_with=["coordinator"],
    ),
    AgentProfile(
        id="platform_assistant",
        display_name="Platform & Methodology Assistant",
        responsibilities=(
            "Helps with search strategy, prompt design, n8n/Groq usage, and product features. "
            "Separate system prompt from the Research Analyst."
        ),
        implements_in=[
            "worker/main.py:webhook_chatbot",
            "multi_agent/orchestrator.py:run_assistant_turn",
        ],
        coordinates_with=["coordinator"],
    ),
    AgentProfile(
        id="discovery_librarian",
        display_name="Discovery Librarian",
        responsibilities=(
            "Queries arXiv, validates PDF accessibility, and optionally persists candidates to SQLite. "
            "Deterministic / retrieval step — no generative LLM in this path."
        ),
        implements_in=[
            "worker/main.py:discover_endpoint",
            "ArxivParser / SerpapiSearcher",
        ],
        coordinates_with=["coordinator", "paper_analyst"],
    ),
    AgentProfile(
        id="paper_analyst",
        display_name="Per-Paper Analyst",
        responsibilities=(
            "For each selected paper, produces structured JSON analysis (problem, methodology, "
            "results, limitations, etc.) via Groq (ResearchReportGenerator / LangChain path)."
        ),
        implements_in=[
            "worker/main.py:report_endpoint (ResearchReportGenerator)",
        ],
        coordinates_with=["coordinator", "discovery_librarian", "topic_synthesizer"],
    ),
    AgentProfile(
        id="topic_synthesizer",
        display_name="Topic Synthesizer",
        responsibilities=(
            "After all per-paper analyses complete, aggregates them into a cross-paper topic summary "
            "(themes, trends, next steps)."
        ),
        implements_in=[
            "worker/main.py:report_endpoint",
        ],
        coordinates_with=["coordinator", "paper_analyst"],
    ),
    AgentProfile(
        id="rag_specialist",
        display_name="RAG Query Specialist",
        responsibilities=(
            "Optional path: retrieves chunks from the Jetson/Qdrant worker and answers strictly "
            "from retrieved context (distinct system prompt from Research Analyst)."
        ),
        implements_in=[
            "worker/main.py:query_endpoint",
        ],
        coordinates_with=["coordinator"],
    ),
)


def agents_manifest() -> dict:
    """JSON-serializable description for GET /multi-agent/architecture."""
    return {
        "multi_agent_architecture": True,
        "pattern": "coordinator_delegation_with_specialists",
        "description": (
            "The API acts as a coordinator: each route or sub-routine maps to a specialist agent "
            "with a defined role. LLM calls use Groq (via LangChain when installed); "
            "other steps use retrieval or HTTP only."
        ),
        "agents": [
            {
                "id": a.id,
                "display_name": a.display_name,
                "responsibilities": a.responsibilities,
                "implements_in": a.implements_in,
                "coordinates_with": a.coordinates_with,
            }
            for a in AGENTS
        ],
        "interaction_flows": [
            {
                "name": "Research Q&A (UI thread)",
                "steps": [
                    "User → Coordinator (POST /webhook/research)",
                    "Coordinator → Research Analyst (build messages + system prompt)",
                    "Research Analyst → Groq (LangChain) or n8n → response → User",
                ],
            },
            {
                "name": "Assistant (floating chat)",
                "steps": [
                    "User → Coordinator (POST /webhook/chatbot)",
                    "Coordinator → Platform Assistant → Groq (LangChain) or n8n → User",
                ],
            },
            {
                "name": "Discover → Ingest → Report",
                "steps": [
                    "User → Coordinator (POST /discover) → Discovery Librarian → paper list",
                    "User selects papers → POST /ingest (persistence / worker)",
                    "User → Coordinator (POST /report) → parallel Paper Analyst agents (asyncio.gather)",
                    "Coordinator → Topic Synthesizer → combined JSON to UI",
                ],
            },
            {
                "name": "Workspace RAG (optional)",
                "steps": [
                    "User → Coordinator (POST /query) → RAG Specialist → Qdrant + Groq (LangChain)",
                ],
            },
        ],
    }
