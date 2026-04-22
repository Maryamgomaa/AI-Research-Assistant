"""
Runtime coordination: handoff trace for observability and assessment.

The coordinator does not invoke models directly; it delegates to the same callables
used in worker/main.py while exposing structure for traces and assessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional


@dataclass
class AgentHandoff:
    step: int
    from_agent: Optional[str]
    to_agent: str
    action: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "action": self.action,
            "detail": self.detail,
        }


@dataclass
class AgentRunTrace:
    """Per-request trace of coordinator → specialist delegation."""

    handoffs: List[AgentHandoff] = field(default_factory=list)
    _step: int = 0

    def handoff(
        self,
        to_agent: str,
        action: str,
        *,
        from_agent: Optional[str] = "coordinator",
        detail: str = "",
    ) -> None:
        self._step += 1
        self.handoffs.append(
            AgentHandoff(
                step=self._step,
                from_agent=from_agent,
                to_agent=to_agent,
                action=action,
                detail=detail[:500] if detail else "",
            )
        )

    def to_payload(self) -> dict:
        return {
            "architecture": "coordinator_delegation",
            "handoffs": [h.as_dict() for h in self.handoffs],
        }


async def run_research_turn(
    trace: AgentRunTrace,
    question_preview: str,
    execute_llm: Callable[[], Awaitable[str]],
) -> tuple[str, dict]:
    """
    Coordinator receives a research question and delegates to the Research Analyst.

    execute_llm: zero-arg async callable that performs the existing Groq/n8n path.
    """
    trace.handoff(
        "research_analyst",
        "delegate_question",
        detail=question_preview,
    )
    output = await execute_llm()
    trace.handoff(
        "coordinator",
        "receive_answer_from_analyst",
        from_agent="research_analyst",
        detail="response_ready",
    )
    return output, trace.to_payload()


async def run_assistant_turn(
    trace: AgentRunTrace,
    message_preview: str,
    execute_llm: Callable[[], Awaitable[str]],
) -> tuple[str, dict]:
    """Coordinator delegates to the Platform Assistant agent."""
    trace.handoff(
        "platform_assistant",
        "delegate_message",
        detail=message_preview,
    )
    output = await execute_llm()
    trace.handoff(
        "platform_assistant",
        "return_answer",
        from_agent="platform_assistant",
        to_agent="coordinator",
        detail="response_ready",
    )
    return output, trace.to_payload()


def trace_report_pipeline(
    trace: AgentRunTrace,
    num_papers: int,
    topic: str,
) -> None:
    """Record logical parallel analysts + synthesizer (actual LLM calls remain in backend)."""
    trace.handoff(
        "paper_analyst",
        "parallel_per_paper_analysis",
        detail=f"papers={num_papers} topic={topic[:120]}",
    )
    trace.handoff(
        "topic_synthesizer",
        "aggregate_reports",
        from_agent="paper_analyst",
        detail="after per-paper JSON analyses",
    )
    trace.handoff(
        "coordinator",
        "return_report_bundle",
        from_agent="topic_synthesizer",
        detail="reports + topic_summary",
    )


def trace_discover(trace: AgentRunTrace, topic: str) -> None:
    trace.handoff(
        "discovery_librarian",
        "search_and_validate",
        detail=topic[:200],
    )
