import json
import logging
import re
from typing import Optional

import httpx

from .config import Config
from .prompt_templates import PromptTemplates

log = logging.getLogger("llm_client")

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_groq import ChatGroq

    _LANGCHAIN_GROQ = True
except ImportError:
    _LANGCHAIN_GROQ = False


class OllamaClient:
    def __init__(
        self,
        base_url: str = Config.OLLAMA_BASE_URL,
        model: str = Config.OLLAMA_MODEL,
        temperature: float = Config.LLM_TEMPERATURE,
        max_tokens: int = Config.LLM_MAX_TOKENS,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        log.info(f"[LLM] Ollama client → {base_url}, model={model}")

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("response", "").strip()

    async def chat(self, messages: list[dict], system: Optional[str] = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["messages"] = [
                {"role": "system", "content": system}
            ] + messages

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "").strip()

    async def list_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
class GroqClient:
    def __init__(
        self,
        api_key: str = Config.GROQ_API_KEY,
        base_url: str = Config.GROQ_BASE_URL,
        model: str = Config.GROQ_MODEL,
        temperature: float = Config.LLM_TEMPERATURE,
        max_tokens: int = Config.LLM_MAX_TOKENS,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        backend = "langchain-groq" if _LANGCHAIN_GROQ else "httpx"
        log.info(f"[LLM] Groq client ({backend}) → {base_url}, model={model}")

    def _openai_messages(self, messages: list[dict], system: Optional[str]) -> list[dict]:
        out = list(messages)
        if system:
            out = [{"role": "system", "content": system}] + out
        return out

    async def _complete_httpx(self, messages: list[dict], stream: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    async def _complete_langchain(self, messages: list[dict]) -> str:
        if not (self.api_key or "").strip():
            raise ValueError("GROQ_API_KEY is not set")
        lc_msgs = []
        for m in messages:
            role, content = m.get("role", "user"), m.get("content", "")
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
            else:
                lc_msgs.append(HumanMessage(content=content))
        llm = ChatGroq(
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        resp = await llm.ainvoke(lc_msgs)
        return (getattr(resp, "content", None) or "").strip()

    async def _complete(self, messages: list[dict], stream: bool = False) -> str:
        if _LANGCHAIN_GROQ:
            if stream:
                log.warning("[LLM] stream=True ignored for LangChain Groq path")
            return await self._complete_langchain(messages)
        return await self._complete_httpx(messages, stream=stream)

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages.insert(0, {"role": "system", "content": system})
        return await self._complete(messages, stream=stream)

    async def chat(self, messages: list[dict], system: Optional[str] = None) -> str:
        messages = self._openai_messages(messages, system)
        return await self._complete(messages, stream=False)


class ResearchReportGenerator:
    def __init__(self, llm: Optional[GroqClient] = None):
        self.llm = llm or GroqClient()
        self.templates = PromptTemplates()

    async def generate_report(
        self,
        paper_meta: dict,
        full_text: str,
        topic: str = "",
    ) -> dict:
        truncated_text = full_text[:3000] if len(full_text) > 3000 else full_text
        prompt = self.templates.report_generation_prompt(
            title=paper_meta.get("title", ""),
            authors=paper_meta.get("authors", []),
            abstract=paper_meta.get("abstract", ""),
            text_excerpt=truncated_text,
            topic=topic,
        )
        system = self.templates.report_system_prompt()
        log.info(f"[REPORT] Generating report for {paper_meta.get('arxiv_id', '?')}...")
        raw = await self.llm.generate(prompt=prompt, system=system)
        report = self._parse_report(raw, paper_meta)
        report["raw_llm_response"] = raw
        return report

    def _parse_report(self, raw: str, paper_meta: dict) -> dict:
        clean = json.loads(raw) if raw.strip().startswith("{") else self._safe_json(raw)
        if not isinstance(clean, dict):
            clean = self._safe_json(raw)
        if not isinstance(clean, dict):
            return {
                "arxiv_id": paper_meta.get("arxiv_id", ""),
                "title": paper_meta.get("title", ""),
                "authors": paper_meta.get("authors", []),
                "problem": "Parse error — see raw_response",
                "methodology": "Not specified",
                "attention_mechanism": "Not specified",
                "dataset": "Not specified",
                "results": "Not specified",
                "contributions": "Not specified",
                "limitations": "Not specified",
                "arabic_nlp_relevance": "Not specified",
                "future_research": "Not specified",
                "keywords": [],
                "raw_response": raw,
                "pdf_url": paper_meta.get("pdf_url", ""),
                "abs_url": paper_meta.get("abs_url", ""),
                "published": paper_meta.get("published", ""),
            }
        clean["arxiv_id"] = paper_meta.get("arxiv_id", "")
        clean["pdf_url"] = paper_meta.get("pdf_url", "")
        clean["abs_url"] = paper_meta.get("abs_url", "")
        clean["published"] = paper_meta.get("published", "")
        return clean

    def _safe_json(self, raw: str) -> Optional[dict]:
        try:
            body = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    async def generate_topic_summary(self, topic: str, paper_reports: list[dict]) -> dict:
        prompt = self.templates.topic_summary_prompt(topic, paper_reports)
        system = self.templates.report_system_prompt()
        raw = await self.llm.generate(prompt=prompt, system=system)
        summary = self._safe_json(raw)
        if isinstance(summary, dict):
            summary["raw_llm_response"] = raw
            return summary
        return {
            "topic": topic,
            "overall_summary": "Could not parse the summary response.",
            "key_insights": [],
            "recommended_papers": [r.get("arxiv_id", "") for r in paper_reports[:3]],
            "research_trends": "Not specified",
            "next_steps": "Not specified",
            "raw_response": raw,
        }


class RAGAnsweringEngine:
    def __init__(self, llm: Optional[GroqClient] = None):
        self.llm = llm or GroqClient()
        self.templates = PromptTemplates()

    async def answer(
        self,
        question: str,
        retrieved_chunks: list[dict],
        language: str = "auto",
    ) -> dict:
        if language == "auto":
            language = "ar" if self._has_arabic(question) else "en"

        context_text = self._format_context(retrieved_chunks)
        prompt = self.templates.rag_prompt(
            question=question,
            context=context_text,
            language=language,
        )
        system = self.templates.rag_system_prompt(language=language)
        log.info(
            f"[RAG] Answering: '{question[:60]}...' | lang={language} | chunks={len(retrieved_chunks)}"
        )
        answer_text = await self.llm.generate(prompt=prompt, system=system)
        return {
            "question": question,
            "answer": answer_text,
            "language": language,
            "sources": self._extract_sources(retrieved_chunks),
            "context_chunks_used": len(retrieved_chunks),
        }

    def _format_context(self, chunks: list[dict]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "Unknown")
            arxiv_id = chunk.get("arxiv_id", "")
            text = chunk.get("text", "")
            score = chunk.get("score", 0)
            parts.append(
                f"[{i}] Source: {title} (arXiv:{arxiv_id}, relevance={score:.2f})\n{text}"
            )
        return "\n\n".join(parts)

    def _extract_sources(self, chunks: list[dict]) -> list[dict]:
        seen = set()
        sources = []
        for chunk in chunks:
            aid = chunk.get("arxiv_id", "")
            if aid and aid not in seen:
                seen.add(aid)
                sources.append({
                    "arxiv_id": aid,
                    "title": chunk.get("title", ""),
                    "authors": chunk.get("authors", []),
                    "published": chunk.get("published", ""),
                    "abs_url": chunk.get("abs_url", ""),
                    "relevance_score": chunk.get("score", 0),
                })
        return sources

    @staticmethod
    def _has_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text))
