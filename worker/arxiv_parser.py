import re
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

import httpx

from .config import Config

log = logging.getLogger("arxiv_parser")

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "openSearch": "http://a9.com/-/spec/opensearch/1.1/",
}


@dataclass
class PaperMetadata:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    categories: list[str]
    pdf_url: str
    abs_url: str
    primary_category: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None

    def to_dict(self) -> dict:
        return self.__dict__


class ArxivParser:
    BASE_URL = Config.ARXIV_BASE_URL

    ARABIC_TERM_MAP = {
        "الانتباه": "attention",
        "معالجة اللغة الطبيعية": "natural language processing NLP",
        "التعلم العميق": "deep learning",
        "الشبكات العصبية": "neural networks",
        "تعلم الآلة": "machine learning",
        "التحويل": "transformer",
        "التضمين": "embedding",
        "المشاعر": "sentiment analysis",
        "التصنيف": "text classification",
        "الترجمة": "machine translation",
        "العربية": "Arabic language",
        "الأسئلة والأجوبة": "question answering",
        "الملخص": "summarization",
        "التسمية التسلسلية": "sequence labeling NER",
    }

    def translate_arabic_query(self, query: str) -> str:
        translated = query
        for arabic, english in self.ARABIC_TERM_MAP.items():
            if arabic in translated:
                translated = translated.replace(arabic, english)
                log.info(f"[TRANSLATE] '{arabic}' → '{english}'")
        return translated

    def build_query(
        self,
        topic: str,
        max_results: int = Config.ARXIV_MAX_RESULTS,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
        categories: Optional[list[str]] = None,
    ) -> str:
        has_arabic = bool(re.search(r"[\u0600-\u06FF]", topic))
        if has_arabic:
            log.info(f"[QUERY] Arabic query detected: '{topic}'")
            topic = self.translate_arabic_query(topic)
            log.info(f"[QUERY] Translated: '{topic}'")

        terms = self._expand_topic(topic)
        query_parts = [f"all:{t}" for t in terms]
        if categories:
            cat_filter = " OR ".join(f"cat:{c}" for c in categories)
            query_parts.append(f"({cat_filter})")

        search_query = " AND ".join(query_parts)
        url = (
            f"{self.BASE_URL}"
            f"?search_query={search_query}"
            f"&sortBy={sort_by}"
            f"&sortOrder={sort_order}"
            f"&max_results={max_results}"
        )
        log.info(f"[QUERY] Built URL: {url}")
        return url

    def _expand_topic(self, topic: str) -> list[str]:
        topic_clean = re.sub(r"[^\w\s]", " ", topic.lower()).strip()
        words = topic_clean.split()

        PHRASES = [
            "natural language processing",
            "machine translation",
            "sentiment analysis",
            "named entity recognition",
            "question answering",
            "text classification",
            "language model",
            "attention mechanism",
            "transformer model",
            "word embedding",
            "Arabic NLP",
            "Arabic language",
        ]

        terms = []
        remaining = topic_clean
        for phrase in PHRASES:
            if phrase in remaining:
                terms.append(phrase.replace(" ", "+"))
                remaining = remaining.replace(phrase, "")

        STOPWORDS = {"the", "a", "an", "of", "in", "for", "and", "or", "to", "with"}
        for w in remaining.split():
            if w not in STOPWORDS and len(w) > 2:
                terms.append(w)

        return terms if terms else [topic_clean.replace(" ", "+")]

    def parse_response(self, xml_text: str) -> list[PaperMetadata]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            raise ValueError(f"Malformed arXiv XML: {e}") from e

        entries = root.findall("atom:entry", NS)
        if not entries:
            log.warning("[PARSE] No entries found in arXiv response")
            return []

        papers = []
        for entry in entries:
            try:
                paper = self._parse_entry(entry)
                papers.append(paper)
            except Exception as e:
                log.warning(f"[PARSE] Skipping entry due to error: {e}")

        log.info(f"[PARSE] Parsed {len(papers)} papers from arXiv response")
        return papers

    def _parse_entry(self, entry: ET.Element) -> PaperMetadata:
        def text(tag: str, default: str = "") -> str:
            el = entry.find(tag, NS)
            return (el.text or "").strip() if el is not None else default

        raw_id = text("atom:id")
        arxiv_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id

        authors = [
            a.find("atom:name", NS).text.strip()
            for a in entry.findall("atom:author", NS)
            if a.find("atom:name", NS) is not None
        ]

        categories = [
            c.get("term", "")
            for c in entry.findall("atom:category", NS)
        ]

        pdf_url = ""
        abs_url = ""
        for link in entry.findall("atom:link", NS):
            href = link.get("href", "")
            rel = link.get("rel", "")
            title = link.get("title", "")
            if title == "pdf" or (rel == "related" and "pdf" in href):
                pdf_url = href
            elif rel == "alternate":
                abs_url = href

        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/") + ".pdf"

        primary_cat = ""
        pc_el = entry.find("arxiv:primary_category", NS)
        if pc_el is not None:
            primary_cat = pc_el.get("term", "")

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=text("atom:title").replace("\n", " "),
            authors=authors,
            abstract=text("atom:summary").replace("\n", " "),
            published=text("atom:published"),
            updated=text("atom:updated"),
            categories=categories,
            pdf_url=pdf_url,
            abs_url=abs_url,
            primary_category=primary_cat,
            comment=text("arxiv:comment") or None,
            journal_ref=text("arxiv:journal_ref") or None,
            doi=text("arxiv:doi") or None,
        )

    async def fetch_papers(
        self,
        topic: str,
        max_results: int = Config.ARXIV_MAX_RESULTS,
        categories: Optional[list[str]] = None,
        sort_by: str = "submittedDate",
        sort_order: str = "descending",
    ) -> list[PaperMetadata]:
        url = self.build_query(
            topic,
            max_results=max_results,
            sort_by=sort_by,
            sort_order=sort_order,
            categories=categories,
        )
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        return self.parse_response(resp.text)

    async def validate_pdf_access(self, pdf_url: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.head(pdf_url)
                ct = resp.headers.get("content-type", "")
                is_pdf = "pdf" in ct or resp.status_code == 200
                log.info(f"[VALIDATE] {pdf_url} → {resp.status_code} ({ct}) → {'✓' if resp.status_code == 200 else '✗'}")
                return resp.status_code == 200 and is_pdf
        except Exception as e:
            log.warning(f"[VALIDATE] HEAD request failed for {pdf_url}: {e}")
            return False
