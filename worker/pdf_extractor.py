import re
import logging
from pathlib import Path

import fitz  # PyMuPDF

log = logging.getLogger("pdf_extractor")

_PAGE_NUM_RE = re.compile(r"^\s*[\d]+\s*$")
_ARXIV_HEADER_RE = re.compile(
    r"arXiv:\d{4}\.\d{4,5}v\d|Preprint|Submitted to|Under review",
    re.IGNORECASE,
)
_SHORT_LINE_RE = re.compile(r"^.{1,4}$")


class PDFExtractor:
    def __init__(
        self,
        remove_headers_footers: bool = True,
        join_hyphenated: bool = True,
        preserve_arabic: bool = True,
    ):
        self.remove_headers_footers = remove_headers_footers
        self.join_hyphenated = join_hyphenated
        self.preserve_arabic = preserve_arabic

    def extract(self, pdf_path: Path | str) -> str:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        log.info(f"[EXTRACT] Opening {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Cannot open PDF {pdf_path}: {e}") from e

        page_texts = []
        for page_num, page in enumerate(doc):
            try:
                page_texts.append(self._extract_page(page, page_num))
            except Exception as e:
                log.warning(f"[EXTRACT] Error on page {page_num}: {e}")

        doc.close()

        full_text = "\n\n".join(t for t in page_texts if t.strip())
        full_text = self._post_process(full_text)

        log.info(
            f"[EXTRACT] {pdf_path.name}: {len(page_texts)} pages → {len(full_text)} chars"
        )
        return full_text

    def _extract_page(self, page: fitz.Page, page_num: int) -> str:
        blocks = page.get_text("blocks", sort=True)
        lines = []

        for block in blocks:
            if block[6] != 0:
                continue
            text = block[4].strip()
            if not text:
                continue
            if self.remove_headers_footers and page_num > 0:
                if _ARXIV_HEADER_RE.search(text):
                    continue
                if _PAGE_NUM_RE.match(text):
                    continue
            lines.append(text)

        return "\n".join(lines)

    def _post_process(self, text: str) -> str:
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if _PAGE_NUM_RE.match(stripped):
                continue
            if _SHORT_LINE_RE.match(stripped) and not self._has_arabic(stripped):
                continue
            cleaned_lines.append(stripped)

        text = "\n".join(cleaned_lines)

        if self.join_hyphenated:
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        if self.preserve_arabic:
            text = re.sub(r"[\u064B-\u065F\u0670]", "", text)

        return text.strip()

    @staticmethod
    def _has_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text))

    def extract_metadata_from_pdf(self, pdf_path: Path | str) -> dict:
        doc = fitz.open(str(pdf_path))
        meta = doc.metadata
        doc.close()
        return {
            "pdf_title": meta.get("title", ""),
            "pdf_author": meta.get("author", ""),
            "pdf_subject": meta.get("subject", ""),
            "pdf_creator": meta.get("creator", ""),
            "pdf_pages": doc.page_count if not doc.is_closed else None,
        }
