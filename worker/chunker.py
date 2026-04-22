import re
import logging
from dataclasses import dataclass

log = logging.getLogger("chunker")

@dataclass
class TextChunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    word_count: int = 0
    has_arabic: bool = False

    def __post_init__(self):
        self.word_count = len(self.text.split())
        self.has_arabic = bool(re.search(r"[\u0600-\u06FF]", self.text))

    def to_dict(self) -> dict:
        return self.__dict__


_SENT_END_RE = re.compile(
    r"(?<=[.!?؟।])\s+"
    r"|(?<=[\u060C\u061B])\s+"
    r"|\n{2,}"
)


class TextChunker:
    def __init__(self, chunk_size: int = 700, overlap: int = 120):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[TextChunk]:
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        chunks = self._pack_into_chunks(sentences)

        log.debug(
            f"[CHUNK] {len(text)} chars → {len(sentences)} sentences → {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.overlap})"
        )
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        parts = _SENT_END_RE.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _pack_into_chunks(self, sentences: list[str]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        current_sentences: list[str] = []
        current_len = 0
        char_cursor = 0

        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                char_start = max(0, char_cursor - current_len)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=len(chunks),
                    char_start=char_start,
                    char_end=char_start + len(chunk_text),
                ))
                overlap_sents = self._get_overlap_sentences(current_sentences)
                current_sentences = overlap_sents
                current_len = sum(len(s) for s in overlap_sents) + len(overlap_sents)

            current_sentences.append(sent)
            current_len += sent_len + 1
            char_cursor += sent_len + 1

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            char_start = max(0, char_cursor - current_len)
            chunks.append(TextChunk(
                text=chunk_text,
                chunk_index=len(chunks),
                char_start=char_start,
                char_end=char_start + len(chunk_text),
            ))

        return chunks

    def _get_overlap_sentences(self, sentences: list[str]) -> list[str]:
        overlap_sents = []
        total = 0
        for sent in reversed(sentences):
            if total + len(sent) > self.overlap:
                break
            overlap_sents.insert(0, sent)
            total += len(sent) + 1
        return overlap_sents

    def chunk_with_metadata(self, text: str, paper_meta: dict) -> list[dict]:
        chunks = self.chunk(text)
        return [
            {
                **chunk.to_dict(),
                "paper": paper_meta,
            }
            for chunk in chunks
        ]
