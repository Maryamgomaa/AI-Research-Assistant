import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Any

# if TYPE_CHECKING:
#     from sentence_transformers import SentenceTransformer

import numpy as np
# from sentence_transformers import SentenceTransformer  # Lazy import

from .chunker import TextChunk
from .config import Config

log = logging.getLogger("embedder")

_THREAD_POOL = ThreadPoolExecutor(max_workers=2)


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = Config.EMBEDDING_MODEL,
        device: str = Config.EMBEDDING_DEVICE,
        batch_size: int = Config.EMBEDDING_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        log.info(f"[EMBED] Initialised with model={model_name} device={device}")

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # Lazy import
            log.info(f"[EMBED] Loading model '{self.model_name}' on {self.device}...")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            log.info(
                f"[EMBED] Model loaded. Vector dim = {self._model.get_sentence_embedding_dimension()}"
            )
        return self._model

    @property
    def vector_dim(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    async def embed_batch(
        self, texts: list[Union[str, TextChunk]]
    ) -> list[list[float]]:
        str_texts = [
            t.text if isinstance(t, TextChunk) else t
            for t in texts
        ]
        if not str_texts:
            return []
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            _THREAD_POOL,
            self._encode_sync,
            str_texts,
        )
        return [v.tolist() for v in vectors]

    def _encode_sync(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors

    def embed_query(self, query: str) -> list[float]:
        vec = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def warmup(self) -> None:
        _ = self.embed_query("warmup sentence for model loading")
        log.info("[EMBED] Warmup complete")

    def get_model_info(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "vector_dim": self.vector_dim,
            "batch_size": self.batch_size,
        }
