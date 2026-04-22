import uuid
import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
)

from .config import Config

log = logging.getLogger("qdrant_manager")


class QdrantManager:
    def __init__(
        self,
        host: str = Config.QDRANT_HOST,
        port: int = Config.QDRANT_PORT,
        url: str = Config.QDRANT_URL,
        api_key: str = Config.QDRANT_API_KEY,
    ):
        # Prefer cloud URL if set, otherwise fall back to local host:port
        if url:
            self.client = QdrantClient(url=url, api_key=api_key or None)
            log.info(f"[QDRANT] Connected to cloud: {url}")
        else:
            self.client = QdrantClient(host=host, port=port)
            log.info(f"[QDRANT] Connected to local: {host}:{port}")

    def ensure_collection(
        self,
        collection: str = Config.QDRANT_COLLECTION,
        vector_size: int = Config.QDRANT_VECTOR_SIZE,
    ) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if collection in existing:
            log.info(f"[QDRANT] Collection '{collection}' already exists")
            return

        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10_000,
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10_000,
                memmap_threshold=50_000,
            ),
        )
        log.info(
            f"[QDRANT] Created collection '{collection}' (dim={vector_size}, distance=COSINE)"
        )

    def create_payload_indexes(self, collection: str = Config.QDRANT_COLLECTION) -> None:
        fields_to_index = ["arxiv_id", "primary_category", "published_year"]
        for field in fields_to_index:
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema="keyword",
                )
                log.info(f"[QDRANT] Indexed payload field: {field}")
            except Exception as e:
                log.warning(f"[QDRANT] Index creation for '{field}' skipped: {e}")

    async def upsert_chunks(
        self,
        collection: str,
        chunks: list,
        vectors: list[list[float]],
        metadata: dict,
    ) -> list[str]:
        self.ensure_collection(collection, vector_size=len(vectors[0]))

        points = []
        point_ids = []

        for chunk, vector in zip(chunks, vectors):
            if hasattr(chunk, "to_dict"):
                chunk_data = chunk.to_dict()
            elif isinstance(chunk, dict):
                chunk_data = chunk
            else:
                chunk_data = {"text": str(chunk), "chunk_index": 0}

            point_id = str(uuid.uuid4())
            point_ids.append(point_id)

            payload = {
                "text": chunk_data.get("text", ""),
                "chunk_index": chunk_data.get("chunk_index", 0),
                "char_start": chunk_data.get("char_start", 0),
                "char_end": chunk_data.get("char_end", 0),
                "has_arabic": chunk_data.get("has_arabic", False),
                "arxiv_id": metadata.get("arxiv_id", ""),
                "title": metadata.get("title", ""),
                "authors": metadata.get("authors", []),
                "abstract": metadata.get("abstract", ""),
                "published": metadata.get("published", ""),
                "pdf_url": metadata.get("pdf_url", ""),
                "abs_url": metadata.get("abs_url", ""),
                "primary_category": metadata.get("primary_category", ""),
                "categories": metadata.get("categories", []),
                "published_year": (metadata.get("published", "") or "")[:4],
            }

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection, points=batch)
            log.debug(f"[QDRANT] Upserted batch {i//batch_size + 1} ({len(batch)} points)")

        log.info(
            f"[QDRANT] ✓ Upserted {len(points)} chunks for arXiv:{metadata.get('arxiv_id', '?')} into '{collection}'"
        )
        return point_ids

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        arxiv_id_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        score_threshold: float = 0.35,
    ) -> list[dict]:
        query_filter = None
        conditions = []

        if arxiv_id_filter:
            conditions.append(
                FieldCondition(key="arxiv_id", match=MatchValue(value=arxiv_id_filter))
            )
        if category_filter:
            conditions.append(
                FieldCondition(key="primary_category", match=MatchValue(value=category_filter))
            )
        if conditions:
            query_filter = Filter(must=conditions)

        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
            search_params=SearchParams(hnsw_ef=64, exact=False),
        )

        return [
            {
                "score": round(r.score, 4),
                "text": r.payload.get("text", ""),
                "arxiv_id": r.payload.get("arxiv_id", ""),
                "title": r.payload.get("title", ""),
                "authors": r.payload.get("authors", []),
                "chunk_index": r.payload.get("chunk_index", 0),
                "published": r.payload.get("published", ""),
                "abs_url": r.payload.get("abs_url", ""),
                "primary_category": r.payload.get("primary_category", ""),
            }
            for r in results
        ]

    def get_collection_info(self, collection: str = Config.QDRANT_COLLECTION) -> dict:
        info = self.client.get_collection(collection)
        return {
            "name": collection,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }
