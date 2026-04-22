"""
semantic_similarity.py — Semantic Similarity Scoring
=====================================================
Uses SentenceTransformers to compute cosine similarity between
ground-truth and predicted answers at the embedding level.

This captures meaning equivalence that token-overlap metrics miss.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

log = logging.getLogger("eval.semantic_similarity")


def compute_semantic_similarity(
    ground_truths: List[str],
    predictions: List[str],
    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device: str = "cpu",
    batch_size: int = 16,
) -> dict:
    """
    Compute pairwise cosine similarity between ground-truth and predicted answers
    using a SentenceTransformer model.

    Parameters
    ----------
    ground_truths : list of ground-truth answer strings.
    predictions   : list of predicted answer strings.
    model_name    : HuggingFace model identifier.
    device        : 'cpu' or 'cuda'.
    batch_size    : encoding batch size.

    Returns
    -------
    dict with per-sample similarities and aggregate stats.
    """
    from sentence_transformers import SentenceTransformer

    assert len(ground_truths) == len(predictions), "Mismatched input lengths"
    n = len(ground_truths)

    log.info(f"[SEMANTIC] Loading model '{model_name}' on {device}...")
    model = SentenceTransformer(model_name, device=device)

    log.info(f"[SEMANTIC] Encoding {n} ground-truth answers...")
    gt_embeddings = model.encode(
        ground_truths,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    log.info(f"[SEMANTIC] Encoding {n} predicted answers...")
    pr_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Pairwise cosine similarity (vectors are already L2-normalised)
    similarities = [
        float(np.dot(gt_embeddings[i], pr_embeddings[i]))
        for i in range(n)
    ]

    return {
        "num_samples": n,
        "mean_similarity": round(float(np.mean(similarities)), 4),
        "median_similarity": round(float(np.median(similarities)), 4),
        "min_similarity": round(float(np.min(similarities)), 4),
        "max_similarity": round(float(np.max(similarities)), 4),
        "std_similarity": round(float(np.std(similarities)), 4),
        "per_sample_similarity": [round(s, 4) for s in similarities],
    }
