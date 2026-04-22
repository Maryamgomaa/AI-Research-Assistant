"""
retrieval_metrics.py — Retrieval Evaluation
============================================
Computes information-retrieval metrics for the RAG pipeline's
document-retrieval stage.

Metrics:
    • Precision@K
    • Recall@K
    • MRR (Mean Reciprocal Rank)
    • Hit Rate @ K
"""

from __future__ import annotations

import logging
from typing import List

log = logging.getLogger("eval.retrieval_metrics")


def precision_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """
    Fraction of the top-K retrieved documents that are relevant.
    """
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / len(top_k)


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """
    Fraction of relevant documents found in the top-K retrieved results.
    """
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    hits = sum(1 for doc in top_k if doc in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(relevant: List[str], retrieved: List[str]) -> float:
    """
    Reciprocal of the rank of the first relevant document in the retrieved list.
    Returns 0 if no relevant document is found.
    """
    relevant_set = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    """
    Binary: 1.0 if at least one relevant document appears in top-K, else 0.0.
    """
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    return 1.0 if any(doc in relevant_set for doc in top_k) else 0.0


def average_precision(relevant: List[str], retrieved: List[str]) -> float:
    """
    Average Precision for a single query — area under the P-R curve
    computed at each recall change point.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    sum_prec = 0.0
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(relevant_set)


# ────────────────────────────────────────────────────────────────
# Batch evaluation
# ────────────────────────────────────────────────────────────────

def evaluate_retrieval(
    all_relevant: List[List[str]],
    all_retrieved: List[List[str]],
    k_values: List[int] | None = None,
) -> dict:
    """
    Evaluate retrieval quality over a batch of queries.

    Parameters
    ----------
    all_relevant  : list of lists — ground-truth relevant doc IDs per query.
    all_retrieved : list of lists — retrieved doc IDs (ranked) per query.
    k_values      : list of K values to evaluate (default [1, 3, 5, 10]).

    Returns
    -------
    dict with per-K and overall metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    assert len(all_relevant) == len(all_retrieved), "Mismatched input lengths"
    n = len(all_relevant)

    results: dict = {"num_queries": n, "k_values": k_values}

    # Per-K metrics
    for k in k_values:
        p_at_k = [precision_at_k(rel, ret, k) for rel, ret in zip(all_relevant, all_retrieved)]
        r_at_k = [recall_at_k(rel, ret, k)    for rel, ret in zip(all_relevant, all_retrieved)]
        hr_at_k = [hit_rate_at_k(rel, ret, k) for rel, ret in zip(all_relevant, all_retrieved)]

        results[f"precision@{k}"] = round(sum(p_at_k) / n, 4)
        results[f"recall@{k}"]    = round(sum(r_at_k) / n, 4)
        results[f"hit_rate@{k}"]  = round(sum(hr_at_k) / n, 4)

    # MRR
    rr_scores = [reciprocal_rank(rel, ret) for rel, ret in zip(all_relevant, all_retrieved)]
    results["mrr"] = round(sum(rr_scores) / n, 4)

    # MAP (Mean Average Precision)
    ap_scores = [average_precision(rel, ret) for rel, ret in zip(all_relevant, all_retrieved)]
    results["map"] = round(sum(ap_scores) / n, 4)

    # Per-query detail
    results["per_query"] = []
    for i, (rel, ret) in enumerate(zip(all_relevant, all_retrieved)):
        query_detail = {"query_index": i, "rr": round(rr_scores[i], 4), "ap": round(ap_scores[i], 4)}
        for k in k_values:
            query_detail[f"recall@{k}"]    = round(recall_at_k(rel, ret, k), 4)
            query_detail[f"precision@{k}"] = round(precision_at_k(rel, ret, k), 4)
        results["per_query"].append(query_detail)

    return results
