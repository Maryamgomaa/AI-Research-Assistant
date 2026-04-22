"""
answer_metrics.py — Answer-Quality Evaluation
==============================================
Computes classification-style metrics and text-overlap scores for
predicted vs ground-truth answers.

Metrics:
    • Token-level Precision / Recall / F1
    • Exact-match Accuracy
    • ROUGE-1, ROUGE-2, ROUGE-L
    • BLEU (sentence-level)
    • Confusion-matrix data (per-token binary match)
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import List, Tuple

import numpy as np

log = logging.getLogger("eval.answer_metrics")

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase + split on non-alphanumeric characters."""
    return re.findall(r"\w+", text.lower())


def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """Return list of n-grams from token list."""
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ────────────────────────────────────────────────────────────────
# Core classification-style metrics (token-level)
# ────────────────────────────────────────────────────────────────

def token_precision_recall_f1(
    ground_truth: str,
    predicted: str,
) -> dict:
    """
    Treat each *unique* token in the ground truth as a positive label.
    TP = tokens present in both; FP = predicted-only; FN = ground-truth-only.
    """
    gt_tokens = set(_tokenize(ground_truth))
    pr_tokens = set(_tokenize(predicted))

    tp = len(gt_tokens & pr_tokens)
    fp = len(pr_tokens - gt_tokens)
    fn = len(gt_tokens - pr_tokens)
    tn = 0  # not meaningful in open-domain text, kept for confusion matrix

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def exact_match(ground_truth: str, predicted: str) -> bool:
    """Case-insensitive exact match after whitespace normalisation."""
    return " ".join(_tokenize(ground_truth)) == " ".join(_tokenize(predicted))


# ────────────────────────────────────────────────────────────────
# ROUGE
# ────────────────────────────────────────────────────────────────

def _rouge_n(gt_tokens: List[str], pr_tokens: List[str], n: int) -> dict:
    """Compute ROUGE-N precision, recall, f1."""
    gt_ngrams = Counter(_ngrams(gt_tokens, n))
    pr_ngrams = Counter(_ngrams(pr_tokens, n))

    overlap = sum((gt_ngrams & pr_ngrams).values())
    gt_total = sum(gt_ngrams.values())
    pr_total = sum(pr_ngrams.values())

    precision = overlap / pr_total if pr_total else 0.0
    recall    = overlap / gt_total if gt_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def _lcs_length(x: List[str], y: List[str]) -> int:
    """Longest common subsequence length (DP)."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(ground_truth: str, predicted: str) -> dict:
    """ROUGE-L via longest common subsequence."""
    gt_tokens = _tokenize(ground_truth)
    pr_tokens = _tokenize(predicted)
    if not gt_tokens or not pr_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(gt_tokens, pr_tokens)
    precision = lcs / len(pr_tokens) if pr_tokens else 0.0
    recall    = lcs / len(gt_tokens) if gt_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def compute_rouge(ground_truth: str, predicted: str) -> dict:
    gt_tokens = _tokenize(ground_truth)
    pr_tokens = _tokenize(predicted)
    return {
        "rouge_1": _rouge_n(gt_tokens, pr_tokens, 1),
        "rouge_2": _rouge_n(gt_tokens, pr_tokens, 2),
        "rouge_l": rouge_l(ground_truth, predicted),
    }


# ────────────────────────────────────────────────────────────────
# BLEU (sentence-level, up to 4-gram)
# ────────────────────────────────────────────────────────────────

def compute_bleu(ground_truth: str, predicted: str, max_n: int = 4) -> float:
    """
    Sentence-level BLEU with brevity penalty.
    Reference = ground_truth; Candidate = predicted.
    """
    ref_tokens  = _tokenize(ground_truth)
    cand_tokens = _tokenize(predicted)

    if not cand_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, np.exp(1 - len(ref_tokens) / len(cand_tokens)))

    precisions = []
    for n in range(1, max_n + 1):
        ref_ng  = Counter(_ngrams(ref_tokens, n))
        cand_ng = Counter(_ngrams(cand_tokens, n))
        clipped = sum((ref_ng & cand_ng).values())
        total   = sum(cand_ng.values())
        precisions.append(clipped / total if total else 0.0)

    # Geometric mean (add smoothing to avoid log(0))
    log_avg = sum(np.log(p + 1e-12) for p in precisions) / max_n
    bleu = bp * np.exp(log_avg)
    return round(float(bleu), 4)


# ────────────────────────────────────────────────────────────────
# Batch evaluation
# ────────────────────────────────────────────────────────────────

def evaluate_answers(
    ground_truths: List[str],
    predictions: List[str],
) -> dict:
    """
    Evaluate a batch of (ground_truth, predicted) answer pairs.
    Returns aggregated metrics + per-sample detail.
    """
    assert len(ground_truths) == len(predictions), "Mismatched input lengths"

    per_sample = []
    total_tp = total_fp = total_fn = total_tn = 0
    exact_matches = 0

    for gt, pr in zip(ground_truths, predictions):
        prf = token_precision_recall_f1(gt, pr)
        rouge = compute_rouge(gt, pr)
        bleu  = compute_bleu(gt, pr)
        em    = exact_match(gt, pr)

        total_tp += prf["tp"]
        total_fp += prf["fp"]
        total_fn += prf["fn"]
        total_tn += prf["tn"]
        exact_matches += int(em)

        per_sample.append({
            "precision": prf["precision"],
            "recall":    prf["recall"],
            "f1":        prf["f1"],
            "exact_match": em,
            "rouge": rouge,
            "bleu":  bleu,
        })

    n = len(ground_truths)

    # Micro-averaged metrics (from summed TP/FP/FN)
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) else 0.0

    # Macro-averaged metrics
    macro_p  = sum(s["precision"] for s in per_sample) / n
    macro_r  = sum(s["recall"]    for s in per_sample) / n
    macro_f1 = sum(s["f1"]        for s in per_sample) / n

    return {
        "num_samples": n,
        "accuracy_exact_match": round(exact_matches / n, 4),
        "micro": {
            "precision": round(micro_p, 4),
            "recall":    round(micro_r, 4),
            "f1":        round(micro_f1, 4),
        },
        "macro": {
            "precision": round(macro_p, 4),
            "recall":    round(macro_r, 4),
            "f1":        round(macro_f1, 4),
        },
        "confusion_matrix": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "tn": total_tn,
        },
        "avg_rouge_1_f1": round(sum(s["rouge"]["rouge_1"]["f1"] for s in per_sample) / n, 4),
        "avg_rouge_2_f1": round(sum(s["rouge"]["rouge_2"]["f1"] for s in per_sample) / n, 4),
        "avg_rouge_l_f1": round(sum(s["rouge"]["rouge_l"]["f1"] for s in per_sample) / n, 4),
        "avg_bleu":       round(sum(s["bleu"] for s in per_sample) / n, 4),
        "per_sample": per_sample,
    }
