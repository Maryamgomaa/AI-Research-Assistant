"""
plot_results.py — Visualisation Utilities
==========================================
Generates and saves evaluation plots as PNG files:
    • Precision-Recall curve (per-sample token-level P/R)
    • Confusion matrix heatmap
    • Semantic similarity distribution
    • Retrieval Recall@K bar chart
    • ROUGE / BLEU score comparison bar chart
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List

import numpy as np

log = logging.getLogger("eval.plot_results")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_precision_recall_curve(
    precisions: List[float],
    recalls: List[float],
    output_dir: str = "evaluation/plots",
    filename: str = "precision_recall_curve.png",
) -> str:
    """Scatter plot of per-sample precision vs recall with a trend line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(output_dir)
    filepath = os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Sort by recall for a cleaner line
    pairs = sorted(zip(recalls, precisions))
    sorted_recalls = [p[0] for p in pairs]
    sorted_precisions = [p[1] for p in pairs]

    ax.plot(sorted_recalls, sorted_precisions, "o-", color="#4F46E5", linewidth=2, markersize=8,
            markerfacecolor="#818CF8", markeredgecolor="#312E81", markeredgewidth=1.5)
    ax.fill_between(sorted_recalls, sorted_precisions, alpha=0.15, color="#818CF8")

    ax.set_xlabel("Recall", fontsize=13, fontweight="bold")
    ax.set_ylabel("Precision", fontsize=13, fontweight="bold")
    ax.set_title("Precision–Recall Curve (Token-Level)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[PLOT] Saved Precision-Recall curve → {filepath}")
    return filepath


def plot_confusion_matrix(
    tp: int, fp: int, fn: int, tn: int,
    output_dir: str = "evaluation/plots",
    filename: str = "confusion_matrix.png",
) -> str:
    """Heatmap of the token-level confusion matrix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(output_dir)
    filepath = os.path.join(output_dir, filename)

    matrix = np.array([[tp, fp], [fn, tn]])
    labels = np.array([
        [f"TP\n{tp}", f"FP\n{fp}"],
        [f"FN\n{fn}", f"TN\n{tn}"],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.cm.Blues
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    # Text annotations
    for i in range(2):
        for j in range(2):
            text_color = "white" if matrix[i, j] > matrix.max() * 0.6 else "#1E1B4B"
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    fontsize=16, fontweight="bold", color=text_color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Positive", "Negative"], fontsize=12)
    ax.set_yticklabels(["Positive", "Negative"], fontsize=12)
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=13, fontweight="bold")
    ax.set_title("Confusion Matrix (Token-Level)", fontsize=15, fontweight="bold", pad=15)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[PLOT] Saved Confusion Matrix → {filepath}")
    return filepath


def plot_semantic_similarity_distribution(
    similarities: List[float],
    output_dir: str = "evaluation/plots",
    filename: str = "semantic_similarity_distribution.png",
) -> str:
    """Histogram of pairwise semantic similarity scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(output_dir)
    filepath = os.path.join(output_dir, filename)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(similarities, bins=20, color="#7C3AED", edgecolor="#4C1D95", alpha=0.85, linewidth=1.2)
    mean_val = np.mean(similarities)
    ax.axvline(mean_val, color="#F59E0B", linestyle="--", linewidth=2.5,
               label=f"Mean = {mean_val:.3f}")

    ax.set_xlabel("Cosine Similarity", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count", fontsize=13, fontweight="bold")
    ax.set_title("Semantic Similarity Distribution", fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[PLOT] Saved Semantic Similarity Distribution → {filepath}")
    return filepath


def plot_retrieval_recall_at_k(
    retrieval_results: dict,
    output_dir: str = "evaluation/plots",
    filename: str = "retrieval_recall_at_k.png",
) -> str:
    """Bar chart of Recall@K for various K values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(output_dir)
    filepath = os.path.join(output_dir, filename)

    k_values = retrieval_results.get("k_values", [1, 3, 5, 10])
    recall_scores = [retrieval_results.get(f"recall@{k}", 0.0) for k in k_values]
    precision_scores = [retrieval_results.get(f"precision@{k}", 0.0) for k in k_values]

    x = np.arange(len(k_values))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 6))
    bars1 = ax.bar(x - width / 2, recall_scores, width, label="Recall@K",
                   color="#10B981", edgecolor="#065F46", linewidth=1.2)
    bars2 = ax.bar(x + width / 2, precision_scores, width, label="Precision@K",
                   color="#3B82F6", edgecolor="#1E3A8A", linewidth=1.2)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("K", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Retrieval: Recall@K & Precision@K", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in k_values], fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=12, loc="upper left", framealpha=0.9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[PLOT] Saved Retrieval Recall@K → {filepath}")
    return filepath


def plot_rouge_bleu_comparison(
    answer_results: dict,
    output_dir: str = "evaluation/plots",
    filename: str = "rouge_bleu_comparison.png",
) -> str:
    """Grouped bar chart comparing ROUGE-1, ROUGE-2, ROUGE-L F1 and BLEU."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ensure_dir(output_dir)
    filepath = os.path.join(output_dir, filename)

    metrics = {
        "ROUGE-1 F1": answer_results.get("avg_rouge_1_f1", 0.0),
        "ROUGE-2 F1": answer_results.get("avg_rouge_2_f1", 0.0),
        "ROUGE-L F1": answer_results.get("avg_rouge_l_f1", 0.0),
        "BLEU":       answer_results.get("avg_bleu", 0.0),
    }

    colors = ["#EF4444", "#F97316", "#EAB308", "#22C55E"]
    edge_colors = ["#991B1B", "#9A3412", "#854D0E", "#166534"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics.keys(), metrics.values(), color=colors, edgecolor=edge_colors, linewidth=1.5)

    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("ROUGE & BLEU Scores (Averaged)", fontsize=15, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[PLOT] Saved ROUGE & BLEU comparison → {filepath}")
    return filepath


def plot_all(
    answer_results: dict,
    retrieval_results: dict,
    semantic_results: dict,
    output_dir: str = "evaluation/plots",
) -> List[str]:
    """Generate all evaluation plots and return list of saved file paths."""
    saved = []

    # 1. Precision-Recall curve
    precisions = [s["precision"] for s in answer_results["per_sample"]]
    recalls    = [s["recall"]    for s in answer_results["per_sample"]]
    saved.append(plot_precision_recall_curve(precisions, recalls, output_dir))

    # 2. Confusion matrix
    cm = answer_results["confusion_matrix"]
    saved.append(plot_confusion_matrix(cm["tp"], cm["fp"], cm["fn"], cm["tn"], output_dir))

    # 3. Semantic similarity distribution
    saved.append(plot_semantic_similarity_distribution(
        semantic_results["per_sample_similarity"], output_dir
    ))

    # 4. Retrieval Recall@K
    saved.append(plot_retrieval_recall_at_k(retrieval_results, output_dir))

    # 5. ROUGE & BLEU comparison
    saved.append(plot_rouge_bleu_comparison(answer_results, output_dir))

    return saved
