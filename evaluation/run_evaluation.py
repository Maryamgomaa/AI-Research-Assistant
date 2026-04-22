"""
run_evaluation.py — Main Evaluation Runner
============================================
Reads a labeled CSV dataset, computes all metrics, prints results
to the terminal, and saves plots as PNG files.

Usage:
    python -m evaluation.run_evaluation
    python -m evaluation.run_evaluation --dataset evaluation/sample_dataset.csv
    python -m evaluation.run_evaluation --dataset path/to/your_data.csv --output-dir evaluation/plots
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Force UTF-8 output on Windows consoles (avoids cp1256 / cp1252 encoding errors)
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

# ── Ensure project root is on sys.path ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.answer_metrics import evaluate_answers
from evaluation.retrieval_metrics import evaluate_retrieval
from evaluation.semantic_similarity import compute_semantic_similarity
from evaluation.plot_results import plot_all

# ── Logging ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval.runner")


# ────────────────────────────────────────────────────────────────
# Dataset loader
# ────────────────────────────────────────────────────────────────

def load_dataset(csv_path: str) -> Tuple[
    List[str],          # questions
    List[str],          # ground_truth_answers
    List[str],          # predicted_answers
    List[List[str]],    # relevant_doc_ids
    List[List[str]],    # retrieved_doc_ids
]:
    """
    Load evaluation dataset from CSV.

    Expected columns:
        question, ground_truth_answer, predicted_answer,
        relevant_doc_ids (comma-separated), retrieved_doc_ids (comma-separated)
    """
    questions, gts, preds, rel_docs, ret_docs = [], [], [], [], []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row["question"].strip())
            gts.append(row["ground_truth_answer"].strip())
            preds.append(row["predicted_answer"].strip())

            rel = [d.strip() for d in row.get("relevant_doc_ids", "").split(",") if d.strip()]
            ret = [d.strip() for d in row.get("retrieved_doc_ids", "").split(",") if d.strip()]
            rel_docs.append(rel)
            ret_docs.append(ret)

    log.info(f"[DATASET] Loaded {len(questions)} samples from '{csv_path}'")
    return questions, gts, preds, rel_docs, ret_docs


# ────────────────────────────────────────────────────────────────
# Pretty terminal output
# ────────────────────────────────────────────────────────────────

DIVIDER    = "=" * 72
SUBDIV     = "-" * 72
HEADER_FMT = "{}"
METRIC_FMT = "  {:<35} {:>10}"
SAMPLE_FMT = "  {:<6} P={:<7} R={:<7} F1={:<7} EM={} ROUGE-L={:<7} BLEU={:<7} SemSim={}"


def _print_header(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(HEADER_FMT.format(f"  {title}"))
    print(DIVIDER)


def _print_metric(label: str, value) -> None:
    print(METRIC_FMT.format(label, value))


def print_answer_results(results: dict, semantic: dict) -> None:
    _print_header("ANSWER-QUALITY METRICS")

    _print_metric("Samples", results["num_samples"])
    _print_metric("Exact-Match Accuracy", results["accuracy_exact_match"])
    print(SUBDIV)

    _print_metric("Micro Precision", results["micro"]["precision"])
    _print_metric("Micro Recall", results["micro"]["recall"])
    _print_metric("Micro F1", results["micro"]["f1"])
    print(SUBDIV)

    _print_metric("Macro Precision", results["macro"]["precision"])
    _print_metric("Macro Recall", results["macro"]["recall"])
    _print_metric("Macro F1", results["macro"]["f1"])
    print(SUBDIV)

    cm = results["confusion_matrix"]
    print("  Confusion Matrix (token-level):")
    print(f"    TP = {cm['tp']:>5}   FP = {cm['fp']:>5}")
    print(f"    FN = {cm['fn']:>5}   TN = {cm['tn']:>5}")
    print(SUBDIV)

    _print_metric("Avg ROUGE-1 F1", results["avg_rouge_1_f1"])
    _print_metric("Avg ROUGE-2 F1", results["avg_rouge_2_f1"])
    _print_metric("Avg ROUGE-L F1", results["avg_rouge_l_f1"])
    _print_metric("Avg BLEU", results["avg_bleu"])
    print(SUBDIV)

    _print_metric("Mean Semantic Similarity", semantic["mean_similarity"])
    _print_metric("Median Semantic Similarity", semantic["median_similarity"])
    _print_metric("Min Semantic Similarity", semantic["min_similarity"])
    _print_metric("Max Semantic Similarity", semantic["max_similarity"])
    print(SUBDIV)

    # Per-sample
    print(HEADER_FMT.format("  PER-SAMPLE DETAILS"))
    for i, sample in enumerate(results["per_sample"]):
        sim = semantic["per_sample_similarity"][i]
        print(SAMPLE_FMT.format(
            f"#{i+1}",
            f"{sample['precision']:.4f}",
            f"{sample['recall']:.4f}",
            f"{sample['f1']:.4f}",
            "Y" if sample["exact_match"] else "N",
            f"{sample['rouge']['rouge_l']['f1']:.4f}",
            f"{sample['bleu']:.4f}",
            f"{sim:.4f}",
        ))


def print_retrieval_results(results: dict) -> None:
    _print_header("RETRIEVAL METRICS")

    _print_metric("Queries", results["num_queries"])
    _print_metric("MRR", results["mrr"])
    _print_metric("MAP", results["map"])
    print(SUBDIV)

    for k in results["k_values"]:
        _print_metric(f"Precision@{k}", results[f"precision@{k}"])
        _print_metric(f"Recall@{k}", results[f"recall@{k}"])
        _print_metric(f"Hit Rate@{k}", results[f"hit_rate@{k}"])
        print()


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Evaluation Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).resolve().parent / "sample_dataset.csv"),
        help="Path to the labeled evaluation CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "plots"),
        help="Directory to save PNG plots.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        help="SentenceTransformer model for semantic similarity.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for embedding model (cpu / cuda).",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="1,3,5,10",
        help="Comma-separated K values for retrieval metrics.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Also save full results as JSON.",
    )
    args = parser.parse_args()

    start_time = time.time()

    # ── 1. Load dataset ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("RAG EVALUATION PIPELINE -- Starting")
    log.info("=" * 60)

    questions, gts, preds, rel_docs, ret_docs = load_dataset(args.dataset)

    # ── 2. Answer-quality metrics ────────────────────────────────
    log.info("[STEP 1/4] Computing answer-quality metrics...")
    answer_results = evaluate_answers(gts, preds)

    # ── 3. Semantic similarity ───────────────────────────────────
    log.info("[STEP 2/4] Computing semantic similarity...")
    semantic_results = compute_semantic_similarity(
        gts, preds,
        model_name=args.embedding_model,
        device=args.device,
    )

    # ── 4. Retrieval metrics ─────────────────────────────────────
    log.info("[STEP 3/4] Computing retrieval metrics...")
    k_values = [int(k.strip()) for k in args.k_values.split(",")]
    retrieval_results = evaluate_retrieval(rel_docs, ret_docs, k_values=k_values)

    # ── 5. Plots ─────────────────────────────────────────────────
    log.info("[STEP 4/4] Generating plots...")
    saved_plots = plot_all(answer_results, retrieval_results, semantic_results, args.output_dir)

    # ── 6. Terminal output ───────────────────────────────────────
    print_answer_results(answer_results, semantic_results)
    print_retrieval_results(retrieval_results)

    _print_header("SAVED PLOTS")
    for p in saved_plots:
        print(f"  [PLOT] {p}")

    # ── 7. Optional JSON export ──────────────────────────────────
    if args.save_json:
        json_path = os.path.join(args.output_dir, "evaluation_results.json")
        os.makedirs(args.output_dir, exist_ok=True)
        full_results = {
            "answer_metrics": answer_results,
            "semantic_similarity": semantic_results,
            "retrieval_metrics": retrieval_results,
            "plots_saved": saved_plots,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        log.info(f"[JSON] Full results saved -> {json_path}")

    elapsed = time.time() - start_time
    print(f"\n{DIVIDER}")
    print(HEADER_FMT.format(f"  EVALUATION COMPLETE -- {elapsed:.1f}s"))
    print(DIVIDER)


if __name__ == "__main__":
    main()
