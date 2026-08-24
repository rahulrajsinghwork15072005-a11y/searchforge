"""IR evaluation: Precision@K, Recall@K, MRR, NDCG@K over TSV qrels."""

from __future__ import annotations

import math


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for d in top if d in relevant) / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    return sum(1 for d in top if d in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], grades: dict[str, int], k: int) -> float:
    """Graded NDCG with log2 discount; ideal ordering by grade descending."""
    def dcg(order: list[str]) -> float:
        total = 0.0
        for position, doc_id in enumerate(order[:k], start=1):
            gain = grades.get(doc_id, 0)
            total += (2**gain - 1) / math.log2(position + 1)
        return total

    ideal = sorted(grades.items(), key=lambda kv: (-kv[1], kv[0]))
    idcg = dcg([doc_id for doc_id, _ in ideal])
    if idcg <= 0:
        return 0.0
    return dcg(retrieved) / idcg


def evaluate(
    run_results: dict[str, list[str]],
    qrels: dict[str, dict],
    k_values: tuple[int, ...] = (3, 5, 10),
) -> dict:
    """qrels: {query_id: {doc_id: binary_flag_or_grade}}.

    Query text is matched to qrels via the caller-supplied mapping in `run_results`
    keyed by the SAME query ids as qrels.
    """
    report: dict = {"queries": len(run_results)}
    for k in k_values:
        report[f"P@{k}"] = 0.0
        report[f"R@{k}"] = 0.0
        report[f"NDCG@{k}"] = 0.0
    report["MRR"] = 0.0

    counted = 0
    for query_id, retrieved in run_results.items():
        judgments = qrels.get(query_id)
        if not judgments:
            continue
        counted += 1
        relevant_binary = {d for d, g in judgments.items() if g > 0}
        for k in k_values:
            report[f"P@{k}"] += precision_at_k(retrieved, relevant_binary, k)
            report[f"R@{k}"] += recall_at_k(retrieved, relevant_binary, k)
            report[f"NDCG@{k}"] += ndcg_at_k(retrieved, judgments, k)
        report["MRR"] += reciprocal_rank(retrieved, relevant_binary)

    if counted:
        for key in list(report):
            if key != "queries":
                report[key] = round(report[key] / counted, 4)
    report["counted_queries"] = counted
    return report
