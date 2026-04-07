"""
Evaluation metrics for information retrieval.

Implements:
    - Precision@K
    - Recall@K
    - Average Precision (AP)
    - Mean Average Precision (MAP)
    - NDCG@K
"""
import math
import logging
from typing import Sequence

logger = logging.getLogger(__name__)


def precision_at_k(retrieved: Sequence, relevant: set, k: int) -> float:
    """
    Precision@K = (# relevant docs in top-K) / K

    Args:
        retrieved: Ordered list of retrieved item IDs.
        relevant:  Set of ground-truth relevant item IDs.
        k:         Cut-off rank.

    Returns:
        Precision value in [0, 1].
    """
    if k <= 0:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(retrieved: Sequence, relevant: set, k: int) -> float:
    """
    Recall@K = (# relevant docs in top-K) / |relevant|

    Args:
        retrieved: Ordered list of retrieved item IDs.
        relevant:  Set of ground-truth relevant item IDs.
        k:         Cut-off rank.

    Returns:
        Recall value in [0, 1].  Returns 0 when relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision(retrieved: Sequence, relevant: set) -> float:
    """
    Average Precision = (1/|R|) * Σ P@k * rel(k)

    Args:
        retrieved: Ordered list of retrieved item IDs.
        relevant:  Set of ground-truth relevant item IDs.

    Returns:
        AP value in [0, 1].
    """
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


def ndcg_at_k(retrieved: Sequence, relevant: set, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.

    Args:
        retrieved: Ordered list of retrieved item IDs.
        relevant:  Set of ground-truth relevant item IDs.
        k:         Cut-off rank.

    Returns:
        NDCG value in [0, 1].
    """
    def dcg(items: list, rel: set, cutoff: int) -> float:
        score = 0.0
        for i, item in enumerate(items[:cutoff], start=1):
            if item in rel:
                score += 1.0 / math.log2(i + 1)
        return score

    top_k = list(retrieved)[:k]
    ideal = sorted(top_k, key=lambda x: x in relevant, reverse=True)
    actual_dcg = dcg(top_k, relevant, k)
    ideal_dcg = dcg(ideal, relevant, k)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def mean_average_precision(
    all_retrieved: list[Sequence],
    all_relevant: list[set],
) -> float:
    """
    Mean Average Precision over multiple queries.

    Args:
        all_retrieved: List of ordered retrieved item lists per query.
        all_relevant:  List of relevant item sets per query.

    Returns:
        MAP value in [0, 1].
    """
    if not all_retrieved:
        return 0.0
    ap_scores = [
        average_precision(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    return sum(ap_scores) / len(ap_scores)


def evaluate_retrieval(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k_values: list[int] | None = None,
) -> dict:
    """
    Convenience function that computes a suite of metrics for a single query.

    Args:
        retrieved_ids: Ordered list of retrieved product IDs.
        relevant_ids:  Ground-truth relevant product IDs.
        k_values:      List of cut-off ranks to evaluate.

    Returns:
        Dictionary with metric names as keys.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    relevant = set(relevant_ids)
    results: dict = {}

    for k in k_values:
        results[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant, k)
        results[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)

    results["ap"] = average_precision(retrieved_ids, relevant)
    return results
