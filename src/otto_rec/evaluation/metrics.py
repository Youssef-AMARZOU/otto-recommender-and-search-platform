"""Offline IR metrics for retrieval and ranking evaluation.

Every session-level function takes an ordered recommendation list (best first),
the ground-truth relevant items for that session, and a cutoff K. All scores
are in [0.0, 1.0]; aggregation helpers average over sessions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from math import log2

OTTO_WEIGHTS: dict[str, float] = {"clicks": 0.10, "carts": 0.25, "orders": 0.65}
OTTO_DEFAULT_K = 20


def _top_k(recommended: Sequence, k: int) -> list:
    return list(recommended)[:k]


def recall_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """Fraction of relevant items present in the top-K recommendations."""
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    return len(set(_top_k(recommended, k)) & rel) / len(rel)


def precision_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """Fraction of the top-K recommendations that are relevant."""
    if k <= 0:
        return 0.0
    rel = set(relevant)
    top = _top_k(recommended, k)
    if not top:
        return 0.0
    return sum(1 for item in top if item in rel) / k


def hit_rate_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """1.0 if at least one relevant item appears in the top-K, else 0.0."""
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    return 1.0 if any(item in rel for item in _top_k(recommended, k)) else 0.0


def mrr_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """Reciprocal rank of the first relevant item within the top-K."""
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    for rank, item in enumerate(_top_k(recommended, k), start=1):
        if item in rel:
            return 1.0 / rank
    return 0.0


def average_precision_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """AP@K with the trec_eval convention: denominator is min(|relevant|, K)."""
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(_top_k(recommended, k), start=1):
        if item in rel:
            hits += 1
            total += hits / rank
    return total / min(len(rel), k)


def ndcg_at_k(recommended: Sequence, relevant: Iterable, k: int) -> float:
    """NDCG@K with binary gains and log2(rank + 1) discounting."""
    rel = set(relevant)
    if k <= 0 or not rel:
        return 0.0
    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, item in enumerate(_top_k(recommended, k), start=1)
        if item in rel
    )
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_session(recommended: Sequence, relevant: Iterable, ks: Sequence[int]) -> dict[str, float]:
    """Full metric suite for one session at each cutoff in ``ks``."""
    scores: dict[str, float] = {}
    for k in ks:
        scores[f"recall@{k}"] = recall_at_k(recommended, relevant, k)
        scores[f"precision@{k}"] = precision_at_k(recommended, relevant, k)
        scores[f"hit_rate@{k}"] = hit_rate_at_k(recommended, relevant, k)
        scores[f"mrr@{k}"] = mrr_at_k(recommended, relevant, k)
        scores[f"map@{k}"] = average_precision_at_k(recommended, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(recommended, relevant, k)
    return scores


def aggregate(scores: Sequence[float]) -> float:
    """Mean of a score list; 0.0 for an empty list."""
    values = list(scores)
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_sessions(
    pairs: Iterable[tuple], ks: Sequence[int]
) -> dict[str, float]:
    """Mean metrics over an iterable of (recommended, relevant) session pairs."""
    collected: dict[str, list[float]] = {}
    for recommended, relevant in pairs:
        for name, value in evaluate_session(recommended, relevant, ks).items():
            collected.setdefault(name, []).append(value)
    return {name: aggregate(values) for name, values in collected.items()}


def otto_weighted_recall(
    predictions: Mapping[str, Mapping[str, Sequence]],
    ground_truth: Mapping[str, Mapping[str, Sequence]],
    k: int = OTTO_DEFAULT_K,
    weights: Mapping[str, float] | None = None,
) -> float:
    """OTTO competition metric: weighted Recall@K across event types.

    ``predictions`` and ``ground_truth`` are keyed by event type
    (clicks / carts / orders) and then by session id. Per event type the
    session-level recall is averaged over sessions holding at least one
    event of that type; the final score is the weighted sum (default
    0.10 clicks, 0.25 carts, 0.65 orders).
    """
    active_weights = dict(weights or OTTO_WEIGHTS)
    total = 0.0
    for event_type, weight in active_weights.items():
        preds = predictions.get(event_type, {})
        recalls = [
            recall_at_k(preds.get(session_id, []), truth, k)
            for session_id, truth in ground_truth.get(event_type, {}).items()
            if truth
        ]
        total += weight * aggregate(recalls)
    return total
