"""Top-N ranking metrics used by CoDARec.

The implementation follows the full-ranking protocol of the original work:
for every user the model scores all unseen items and the top-N list is
compared against the held-out interactions.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

MetricSet = Tuple[List[float], List[float], List[float], List[float]]
"""``(precision, recall, ndcg, mrr)``, each aligned with the ``topn`` list."""

METRIC_NAMES = ("precision", "recall", "ndcg", "mrr")


def compute_topn_accuracy(
    ground_truth: Sequence[Sequence[int]],
    predicted_indices: Sequence[Sequence[int]],
    topn: Sequence[int],
) -> MetricSet:
    """Compute Precision/Recall/NDCG/MRR at every cut-off in ``topn``.

    Parameters
    ----------
    ground_truth:
        One list of relevant item ids per user.
    predicted_indices:
        One ranked list of (at least ``max(topn)``) item ids per user.
    topn:
        Ascending cut-off values.

    Returns
    -------
    A 4-tuple of lists, ordered as Precision, Recall, NDCG, MRR.
    """
    n_users = len(predicted_indices)
    if n_users != len(ground_truth):
        raise ValueError("ground_truth and predicted_indices must have equal length")
    if n_users == 0:
        raise ValueError("cannot evaluate an empty prediction set")

    precision: List[float] = []
    recall: List[float] = []
    ndcg: List[float] = []
    mrr: List[float] = []

    for cut_off in topn:
        sum_precision = sum_recall = sum_ndcg = sum_mrr = 0.0
        for user in range(n_users):
            relevant = ground_truth[user]
            if len(relevant) == 0:
                continue

            first_hit_rank = 0
            hits = 0
            dcg = 0.0
            for rank in range(cut_off):
                if predicted_indices[user][rank] in relevant:
                    # DCG gain; a hit at rank r contributes 1 / log2(r + 2).
                    dcg += 1.0 / math.log2(rank + 2)
                    if first_hit_rank == 0:
                        first_hit_rank = rank + 1
                    hits += 1

            # Ideal DCG: all relevant items placed at the top positions.
            ideal_hits = min(len(relevant), cut_off)
            idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))

            sum_precision += hits / cut_off
            sum_recall += hits / len(relevant)
            sum_ndcg += (dcg / idcg) if idcg > 0 else 0.0
            sum_mrr += (1.0 / first_hit_rank) if first_hit_rank else 0.0

        precision.append(round(sum_precision / n_users, 4))
        recall.append(round(sum_recall / n_users, 4))
        ndcg.append(round(sum_ndcg / n_users, 4))
        mrr.append(round(sum_mrr / n_users, 4))

    return precision, recall, ndcg, mrr


def as_dict(results: MetricSet, topn: Sequence[int]) -> Dict[str, float]:
    """Flatten a metric set into ``{"recall@20": 0.123, ...}``."""
    return {
        f"{name}@{cut_off}": value
        for name, values in zip(METRIC_NAMES, results)
        for cut_off, value in zip(topn, values)
    }


def format_results(results: MetricSet) -> str:
    """Render a metric set as one readable line per metric."""
    return "\n".join(
        f"  {name.capitalize():<9}: " + "  ".join(f"{value:.4f}" for value in values)
        for name, values in zip(METRIC_NAMES, results)
    )


def select_value(
    results: MetricSet,
    topn: Sequence[int],
    metric: str = "recall@20",
) -> float:
    """Look up ``metric`` (e.g. ``"recall@20"``) inside a metric set."""
    name, _, cut_off_text = metric.partition("@")
    name = name.strip().lower()
    if name not in METRIC_NAMES:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRIC_NAMES}")
    try:
        cut_off = int(cut_off_text)
    except ValueError as exc:
        raise ValueError(f"metric {metric!r} must look like 'recall@20'") from exc
    if cut_off not in topn:
        raise ValueError(f"cut-off {cut_off} of metric {metric!r} is not in topn={list(topn)}")
    return results[METRIC_NAMES.index(name)][list(topn).index(cut_off)]


def report(
    log,
    valid_results: MetricSet | None,
    test_results: MetricSet | None,
    title: str = "",
) -> None:
    """Print a labelled metric block through ``log`` (a Logger or print)."""
    if title:
        log.info(title)
    for label, results in (("Valid", valid_results), ("Test", test_results)):
        if results is None:
            continue
        log.info(f"[{label}]")
        log.rule("-", 60)
        log.info(format_results(results))


__all__ = [
    "METRIC_NAMES",
    "MetricSet",
    "as_dict",
    "compute_topn_accuracy",
    "format_results",
    "report",
    "select_value",
]
