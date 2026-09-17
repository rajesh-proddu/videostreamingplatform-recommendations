"""Ranking-quality metrics over graded relevance labels.

`relevance` maps video_id -> grade (0 = irrelevant, 3 = ideal); unlisted IDs
count as 0. `ranked` is the served order.
"""

import math


def precision_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    """Fraction of the top-k slots holding a relevant (grade > 0) video.

    Divides by k, not by the number served, so returning fewer items than k
    costs precision.
    """
    if k <= 0:
        return 0.0
    return sum(1 for vid in ranked[:k] if relevance.get(vid, 0) > 0) / k


def recall_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    """Fraction of all relevant videos that appear in the top k. 1.0 when none are relevant."""
    relevant = {vid for vid, grade in relevance.items() if grade > 0}
    if not relevant:
        return 1.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def _dcg(grades: list[int]) -> float:
    return sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    """Normalized discounted cumulative gain. 1.0 when nothing is relevant and nothing is served."""
    ideal = _dcg(sorted((g for g in relevance.values() if g > 0), reverse=True)[:k])
    if ideal == 0:
        return 1.0 if not ranked else 0.0
    return _dcg([relevance.get(vid, 0) for vid in ranked[:k]]) / ideal
