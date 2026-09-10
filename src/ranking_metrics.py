"""Official-style per-impression ranking metrics: AUC, MRR, nDCG@k.
All take (scores, labels) for ONE impression's candidate set:
    scores: array-like of predicted relevance scores, one per candidate
    labels: array-like of binary ground-truth relevance (1=clicked, 0=not)
Return None when the metric is undefined for that impression (e.g. AUC
needs at least one positive AND one negative candidate) -- callers should
filter out Nones before averaging, and are expected to report how many
impressions were skipped and why.
"""
import numpy as np


def auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None  # undefined: no negatives or no positives to compare against
    # Mann-Whitney U statistic == AUC, avoids needing sklearn here
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    avg_rank_per_val = sums / counts
    ranks = avg_rank_per_val[inv]

    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    sum_ranks_pos = ranks[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)


def mrr(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        return None  # no relevant item to rank at all
    order = np.argsort(-scores)  # descending
    ranked_labels = labels[order]
    first_hit = np.argmax(ranked_labels == 1) + 1  # 1-indexed rank of first positive
    return 1.0 / first_hit


def _dcg_at_k(sorted_labels, k):
    k = min(k, len(sorted_labels))
    if k == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, k + 2))  # positions 1..k -> log2(i+1)
    return float(np.sum(sorted_labels[:k] * discounts))


def ndcg_at_k(scores, labels, k):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if labels.sum() == 0:
        return None  # IDCG would be 0, nDCG undefined
    order = np.argsort(-scores)
    dcg = _dcg_at_k(labels[order], k)
    ideal_order = np.argsort(-labels)  # all positives first
    idcg = _dcg_at_k(labels[ideal_order], k)
    return dcg / idcg if idcg > 0 else None