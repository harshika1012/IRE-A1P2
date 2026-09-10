"""Beyond-accuracy metrics: intra-list diversity, novelty, catalog coverage.
Diversity/novelty are computed per recommended list (top-K); coverage is a
single number over the whole evaluation run.
"""
import numpy as np


def intra_list_diversity(article_ids, id_to_embedding: dict) -> float:
    """Average pairwise (1 - cosine similarity) among the recommended list's
    article embeddings. Embeddings are assumed L2-normalized (as produced by
    src/embeddings.py), so cosine similarity == dot product. Returns None if
    fewer than 2 articles have an embedding (diversity undefined for a
    singleton list)."""
    vecs = [id_to_embedding[a] for a in article_ids if a in id_to_embedding]
    if len(vecs) < 2:
        return None
    vecs = np.stack(vecs)
    sims = vecs @ vecs.T
    n = len(vecs)
    iu = np.triu_indices(n, k=1)
    avg_sim = sims[iu].mean()
    return float(1 - avg_sim)


def novelty(article_ids, popularity: dict, total_clicks: int) -> float:
    """Self-information novelty: -log2(p(item)), averaged over the list.
    popularity: dict article_id -> click count (from TRAIN split only).
    Higher = more novel (less popular). Unseen-in-train articles are treated
    as maximally novel (smoothed with 1 pseudo-click to avoid log(0))."""
    if not article_ids:
        return None
    scores = []
    for a in article_ids:
        clicks = popularity.get(a, 0) + 1  # +1 Laplace smoothing
        p = clicks / (total_clicks + len(popularity) + 1)
        scores.append(-np.log2(p))
    return float(np.mean(scores))


def catalog_coverage(all_recommended_lists, catalog_size: int) -> float:
    """Fraction of the full article catalog that appears in AT LEAST ONE
    recommended list across the whole evaluation run. Computed once, not
    per-impression -- pass every impression's top-K list in."""
    seen = set()
    for lst in all_recommended_lists:
        seen.update(lst)
    return len(seen) / catalog_size if catalog_size else 0.0