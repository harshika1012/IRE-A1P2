"""Turn a user's click history into a BM25 query string.

Per the assignment's example: concatenate titles (+ abstracts, for a richer
query) of the user's N most recently clicked articles. click_history in
user_features.parquet is already time-ordered ascending, so the last N
entries are the most recent.
"""
N_RECENT = 5


def build_query(click_history, article_lookup: dict, n_recent: int = N_RECENT) -> str:
    """click_history: list/array of article_ids (time-ordered ascending).
    article_lookup: dict article_id -> "title abstract" string.
    """
    if click_history is None or len(click_history) == 0:
        return ""
    recent = list(click_history)[-n_recent:]
    parts = [article_lookup[aid] for aid in recent if aid in article_lookup]
    return " ".join(parts)


def build_user_embedding(click_history, id_to_embedding: dict, n_recent: int = N_RECENT):
    """Mean-pool the embeddings of the user's N most recently clicked articles.
    id_to_embedding: dict article_id -> np.ndarray embedding vector.
    Returns None if no clicked article has an embedding (cold-start / OOV).
    """
    import numpy as np
    if click_history is None or len(click_history) == 0:
        return None
    recent = list(click_history)[-n_recent:]
    vecs = [id_to_embedding[aid] for aid in recent if aid in id_to_embedding]
    if not vecs:
        return None
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm  # re-normalize after averaging, for cosine similarity
    return mean_vec