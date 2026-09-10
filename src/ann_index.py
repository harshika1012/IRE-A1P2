"""Approximate/exact nearest-neighbour index over article embeddings.

At ~77K articles and a few hundred dims, brute-force cosine similarity
(one matrix-vector product per query) is fast and exact -- no approximation
needed at this scale. FAISS is used automatically if installed, purely as
a speed-up (IndexFlatIP = exact inner-product search, not approximate; if
you want true ANN speedups at larger scale, swap in IndexIVFFlat or HNSW).
"""
import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class ANNIndex:
    def __init__(self):
        self.article_ids = None
        self.embeddings = None  # assumed L2-normalized -> dot product == cosine similarity
        self._faiss_index = None

    def fit(self, article_ids, embeddings: np.ndarray):
        self.article_ids = np.asarray(article_ids)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self._id_to_row = {aid: i for i, aid in enumerate(self.article_ids)}
        if _HAS_FAISS:
            dim = self.embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(self.embeddings)
        return self

    def top_k(self, query_vec: np.ndarray, k: int = 200):
        """query_vec: (dim,) array, ideally L2-normalized.
        Returns list of (article_id, score), highest first."""
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        n = self.embeddings.shape[0]
        k = min(k, n)

        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q, k)
            scores, idx = scores[0], idx[0]
        else:
            sims = self.embeddings @ q[0]  # (n,) cosine similarity since both normalized
            if k >= n:
                idx = np.argsort(-sims)
            else:
                part = np.argpartition(-sims, k)[:k]
                idx = part[np.argsort(-sims[part])]
            scores = sims[idx]

        return list(zip(self.article_ids[idx], scores))

    def score_docs(self, doc_ids, query_vec) -> dict:
        """Cosine similarity of ONLY the given candidate doc_ids against the
        query vector -- avoids a full-corpus search when the candidate set
        is already known and small (one impression's inview list). Unknown
        doc_ids (not in the index) get score 0.0 rather than raising.

        Internally does ONE matrix-vector product for all candidates at
        once (via score_candidates), rather than one dot product per doc_id
        in a loop -- cheap since dense vector ops are BLAS-backed."""
        if query_vec is None:
            return {d: 0.0 for d in doc_ids}
        scores = self.score_candidates(doc_ids, query_vec)
        return dict(zip(doc_ids, scores))

    def score_candidates(self, doc_ids, query_vec) -> np.ndarray:
        """Same as score_docs but returns a plain array aligned with
        doc_ids, not a dict -- avoids dict-construction overhead when
        called in a tight per-impression loop across millions of rows."""
        q = np.asarray(query_vec, dtype=np.float32)
        rows = np.array([self._id_to_row.get(d, -1) for d in doc_ids])
        valid = rows >= 0
        scores = np.zeros(len(doc_ids), dtype=np.float64)
        if valid.any():
            scores[valid] = self.embeddings[rows[valid]] @ q  # ONE matvec for all candidates
        return scores