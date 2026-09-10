"""BM25 (Okapi) over a document collection, backed by a sparse term-document
matrix. The CountVectorizer's vocabulary + CSC column structure IS the
inverted index: each column holds the posting list (doc ids + raw term
counts) for one term. We then precompute BM25 weights once at index time,
so scoring a query is a single sparse matrix-vector product instead of a
per-document Python loop.
"""
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from src.tokenizer import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids = None
        self.vectorizer = None
        self.W = None  # precomputed BM25 weight matrix (n_docs x n_terms), sparse

    def fit(self, doc_ids, texts):
        """doc_ids: list-like of article ids (aligned with texts).
        texts: list-like of raw title+abstract strings.
        """
        self.doc_ids = np.asarray(doc_ids)
        self.vectorizer = CountVectorizer(tokenizer=tokenize, preprocessor=None,
                                           lowercase=False, token_pattern=None)
        X = self.vectorizer.fit_transform(texts)  # (n_docs, n_terms) raw term counts

        n_docs = X.shape[0]
        doc_len = np.asarray(X.sum(axis=1)).flatten()
        avgdl = doc_len.mean() if n_docs else 1.0

        df = np.diff(X.tocsc().indptr)  # document frequency per term
        idf = np.log(((n_docs - df + 0.5) / (df + 0.5)) + 1.0)

        Xc = X.tocoo()
        rows, cols, tf = Xc.row, Xc.col, Xc.data.astype(np.float64)
        len_norm = 1 - self.b + self.b * (doc_len[rows] / avgdl)
        denom = tf + self.k1 * len_norm
        w_data = idf[cols] * (tf * (self.k1 + 1)) / denom

        self.W = csr_matrix((w_data, (rows, cols)), shape=X.shape)
        self._n_docs = n_docs
        self._id_to_row = {doc_id: i for i, doc_id in enumerate(self.doc_ids)}
        return self

    def _query_vector(self, query_text: str):
        tokens = tokenize(query_text)
        vocab = self.vectorizer.vocabulary_
        idx = sorted({vocab[t] for t in tokens if t in vocab})
        from scipy.sparse import csr_matrix as _csr
        if not idx:
            return None
        data = np.ones(len(idx))
        q = _csr((data, (idx, [0] * len(idx))), shape=(self.W.shape[1], 1))
        return q

    def vectorize_query(self, query_text: str):
        """Public wrapper around _query_vector, returns a (1, n_terms) row
        vector (or None if no query terms are in the vocabulary) -- used by
        batch_score for building a stacked query matrix."""
        q = self._query_vector(query_text)
        return q.T.tocsr() if q is not None else None  # (n_terms,1) -> (1,n_terms)

    def batch_score(self, doc_ids, query_matrix, query_row_idx) -> np.ndarray:
        """Vectorized scoring for MANY (doc_id, query) pairs at once --
        the fast alternative to calling score_docs() once per impression
        in a Python loop.

        doc_ids: array-like, length P (P = total candidate instances across
                 however many impressions you're batching, e.g. sum of
                 candidate-list lengths for a whole chunk)
        query_matrix: sparse matrix (Q, n_terms) -- one row per DISTINCT
                 query (e.g. one row per unique user in the chunk), built by
                 stacking vectorize_query() outputs
        query_row_idx: array-like, length P -- which row of query_matrix
                 each doc_id's pair should be scored against

        Returns: np.ndarray, length P, aligned with doc_ids (0.0 for any
                 doc_id not present in this index).
        """
        from scipy.sparse import vstack
        P = len(doc_ids)
        rows = np.array([self._id_to_row.get(d, -1) for d in doc_ids])
        valid = rows >= 0
        scores = np.zeros(P, dtype=np.float64)
        if valid.any():
            W_sel = self.W[rows[valid]]
            Q_sel = query_matrix[np.asarray(query_row_idx)[valid]]
            # elementwise multiply + row-sum -- ONE vectorized op for all
            # valid pairs at once, instead of P separate small matvecs
            prod = W_sel.multiply(Q_sel)
            scores[valid] = np.asarray(prod.sum(axis=1)).ravel()
        return scores

    def top_k(self, query_text: str, k: int = 200):
        """Returns list of (article_id, score), highest first. Empty list if
        no query terms match the vocabulary (e.g. cold-start / OOV query)."""
        q = self._query_vector(query_text)
        if q is None:
            return []
        scores = (self.W @ q).toarray().ravel()
        if k >= len(scores):
            top_idx = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, k)[:k]
            top_idx = part[np.argsort(-scores[part])]
        top_idx = top_idx[scores[top_idx] > 0]  # drop zero-score (no overlap) docs
        return list(zip(self.doc_ids[top_idx], scores[top_idx]))

    def score_docs(self, doc_ids, query_text: str) -> dict:
        """Score ONLY the given candidate doc_ids against the query -- much
        cheaper than top_k() when you already know the small candidate set
        (e.g. the ~10-40 articles in one impression), as opposed to
        searching the full corpus. Unknown doc_ids (not in the index) get
        score 0.0 rather than raising, so they sort last but the submission
        still covers every candidate."""
        q = self._query_vector(query_text)
        result = {}
        if q is None:
            return {d: 0.0 for d in doc_ids}
        rows = [self._id_to_row[d] for d in doc_ids if d in self._id_to_row]
        if rows:
            sub_scores = (self.W[rows] @ q).toarray().ravel()
            row_to_score = dict(zip(rows, sub_scores))
        else:
            row_to_score = {}
        for d in doc_ids:
            r = self._id_to_row.get(d)
            result[d] = float(row_to_score.get(r, 0.0)) if r is not None else 0.0
        return result