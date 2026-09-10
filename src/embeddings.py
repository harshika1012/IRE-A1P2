"""Article embeddings via a multilingual sentence encoder.

Using ONE multilingual model for both MIND (English) and EB-NeRD (Danish)
keeps both datasets in the same embedding space -- important since Q3.5
asks you to compare lexical vs semantic retrieval, and later parts may
compare across datasets too.

If you'd rather use the pretrained Ekstra_Bladet_word2vec.zip /
google_bert_base_multilingual_cased.zip artifacts instead: unzip them,
inspect the parquet/npy inside (EB-NeRD's word2vec bundle is typically a
`document_vector.parquet` with columns article_id, document_vector), and
write a small loader that returns (article_ids, embeddings) in the same
shape this module produces -- then swap it in wherever `compute_embeddings`
is called below. Not implemented here since MIND has no equivalent
pretrained artifact, so you'd need two different pipelines anyway.
"""
import numpy as np
from pathlib import Path
from src.config import FEATURES_DIR

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        print(f"  loading embedding model: {MODEL_NAME} (first call downloads it, ~470MB)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def compute_embeddings(article_ids, texts, batch_size: int = 128):
    """Returns (article_ids as np.array, embeddings as np.ndarray [n, dim])."""
    model = _get_model()
    embeddings = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,  # pre-normalize for cosine via dot product
    )
    return np.asarray(article_ids), embeddings


def save_embeddings(dataset: str, article_ids, embeddings):
    path = FEATURES_DIR / f"embeddings_{dataset}.npz"
    np.savez(path, article_ids=np.asarray(article_ids, dtype=object), embeddings=embeddings)
    print(f"  saved {path} ({embeddings.shape[0]} articles, dim={embeddings.shape[1]})")


def load_embeddings(dataset: str):
    path = FEATURES_DIR / f"embeddings_{dataset}.npz"
    data = np.load(path, allow_pickle=True)
    return data["article_ids"], data["embeddings"]


def embeddings_exist(dataset: str) -> bool:
    return (FEATURES_DIR / f"embeddings_{dataset}.npz").exists()