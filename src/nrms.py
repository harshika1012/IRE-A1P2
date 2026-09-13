"""Assignment 2, Part I Q3: NRMS baseline (Wu et al., 2019) + one principled
improvement -- category-aware news encoding.

Baseline: NRMS's two defining encoders --
  News encoder:  word embedding -> multi-head self-attention over the
                 title's tokens -> additive attention pooling -> news vector.
  User encoder:  the same self-attention + additive-attention pattern,
                 applied over the sequence of the user's recently clicked
                 NEWS VECTORS (from the news encoder above) -> user vector.
Scoring is a dot product between the user vector and a candidate's news
vector (as in the paper).

Improvement (Q3.2): concatenate a learned CATEGORY embedding onto the
title-attention output before it leaves the news encoder, then project
back down to the same width -- so the user encoder and scoring function
are byte-for-byte identical between the baseline and improved model. That
keeps the ablation isolated to exactly one change (`use_category=True`),
per the assignment's "run an ablation study isolating the contribution of
your improvement."

Trained with pointwise BCE over (history, candidate, label) examples built
from 1 positive + a handful of sampled negatives per impression, rather
than the paper's shared K-negative softmax -- NRMS's defining contribution
is its two encoders, not its training objective, and pointwise BCE lets
both models reuse Q2's exact per-impression AUC/MRR/nDCG harness
(src.reranker.evaluate_scores) for evaluation.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.tokenizer import tokenize

PAD_IDX = 0
UNK_IDX = 1


class Vocab:
    """Word -> index, built from a corpus of titles. <pad>=0, <unk>=1."""

    def __init__(self, texts, max_size: int = 20000):
        counts = {}
        for t in texts:
            for tok in tokenize(t):
                counts[tok] = counts.get(tok, 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:max_size - 2]
        self.word2idx = {"<pad>": PAD_IDX, "<unk>": UNK_IDX}
        for w, _ in top:
            self.word2idx[w] = len(self.word2idx)

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text, max_len: int):
        ids = [self.word2idx.get(tok, UNK_IDX) for tok in tokenize(text)][:max_len]
        ids += [PAD_IDX] * (max_len - len(ids))
        return ids


class CategoryVocab:
    """Category string -> index. <unk>=0 covers unseen/NaN categories."""

    def __init__(self, categories):
        uniq = sorted({c for c in categories if isinstance(c, str)})
        self.cat2idx = {"<unk>": 0}
        for c in uniq:
            self.cat2idx[c] = len(self.cat2idx)

    def __len__(self):
        return len(self.cat2idx)

    def encode(self, cat):
        return self.cat2idx.get(cat, 0)


def _additive_attention_pool(x: torch.Tensor, mask: torch.Tensor, attn_layer: nn.Module) -> torch.Tensor:
    """x: (B, L, D), mask: (B, L) bool (True = real, non-pad position).
    Returns (B, D), the attention-weighted sum over the L axis."""
    scores = attn_layer(x).squeeze(-1)
    scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return (x * weights.unsqueeze(-1)).sum(dim=1)


def _guard_all_pad_rows(mask: torch.Tensor) -> torch.Tensor:
    """MultiheadAttention needs >=1 unmasked key per row, and additive
    attention's softmax is undefined over an all -inf row. A row is all-pad
    when a title is empty (shouldn't happen here) or a history slot is
    unused padding -- give it one fake valid position so the encoder stays
    numerically defined; its content doesn't matter since UserEncoder masks
    padded history slots out of its own attention/pooling regardless."""
    all_pad = ~mask.any(dim=1)
    if all_pad.any():
        mask = mask.clone()
        mask[all_pad, 0] = True
    return mask


class NewsEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4,
                 num_categories: int = 1, use_category: bool = False,
                 cat_embed_dim: int = 16, dropout: float = 0.2):
        super().__init__()
        self.use_category = use_category
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.attn_pool = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Tanh(), nn.Linear(embed_dim, 1))
        self.out_dim = embed_dim
        if use_category:
            self.category_embedding = nn.Embedding(num_categories, cat_embed_dim)
            self.project = nn.Linear(embed_dim + cat_embed_dim, embed_dim)

    def forward(self, title_ids: torch.Tensor, category_ids: torch.Tensor = None) -> torch.Tensor:
        """title_ids: (B, L) long. category_ids: (B,) long, required iff use_category."""
        mask = _guard_all_pad_rows(title_ids != PAD_IDX)
        x = self.dropout(self.embedding(title_ids))
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=~mask)
        vec = _additive_attention_pool(attn_out, mask, self.attn_pool)
        if self.use_category:
            cat_vec = self.category_embedding(category_ids)
            vec = self.project(torch.cat([vec, cat_vec], dim=-1))
        return vec


class UserEncoder(nn.Module):
    def __init__(self, news_dim: int = 64, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(news_dim, num_heads, batch_first=True, dropout=dropout)
        self.attn_pool = nn.Sequential(nn.Linear(news_dim, news_dim), nn.Tanh(), nn.Linear(news_dim, 1))

    def forward(self, hist_vecs: torch.Tensor, hist_mask: torch.Tensor) -> torch.Tensor:
        """hist_vecs: (B, H, D). hist_mask: (B, H) bool, True = real history item.
        A user with zero history (all pad, e.g. cold-start) gets a ~0 vector."""
        mask = _guard_all_pad_rows(hist_mask)
        attn_out, _ = self.self_attn(hist_vecs, hist_vecs, hist_vecs, key_padding_mask=~mask)
        return _additive_attention_pool(attn_out, mask, self.attn_pool)


class NRMS(nn.Module):
    def __init__(self, vocab_size: int, num_categories: int, embed_dim: int = 64,
                 num_heads: int = 4, use_category: bool = False, dropout: float = 0.2):
        super().__init__()
        self.news_encoder = NewsEncoder(vocab_size, embed_dim, num_heads, num_categories,
                                         use_category=use_category, dropout=dropout)
        self.user_encoder = UserEncoder(embed_dim, num_heads, dropout=dropout)

    def encode_news(self, title_ids: torch.Tensor, category_ids: torch.Tensor) -> torch.Tensor:
        return self.news_encoder(title_ids, category_ids)

    def encode_user(self, hist_title_ids: torch.Tensor, hist_category_ids: torch.Tensor,
                     hist_mask: torch.Tensor) -> torch.Tensor:
        B, H, L = hist_title_ids.shape
        hist_vecs = self.news_encoder(hist_title_ids.reshape(B * H, L),
                                       hist_category_ids.reshape(B * H)).reshape(B, H, -1)
        hist_vecs = hist_vecs * hist_mask.unsqueeze(-1)  # zero pad slots regardless of encoder output
        return self.user_encoder(hist_vecs, hist_mask)

    def forward(self, hist_title_ids, hist_category_ids, hist_mask, cand_title_ids, cand_category_ids):
        """Returns raw logits (B,) -- apply sigmoid / BCEWithLogitsLoss outside."""
        user_vec = self.encode_user(hist_title_ids, hist_category_ids, hist_mask)
        cand_vec = self.news_encoder(cand_title_ids, cand_category_ids)
        return (user_vec * cand_vec).sum(dim=-1)


class NRMSExampleDataset(Dataset):
    """rows: list of dicts with history_titles/history_categories (parallel
    lists, point-in-time recent clicks), candidate_title, candidate_category,
    label, impression_id. Tokenizes/pads to fixed shapes at __getitem__ time."""

    def __init__(self, rows, vocab: Vocab, cat_vocab: CategoryVocab,
                 max_title_len: int = 20, max_hist_len: int = 20):
        self.rows = rows
        self.vocab = vocab
        self.cat_vocab = cat_vocab
        self.max_title_len = max_title_len
        self.max_hist_len = max_hist_len

    def __len__(self):
        return len(self.rows)

    def _encode_row(self, r):
        hist_titles = r["history_titles"][-self.max_hist_len:]
        hist_cats = r["history_categories"][-self.max_hist_len:]
        n_hist = len(hist_titles)
        pad_needed = self.max_hist_len - n_hist

        hist_ids = [self.vocab.encode(t, self.max_title_len) for t in hist_titles]
        hist_ids += [[PAD_IDX] * self.max_title_len] * pad_needed
        hist_cat_ids = [self.cat_vocab.encode(c) for c in hist_cats] + [0] * pad_needed
        hist_mask = [True] * n_hist + [False] * pad_needed

        return {
            "hist_ids": torch.tensor(hist_ids, dtype=torch.long),
            "hist_cats": torch.tensor(hist_cat_ids, dtype=torch.long),
            "hist_mask": torch.tensor(hist_mask, dtype=torch.bool),
            "cand_ids": torch.tensor(self.vocab.encode(r["candidate_title"], self.max_title_len), dtype=torch.long),
            "cand_cat": torch.tensor(self.cat_vocab.encode(r["candidate_category"]), dtype=torch.long),
        }

    def __getitem__(self, idx):
        r = self.rows[idx]
        item = self._encode_row(r)
        item["label"] = torch.tensor(r["label"], dtype=torch.float)
        return item
