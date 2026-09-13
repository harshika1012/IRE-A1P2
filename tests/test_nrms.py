"""Q3: unit tests for the NRMS building blocks (src/nrms.py). Focused on
the edge cases that silently produce NaN in attention-based encoders:
all-pad titles and cold-start (zero-history) users.

Run with: pytest tests/test_nrms.py -q
"""
import torch

from src.nrms import NRMS, Vocab, CategoryVocab, NewsEncoder, UserEncoder, PAD_IDX


def test_vocab_encode_pads_and_truncates():
    vocab = Vocab(["Breaking News Today", "Sports Update"])
    ids = vocab.encode("Breaking News", max_len=5)
    assert len(ids) == 5
    assert ids[-2:] == [PAD_IDX, PAD_IDX]  # 2 real tokens + 3 pad
    truncated = vocab.encode("Breaking News Today Extra Words Here", max_len=3)
    assert len(truncated) == 3


def test_category_vocab_maps_unseen_to_unk():
    cv = CategoryVocab(["sports", "news", "sports"])
    assert cv.encode("sports") != 0
    assert cv.encode("totally_unseen_category") == 0
    assert cv.encode(None) == 0


def test_news_encoder_handles_all_pad_title_without_nan():
    encoder = NewsEncoder(vocab_size=50, embed_dim=16, num_heads=2)
    title_ids = torch.zeros((3, 10), dtype=torch.long)  # every row all-pad
    out = encoder(title_ids)
    assert out.shape == (3, 16)
    assert not torch.isnan(out).any()


def test_user_encoder_handles_zero_history_without_nan():
    encoder = UserEncoder(news_dim=16, num_heads=2)
    hist_vecs = torch.zeros((2, 5, 16))
    hist_mask = torch.zeros((2, 5), dtype=torch.bool)  # cold-start: no history at all
    out = encoder(hist_vecs, hist_mask)
    assert out.shape == (2, 16)
    assert not torch.isnan(out).any()


def test_nrms_forward_and_one_training_step_reduces_loss():
    torch.manual_seed(0)
    vocab = Vocab(["breaking news today", "sports update now", "weather report here"])
    cat_vocab = CategoryVocab(["sports", "news"])
    model = NRMS(len(vocab), len(cat_vocab), embed_dim=16, num_heads=2, use_category=True)

    B, H, L = 4, 3, 6
    hist_ids = torch.randint(1, len(vocab), (B, H, L))
    hist_cats = torch.randint(0, len(cat_vocab), (B, H))
    hist_mask = torch.ones((B, H), dtype=torch.bool)
    cand_ids = torch.randint(1, len(vocab), (B, L))
    cand_cats = torch.randint(0, len(cat_vocab), (B,))
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0])

    loss_fn = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.05)

    logits = model(hist_ids, hist_cats, hist_mask, cand_ids, cand_cats)
    assert logits.shape == (B,)
    assert not torch.isnan(logits).any()
    first_loss = loss_fn(logits, labels).item()

    for _ in range(20):
        opt.zero_grad()
        logits = model(hist_ids, hist_cats, hist_mask, cand_ids, cand_cats)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

    assert loss.item() < first_loss  # model can fit this tiny batch
