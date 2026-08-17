"""Multi-Branch Tree-Drafting: K parallele Branches, ein Batch-Forward je Schritt.

Prueft Top-K-Branch-Konstruktion, Baub-Verifikation (laengster gueltiger Pfad)
und Greedy-Aequivalenz des Tree-Drafting gegen Standard-Greedy.
"""
import torch

from k4n0n3.mtp_engine import MTPVerificationEngine


def make_forward(vocab: int, num_drafts: int):
    """Batch-faehiger Mock: next(t) = (t+1) % vocab; perfekte Drafts d_d = next^(d+1)(last)."""

    def forward_fn(ids):
        B, L = ids.shape
        nxt = (ids + 1) % vocab  # [B, L]
        logits = torch.zeros(B, L, vocab)
        logits.scatter_(2, nxt.unsqueeze(-1), 1.0)
        drafts = []
        if B == 1:
            last = int(ids[0, -1].item())
            for d in range(1, num_drafts + 1):
                dl = torch.zeros(1, vocab)
                dl[0, (last + 1 + d) % vocab] = 1.0
                drafts.append(dl)
        return logits, drafts

    return forward_fn


def standard_greedy(forward_fn, input_ids, max_new_tokens):
    ids = input_ids
    out = []
    for _ in range(max_new_tokens):
        logits, _ = forward_fn(ids)
        t = int(torch.argmax(logits[:, -1]))
        out.append(t)
        ids = torch.cat([ids, ids.new_tensor([[t]])], dim=1)
    return out


def test_topk_branches():
    engine = MTPVerificationEngine(num_branches=3)
    d1 = torch.tensor([[0.1, 0.5, 0.3, 0.05, 0.05]])  # top3: [1, 2, 0]
    d2 = torch.tensor([[0.2, 0.1, 0.6, 0.05, 0.05]])  # top3: [2, 0, 1]
    branches = engine.topk_branches([d1, d2])
    assert branches == [[1, 2], [2, 0], [0, 1]]


def test_verify_tree_selects_best():
    engine = MTPVerificationEngine(num_branches=2)
    vocab = 10

    def forward_fn(ids):
        B, L = ids.shape
        nxt = (ids + 1) % vocab
        logits = torch.zeros(B, L, vocab)
        logits.scatter_(2, nxt.unsqueeze(-1), 1.0)
        return logits, []

    ids = torch.tensor([[0]])  # L=1
    branches = [[1, 2], [3, 4]]  # Branch 0 perfekt, Branch 1 falsch
    n_acc, prefix = engine._verify_tree(forward_fn, ids, branches)
    assert n_acc == 2
    assert prefix == [1, 2]


def test_verify_tree_short_valid_prefix():
    engine = MTPVerificationEngine(num_branches=2)
    vocab = 10

    def forward_fn(ids):
        B, L = ids.shape
        nxt = (ids + 1) % vocab
        logits = torch.zeros(B, L, vocab)
        logits.scatter_(2, nxt.unsqueeze(-1), 1.0)
        return logits, []

    ids = torch.tensor([[0]])
    branches = [[1, 9], [9, 9]]  # Branch 0 nur Position 1 korrekt
    n_acc, prefix = engine._verify_tree(forward_fn, ids, branches)
    assert n_acc == 1
    assert prefix == [1]


def test_generate_tree_matches_greedy():
    vocab = 10
    engine = MTPVerificationEngine(num_branches=2)
    forward_fn = make_forward(vocab, num_drafts=2)
    input_ids = torch.tensor([[0]])
    out = engine.generate(forward_fn, input_ids, max_new_tokens=8)
    ref = standard_greedy(forward_fn, input_ids, 8)
    assert out == ref == [1, 2, 3, 4, 5, 6, 7, 8]


def test_num_branches_one_identical_to_tree():
    vocab = 10
    forward_fn = make_forward(vocab, num_drafts=2)
    input_ids = torch.tensor([[0]])
    out1 = MTPVerificationEngine(num_branches=1).generate(forward_fn, input_ids, 8)
    out2 = MTPVerificationEngine(num_branches=2).generate(forward_fn, input_ids, 8)
    ref = standard_greedy(forward_fn, input_ids, 8)
    assert out1 == out2 == ref


def test_tree_stops_at_eos():
    vocab = 10
    engine = MTPVerificationEngine(num_branches=2)
    forward_fn = make_forward(vocab, num_drafts=2)
    out = engine.generate(forward_fn, torch.tensor([[0]]), max_new_tokens=10, eos_token_id=4)
    assert out == [1, 2, 3, 4]


def test_tree_respects_max_new_tokens():
    vocab = 10
    engine = MTPVerificationEngine(num_branches=3)
    forward_fn = make_forward(vocab, num_drafts=5)
    out = engine.generate(forward_fn, torch.tensor([[0]]), max_new_tokens=4)
    assert out == [1, 2, 3, 4]
