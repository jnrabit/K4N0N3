"""Phase 3 — Speculative Verification Engine & Multi-Token Generation Loop.

Prueft Draft-Extraktion, greedy-Verifikation, 1-Token-Fallback und die
Greedy-Aequivalenz des MTP-Decode-Loops gegen Standard-Greedy.
"""
import torch

from k4n0n3.mtp_engine import MTPVerificationEngine


def _one_hot_logits(token: int, vocab: int) -> torch.Tensor:
    logits = torch.zeros(vocab)
    logits[token] = 1.0
    return logits


def test_extract_draft_tokens():
    dim, vocab = 8, 5
    head = torch.nn.Linear(dim, vocab)
    head.weight.data.zero_()
    head.bias.data = torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0])  # argmax = 2

    engine = MTPVerificationEngine()
    buffer = {
        "model.mtp_layers.0": [torch.randn(1, 4, dim)],
        "model.mtp_layers.1": [torch.randn(1, 4, dim)],
    }
    logits = engine.extract_draft_logits(buffer, head)
    assert len(logits) == 2
    assert logits[0].shape == (1, vocab)
    assert engine.greedy_tokens(logits) == [2, 2]


def test_extract_draft_logits_without_head():
    # Output hat bereits Vokabular-Breite -> direkt als Logits (head=None).
    vocab = 6
    engine = MTPVerificationEngine()
    buffer = {
        "model.mtp_layers.0": [_one_hot_logits(4, vocab).unsqueeze(0).unsqueeze(0)],
    }
    logits = engine.extract_draft_logits(buffer, None)
    assert engine.greedy_tokens(logits) == [4]


def test_verify_accepts_correct():
    engine = MTPVerificationEngine()
    t1 = _one_hot_logits(1, 3).unsqueeze(0)
    t2 = _one_hot_logits(2, 3).unsqueeze(0)
    assert engine.verify_drafts([1, 2], [t1, t2]) == 2


def test_verify_fallback_on_incorrect():
    engine = MTPVerificationEngine()
    t1 = _one_hot_logits(1, 10).unsqueeze(0)
    t2 = _one_hot_logits(2, 10).unsqueeze(0)
    # Erster Draft falsch -> 0 akzeptierte Drafts (1-Token-Fallback).
    assert engine.verify_drafts([9, 2], [t1, t2]) == 0
    # Erster richtig, zweiter falsch -> 1 akzeptierter Draft.
    assert engine.verify_drafts([1, 99], [t1, t2]) == 1


def _make_cyclic_forward(vocab: int, num_drafts: int):
    """next(t) = (t+1) % vocab; perfekte Drafts d_i = next^i(t)."""

    def forward_fn(ids):
        last = int(ids[0][-1].item())
        logits = _one_hot_logits((last + 1) % vocab, vocab).unsqueeze(0).unsqueeze(0)
        drafts = []
        for i in range(1, num_drafts + 1):
            drafts.append(_one_hot_logits((last + i + 1) % vocab, vocab).unsqueeze(0))
        return logits, drafts

    return forward_fn


def _standard_greedy(forward_fn, input_ids, max_new_tokens):
    ids = input_ids
    out = []
    for _ in range(max_new_tokens):
        logits, _ = forward_fn(ids)
        t = int(torch.argmax(logits[:, -1]))
        out.append(t)
        ids = torch.cat([ids, ids.new_tensor([[t]])], dim=1)
    return out


def test_generate_matches_greedy():
    vocab = 10
    engine = MTPVerificationEngine()
    forward_fn = _make_cyclic_forward(vocab, num_drafts=2)
    input_ids = torch.tensor([[0]])
    out = engine.generate(forward_fn, input_ids, max_new_tokens=8)
    ref = _standard_greedy(forward_fn, input_ids, 8)
    assert out == ref == [1, 2, 3, 4, 5, 6, 7, 8]


def test_generate_fallback_still_matches_greedy():
    vocab = 10
    engine = MTPVerificationEngine()

    def broken_draft_forward(ids):
        last = int(ids[0][-1].item())
        logits = _one_hot_logits((last + 1) % vocab, vocab).unsqueeze(0).unsqueeze(0)
        # Absichtlich falscher Draft -> muss auf 1 Token zurueckfallen.
        drafts = [_one_hot_logits(99 % vocab, vocab).unsqueeze(0)]
        return logits, drafts

    input_ids = torch.tensor([[0]])
    out = engine.generate(broken_draft_forward, input_ids, max_new_tokens=6)
    ref = _standard_greedy(broken_draft_forward, input_ids, 6)
    assert out == ref == [1, 2, 3, 4, 5, 6]


def test_generate_stops_at_eos():
    vocab = 10
    engine = MTPVerificationEngine()

    def forward_fn(ids):
        last = int(ids[0][-1].item())
        logits = _one_hot_logits(last + 1, vocab).unsqueeze(0).unsqueeze(0)
        return logits, []

    out = engine.generate(forward_fn, torch.tensor([[0]]), max_new_tokens=10, eos_token_id=4)
    assert out == [1, 2, 3, 4]


def test_generate_respects_max_new_tokens():
    vocab = 10
    engine = MTPVerificationEngine()
    forward_fn = _make_cyclic_forward(vocab, num_drafts=5)
    out = engine.generate(forward_fn, torch.tensor([[0]]), max_new_tokens=3)
    assert len(out) == 3
    assert out == [1, 2, 3]
