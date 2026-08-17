"""Phase 4 — Real-World MTP Module Signatures & Tensor Alignment.

Prueft, dass _run_mtp_pass reale MTP-Signaturen (position_ids, attention_mask,
input_ids) korrekt bedient, position_ids um die Draft-Tiefe verschiebt und
single-argument MTP-Module weiterhin direkt aufruft.
"""
import torch

from k4n0n3.hooks import LayerManager


class ComplexMTP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.received_position_ids = None
        self.received_attention_mask = None
        self.received_input_ids = None
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, hidden_states, position_ids, attention_mask=None):
        self.received_position_ids = position_ids
        self.received_attention_mask = attention_mask
        return self.proj(hidden_states)


class InputIdsMTP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.received_input_ids = None
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, hidden_states, input_ids):
        self.received_input_ids = input_ids
        return self.proj(hidden_states)


class SingleArgMTP(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.calls = 0
        self.proj = torch.nn.Linear(dim, dim)

    def forward(self, hidden_states):
        self.calls += 1
        return self.proj(hidden_states)


def _make_model(mtp_modules, dim: int = 16):
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleDict({
                "layers": torch.nn.ModuleList([torch.nn.Linear(dim, dim)]),
                "mtp_layers": torch.nn.ModuleList(mtp_modules),
            })

        def forward(self, input_ids, attention_mask=None, position_ids=None):
            x = input_ids.float().unsqueeze(-1).expand(-1, -1, dim)
            for layer in self.model["layers"]:
                x = layer(x)
            return x

    return M()


def test_complex_signature_no_typeerror():
    dim = 16
    model = _make_model([ComplexMTP(dim)])
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    model(input_ids=ids)  # darf keinen TypeError werfen
    mtp = model.model["mtp_layers"][0]
    assert mtp.received_position_ids is not None
    assert mtp.received_attention_mask is None


def test_position_offsets_align_with_depth():
    dim = 16
    model = _make_model([ComplexMTP(dim), ComplexMTP(dim)])
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    seq = 5
    ids = torch.tensor([list(range(seq))])
    model(input_ids=ids)

    m0 = model.model["mtp_layers"][0]  # Tiefe 1 -> base + 1
    m1 = model.model["mtp_layers"][1]  # Tiefe 2 -> base + 2
    expected_base = torch.arange(seq, dtype=torch.long).unsqueeze(0)
    assert torch.equal(m0.received_position_ids, expected_base + 1)
    assert torch.equal(m1.received_position_ids, expected_base + 2)


def test_position_ids_from_context_shifted():
    dim = 16
    model = _make_model([ComplexMTP(dim)])
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    seq = 4
    ids = torch.tensor([list(range(seq))])
    pos = torch.tensor([[10, 11, 12, 13]])
    model(input_ids=ids, position_ids=pos)
    mtp = model.model["mtp_layers"][0]
    assert torch.equal(mtp.received_position_ids, pos + 1)


def test_input_ids_passed_when_required():
    dim = 16
    model = _make_model([InputIdsMTP(dim)])
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    ids = torch.tensor([[7, 8, 9]])
    model(input_ids=ids)
    mtp = model.model["mtp_layers"][0]
    assert torch.equal(mtp.received_input_ids, ids)


def test_single_arg_mtp_still_works():
    dim = 16
    model = _make_model([SingleArgMTP(dim)])
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    ids = torch.tensor([[1, 2, 3]])
    model(input_ids=ids)
    mtp = model.model["mtp_layers"][0]
    assert mtp.calls == 1
    assert len(mgr.get_mtp_buffer()["model.mtp_layers.0"]) == 1
