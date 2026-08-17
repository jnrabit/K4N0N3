"""MTP Weight Reconstruction — liest MTP/Draft-Gewichte aus .safetensors und
baut funktionale MTPDraftHead-Submodule, die die Discovery findet."""
import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from k4n0n3.hooks import LayerManager
from k4n0n3.mtp_loader import MTPDraftHead, reconstruct_and_attach_mtp


class HostModel(nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 8):
        super().__init__()
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)]),
        })

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return x


def test_mtp_draft_head_forward():
    dim, vocab = 8, 5
    head = MTPDraftHead([
        nn.Linear(dim, dim),
        nn.Linear(dim, vocab),
    ])
    out = head(torch.randn(2, 3, dim))
    assert out.shape == (2, 3, vocab)


def test_reconstruct_from_safetensors(tmp_path):
    dim, vocab = 8, 5
    state = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(dim, dim),
        "mtp.0.proj.weight": torch.full((dim, dim), 1.0),
        "mtp.0.proj.bias": torch.full((dim,), 2.0),
        "mtp.0.head.weight": torch.full((vocab, dim), 3.0),
        "mtp.1.proj.weight": torch.full((dim, dim), 4.0),
        "mtp.1.head.weight": torch.full((vocab, dim), 5.0),
    }
    save_file(state, str(tmp_path / "model.safetensors"))

    model = HostModel(num_layers=4, dim=dim)
    heads = reconstruct_and_attach_mtp(model, str(tmp_path))
    assert heads is not None and len(heads) == 2

    def head_vals(head):
        return {float(l.weight.flatten()[0].item()) for l in head.layers}

    assert head_vals(model.mtp_layers["0"]) == {1.0, 3.0}
    assert head_vals(model.mtp_layers["1"]) == {4.0, 5.0}
    biases = [l for l in model.mtp_layers["0"].layers if l.bias is not None]
    assert len(biases) == 1 and float(biases[0].bias[0].item()) == 2.0


def test_attach_and_discover(tmp_path):
    dim, vocab = 8, 5
    state = {
        "mtp.0.proj.weight": torch.randn(dim, dim),
        "mtp.0.head.weight": torch.randn(vocab, dim),
        "mtp.1.proj.weight": torch.randn(dim, dim),
        "mtp.1.head.weight": torch.randn(vocab, dim),
    }
    save_file(state, str(tmp_path / "model.safetensors"))

    model = HostModel(num_layers=4, dim=dim)
    reconstruct_and_attach_mtp(model, str(tmp_path))

    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    assert set(mgr._mtp_layer_list) == {"mtp_layers.0", "mtp_layers.1"}
    assert mgr._layer_to_mtp["model.layers.0"] == ["mtp_layers.0"]
    assert mgr._layer_to_mtp["model.layers.1"] == ["mtp_layers.1"]
    assert len(mgr.mtp_layers) == 2


def test_fallback_no_mtp(tmp_path):
    state = {"model.layers.0.weight": torch.randn(8, 8)}
    save_file(state, str(tmp_path / "model.safetensors"))

    model = HostModel(num_layers=4, dim=8)
    with pytest.warns(UserWarning):
        heads = reconstruct_and_attach_mtp(model, str(tmp_path))
    assert heads is None
    assert not hasattr(model, "mtp_layers")


def test_fallback_no_safetensors(tmp_path):
    model = HostModel(num_layers=4, dim=8)
    with pytest.warns(UserWarning):
        heads = reconstruct_and_attach_mtp(model, str(tmp_path))
    assert heads is None


def test_layer_index_based_detection(tmp_path):
    """model.layers.{i} mit i >= num_hidden_layers wird als MTP erkannt."""
    dim, vocab = 8, 5
    state = {
        "model.layers.0.weight": torch.randn(dim, dim),
        "model.layers.1.weight": torch.randn(dim, dim),
        "model.layers.2.weight": torch.randn(dim, dim),  # MTP (>= 2)
        "model.layers.2.bias": torch.randn(dim),
    }
    save_file(state, str(tmp_path / "model.safetensors"))

    model = HostModel(num_layers=2, dim=dim)
    model.config = type("Cfg", (), {"num_hidden_layers": 2})()
    heads = reconstruct_and_attach_mtp(model, str(tmp_path))
    assert heads is not None and len(heads) == 1
    assert len(model.mtp_layers["0"].layers) == 1
    assert torch.equal(model.mtp_layers["0"].layers[0].weight, state["model.layers.2.weight"])
