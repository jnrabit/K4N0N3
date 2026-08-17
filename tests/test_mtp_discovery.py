"""Phase 1 — MTP Submodule Discovery & Layer Mapping.

Prueft, dass der LayerManager MTP/Draft-Head-Module separat von den
Standard-Layern erkennt und mappt, ohne Modelle ohne MTP zu brechen.
"""
import torch
from k4n0n3.hooks import LayerManager


class NoMTPTransformer(torch.nn.Module):
    def __init__(self, num_layers: int = 3, dim: int = 32):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([
                torch.nn.Linear(dim, dim) for _ in range(num_layers)
            ]),
            "norm": torch.nn.LayerNorm(dim),
        })
        self.lm_head = torch.nn.Linear(dim, 10)

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return self.lm_head(self.model["norm"](x))


class MTPTransformer(torch.nn.Module):
    def __init__(self, num_layers: int = 3, num_mtp: int = 2, dim: int = 32):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([
                torch.nn.Linear(dim, dim) for _ in range(num_layers)
            ]),
            "mtp_layers": torch.nn.ModuleList([
                torch.nn.Linear(dim, dim) for _ in range(num_mtp)
            ]),
            "norm": torch.nn.LayerNorm(dim),
        })
        self.lm_head = torch.nn.Linear(dim, 10)

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return self.lm_head(self.model["norm"](x))


class MTPVariantsTransformer(torch.nn.Module):
    """draft_heads (ModuleList) + blk.*.nextn als zusaetzliche MTP-Muster."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([torch.nn.Linear(dim, dim)]),
            "draft_heads": torch.nn.ModuleList([torch.nn.Linear(dim, dim)]),
        })
        self.blk = torch.nn.ModuleList([
            torch.nn.ModuleDict({"nextn": torch.nn.Linear(dim, dim)})
        ])

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return x


def test_default_use_mtp_false():
    mgr = LayerManager(NoMTPTransformer(), layer_prefix="model.layers")
    assert mgr.use_mtp is False


def test_standard_model_mtp_layers_empty():
    mgr = LayerManager(NoMTPTransformer(), layer_prefix="model.layers")
    assert mgr.mtp_layers == []
    assert len(mgr.standard_layers) == 3
    assert mgr.layer_map["mtp_layers"] == []
    assert len(mgr.layer_map["standard_layers"]) == 3


def test_standard_model_use_mtp_true_empty():
    mgr = LayerManager(NoMTPTransformer(), layer_prefix="model.layers", use_mtp=True)
    assert mgr.mtp_layers == []


def test_mtp_model_registers_both():
    mgr = LayerManager(MTPTransformer(), layer_prefix="model.layers", use_mtp=True)
    assert len(mgr.standard_layers) == 3
    assert len(mgr.mtp_layers) == 2
    mapping = mgr.layer_map
    assert len(mapping["standard_layers"]) == 3
    assert len(mapping["mtp_layers"]) == 2
    assert mapping["mtp_layers"] == list(mgr._mtp_layers.values())
    assert mgr._mtp_layer_list == ["model.mtp_layers.0", "model.mtp_layers.1"]


def test_mtp_model_use_mtp_false_ignores_mtp():
    mgr = LayerManager(MTPTransformer(), layer_prefix="model.layers", use_mtp=False)
    assert mgr.mtp_layers == []
    assert len(mgr.standard_layers) == 3


def test_mtp_variant_patterns():
    mgr = LayerManager(MTPVariantsTransformer(), layer_prefix="model.layers", use_mtp=True)
    assert len(mgr.standard_layers) == 1
    names = set(mgr._mtp_layer_list)
    assert "model.draft_heads.0" in names
    assert "blk.0.nextn" in names
    assert len(mgr.mtp_layers) == 2


def test_mtp_not_in_standard_hooks():
    mgr = LayerManager(MTPTransformer(), layer_prefix="model.layers", use_mtp=True)
    assert len(mgr._hook_handles) == 2 * 3
