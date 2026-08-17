"""Phase 2 — Dual-Pass Forward Hooks & MTP Output Buffering.

Prueft, dass bei use_mtp=True der MTP/Draft-Pass im Post-Hook des zugehoerigen
Layers laeuft, seine Outputs buffert und die Eviction erst danach passiert.
"""
import torch
import torch.nn.functional as F

from k4n0n3.hooks import LayerManager


class MTPModel(torch.nn.Module):
    """Standard-Layer + parallele MTP-Liste (Leaf-Linear-Module)."""

    def __init__(self, num_layers: int = 4, num_mtp: int = 2, dim: int = 16, mtp_dim: int = 8):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)]),
            "mtp_layers": torch.nn.ModuleList([torch.nn.Linear(dim, mtp_dim) for _ in range(num_mtp)]),
            "norm": torch.nn.LayerNorm(dim),
        })
        self.lm_head = torch.nn.Linear(dim, 5)

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return self.lm_head(self.model["norm"](x))


class RecordingMTP(torch.nn.Module):
    """MTP-Modul mit eigenen Parametern, das jede Ausfuehrung protokolliert."""

    def __init__(self, event_name: str, dim: int, out_dim: int, timeline: list):
        super().__init__()
        self.event_name = event_name
        self.timeline = timeline
        self.weight = torch.nn.Parameter(torch.randn(out_dim, dim))
        self.bias = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        self.timeline.append(f"mtp:{self.event_name}")
        return F.linear(x, self.weight, self.bias)


class RecordingModel(torch.nn.Module):
    def __init__(self, num_layers: int = 6, dim: int = 16, mtp_dim: int = 8, timeline: list | None = None):
        super().__init__()
        self.timeline = timeline if timeline is not None else []
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)]),
            "mtp_layers": torch.nn.ModuleList([
                RecordingMTP(f"mtp.{i}", dim, mtp_dim, self.timeline) for i in range(2)
            ]),
            "norm": torch.nn.LayerNorm(dim),
        })

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return self.model["norm"](x)


def test_mtp_forward_populates_buffer():
    model = MTPModel(num_layers=4, num_mtp=2, dim=16, mtp_dim=8)
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    model(torch.randn(2, 5, 16))
    buf = mgr.get_mtp_buffer()
    assert set(buf.keys()) == {"model.mtp_layers.0", "model.mtp_layers.1"}
    assert buf["model.mtp_layers.0"][0].shape == (2, 5, 8)
    assert buf["model.mtp_layers.1"][0].shape == (2, 5, 8)


def test_eviction_after_mtp():
    timeline: list[str] = []
    model = RecordingModel(timeline=timeline)
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True, prefetch_depth=1)
    orig_offload = mgr._offload

    def spy_offload(name: str):
        timeline.append(f"offload:{name}")
        return orig_offload(name)

    mgr._offload = spy_offload
    model(torch.randn(2, 5, 16))

    assert timeline.index("mtp:mtp.0") < timeline.index("offload:model.layers.0")
    assert timeline.index("mtp:mtp.1") < timeline.index("offload:model.layers.1")


def test_clear_mtp_buffer():
    model = MTPModel(num_layers=4, num_mtp=2, dim=16, mtp_dim=8)
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    model(torch.randn(2, 5, 16))
    assert len(mgr.get_mtp_buffer()) == 2
    mgr.clear_mtp_buffer()
    assert mgr.get_mtp_buffer() == {}


def test_buffer_reset_between_steps():
    model = MTPModel(num_layers=4, num_mtp=1, dim=16, mtp_dim=8)
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    model(torch.randn(2, 5, 16))
    first = mgr.get_mtp_buffer()["model.mtp_layers.0"][0]
    model(torch.randn(2, 5, 16))
    buf = mgr.get_mtp_buffer()
    assert len(buf["model.mtp_layers.0"]) == 1
    assert buf["model.mtp_layers.0"][0] is not first


def test_use_mtp_false_no_buffer():
    model = MTPModel(num_layers=4, num_mtp=2, dim=16, mtp_dim=8)
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=False)
    model(torch.randn(2, 5, 16))
    assert mgr.get_mtp_buffer() == {}


def test_mtp_output_no_grad():
    model = MTPModel(num_layers=4, num_mtp=2, dim=16, mtp_dim=8)
    mgr = LayerManager(model, layer_prefix="model.layers", use_mtp=True)
    model(torch.randn(2, 5, 16))
    out = mgr.get_mtp_buffer()["model.mtp_layers.0"][0]
    assert out.requires_grad is False
