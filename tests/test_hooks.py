import torch
from k4n0n3.hooks import LayerManager, LayerState


class SimpleTransformer(torch.nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 64):
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
        x = self.model["norm"](x)
        return self.lm_head(x)


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(32, 32),
            torch.nn.Linear(32, 32),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def test_layer_discovery():
    model = SimpleTransformer(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers")
    assert len(mgr._layers) == 4
    assert mgr._layer_list == [
        "model.layers.0", "model.layers.1", "model.layers.2", "model.layers.3"
    ]


def test_layer_initial_state():
    model = SimpleTransformer(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers")
    for info in mgr._layer_info.values():
        assert info.state == LayerState.ON_CPU


def test_auto_discover():
    model = TinyTransformer()
    mgr = LayerManager(model, layer_prefix="layers")
    assert len(mgr._layers) == 2


def test_prepare_offloads_all():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("No CUDA available")
    model = SimpleTransformer(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers")
    mgr.prepare()
    assert mgr.memory.used_bytes() >= 0


def test_hooks_registered():
    model = SimpleTransformer(num_layers=2)
    mgr = LayerManager(model, layer_prefix="model.layers")
    assert len(mgr._hook_handles) == 4


def test_remove_hooks():
    model = SimpleTransformer(num_layers=2)
    mgr = LayerManager(model, layer_prefix="model.layers")
    mgr.remove_hooks()
    assert len(mgr._hook_handles) == 0


def test_layer_sizes_measured():
    model = SimpleTransformer(num_layers=3, dim=128)
    mgr = LayerManager(model, layer_prefix="model.layers")
    for name in mgr._layer_list:
        assert mgr._layer_info[name].size_mb > 0


def test_stats_returns_dict():
    model = TinyTransformer()
    mgr = LayerManager(model, layer_prefix="layers")
    s = mgr.stats()
    assert len(s) == 2
    assert "size_mb" in s["layers.0"]


def test_verbose_mode():
    model = TinyTransformer()
    mgr = LayerManager(model, layer_prefix="layers", verbose=True)
    assert mgr.verbose is True
