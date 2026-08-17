"""Layer-Pinning: gepinnte Layer bleiben dauerhaft auf der GPU (resident),
unpinned Layer folgen weiter dem normalen Offload/LRU-Verhalten."""
import pytest
import torch

from k4n0n3.hooks import LayerManager, LayerState


class TinyModel(torch.nn.Module):
    def __init__(self, num_layers: int = 4, dim: int = 32):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(num_layers)]),
        })

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return x


def _params_device(mgr: LayerManager, name: str) -> str:
    devices = {p.device.type for p in mgr._layers[name].parameters()}
    return devices.pop() if len(devices) == 1 else "mixed"


def test_pinned_index_normalization():
    model = TinyModel(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers", pinned_layers=[0, -1])
    assert mgr.pinned_layer_indices == {0, 3}
    assert mgr._pinned_names == {"model.layers.0", "model.layers.3"}


def test_pinned_string_resolution():
    model = TinyModel(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers",
                       pinned_layers=["0", "model.layers.2", "3"])
    assert mgr.pinned_layer_indices == {0, 2, 3}


def test_pinned_negative_string():
    model = TinyModel(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers", pinned_layers=["-1"])
    assert mgr.pinned_layer_indices == {3}


def test_pinned_empty_default():
    model = TinyModel(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers")
    assert mgr.pinned_layer_indices == set()
    assert mgr._pinned_names == set()


def test_pinned_invalid_raises():
    model = TinyModel(num_layers=4)
    with pytest.raises(ValueError):
        LayerManager(model, layer_prefix="model.layers", pinned_layers=[10])
    with pytest.raises(ValueError):
        LayerManager(model, layer_prefix="model.layers", pinned_layers=["nope"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm noetig")
def test_pinned_stay_on_gpu_after_offload():
    model = TinyModel(num_layers=4, dim=64)
    mgr = LayerManager(model, layer_prefix="model.layers", vram_budget_mb=512,
                       pinned_layers=[0, -1])
    mgr.prepare()
    # Gepinnte Layer vorgeladen
    assert _params_device(mgr, "model.layers.0") == "cuda"
    assert _params_device(mgr, "model.layers.3") == "cuda"
    # Offload eines gepinnten Layers ist No-op
    mgr._offload("model.layers.0")
    assert mgr._layer_info["model.layers.0"].state == LayerState.ON_GPU
    assert _params_device(mgr, "model.layers.0") == "cuda"
    # Offload eines unpinned Layers verschiebt nach CPU
    mgr._offload("model.layers.1")
    assert mgr._layer_info["model.layers.1"].state == LayerState.ON_CPU
    assert _params_device(mgr, "model.layers.1") == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm noetig")
def test_forward_keeps_pinned_on_gpu():
    model = TinyModel(num_layers=4, dim=64)
    mgr = LayerManager(model, layer_prefix="model.layers", vram_budget_mb=512,
                       pinned_layers=[0, -1])
    mgr.prepare()
    x = torch.randn(2, 8, 64, device="cuda")
    with torch.no_grad():
        model(x)
    assert _params_device(mgr, "model.layers.0") == "cuda"
    assert _params_device(mgr, "model.layers.3") == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/ROCm noetig")
def test_offload_all_keeps_pinned_resident():
    model = TinyModel(num_layers=4, dim=64)
    mgr = LayerManager(model, layer_prefix="model.layers", pinned_layers=[0, -1])
    mgr.prepare()
    mgr.offload_all()
    assert _params_device(mgr, "model.layers.0") == "cuda"
    assert _params_device(mgr, "model.layers.3") == "cuda"
    assert _params_device(mgr, "model.layers.1") == "cpu"
