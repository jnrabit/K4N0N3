import torch
from k4n0n3.parallel import PipelineParallel, _PartialModel, _build_partial


class DeepTransformer(torch.nn.Module):
    def __init__(self, num_layers: int = 8, dim: int = 32):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([
                torch.nn.Linear(dim, dim) for _ in range(num_layers)
            ]),
        })

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        return x


def test_partial_model():
    layers = [torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)]
    pm = _PartialModel(layers)
    x = torch.randn(4, 16)
    out = pm(x)
    assert out.shape == (4, 16)


def test_build_partial():
    layers = [torch.nn.Linear(16, 16) for _ in range(3)]
    device = torch.device("cpu")
    pm = _build_partial(layers, device, "prefix", 0)
    assert len(pm.layers) == 3


def test_pipeline_requires_gpus():
    if torch.cuda.device_count() >= 2:
        model = DeepTransformer(num_layers=4, dim=32)
        pp = PipelineParallel(
            model,
            layer_prefix="model.layers",
            vram_budget_mb=512,
            verbose=False,
        )
        assert pp.n_gpus >= 2
        assert len(pp.managers) == pp.n_gpus
        pp.prepare()
        pp.remove_hooks()
    else:
        import pytest
        try:
            PipelineParallel(DeepTransformer(num_layers=4))
            assert False, "Should have raised"
        except RuntimeError:
            pass
