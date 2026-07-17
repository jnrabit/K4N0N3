import torch
from k4n0n3.utils import auto_vram_budget, estimate_model_size, get_gpu_info, list_layers


class SampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(64, 128)
        self.linear2 = torch.nn.Linear(128, 10)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def test_get_gpu_info():
    info = get_gpu_info()
    if torch.cuda.is_available():
        assert info is not None
        assert "total_mb" in info
        assert "free_mb" in info
        assert info["total_mb"] > 0
    else:
        assert info is None


def test_auto_vram_budget():
    budget = auto_vram_budget()
    assert isinstance(budget, int)
    assert budget > 0


def test_estimate_model_size():
    model = SampleModel()
    size_mb = estimate_model_size(model)
    assert size_mb > 0
    assert size_mb < 10


def test_list_layers():
    model = SampleModel()
    layers = list_layers(model)
    assert "linear1" in layers
    assert "linear2" in layers
