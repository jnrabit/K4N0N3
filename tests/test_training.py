import torch
from k4n0n3.training import TrainingManager


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


def test_training_hooks_registered():
    model = TinyTransformer()
    mgr = TrainingManager(model, layer_prefix="layers", vram_budget_mb=1024)
    # 2 layers × 4 hooks (fw_pre, fw_post, bw_pre, bw_post) = 8
    assert len(mgr._hook_handles) == 8


def test_training_layer_discovery():
    model = TinyTransformer()
    mgr = TrainingManager(model, layer_prefix="layers")
    assert len(mgr._layers) == 2
    assert mgr._layer_list == ["layers.0", "layers.1"]


def test_training_prepare():
    model = TinyTransformer()
    mgr = TrainingManager(model, layer_prefix="layers")
    mgr.prepare()
    assert mgr.memory.used_bytes() >= 0


def test_training_forward_backward():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("No CUDA available")

    model = TinyTransformer()
    model.train()
    mgr = TrainingManager(model, layer_prefix="layers", vram_budget_mb=4096)
    mgr.prepare()

    x = torch.randn(4, 32, requires_grad=True, device="cuda")
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    for p in model.parameters():
        assert p.grad is not None

    mgr.remove_hooks()


def test_training_remove_hooks():
    model = TinyTransformer()
    mgr = TrainingManager(model, layer_prefix="layers")
    mgr.remove_hooks()
    assert len(mgr._hook_handles) == 0


def test_training_report():
    model = TinyTransformer()
    mgr = TrainingManager(model, layer_prefix="layers")
    r = mgr.report()
    assert "Budget" in r or "No data" in r
