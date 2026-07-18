"""Tests fuer Q1: TrainingManager auf Master/Drop, Frozen-Base-Guard, Backward-Prefetch."""
import pytest
import torch

from k4n0n3.training import TrainingManager


class TinyTransformer(torch.nn.Module):
    def __init__(self, dim: int = 256):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dim, dim),
            torch.nn.Linear(dim, dim),
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TinyAdapter(torch.nn.Module):
    """Mini-LoRA: base frozen + kleine trainierbare A/B."""

    def __init__(self, base: torch.nn.Linear, r: int = 2):
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Parameter(torch.randn(r, base.in_features) * 0.02)
        self.lora_B = torch.nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        out = self.base(x)
        delta = x.float() @ self.lora_A.T @ self.lora_B.T
        return out + delta.to(out.dtype)


def frozen_model_with_adapters(dim: int = 256) -> TinyTransformer:
    model = TinyTransformer(dim)
    for p in model.parameters():
        p.requires_grad_(False)
    for i, layer in enumerate(model.layers):
        model.layers[i] = TinyAdapter(layer)
    return model


def test_training_hooks_registered():
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers", vram_budget_mb=1024)
    # 2 layers × 4 hooks (fw_pre, fw_post, bw_pre, bw_post) = 8
    assert len(mgr._hook_handles) == 8


def test_training_layer_discovery():
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers")
    assert len(mgr._layers) == 2
    assert mgr._layer_list == ["layers.0", "layers.1"]


def test_unfrozen_base_raises():
    """Q1-Guard: unfrozen Basis muss hart abgelehnt werden."""
    model = TinyTransformer()  # alles requires_grad=True
    mgr = TrainingManager(model, layer_prefix="layers")
    with pytest.raises(ValueError, match="frozen"):
        mgr.prepare()


def test_frozen_base_with_adapters_prepares():
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers")
    mgr.prepare()
    assert mgr.memory.used_bytes() >= 0


def test_adapters_not_in_master():
    """Trainierbare Params (Adapter) duerfen keinen Master-Eintrag bekommen."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA available")
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers")
    for name in mgr._layer_list:
        master = mgr._cpu_master[name]
        assert not any("lora" in k for k in master), master.keys()
        assert any("base" in k for k in master)


def test_backward_prefetch_direction():
    """bw_post prefetcht rueckwaerts (idx-1) und setzt die Phase zurueck."""
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers", prefetch_depth=1)
    calls = []
    mgr._prefetch_async = lambda name: calls.append(name)  # type: ignore[method-assign]
    mgr._offload = lambda name: None  # type: ignore[method-assign]
    mgr._phase = "bw"
    hook = mgr._make_bw_post("layers.1")
    hook(mgr._layers["layers.1"], None, None)
    assert calls == ["layers.0"]
    hook0 = mgr._make_bw_post("layers.0")
    hook0(mgr._layers["layers.0"], None, None)
    assert mgr._phase == "fw"


def test_fw_post_passive_during_backward_recompute():
    """Waehrend der Checkpointing-Rekomputation darf fw_post nicht vorwaerts prefetchen."""
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers", prefetch_depth=1)
    calls = []
    mgr._prefetch_async = lambda name: calls.append(name)  # type: ignore[method-assign]
    mgr._phase = "bw"
    hook = mgr._make_fw_post("layers.0")
    hook(mgr._layers["layers.0"], None, None)
    assert calls == []


def test_training_forward_backward_adapter_grads():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA available")

    model = frozen_model_with_adapters()
    model.train()
    mgr = TrainingManager(model, layer_prefix="layers", vram_budget_mb=4096)
    mgr.prepare()

    x = torch.randn(4, 256, requires_grad=True, device="cuda")
    loss = model(x).sum()
    loss.backward()

    assert x.grad is not None
    for n, p in model.named_parameters():
        if "lora" in n:
            assert p.grad is not None, n
            assert p.device.type == "cuda"
        else:
            assert p.grad is None, n
    mgr.remove_hooks()


def test_training_quantize_transfer_smoke():
    if not torch.cuda.is_available():
        pytest.skip("No CUDA available")
    model = frozen_model_with_adapters()
    model.half()
    # Adapter zurueck auf fp32 (AdamW-Stabilitaet, wie in bench/train_lora.py)
    for n, p in model.named_parameters():
        if "lora" in n:
            p.data = p.data.float()
    model.train()
    mgr = TrainingManager(model, layer_prefix="layers", quantize_transfer=True)
    mgr.prepare()
    x = torch.randn(2, 256, dtype=torch.float16, device="cuda")
    out = model(x)
    assert torch.isfinite(out.float()).all()
    mgr.remove_hooks()


def test_training_remove_hooks():
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers")
    mgr.remove_hooks()
    assert len(mgr._hook_handles) == 0


def test_training_report():
    model = frozen_model_with_adapters()
    mgr = TrainingManager(model, layer_prefix="layers")
    r = mgr.report()
    assert "Budget" in r or "No data" in r
