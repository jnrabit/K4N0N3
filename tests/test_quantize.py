"""Tests fuer Auftrag M: custom weight-only int8 + On-GPU-Dequant."""
import pytest
import torch

from k4n0n3.hooks import (
    LayerManager,
    dequantize_groupwise_int4,
    dequantize_int8,
    quantize_groupwise_int4,
    quantize_per_channel_int8,
)


class HalfTransformer(torch.nn.Module):
    """Kleines fp16-Modell mit Linear-Layern als getrackte Layer."""

    def __init__(self, num_layers: int = 4, dim: int = 64):
        super().__init__()
        self.model = torch.nn.ModuleDict({
            "layers": torch.nn.ModuleList([
                torch.nn.Linear(dim, dim) for _ in range(num_layers)
            ]),
            "norm": torch.nn.LayerNorm(dim),
        })
        self.lm_head = torch.nn.Linear(dim, 10)
        self.half()

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer(x)
        x = self.model["norm"](x)
        return self.lm_head(x)


# -- Quantisierungs-Mathematik (CPU, immer lauffaehig) -----------------------


def test_quantize_shapes_dtypes():
    w = torch.randn(32, 64, dtype=torch.float16)
    q, scale = quantize_per_channel_int8(w)
    assert q.dtype == torch.int8 and q.shape == w.shape
    assert scale.dtype == torch.float16 and scale.shape == (32,)


def test_quantize_roundtrip_error_bounded():
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.float16)
    q, scale = quantize_per_channel_int8(w)
    deq = dequantize_int8(q, scale)
    # Fehler pro Element <= scale/2 (Rundung) + fp16-Rundung von Scale und
    # Produkt (je <= 127*scale*2^-11): theoretisch <= 0.625*scale
    bound = scale.float().unsqueeze(1) * 0.65 + 1e-6
    assert ((w.float() - deq.float()).abs() <= bound).all()


def test_quantize_zero_row_stable():
    w = torch.zeros(4, 16, dtype=torch.float16)
    w[1, :] = 3.0
    q, scale = quantize_per_channel_int8(w)
    deq = dequantize_int8(q, scale)
    assert torch.isfinite(deq).all()
    assert (deq[0] == 0).all()
    assert torch.allclose(deq[1].float(), w[1].float(), rtol=0.02)


def test_quantize_range_full():
    w = torch.randn(8, 8, dtype=torch.float16)
    q, _ = quantize_per_channel_int8(w)
    assert q.max() <= 127 and q.min() >= -127
    # Pro Kanal wird das Betragsmaximum auf 127 abgebildet
    assert (q.abs().amax(dim=1) == 127).all()


# -- P: int4 group-wise gepackt ----------------------------------------------


def test_int4_pack_shapes():
    w = torch.randn(16, 64, dtype=torch.float16)
    packed, scale, meta = quantize_groupwise_int4(w, group_size=32)
    assert packed.dtype == torch.uint8 and packed.shape == (16, 32)
    assert scale.dtype == torch.float16 and scale.shape == (16, 2)
    assert meta == {"group_size": 32, "orig_shape": (16, 64)}


def test_int4_roundtrip_error_bounded():
    torch.manual_seed(3)
    w = torch.randn(64, 256, dtype=torch.float16)
    packed, scale, meta = quantize_groupwise_int4(w, group_size=128)
    deq = dequantize_groupwise_int4(packed, scale, meta)
    assert deq.shape == w.shape and deq.dtype == torch.float16
    # Rundung <= scale/2 plus fp16-Terme, pro Gruppe expandiert
    bound = scale.float().repeat_interleave(128, dim=1) * 0.65 + 1e-6
    assert ((w.float() - deq.float()).abs() <= bound).all()


def test_int4_last_group_shorter():
    torch.manual_seed(5)
    w = torch.randn(8, 96, dtype=torch.float16)  # 96 = 64 + Restgruppe 32
    packed, scale, meta = quantize_groupwise_int4(w, group_size=64)
    assert scale.shape == (8, 2)
    deq = dequantize_groupwise_int4(packed, scale, meta)
    assert deq.shape == w.shape
    reps = torch.tensor([64, 32])
    bound = scale.float().repeat_interleave(reps, dim=1) * 0.65 + 1e-6
    assert ((w.float() - deq.float()).abs() <= bound).all()


def test_int4_negative_values_roundtrip():
    w = torch.tensor([[-7.0, 7.0, -1.0, 0.0]], dtype=torch.float16)
    packed, scale, meta = quantize_groupwise_int4(w, group_size=4)
    deq = dequantize_groupwise_int4(packed, scale, meta)
    assert torch.allclose(deq.float(), w.float(), atol=0.51)


def test_int4_groupwise_beats_per_channel_on_outliers():
    # Ein Ausreisser am Kanalende darf die Quantisierung der ersten Gruppe
    # nicht mehr verzerren — der Kern des group-wise-Arguments.
    w = torch.randn(4, 256, dtype=torch.float16) * 0.1
    w[:, -1] = 50.0
    packed, scale, meta = quantize_groupwise_int4(w, group_size=128)
    deq = dequantize_groupwise_int4(packed, scale, meta)
    err_first_group = (w[:, :128].float() - deq[:, :128].float()).abs().mean()
    assert err_first_group < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_int4_mechanik_offload_vs_full_gpu_identisch():
    from k4n0n3.hooks import _upload_layer

    torch.manual_seed(4)
    model = HalfTransformer(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers",
                       vram_budget_mb=1, quantize_transfer="int4")
    m0 = mgr._cpu_master["model.layers.0"]
    assert "q4" in m0["weight"] and m0["weight"]["q4"].dtype == torch.uint8
    x = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
    mgr.prepare()
    with torch.no_grad():
        y_offload = model(x).clone()
    mgr.remove_hooks()
    for name in mgr._layer_list:
        _upload_layer(mgr._layers[name], mgr._cpu_master[name], mgr._param_refs[name])
    torch.cuda.synchronize()
    with torch.no_grad():
        y_full = model(x)
    assert torch.equal(y_offload, y_full)


# -- O: Pin-Budget (Zwei-Pass + Per-Layer-Reprobe) ---------------------------


def test_can_pin_fraction_zero_forces_unpinned():
    model = HalfTransformer()
    mgr = LayerManager(model, layer_prefix="model.layers", pin_ram_fraction=0.0)
    assert mgr._can_pin(1) is False
    assert not any(mgr._pinned.values())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_quantized_two_pass_pins_with_fresh_budget(monkeypatch):
    """O1: Budget wird nach der Quantisierung geprobt, nicht davor."""
    import k4n0n3.hooks as hooks_mod

    probes = []

    real = hooks_mod._available_ram_bytes

    def tracking_probe():
        probes.append(True)
        return real()

    monkeypatch.setattr(hooks_mod, "_available_ram_bytes", tracking_probe)
    model = HalfTransformer(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers", quantize_transfer=True)
    # Per-Layer-Reprobe: eine Probe pro Layer (Pass 2), nicht eine einzige
    assert len(probes) == 4
    assert all(mgr._pinned.values())  # Winz-Modell muss komplett pinnbar sein


# -- LayerManager-Integration ------------------------------------------------


def test_quantize_transfer_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = HalfTransformer()
    with pytest.raises(ValueError, match="quantize_transfer"):
        LayerManager(model, layer_prefix="model.layers", quantize_transfer=True)


def test_default_flag_off_changes_nothing():
    model = HalfTransformer()
    mgr = LayerManager(model, layer_prefix="model.layers")
    assert mgr._quantize_transfer is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_master_structure_quantized():
    model = HalfTransformer()
    mgr = LayerManager(model, layer_prefix="model.layers", quantize_transfer=True)
    m0 = mgr._cpu_master["model.layers.0"]
    entry = m0["weight"]
    assert isinstance(entry, dict) and set(entry) == {"q", "scale"}
    assert entry["q"].dtype == torch.int8
    assert entry["scale"].dtype == torch.float16
    # Bias bleibt unquantisierter direkter Master
    assert isinstance(m0["bias"], torch.Tensor)
    # Drop-Zustand: p.data zeigt auf den int8-Master
    w = model.model["layers"][0].weight
    assert w.data.dtype == torch.int8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_mechanik_offload_vs_full_gpu_identisch():
    """M3 Punkt 1 im Kleinen: gleiche int8-Master, mit vs. ohne Offloading."""
    from k4n0n3.hooks import _upload_layer

    torch.manual_seed(1)
    model = HalfTransformer(num_layers=4)
    mgr = LayerManager(model, layer_prefix="model.layers",
                       vram_budget_mb=1, quantize_transfer=True)
    x = torch.randn(2, 8, 64, dtype=torch.float16, device="cuda")
    mgr.prepare()
    with torch.no_grad():
        y_offload = model(x).clone()

    mgr.remove_hooks()
    for name in mgr._layer_list:
        _upload_layer(mgr._layers[name], mgr._cpu_master[name], mgr._param_refs[name])
    torch.cuda.synchronize()
    with torch.no_grad():
        y_full = model(x)
    assert torch.equal(y_offload, y_full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA available")
def test_quantized_forward_close_to_fp16():
    torch.manual_seed(2)
    model = HalfTransformer(num_layers=2)
    x = torch.randn(2, 8, 64, dtype=torch.float16)
    with torch.no_grad():
        y_ref = model(x.clone()).float()

    mgr = LayerManager(model, layer_prefix="model.layers", quantize_transfer=True)
    mgr.prepare()
    with torch.no_grad():
        y_q = model(x.to("cuda")).float().cpu()
    assert torch.allclose(y_ref, y_q, atol=0.05, rtol=0.05)