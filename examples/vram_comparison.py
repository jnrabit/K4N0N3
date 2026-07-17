"""K4N0N3 VRAM Benchmark: Standard PyTorch vs K4N0N3 — same model, same forward."""
from __future__ import annotations

import time
import torch
from k4n0n3 import LayerManager

torch.backends.cudnn.benchmark = False


def banner(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


def vram_used_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2


def vram_peak_reset() -> None:
    torch.cuda.reset_peak_memory_stats()


def vram_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


class _GPT2StyleModel(torch.nn.Module):
    """12-layer decoder-only transformer (GPT-2 scale approx)."""
    def __init__(self, n_layers: int = 12, dim: int = 768, heads: int = 12):
        super().__init__()
        self.wte = torch.nn.Embedding(50257, dim)
        self.wpe = torch.nn.Embedding(1024, dim)
        self.layers = torch.nn.ModuleList([
            _TransformerBlock(dim, heads) for _ in range(n_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(dim)
        self.lm_head = torch.nn.Linear(dim, 50257, bias=False)

    def forward(self, x):
        x = self.wte(x) + self.wpe(torch.arange(x.size(1), device=x.device))
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class _TransformerBlock(torch.nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln_2 = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x


def layer_size_mb(layers: dict[str, torch.nn.Module]) -> float:
    return sum(
        sum(p.numel() * p.element_size() for p in m.parameters()) / 1024**2
        for m in layers.values()
    )


def fixed_size_mb(model: torch.nn.Module, layer_ids: set[int]) -> float:
    total = 0.0
    for m in model.modules():
        if id(m) in layer_ids:
            continue
        for p in m.parameters(recurse=False):
            total += p.numel() * p.element_size()
    return total / 1024**2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    n_layers = 12
    dim = 768

    print(f"GPU: {name}  |  Model: {n_layers} layers × {dim} dim  |  float32")

    # ── Build reference weights ──────────────────────────────────────────

    ref = _GPT2StyleModel(n_layers=n_layers, dim=dim).to("cpu")
    total_mb = sum(p.numel() * p.element_size() for p in ref.parameters()) / 1024**2
    print(f"Total params: {total_mb:.0f} MB")

    x = torch.randint(0, 50257, (4, 64), device=device)

    # ── 1. Standard PyTorch ──────────────────────────────────────────────

    banner("1. Standard PyTorch — all layers on GPU")

    model_std = _GPT2StyleModel(n_layers=n_layers, dim=dim).to("cpu")
    model_std.load_state_dict(ref.state_dict())
    model_std.to(device)

    # Warmup (kernel compilation)
    with torch.no_grad():
        model_std(x)
    torch.cuda.synchronize()

    vram_peak_reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_std = model_std(x)
    torch.cuda.synchronize()
    dt_std = (time.perf_counter() - t0) * 1000
    peak_std = vram_peak_mb()

    print(f"Forward: {dt_std:.1f} ms  |  Peak VRAM: {peak_std:.0f} MB  |  All {n_layers} layers GPU")
    del model_std, out_std
    torch.cuda.empty_cache()

    # ── 2. K4N0N3 — streaming ────────────────────────────────────────────

    for prefetch in [1, 2]:
        label = f"prefetch={prefetch} (max {prefetch+1} layers)"
        banner(f"2. K4N0N3 — {label}")

        model_k = _GPT2StyleModel(n_layers=n_layers, dim=dim).to("cpu")
        model_k.load_state_dict(ref.state_dict())

        mgr = LayerManager(
            model_k, layer_prefix="layers",
            vram_budget_mb=4096,
            prefetch_depth=prefetch,
            verbose=False,
        )
        mgr.prepare()

        # Warmup
        with torch.no_grad():
            model_k(x)
        torch.cuda.synchronize()

        vram_peak_reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            out_k = model_k(x)
        torch.cuda.synchronize()
        dt_k = (time.perf_counter() - t0) * 1000
        peak_k = vram_peak_mb()

        layer_ids: set[int] = set()
        for m in mgr._layers.values():
            for sub in m.modules():
                layer_ids.add(id(sub))

        fixed_mb = fixed_size_mb(model_k, layer_ids)
        one_layer_mb = layer_size_mb({mgr._layer_list[0]: mgr._layers[mgr._layer_list[0]]})
        max_layers_on_gpu = prefetch + 1
        theory_peak = fixed_mb + one_layer_mb * max_layers_on_gpu

        print(f"Forward: {dt_k:.1f} ms  |  Peak VRAM: {peak_k:.0f} MB")
        print(f"Theory: {theory_peak:.0f} MB  (fixed: {fixed_mb:.0f} + {max_layers_on_gpu}×{one_layer_mb:.0f})")
        print(f"VRAM saved: {peak_std - peak_k:.0f} MB ({100*(peak_std-peak_k)/peak_std:.0f}%)")
        print(mgr.memory.report())

        mgr.remove_hooks()
        del model_k, out_k, mgr
        torch.cuda.empty_cache()

        # Store for summary
        if prefetch == 1:
            dt_p1, peak_p1 = dt_k, peak_k
        else:
            dt_p2, peak_p2 = dt_k, peak_k

    # ── 3. Summary ───────────────────────────────────────────────────────

    banner("3. Summary")
    print(f"{'Method':<30} {'Peak VRAM':>10} {'Time':>10} {'Layers GPU':>12} {'vs Std':>10}")
    print("-" * 75)
    print(f"{'Standard PyTorch':<30} {peak_std:>8.0f} MB {dt_std:>8.1f} ms {'all 12':>12} {'—':>10}")
    print(f"{'K4N0N3 (prefetch=1)':<30} {peak_p1:>8.0f} MB {dt_p1:>8.1f} ms {'max 2':>12} {'-'+str(int(peak_std-peak_p1))+' MB':>10}")
    print(f"{'K4N0N3 (prefetch=2)':<30} {peak_p2:>8.0f} MB {dt_p2:>8.1f} ms {'max 3':>12} {'-'+str(int(peak_std-peak_p2))+' MB':>10}")
    print()
    print(f"VRAM efficiency (prefetch=1): {peak_p1/peak_std*100:.0f}% of standard")
    print(f"VRAM efficiency (prefetch=2): {peak_p2/peak_std*100:.0f}% of standard")


if __name__ == "__main__":
    main()
