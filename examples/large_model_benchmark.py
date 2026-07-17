"""K4N0N3: Large synthetic model (1.5 GB) — clear VRAM savings demonstration."""
from __future__ import annotations

import time
import torch
from k4n0n3 import LayerManager


def banner(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


def vram_reset() -> None:
    torch.cuda.reset_peak_memory_stats()


def vram_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


class _LargeTransformer(torch.nn.Module):
    def __init__(self, n_layers: int, dim: int, heads: int, vocab: int):
        super().__init__()
        self.wte = torch.nn.Embedding(vocab, dim)
        self.wpe = torch.nn.Embedding(2048, dim)
        self.layers = torch.nn.ModuleList([
            _Block(dim, heads) for _ in range(n_layers)
        ])
        self.ln_f = torch.nn.LayerNorm(dim)
        self.lm_head = torch.nn.Linear(dim, vocab, bias=False)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        x = self.wte(x) + self.wpe(pos)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


class _Block(torch.nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(dim)
        self.ln2 = torch.nn.LayerNorm(dim)
        self.attn = torch.nn.MultiheadAttention(dim, heads, bias=False, batch_first=True)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(dim * 4, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    free, total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    print(f"GPU: {gpu}  |  VRAM: {free/1024**3:.1f}/{total/1024**3:.1f} GB free")

    n_layers = 24
    dim = 2048
    heads = 16
    vocab = 32000
    batch_size = 1
    seq_len = 64

    print(f"Model: {n_layers}L × d{dim} × {heads}heads  |  vocab: {vocab}  |  batch: {batch_size}×{seq_len}")

    # ── Build ────────────────────────────────────────────────────────────

    banner("1. Build model")
    t0 = time.perf_counter()
    model = _LargeTransformer(n_layers, dim, heads, vocab).to("cpu")
    dt_build = time.perf_counter() - t0
    params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

    print(f"Built in {dt_build:.1f}s  |  Params: {params_mb:.0f} MB  |  dtype: float32")

    # Measure fixed vs layer sizes
    layer_ids: set[int] = set()
    for m in model.layers:
        for sub in m.modules():
            layer_ids.add(id(sub))

    fixed_mb = 0.0
    for m in model.modules():
        if id(m) in layer_ids:
            continue
        for p in m.parameters(recurse=False):
            fixed_mb += p.numel() * p.element_size()
    fixed_mb /= 1024**2

    one_layer = sum(p.numel() * p.element_size() for p in model.layers[0].parameters()) / 1024**2
    all_layers = one_layer * n_layers

    print(f"Fixed params:  {fixed_mb:.0f} MB  (embeddings, head, norms)")
    print(f"Layer params:  {all_layers:.0f} MB  ({n_layers} × {one_layer:.0f} MB)")
    print(f"K4N0N3 theory (2 layers): {fixed_mb + 2*one_layer:.0f} MB  ← vs {params_mb:.0f} MB all-on-GPU")

    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)

    # ── 2. Standard PyTorch ──────────────────────────────────────────────

    banner("2. Standard PyTorch — all layers on GPU")

    model.to(device)
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()

    vram_reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x)
    torch.cuda.synchronize()
    dt_std = (time.perf_counter() - t0) * 1000
    peak_std = vram_peak_mb()

    print(f"Forward: {dt_std:.0f} ms  |  Peak VRAM: {peak_std:.0f} MB")
    print(f"All {n_layers} layers on GPU = {params_mb:.0f} MB + activations")

    model.to("cpu")
    torch.cuda.empty_cache()

    # ── 3. K4N0N3 streaming ──────────────────────────────────────────────

    for prefetch in [1, 2]:
        max_layers = prefetch + 1
        banner(f"3. K4N0N3 — prefetch={prefetch} (max {max_layers} layers GPU)")

        model.to("cpu")
        mgr = LayerManager(
            model, layer_prefix="layers",
            vram_budget_mb=4096,
            prefetch_depth=prefetch,
            verbose=False,
        )
        mgr.prepare()

        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()

        vram_reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        peak = vram_peak_mb()

        theory = fixed_mb + max_layers * one_layer
        saved = peak_std - peak
        pct = 100 * saved / peak_std if peak_std > 0 else 0

        print(f"Forward: {dt:.0f} ms  |  Peak VRAM: {peak:.0f} MB")
        print(f"Theory:   {theory:.0f} MB  (fixed {fixed_mb:.0f} + {max_layers}×{one_layer:.0f})")
        print(f"Saved:    {saved:.0f} MB ({pct:.0f}%)")
        print(mgr.memory.report())

        mgr.remove_hooks()

    # ── 4. Summary ──────────────────────────────────────────────────────

    banner("4. Summary")

    # Re-run both for clean summary
    results = [("Standard PyTorch", dt_std, peak_std, n_layers)]

    model.to("cpu")
    torch.cuda.empty_cache()

    for prefetch in [1, 2]:
        model.to("cpu")
        mgr = LayerManager(
            model, layer_prefix="layers",
            vram_budget_mb=4096,
            prefetch_depth=prefetch,
            verbose=False,
        )
        mgr.prepare()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        vram_reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000
        peak = vram_peak_mb()
        label = f"K4N0N3 (prefetch={prefetch})"
        results.append((label, dt, peak, prefetch + 1))
        mgr.remove_hooks()

    print(f"{'Method':<28} {'Peak VRAM':>10} {'Time':>8} {'Layers GPU':>12} {'Saved':>10}")
    print("-" * 72)
    std_peak = results[0][2]
    for label, dt, peak, active in results:
        saved = std_peak - peak
        pct = 100 * saved / std_peak
        print(f"{label:<28} {peak:>8.0f} MB {dt:>6.0f} ms {str(active):>12} {saved:>8.0f} MB ({pct:.0f}%)")

    print(f"\nModel: {params_mb:.0f} MB total  |  GPU: {gpu} ({total/1024**3:.0f} GB)")
    print(f"K4N0N3 enables running {params_mb/1024:.1f} GB model with only "
          f"{(fixed_mb + 2*one_layer)/1024:.1f} GB VRAM budget")


if __name__ == "__main__":
    main()
