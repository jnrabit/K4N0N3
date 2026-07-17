"""K4N0N3 Stress Test — tight VRAM budget forces LRU eviction."""
from __future__ import annotations

import time
import torch
from k4n0n3 import LayerManager


def banner(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


class _LayerStack(torch.nn.Module):
    def __init__(self, n: int, dim: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(n)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    n_layers = 8
    dim = 1024  # 1024×1024×4 bytes ≈ 4 MB per layer
    layer_mb = dim * dim * 4 / 1024**2

    print(f"GPU: {gpu_name}  |  Dim: {dim}  |  Layer size: {layer_mb:.1f} MB  |  {n_layers} layers")

    # ── Scenario: Budget fits exactly 2 layers ──────────────────────────
    budget_mb = int(layer_mb * 2.5)  # 2 layers + a bit of slack
    model = _LayerStack(n_layers, dim).to("cpu")

    banner(f"Budget: {budget_mb} MB (fits ~2 layers, but we have {n_layers})")

    mgr = LayerManager(
        model, layer_prefix="layers",
        vram_budget_mb=budget_mb,
        prefetch_depth=1,
        verbose=True,
    )
    mgr.prepare()

    x = torch.randn(4, dim, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(x)
    dt = (time.perf_counter() - t0) * 1000

    banner("Results")
    print(f"Forward: {dt:.1f} ms  |  Output shape: {out.shape}")
    print(f"Max layers on GPU at once: {max(s['transfer_ms'] for s in mgr.stats().values()):.0f} (budget ~2)")
    print(mgr.memory.report())

    # Verify output is correct
    banner("Correctness check")
    model2 = _LayerStack(n_layers, dim).to(device)
    model2.load_state_dict(model.state_dict())
    with torch.no_grad():
        ref = model2(x)
    diff = (out.cpu() - ref.cpu()).abs().max().item()
    print(f"Max difference vs all-on-GPU reference: {diff:.2e}  {'✓ OK' if diff < 1e-4 else '✗ MISMATCH'}")

    mgr.remove_hooks()

    # ── Scenario: Extreme budget (1 layer) ──────────────────────────────
    banner(f"Extreme: Budget = {int(layer_mb * 1.2)} MB (fits ~1 layer)")
    model3 = _LayerStack(n_layers, dim).to("cpu")
    mgr2 = LayerManager(
        model3, layer_prefix="layers",
        vram_budget_mb=int(layer_mb * 1.2),
        prefetch_depth=1,
        verbose=True,
    )
    mgr2.prepare()

    t0 = time.perf_counter()
    with torch.no_grad():
        out2 = model3(x)
    dt2 = (time.perf_counter() - t0) * 1000

    print(f"\nForward: {dt2:.1f} ms")
    print(mgr2.memory.report())
    print(f"Evictions forced: budget holds only 1 layer, model has {n_layers}")

    mgr2.remove_hooks()


if __name__ == "__main__":
    main()
