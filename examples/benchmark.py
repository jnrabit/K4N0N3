"""K4N0N3 Benchmark — layer offloading, VRAM tracking, prefetch effectiveness."""
from __future__ import annotations

import time
import torch
from k4n0n3 import LayerManager, MemoryManager


def banner(title: str) -> None:
    w = 64
    print(f"\n{'='*w}\n  {title}\n{'='*w}")


class _LayerStack(torch.nn.Module):
    def __init__(self, n_layers: int, dim: int):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(dim, dim) for _ in range(n_layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ── Benchmark 1: MemoryManager Budget & Eviction ──────────────────────────

def bench_memory_manager():
    banner("1. MemoryManager Budget + LRU Eviction")

    mgr = MemoryManager(vram_budget_mb=1)  # Tight 1 MB budget

    layers = {
        "L0": torch.nn.Linear(512, 512),   # ~1 MB  (512*512*4 = 1,048,576 bytes)
        "L1": torch.nn.Linear(512, 512),   # ~1 MB
        "L2": torch.nn.Linear(64, 64),     # ~16 KB
    }

    print(f"{'Step':<8} {'Action':<18} {'Used (MB)':<12} {'Peak (MB)':<12} {'Ratio':<8}")
    print("-" * 60)

    for i, (name, mod) in enumerate(layers.items()):
        size_mb = sum(p.numel() * p.element_size() for p in mod.parameters()) / 1024**2
        mgr.mark_on_gpu(name, mod)
        print(
            f"{i+1:<8} add {name:<14} "
            f"{mgr.used_bytes()/1024**2:<12.3f} "
            f"{mgr._peak_bytes/1024**2:<12.3f} "
            f"{mgr.usage_ratio():<8.2f}"
        )

    print(f"\n{mgr.report()}")


# ── Benchmark 2: Layer Discovery + Size Measurement ───────────────────────

def bench_layer_discovery():
    banner("2. Layer Discovery + Size Measurement")

    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.ModuleDict({
                "layers": torch.nn.ModuleList([
                    torch.nn.Linear(256, 256),
                    torch.nn.Linear(256, 128),
                    torch.nn.Linear(128, 64),
                    torch.nn.Linear(64, 32),
                    torch.nn.Linear(32, 16),
                ]),
            })

        def forward(self, x):
            for layer in self.model["layers"]:
                x = layer(x)
            return x

    model = TestModel()
    mgr = LayerManager(model, layer_prefix="model.layers", verbose=True)

    stats = mgr.stats()
    print(f"\n{'Layer':<20} {'Size (MB)':<12}")
    print("-" * 35)
    for name, s in stats.items():
        print(f"{name:<20} {s['size_mb']:<12.4f}")

    print(f"\nTotal layers discovered: {len(stats)}")
    print(f"Total hooks registered: {len(mgr._hook_handles)}")
    mgr.remove_hooks()


# ── Benchmark 3: Hook Lifecycle (Forward Pass) ────────────────────────────

def bench_hook_lifecycle():
    banner("3. Hook Lifecycle — Forward Pass")

    n = 6
    dim = 256
    model = _LayerStack(n, dim)

    mgr = LayerManager(
        model, layer_prefix="layers",
        vram_budget_mb=4096, prefetch_depth=2, verbose=True,
    )
    mgr.prepare()

    x = torch.randn(8, dim)
    if torch.cuda.is_available():
        x = x.to("cuda")
    t0 = time.perf_counter()

    with torch.no_grad():
        out = model(x)

    dt = time.perf_counter() - t0
    print(f"\nForward pass: {dt*1000:.2f} ms ({n} layers x {dim}x{dim})")
    print(mgr.memory.report())

    mgr.remove_hooks()


# ── Benchmark 4: Training Backward Pass ───────────────────────────────────

def bench_training():
    banner("4. TrainingManager — Forward + Backward")

    from k4n0n3 import TrainingManager

    class TrainModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([
                torch.nn.Linear(128, 128) for _ in range(4)
            ])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = TrainModel()
    mgr = TrainingManager(model, layer_prefix="layers", vram_budget_mb=4096, verbose=False)
    mgr.prepare()

    x = torch.randn(16, 128, requires_grad=True)
    if torch.cuda.is_available():
        x = x.to("cuda")

    t0 = time.perf_counter()
    out = model(x)
    loss = out.sum()
    loss.backward()
    dt = time.perf_counter() - t0

    print(f"Forward + backward: {dt*1000:.2f} ms (4 layers, 128x128)")
    print(f"Hooks active: {len(mgr._hook_handles)} (fw_pre, fw_post, bw_pre, bw_post per layer)")
    print(mgr.report())

    mgr.remove_hooks()


# ── Benchmark 5: Stats / Profiling Output ─────────────────────────────────

def bench_profiling():
    banner("5. LayerManager.stats() — Profiling Output")

    model = _LayerStack(4, 128)
    mgr = LayerManager(model, layer_prefix="layers", prefetch_depth=1)
    mgr.prepare()

    x = torch.randn(8, 128)
    if torch.cuda.is_available():
        x = x.to("cuda")
    with torch.no_grad():
        model(x)
        model(x)

    stats = mgr.stats()
    print(f"\n{'Layer':<8} {'Size MB':<10} {'Transfer ms':<14} {'Compute ms':<14}")
    print("-" * 50)
    for name, s in stats.items():
        print(
            f"{name:<8} {s['size_mb']:<10.4f} "
            f"{s['transfer_ms']:<14.3f} {s['compute_ms']:<14.3f}"
        )

    mgr.remove_hooks()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__}  |  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    bench_memory_manager()
    bench_layer_discovery()
    bench_hook_lifecycle()
    bench_training()
    bench_profiling()

    print(f"\n{'='*64}\n  All benchmarks complete.\n{'='*64}")
