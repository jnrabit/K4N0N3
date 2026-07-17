"""K4N0N3 with DistilBERT (cached) — VRAM comparison, generate via CausalLM fallback."""
from __future__ import annotations

import time
import torch
from k4n0n3 import ZeroFlushModel


def banner(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


def vram_reset() -> None:
    torch.cuda.reset_peak_memory_stats()


def vram_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def vram_current_mb() -> float:
    return torch.cuda.memory_allocated() / 1024**2


def main():
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {name}  |  VRAM: {free/1024**3:.1f}/{total/1024**3:.1f} GB free")

    model_name = "distilbert/distilbert-base-uncased"

    # ── 1. Load ──────────────────────────────────────────────────────────

    banner(f"1. Load: {model_name}")
    t0 = time.perf_counter()

    model = ZeroFlushModel(
        model_name,
        vram_budget_mb=1024,
        prefetch_depth=1,
        verbose=False,
    )
    dt_load = time.perf_counter() - t0

    n_layers = len(model.layer_manager._layers)
    layer_mb = model.layer_manager._layer_info[model.layer_manager._layer_list[0]].size_mb
    params_mb = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1024**2
    fixed_mb = params_mb - n_layers * layer_mb

    print(f"Loaded in {dt_load:.0f}s")
    print(f"Params: {params_mb:.0f} MB  |  {n_layers} layers × {layer_mb:.1f} MB  |  Fixed: {fixed_mb:.0f} MB")
    print(f"With prefetch=1: theory peak = {fixed_mb:.0f} + 2×{layer_mb:.1f} = {fixed_mb + 2*layer_mb:.0f} MB")

    prompt = "The capital of France is"
    inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda")

    # ── 2. Standard PyTorch ──────────────────────────────────────────────

    banner("2. Standard PyTorch (all layers on GPU)")
    model.offload_all()
    torch.cuda.empty_cache()
    model.model.to("cuda")

    with torch.no_grad():
        model.model(**inputs)
    torch.cuda.synchronize()

    vram_reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_std = model.model(**inputs)
    torch.cuda.synchronize()
    dt_std = (time.perf_counter() - t0) * 1000
    peak_std = vram_peak_mb()
    vram_std = vram_current_mb()

    print(f"Forward:  {dt_std:6.1f} ms")
    print(f"VRAM now: {vram_std:6.0f} MB  |  Peak: {peak_std:.0f} MB")
    print(f"Layers: all {n_layers} on GPU concurrently")

    # ── 3. K4N0N3 streaming ──────────────────────────────────────────────

    banner("3. K4N0N3 streaming (max 2 layers on GPU)")
    model.offload_all()
    torch.cuda.empty_cache()
    model.prepare()

    with torch.no_grad():
        model.model(**inputs)
    torch.cuda.synchronize()

    vram_reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_k = model.model(**inputs)
    torch.cuda.synchronize()
    dt_k = (time.perf_counter() - t0) * 1000
    peak_k = vram_peak_mb()
    vram_k = vram_current_mb()

    print(f"Forward:  {dt_k:6.1f} ms")
    print(f"VRAM now: {vram_k:6.0f} MB  |  Peak: {peak_k:.0f} MB")
    print(model.report())
    print(f"Theory: {fixed_mb + 2*layer_mb:.0f} MB (fixed + 2 layers) — actual includes activation overhead")

    # ── 4. Batch test ────────────────────────────────────────────────────

    banner("4. Batch (4 inputs × 16 tokens)")
    texts = [
        "The capital of France is",
        "Machine learning is a field of",
        "The theory of relativity was",
        "Python is a programming language",
    ]
    batch = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to("cuda")

    model.offload_all()
    torch.cuda.empty_cache()
    model.prepare()

    with torch.no_grad():
        model.model(**batch)
    torch.cuda.synchronize()

    vram_reset()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_batch = model.model(**batch)
    torch.cuda.synchronize()
    dt_batch = (time.perf_counter() - t0) * 1000
    peak_batch = vram_peak_mb()
    vram_batch = vram_current_mb()

    print(f"Forward:  {dt_batch:6.1f} ms")
    print(f"VRAM now: {vram_batch:6.0f} MB  |  Peak: {peak_batch:.0f} MB")
    print(f"Shape: {list(out_batch.last_hidden_state.shape)}")
    print(model.report())

    # ── 5. Summary ──────────────────────────────────────────────────────

    banner("5. Summary")
    print(f"{'Method':<30} {'Batch':>6} {'VRAM now':>10} {'Peak':>8} {'Time':>8} {'Layers GPU':>12}")
    print("-" * 75)
    print(f"{'Standard PyTorch':<30} {'1×9':>6} {vram_std:>8.0f} MB {peak_std:>6.0f} MB {dt_std:>6.1f} ms {'all ' + str(n_layers):>12}")
    print(f"{'K4N0N3 (prefetch=1)':<30} {'1×9':>6} {vram_k:>8.0f} MB {peak_k:>6.0f} MB {dt_k:>6.1f} ms {'max 2':>12}")
    print(f"{'K4N0N3 (prefetch=1)':<30} {'4×16':>6} {vram_batch:>8.0f} MB {peak_batch:>6.0f} MB {dt_batch:>6.1f} ms {'max 2':>12}")
    print()
    print(f"Theory (2 layers): {fixed_mb + 2*layer_mb:.0f} MB")
    print(f"Theory (6 layers): {fixed_mb + 6*layer_mb:.0f} MB  ← Standard PyTorch")
    print(f"K4N0N3 now:        {vram_k:.0f} MB  (after post-hook offload)")
    print(f"Standard now:      {vram_std:.0f} MB  (all layers stay on GPU)")

    model.offload_all()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
