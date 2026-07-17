"""K4N0N3 with Qwen2.5-0.5B — real HF model, VRAM comparison."""
from __future__ import annotations

import time
import torch
from k4n0n3 import ZeroFlushModel
from k4n0n3.utils import estimate_model_size


def banner(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


def vram_reset() -> None:
    torch.cuda.reset_peak_memory_stats()


def vram_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def main():
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {name}  |  VRAM: {free/1024**3:.1f}/{total/1024**3:.1f} GB free")

    model_name = "openai-community/gpt2"

    # ── 1. Load ──────────────────────────────────────────────────────────

    banner(f"1. Loading {model_name}")
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
    params_mb = estimate_model_size(model.model)
    fixed_mb = params_mb - n_layers * layer_mb

    print(f"Load: {dt_load:.0f}s  |  Model: {model_name}")
    print(f"Params: {params_mb:.0f} MB  |  Layers: {n_layers} × {layer_mb:.1f} MB  |  Fixed: {fixed_mb:.0f} MB")
    print(f"Theory peak VRAM: {fixed_mb:.0f} + 2×{layer_mb:.0f} = {fixed_mb + 2*layer_mb:.0f} MB (prefetch=1)")

    # ── 2. Standard PyTorch ──────────────────────────────────────────────

    banner("2. Standard PyTorch (all layers GPU)")
    model.prepare()
    model.model.to("cuda")

    prompt = "The capital of France is"
    inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda")

    # Warmup
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

    print(f"Forward: {dt_std:.1f} ms  |  Peak VRAM: {peak_std:.0f} MB")
    print(f"All {n_layers} layers on GPU concurrently")
    print(model.report())

    # ── 3. K4N0N3 (streaming) ───────────────────────────────────────────

    banner("3. K4N0N3 — streaming (max 2 layers GPU)")
    model.prepare()

    # Warmup
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

    print(f"Forward: {dt_k:.1f} ms  |  Peak VRAM: {peak_k:.0f} MB")
    print(model.report())

    # ── 4. Generate ──────────────────────────────────────────────────────

    banner("4. Generate text")
    model.offload_all()
    model.prepare()

    result = model.generate(prompt, max_length=30)
    print(f"Prompt: {prompt}")
    print(f"Result: {result}")

    # ── 5. Summary ──────────────────────────────────────────────────────

    banner("5. Summary")
    print(f"{'Method':<25} {'Peak VRAM':>10} {'Time':>10} {'Layers GPU':>12}")
    print("-" * 60)
    print(f"{'Standard PyTorch':<25} {peak_std:>8.0f} MB {dt_std:>8.1f} ms {'all ' + str(n_layers):>12}")
    print(f"{'K4N0N3 (prefetch=1)':<25} {peak_k:>8.0f} MB {dt_k:>8.1f} ms {'max 2':>12}")
    print()
    print(f"VRAM saved: {peak_std - peak_k:.0f} MB ({100*(peak_std-peak_k)/peak_std:.0f}%)")
    print(f"VRAM efficiency: {peak_k/peak_std*100:.0f}% of standard")

    model.offload_all()


if __name__ == "__main__":
    main()
