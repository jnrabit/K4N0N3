"""K4N0N3 × GPT-2 (124M) — Real CausalLM, VRAM comparison, text generation."""
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
    gpu = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {gpu}  |  VRAM: {free/1024**3:.1f}/{total/1024**3:.1f} GB free")

    model_name = "openai-community/gpt2"

    # ── 1. Load ──────────────────────────────────────────────────────────

    banner(f"1. Load {model_name}")
    t0 = time.perf_counter()

    model = ZeroFlushModel(
        model_name,
        vram_budget_mb=512,
        prefetch_depth=1,
        verbose=False,
    )
    dt_load = time.perf_counter() - t0

    n_layers = len(model.layer_manager._layers)
    layer_mb = model.layer_manager._layer_info[model.layer_manager._layer_list[0]].size_mb
    params_mb = estimate_model_size(model.model)
    fixed_mb = params_mb - n_layers * layer_mb
    prefetch_layers_mb = layer_mb * 2

    print(f"Loaded in {dt_load:.0f}s")
    print(f"Params: {params_mb:.0f} MB  |  dtype: float16")
    print(f"Layers: {n_layers} × {layer_mb:.1f} MB  |  Fixed: {fixed_mb:.0f} MB")
    print(f"Standard VRAM:  ~{params_mb + 50:.0f} MB  (model + activations)")
    print(f"K4N0N3 theory:  ~{fixed_mb + prefetch_layers_mb + 50:.0f} MB  (fixed + 2 layers + activations)")
    print(f"K4N0N3 savings: ~{params_mb - prefetch_layers_mb:.0f} MB  "
          f"({100*(params_mb - prefetch_layers_mb)/params_mb:.0f}%)")

    prompt = "The capital of France is"

    # ── 2. Standard PyTorch ──────────────────────────────────────────────

    banner("2. Standard PyTorch — all layers GPU")
    model.model.to("cuda")
    inputs = model.tokenizer(prompt, return_tensors="pt").to("cuda")

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
    print(f"All {n_layers} layers GPU concurrently")

    # ── 3. K4N0N3 (prefetch=1) ───────────────────────────────────────────

    banner("3. K4N0N3 — prefetch=1 (max 2 layers GPU)")
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

    saved = peak_std - peak_k
    pct = 100 * saved / peak_std

    print(f"Forward: {dt_k:.1f} ms  |  Peak VRAM: {peak_k:.0f} MB")
    print(f"VRAM saved: {saved:.0f} MB ({pct:.0f}%)")
    print(model.report())

    # ── 4. Generate ──────────────────────────────────────────────────────

    banner("4. Text Generation (streaming layers)")
    model.offload_all()
    torch.cuda.empty_cache()
    model.prepare()

    t0 = time.perf_counter()
    result = model.generate(prompt, max_length=40, do_sample=True, temperature=0.7)
    dt_gen = time.perf_counter() - t0

    print(f"Prompt:   {prompt}")
    print(f"Generate: {result}")
    print(f"Time: {dt_gen:.1f}s  ({len(model.tokenizer.encode(result))} tokens)")
    print(model.report())

    # ── 5. Summary ──────────────────────────────────────────────────────

    banner("5. Summary")
    print(f"{'Method':<25} {'Peak VRAM':>10} {'Time':>8} {'Layers GPU':>12} {'Saved':>10}")
    print("-" * 68)
    print(f"{'Standard PyTorch':<25} {peak_std:>8.0f} MB {dt_std:>6.1f} ms {'all ' + str(n_layers):>12} {'—':>10}")
    print(f"{'K4N0N3 (prefetch=1)':<25} {peak_k:>8.0f} MB {dt_k:>6.1f} ms {'max 2':>12} {saved:>6.0f} MB ({pct:.0f}%)")
    print()
    print(f"Model: {model_name} ({params_mb:.0f} MB float16, {n_layers} layers)")
    print(f"GPU:  {gpu} ({total/1024**3:.0f} GB)")
    print(f"K4N0N3 runs GPT-2 with {peak_k/peak_std*100:.0f}% of standard VRAM")

    model.offload_all()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
