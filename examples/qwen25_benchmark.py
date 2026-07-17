"""K4N0N3 × Qwen2.5-0.5B (494M) — Real CausalLM, VRAM, generate."""
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

    model_name = "Qwen/Qwen2.5-0.5B"

    banner(f"1. Load {model_name}")
    t0 = time.perf_counter()

    model = ZeroFlushModel(
        model_name,
        vram_budget_mb=1024,
        prefetch_depth=1,
        verbose=False,
        torch_dtype=torch.float16,
    )
    dt_load = time.perf_counter() - t0

    n_layers = len(model.layer_manager._layers)
    layer_mb = model.layer_manager._layer_info[model.layer_manager._layer_list[0]].size_mb
    params_mb = estimate_model_size(model.model)
    fixed_mb = params_mb - n_layers * layer_mb
    k4_theory = fixed_mb + 2 * layer_mb

    print(f"Loaded: {dt_load:.0f}s  |  {params_mb:.0f} MB  |  {n_layers} layers × {layer_mb:.1f} MB")
    print(f"Fixed: {fixed_mb:.0f} MB  |  K4N0N3 theory (2L): {k4_theory:.0f} MB")
    print(f"Arch: {model.model.config.model_type}  |  Generate: {hasattr(model.model, 'generate')}")

    prompt = "The capital of France is"

    # ── 2. Standard ──────────────────────────────────────────────────────

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

    print(f"Forward: {dt_std:.0f} ms  |  Peak VRAM: {peak_std:.0f} MB")
    print(f"All {n_layers} layers GPU concurrent")

    # ── 3. K4N0N3 ────────────────────────────────────────────────────────

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
    pct = 100 * saved / max(peak_std, 1)

    print(f"Forward: {dt_k:.0f} ms  |  Peak VRAM: {peak_k:.0f} MB")
    print(f"VRAM saved: {saved:.0f} MB ({pct:.0f}%)")
    print(model.report())

    # ── 4. Generate ──────────────────────────────────────────────────────

    banner("4. Text Generation")
    model.offload_all()
    torch.cuda.empty_cache()
    model.prepare()

    t0 = time.perf_counter()
    result = model.generate(prompt, max_length=50, do_sample=True, temperature=0.7)
    dt_gen = time.perf_counter() - t0

    print(f"Prompt:   {prompt}")
    print(f"Generate: {result}")
    n_tok = len(model.tokenizer.encode(result))
    print(f"Time: {dt_gen:.1f}s  ({n_tok} tokens, {n_tok/dt_gen:.1f} tok/s)")
    print(model.report())

    # ── 5. Summary ──────────────────────────────────────────────────────

    banner("5. Summary")
    print(f"{'Method':<28} {'Peak VRAM':>10} {'Time':>8} {'Layers GPU':>12} {'Saved':>10}")
    print("-" * 72)
    print(f"{'Standard PyTorch':<28} {peak_std:>8.0f} MB {dt_std:>6.0f} ms {'all ' + str(n_layers):>12} {'—':>10}")
    print(f"{'K4N0N3 (prefetch=1)':<28} {peak_k:>8.0f} MB {dt_k:>6.0f} ms {'max 2':>12} {saved:>6.0f} MB ({pct:.0f}%)")
    print()
    print(f"Model: {model_name} ({params_mb:.0f} MB float16, {n_layers}L)")
    print(f"K4N0N3 VRAM: {peak_k/peak_std*100:.0f}% of standard")

    model.offload_all()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
