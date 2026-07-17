"""K4N0N3 HF Benchmark — real model: DistilBERT on GPU with layer offloading."""
from __future__ import annotations

import time
import torch
from k4n0n3 import ZeroFlushModel


def banner(title: str) -> None:
    print(f"\n{'='*64}\n  {title}\n{'='*64}")


def main():
    print(f"PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {free/1024**3:.1f}/{total/1024**3:.1f} GB free")

    model_name = "distilbert/distilbert-base-uncased"

    banner(f"1. Loading {model_name}")
    t0 = time.perf_counter()

    model = ZeroFlushModel(
        model_name,
        vram_budget_mb=1024,      # tight budget: only 1 GB
        prefetch_depth=2,
        verbose=True,
    )
    print(f"Load time: {time.perf_counter() - t0:.1f}s")
    print(f"VRAM budget: {model.vram_budget_mb} MB")
    print(f"Layer prefix: {model.layer_manager._layer_list[0].rsplit('.', 1)[0]}")
    print(f"Layers found: {len(model.layer_manager._layers)}")

    banner("2. First forward pass (cold)")
    model.prepare()

    prompt = "The capital of France is"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.model(**inputs)
    dt1 = (time.perf_counter() - t0) * 1000
    print(f"Cold forward: {dt1:.1f} ms")

    banner("3. Second forward (warm, layers already on GPU)")
    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.model(**inputs)
    dt2 = (time.perf_counter() - t0) * 1000
    print(f"Warm forward: {dt2:.1f} ms")
    print(f"Speedup warm/cold: {dt1/dt2:.1f}x")

    banner("4. VRAM Report")
    print(model.report())

    banner("5. Layer Stats")
    stats = model.layer_manager.stats()
    total_params = sum(s["size_mb"] for s in stats.values())
    print(f"{'Layer':<30} {'Size MB':<10}")
    print("-" * 42)
    for name, s in stats.items():
        print(f"{name:<30} {s['size_mb']:<10.2f}")
    print(f"\nTotal: {total_params:.1f} MB across {len(stats)} layers")

    banner("6. Forward (batch of 4 inputs)")
    model.prepare()

    texts = [
        "The capital of France is",
        "Machine learning is a subset of",
        "The theory of relativity was developed by",
        "Python is a programming language that",
    ]
    inputs = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.model(**inputs)
    dt = (time.perf_counter() - t0) * 1000
    print(f"Batch-4 forward: {dt:.1f} ms")
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print(model.report())

    model.offload_all()
    print(f"\nOffloaded — all layers on CPU.")
    print(model.report())


if __name__ == "__main__":
    main()
