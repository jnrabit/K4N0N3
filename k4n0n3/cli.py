"""K4N0N3 CLI — transparent layer offloading for LLMs."""
from __future__ import annotations

import argparse
import os
import sys
import time


def cmd_list_models(args: argparse.Namespace) -> None:
    try:
        from .gguf_reader import list_ollama_models
        models = list_ollama_models()
        if not models:
            print("No Ollama GGUF models found in ~/.ollama/models/blobs/")
            return

        print(f"{'Name':<38} {'Size':>6} {'Arch':<8} {'Layers':>6} {'Dim':>5} {'Heads':>6} {'Type':>8}")
        print("-" * 85)
        for m in models:
            print(
                f"{m.name[:37]:<38} {m.size_gb:>5.1f}G {m.architecture:<8} "
                f"{m.n_layers:>4}L {m.dim:>5} {m.n_heads:>4}h {m.file_type:>8}"
            )
        print(f"\n{len(models)} models  |  Use HF equivalents with: k4n0n3 run MODEL_NAME")
    except ImportError:
        print("gguf not installed. Run: pip install gguf")


def cmd_info(args: argparse.Namespace) -> None:
    print(f"K4N0N3 v0.2.0 — Zero-Flush Memory Management")
    print(f"Python:  {sys.version.split()[0]}")
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA:    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"GPU:     {torch.cuda.get_device_name(0)}  ({total/1024**3:.1f} GB)")
            print(f"VRAM free: {free/1024**3:.1f} GB")
    except ImportError:
        print("PyTorch: not installed")


def cmd_run(args: argparse.Namespace) -> None:
    model_name = args.model
    budget = args.budget or 2048
    prefetch = args.prefetch or 1
    prompt = args.prompt or "Explain quantum computing in one sentence."
    max_tokens = args.max_tokens or 50

    try:
        from .huggingface import ZeroFlushModel
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install transformers")
        sys.exit(1)

    print(f"Loading {model_name}...")
    t0 = time.perf_counter()

    model = ZeroFlushModel(
        model_name,
        vram_budget_mb=budget,
        prefetch_depth=prefetch,
        verbose=not args.quiet,
    )
    dt = time.perf_counter() - t0
    n_layers = len(model.layer_manager._layers)
    layer_mb = model.layer_manager._layer_info[model.layer_manager._layer_list[0]].size_mb

    if not args.quiet:
        print(f"Loaded in {dt:.0f}s  |  {n_layers} layers × {layer_mb:.0f} MB  |  Budget: {budget} MB")

    print(f"\nPrompt: {prompt}\n")
    t0 = time.perf_counter()
    result = model.generate(
        prompt, max_length=max_tokens,
        do_sample=True, temperature=0.7,
    )
    dt = time.perf_counter() - t0
    n_tok = len(model.tokenizer.encode(result))

    print(result)
    print(f"\n—— {n_tok} tokens in {dt:.1f}s ({n_tok/dt:.1f} tok/s) ——")
    if not args.quiet:
        print(model.report())

    model.offload_all()


def cmd_bench(args: argparse.Namespace) -> None:
    import torch

    model_name = args.model
    budget = args.budget or 1024
    prefetch = args.prefetch or 1

    try:
        from .huggingface import ZeroFlushModel
    except ImportError as e:
        print(f"Missing dependency: {e}")
        sys.exit(1)

    print(f"Loading {model_name}...")
    model = ZeroFlushModel(
        model_name,
        vram_budget_mb=budget,
        prefetch_depth=prefetch,
        verbose=False,
        torch_dtype=torch.float16,
    )

    n = len(model.layer_manager._layers)
    layer_mb = model.layer_manager._layer_info[model.layer_manager._layer_list[0]].size_mb
    params = sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1024**2
    fixed = params - n * layer_mb

    prompt = "Hello world"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Standard baseline: remove hooks, load fresh to GPU
    model.layer_manager.remove_hooks()
    try:
        model.model.to("cuda" if torch.cuda.is_available() else "cpu")
    except torch.cuda.OutOfMemoryError:
        print("Standard PyTorch: OOM — model too large for VRAM without offloading")
        model.layer_manager._register_hooks()  # re-register hooks
        sys.exit(0)
    with torch.no_grad():
        model.model(**inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    with torch.no_grad():
        model.model(**inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt_std = (time.perf_counter() - t0) * 1000

    # Re-register hooks before K4N0N3 run
    model.layer_manager._register_hooks()

    # K4N0N3
    model.offload_all()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.prepare()

    with torch.no_grad():
        model.model(**inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.perf_counter()
    with torch.no_grad():
        model.model(**inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt_k = (time.perf_counter() - t0) * 1000

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  {n} layers × {layer_mb:.0f} MB  |  Fixed: {fixed:.0f} MB  |  Total: {params:.0f} MB")
    print(f"{'='*60}")
    print(f"  Standard PyTorch:  {dt_std:6.0f} ms  |  ALL {n:>3} layers GPU  |  ~{params:.0f} MB")
    print(f"  K4N0N3 (p={prefetch}):      {dt_k:6.0f} ms  |  MAX {prefetch+1:>3} layers GPU  |  ~{fixed + (prefetch+1)*layer_mb:.0f} MB")
    print(f"  Layer VRAM saved:  {(n - prefetch - 1) * layer_mb:.0f} MB  ({100*(n-prefetch-1)/n:.0f}%)")
    print(f"{'='*60}")
    print(f"  {model.report()}")

    model.offload_all()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="k4n0n3",
        description="K4N0N3 — Zero-Flush Memory Management for LLMs",
    )
    sub = parser.add_subparsers(dest="command")

    p_info = sub.add_parser("info", help="Show system info")
    p_info.set_defaults(func=cmd_info)

    p_list = sub.add_parser("list-models", help="List Ollama GGUF models")
    p_list.set_defaults(func=cmd_list_models)

    p_run = sub.add_parser("run", help="Run a model with K4N0N3 offloading")
    p_run.add_argument("model", help="HF model name (e.g. Qwen/Qwen2.5-0.5B)")
    p_run.add_argument("--budget", "-b", type=int, help="VRAM budget in MB")
    p_run.add_argument("--prefetch", "-p", type=int, help="Prefetch depth")
    p_run.add_argument("--prompt", help="Input prompt")
    p_run.add_argument("--max-tokens", "-n", type=int, help="Max tokens to generate")
    p_run.add_argument("--quiet", "-q", action="store_true", help="Less output")
    p_run.set_defaults(func=cmd_run)

    p_bench = sub.add_parser("bench", help="Benchmark VRAM: standard vs K4N0N3")
    p_bench.add_argument("model", help="HF model name")
    p_bench.add_argument("--budget", "-b", type=int, help="VRAM budget in MB")
    p_bench.add_argument("--prefetch", "-p", type=int, help="Prefetch depth")
    p_bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
