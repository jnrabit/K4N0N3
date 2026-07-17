"""List all Ollama GGUF models with architecture & sizing info."""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/jnrabit/projekte/K4N0N3")

from k4n0n3.gguf_reader import list_ollama_models


def main():
    models = list_ollama_models()
    if not models:
        print("No Ollama GGUF models found.")
        return

    print(f"{'Name':<35} {'Size':>6} {'Arch':<8} {'Layers':>6} {'Dim':>5} {'Heads':>6} {'Type':>8}")
    print("-" * 85)

    for m in models:
        name = m.name[:34] if m.name else "(unknown)"
        ftype = m.file_type.replace("LLAMA_FILE_TYPE_", "").replace("_", " ") if m.file_type else "?"
        print(
            f"{name:<35} {m.size_gb:>5.1f}G {m.architecture:<8} "
            f"{m.n_layers:>4}L {m.dim:>5} {m.n_heads:>4}h {ftype:>8}"
        )

    total_gb = sum(m.size_gb for m in models)
    print(f"\n{len(models)} models, {total_gb:.1f} GB total")

    # Show K4N0N3-relevant info
    print(f"\n{'='*64}")
    print("K4N0N3 Compatibility Notes:")
    print(f"{'='*64}")
    for m in models:
        if m.n_layers <= 0 or m.dim <= 0:
            continue
        layer_params = m.n_layers * (4 * m.dim * m.dim)  # rough estimate
        layer_mb = layer_params * 2 / 1024**2  # float16
        print(f"\n{m.name}:")
        print(f"  HF equivalent needed (GGUF not directly supported)")
        print(f"  ~{m.n_layers} layers × ~{layer_mb/m.n_layers:.0f} MB each")
        print(f"  K4N0N3 with 2 layers: ~{m.n_layers * layer_mb/m.n_layers * 2:.0f} MB GPU")
        print(f"  Full model (float16): ~{layer_mb:.0f} MB")
        print(f"  VRAM savings: ~{(1 - 2/m.n_layers)*100:.0f}%")


if __name__ == "__main__":
    main()
