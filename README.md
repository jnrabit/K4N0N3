# K4N0N3: Zero-Flush Memory Management for LLMs

Transparent **layer-level CPU/GPU offloading** for Hugging Face models. Run large
LLMs on limited VRAM — no manual `.to("cuda")` calls needed. Includes an
experimental **MTP (Multi-Token Prediction) / self-speculative decoding** stack.

## How It Works

K4N0N3 registers **forward hooks** on every transformer layer. Before a layer
executes, its parameters are moved to GPU. After execution, they are offloaded
back to CPU. While the current layer runs, upcoming layers are **asynchronously
prefetched** to overlap transfer with computation.

```
Layer 0 ████████░░░░░░░░
Layer 1     ░░░░████████░░░░
Layer 2         ░░░░████████
           ↑prefetch  ↑compute
```

## Features

- **Transparent offloading** — wrap any Hugging Face model, no code changes
- **Async prefetch** — overlaps CPU→GPU transfers with layer execution
- **VRAM budgeting** — specify a memory limit, older layers get evicted automatically
- **Layer pinning** — keep designated layers (e.g. `[0, -1]`) permanently resident
  in VRAM, immune to LRU eviction
- **Auto-discovery** — detects transformer layer structure for Llama, Mistral,
  GPT-2, Falcon, BLOOM, Gemma, Phi, Qwen2, and more
- **MTP draft-head discovery & dual-pass hooks** — recognizes `mtp.*` / `draft_head`
  / `nextn` modules and executes them alongside their host layer
- **MTP verification engine** — greedy-verlustfreies Multi-Token-Prediction
  decoding with optional multi-branch tree-drafting (`num_branches`)
- **MTP weight reconstruction** — reads MTP/draft weights that HF loaders ignore
  (`mtp.*`, `model.layers.61+`) and re-attaches them as functional submodules

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from k4n0n3 import ZeroFlushModel

model = ZeroFlushModel(
    "mistralai/Mistral-7B-Instruct-v0.1",
    vram_budget_mb=4096,
    prefetch_depth=2,
    pinned_layers=[0, -1],   # Layer 0 und letzter Layer bleiben resident
)

print(model.generate("Explain quantum computing in simple terms.", max_length=100))
```

### MTP / Self-Speculative Decoding

```python
from k4n0n3 import ZeroFlushModel

# use_mtp=True aktiviert die MTP-Engine; mtp_checkpoint lädt separat trainierte
# Draft-Head-Gewichte (siehe Mini-MTP-Training unten).
model = ZeroFlushModel(
    "Qwen/Qwen2.5-0.5B",
    vram_budget_mb=4096,
    use_mtp=True,
    mtp_checkpoint="checkpoints/qwen2.5-0.5b-mtp",
    mtp_num_branches=2,            # 1 = single-path, >1 = tree-drafting
)

print(model.generate("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20", max_new_tokens=32))
print(model._mtp_stats)            # {"steps": …, "accepted_per_step": …}
```

### Mini-MTP Micro-Training (Proof-of-Concept)

Trainiert einen leichten Draft-Head (2-Layer-MLP) auf dem letzten Layer-Output
eines eingefrorenen Basis-Modells und validiert die MTP-Akzeptanz end-to-end:

```bash
python scripts/train_mini_mtp.py --steps 100
```

Ergebnis auf AMD RX 7600 (ROCm): `accepted_per_step = 1.455` (> 1.0), d. h. die
Draft-Heads erzeugen messbare Multi-Token-Gewinne. Ausgabe:
`checkpoints/qwen2.5-0.5b-mtp/model.safetensors` (Keys `mtp.23.*`).

## API

### `ZeroFlushModel`
Main entry point. Wraps a Hugging Face model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | *(required)* | HuggingFace model name or path |
| `vram_budget_mb` | `4096` | VRAM budget in MB |
| `prefetch_depth` | `1` | Number of layers to prefetch ahead |
| `device` | `"cuda"` | Target compute device |
| `quantize_transfer` | `False` | `False` / `True`/`"int8"` / `"int4"` weight-only transfer |
| `pin_ram_fraction` | `0.7` | Fraction of available RAM used for pinned masters |
| `pinned_layers` | `None` | `[int\|str]` layer indices/names kept resident in VRAM (negative = from end) |
| `use_mtp` | `False` | Enable MTP/self-speculative decoding |
| `mtp_num_branches` | `1` | Multi-branch tree-drafting width (1 = single-path) |

### `LayerManager`
Low-level API for manual hook management on any `nn.Module`.

```python
from k4n0n3 import LayerManager

manager = LayerManager(model, layer_prefix="model.layers", vram_budget_mb=2048,
                       pinned_layers=[0, -1])
manager.prepare()
output = model(input)
manager.remove_hooks()
```

### `MTPVerificationEngine`
Standalone greedy-verlustfreie MTP-Verifikation (model-agnostisch, CPU-testbar).

```python
from k4n0n3 import MTPVerificationEngine
engine = MTPVerificationEngine(num_branches=2)   # 1 = single-path, >1 = tree-drafting
```

### `reconstruct_and_attach_mtp`
Reads ignored MTP weights from a checkpoint and re-attaches them.

```python
from k4n0n3.mtp_loader import reconstruct_and_attach_mtp
reconstruct_and_attach_mtp(model, "path/to/model", dtype=torch.float16)
```

### `MemoryManager`
Tracks VRAM usage and handles LRU eviction.

### `ManagedTensor`
Simple tensor wrapper with device tracking (no `torch.Tensor` subclass).

## Benchmark

```bash
# Synthetisch (CPU, kein Netzwerk/GPU)
python bench/bench_mtp_k4n0n3.py --synthetic
# Echtes Modell (GPU/ROCm)
python bench/bench_mtp_k4n0n3.py --model Qwen/Qwen2.5-3B --json
```

## Testing

```bash
python -m pytest tests/ -v          # 105 Tests (inkl. Integrationstests)
python -m pytest tests/ -m "not integration"   # ohne echte-Modell-Tests
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ (ROCm build for AMD GPUs)
- Hugging Face `transformers` 4.30+
- CUDA-compatible GPU (NVIDIA) or ROCm (AMD)

## MTP Status (Experimental)

The MTP stack is functionally complete and greedy-lossless. The engine, tree-
drafting, signature adaptation and weight reconstruction are verified by tests,
and the Mini-MTP micro-training demonstrates real acceptance (`accepted_per_step
= 1.455 > 1.0`). It does **not yet deliver a real speedup** end-to-end:
verification uses recompute (no KV-cache), and no small MTP-capable HF model
exists (Qwen3-Next ≈ 80B, DeepSeek-V3 = 671B). Weight reconstruction is generic
(linear approximation), not architecture-faithful. Use it for research/validation;
`use_mtp=False` is the production default.

## License

MIT
