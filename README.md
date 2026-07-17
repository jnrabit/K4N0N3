# K4N0N3: Zero-Flush Memory Management for LLMs

Transparent **layer-level CPU/GPU offloading** for Hugging Face models. Run large
LLMs on limited VRAM — no manual `.to("cuda")` calls needed.

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
- **Auto-discovery** — detects transformer layer structure for Llama, Mistral,
  GPT-2, Falcon, BLOOM, Gemma, Phi, Qwen2, and more

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
)

print(model.generate("Explain quantum computing in simple terms.", max_length=100))
```

## API

### `ZeroFlushModel`
Main entry point. Wraps a Hugging Face model.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | *(required)* | HuggingFace model name or path |
| `vram_budget_mb` | `4096` | VRAM budget in MB |
| `prefetch_depth` | `1` | Number of layers to prefetch ahead |
| `device` | `"cuda"` | Target compute device |

### `LayerManager`
Low-level API for manual hook management on any `nn.Module`.

```python
from k4n0n3 import LayerManager

manager = LayerManager(model, layer_prefix="model.layers", vram_budget_mb=2048)
manager.prepare()
output = model(input)
manager.remove_hooks()
```

### `MemoryManager`
Tracks VRAM usage and handles LRU eviction.

### `ManagedTensor`
Simple tensor wrapper with device tracking (no `torch.Tensor` subclass).

## Testing

```bash
python -m pytest tests/ -v
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- Hugging Face `transformers` 4.30+
- CUDA-compatible GPU (NVIDIA) or ROCm (AMD)

## License

MIT
