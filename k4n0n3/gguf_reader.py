"""Read Ollama / GGUF model metadata — architecture, layers, sizes."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class GGUFModelInfo:
    path: str
    size_gb: float
    architecture: str
    name: str
    file_type: str
    n_layers: int
    n_heads: int
    n_kv_heads: int
    dim: int
    ffn_dim: int
    vocab_size: int
    ctx_len: int
    tensor_count: int


_FILE_TYPES = {
    1: "F32", 2: "F16", 3: "Q4_0", 4: "Q4_1", 5: "Q4_K_M", 6: "Q5_K_M",
    7: "Q4_K_S", 8: "Q5_K_S", 9: "Q8_0", 10: "Q8_1",
    11: "Q2_K", 12: "Q3_K_S", 13: "Q3_K_M", 14: "Q3_K_L",
    15: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ2_S",
    19: "IQ3_XXS", 20: "IQ3_S", 21: "IQ1_S", 22: "IQ4_NL",
    23: "IQ3_XS", 24: "IQ4_XS", 25: "I8", 26: "I16", 27: "I32",
    28: "I64", 29: "F64", 30: "IQ1_M", 31: "BF16",
}


def read_gguf(path: str) -> GGUFModelInfo:
    from gguf import GGUFReader

    r = GGUFReader(path)

    def _val(key: str, default: int = 0) -> int:
        f = r.fields.get(key)
        if f is None or not f.data:
            return default
        v = f.parts[f.data[0]]
        if hasattr(v, "tobytes"):
            v = int.from_bytes(v.tobytes(), "little")
        return int(v)

    def _str(key: str, default: str = "?") -> str:
        f = r.fields.get(key)
        if f is None or not f.data:
            return default
        idx = f.data[0]
        raw = f.parts[idx]
        if hasattr(raw, "tobytes"):
            b = raw.tobytes()
            if key == "general.file_type":
                val = int.from_bytes(b, "little")
                return _FILE_TYPES.get(val, f"UNK({val})")
            return b.decode("utf-8", errors="replace")
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    arch = _str("general.architecture")

    return GGUFModelInfo(
        path=path,
        size_gb=os.path.getsize(path) / 1024**3,
        architecture=arch,
        name=_str("general.name"),
        file_type=_str("general.file_type"),
        n_layers=_val(f"{arch}.block_count"),
        n_heads=_val(f"{arch}.attention.head_count"),
        n_kv_heads=_val(f"{arch}.attention.head_count_kv", _val(f"{arch}.attention.head_count")),
        dim=_val(f"{arch}.embedding_length"),
        ffn_dim=_val(f"{arch}.feed_forward_length"),
        vocab_size=_val(f"{arch}.vocab_size", _val("tokenizer.ggml.tokens", 0)),
        ctx_len=_val(f"{arch}.context_length"),
        tensor_count=len(r.tensors),
    )


def list_ollama_models(models_dir: str | None = None) -> list[GGUFModelInfo]:
    if models_dir is None:
        models_dir = os.path.expanduser("~/.ollama/models/blobs/")
    results = []
    for fname in sorted(os.listdir(models_dir)):
        fpath = os.path.join(models_dir, fname)
        try:
            info = read_gguf(fpath)
            if info.architecture != "?":
                results.append(info)
        except Exception:
            pass
    results.sort(key=lambda m: -m.size_gb)
    return results
