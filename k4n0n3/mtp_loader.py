"""MTP Weight Reconstruction — liest ignorierte MTP/Draft-Gewichte aus dem
Checkpoint und baut funktionale Draft-Head-Submodule daraus.

Standard-HF-Loader ignorieren MTP-Gewichte (z. B. ``mtp.*`` bei Qwen3-Next,
``model.layers.61+`` bei DeepSeek-V3). Dieses Modul liest diese Keys direkt
aus den ``.safetensors``-Dateien (memory-mapped, ohne Voll-Reload ins RAM) und
haengt rekonstruierte ``MTPDraftHead``-Module an das Host-Modell, damit die
K4N0N3-Discovery sie finden kann.
"""
from __future__ import annotations

import os
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

_MTP_NAME_PREFIXES = ("mtp.", "mtp_module")
_LAYER_PREFIX = "model.layers."


class MTPDraftHead(nn.Module):
    """Generischer Draft-Head: Sequenz von Linears aus MTP-Gewichten.

    ``forward(hidden_states)`` wendet die Linear-Layer sequentiell (mit
    Aktivierung dazwischen) an. Ist der letzte Linear eine Projektion auf
    Vokabular-Breite, ist der Output bereits Logits.
    """

    def __init__(self, linear_layers: list[nn.Linear], act: nn.Module | None = None):
        super().__init__()
        self.layers = nn.ModuleList(linear_layers)
        self.act = act if act is not None else nn.ReLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        n = len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < n - 1:
                x = self.act(x)
        return x


def reconstruct_and_attach_mtp(
    model: nn.Module,
    model_name_or_path: str,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> list[nn.Module] | None:
    """Rekonstruiert MTP/Draft-Heads aus dem Checkpoint und haengt sie an.

    Gibt die Liste der angehaengten ``MTPDraftHead``-Module zurueck, oder
    ``None`` (mit Warnung), wenn keine MTP-Gewichte gefunden wurden.
    """
    try:
        import safetensors
    except ImportError:
        warnings.warn("safetensors fehlt — MTP-Rekonstruktion uebersprungen.")
        return None

    files = _resolve_safetensors_files(model_name_or_path)
    if not files:
        warnings.warn(
            f"Keine .safetensors-Dateien fuer MTP-Rekonstruktion gefunden "
            f"({model_name_or_path}). Standard-Generation ohne MTP."
        )
        return None

    num_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    mtp_keys = _collect_mtp_keys(files, num_layers)
    if not mtp_keys:
        warnings.warn(
            "Keine MTP/Draft-Gewichte im Checkpoint gefunden — "
            "Standard-Generation ohne MTP."
        )
        return None

    tensors = _load_tensors(files, mtp_keys)
    grouped: dict[int, dict[str, torch.Tensor]] = {}
    for key, tensor in tensors.items():
        grouped.setdefault(_mtp_module_index(key, num_layers), {})[key] = tensor

    heads: dict[int, nn.Module] = {}
    for idx in sorted(grouped.keys()):
        heads[idx] = _build_draft_head(grouped[idx], dtype=dtype, device=device)

    # ModuleDict statt ModuleList: der Modul-Index (aus dem Key) bleibt im
    # Namen erhalten, damit _associate_mtp_layers das MTP-Modul dem richtigen
    # Standard-Layer zuordnen kann (z. B. mtp.23 -> model.layers.23).
    model.mtp_layers = nn.ModuleDict({str(idx): head for idx, head in heads.items()})
    return list(model.mtp_layers.values())


# -- helpers -----------------------------------------------------------------


def _resolve_safetensors_files(model_name_or_path: str) -> list[str]:
    if os.path.isdir(str(model_name_or_path)):
        directory = str(model_name_or_path)
    else:
        try:
            from huggingface_hub import snapshot_download
            directory = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["*.safetensors"],
            )
        except Exception:
            return []
    files = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".safetensors") and not name.endswith(".index.json"):
            files.append(os.path.join(directory, name))
    return files


def _collect_mtp_keys(files: list[str], num_layers: int | None) -> set[str]:
    import safetensors
    keys: set[str] = set()
    for f in files:
        with safetensors.safe_open(f, framework="pt", device="cpu") as sf:
            keys.update(sf.keys())
    return {k for k in keys if _is_mtp_key(k, num_layers)}


def _is_mtp_key(key: str, num_layers: int | None) -> bool:
    if key.startswith(_MTP_NAME_PREFIXES):
        return True
    if num_layers is not None and key.startswith(_LAYER_PREFIX):
        rest = key[len(_LAYER_PREFIX):]
        idx_str = rest.split(".", 1)[0]
        if idx_str.isdigit() and int(idx_str) >= num_layers:
            return True
    return False


def _load_tensors(files: list[str], keys: set[str]) -> dict[str, torch.Tensor]:
    import safetensors
    tensors: dict[str, torch.Tensor] = {}
    key_set = set(keys)
    for f in files:
        with safetensors.safe_open(f, framework="pt", device="cpu") as sf:
            for key in key_set:
                if key in sf.keys():
                    tensors[key] = sf.get_tensor(key)
    return tensors


def _mtp_module_index(key: str, num_layers: int | None) -> int:
    if key.startswith("mtp."):
        rest = key[len("mtp."):]
        idx_str = rest.split(".", 1)[0]
        return int(idx_str) if idx_str.isdigit() else 0
    if key.startswith("mtp_module"):
        return 0
    if key.startswith(_LAYER_PREFIX) and num_layers is not None:
        rest = key[len(_LAYER_PREFIX):]
        idx_str = rest.split(".", 1)[0]
        if idx_str.isdigit():
            return int(idx_str) - num_layers
    return 0


def _build_draft_head(
    tensors: dict[str, torch.Tensor],
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> MTPDraftHead:
    weight_keys = sorted(
        k for k, t in tensors.items()
        if k.endswith(".weight") and t.dim() == 2
    )
    layers: list[nn.Linear] = []
    for wk in weight_keys:
        w = tensors[wk]
        out_f, in_f = w.shape
        bias_key = wk[:-len(".weight")] + ".bias"
        linear = nn.Linear(in_f, out_f, bias=(bias_key in tensors))
        linear.weight.data = w.to(dtype=dtype, device=device)
        if bias_key in tensors:
            linear.bias.data = tensors[bias_key].to(dtype=dtype, device=device)
        layers.append(linear)
    return MTPDraftHead(layers)
