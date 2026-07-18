"""Training-aware offload: Master/Drop + Backward-Hooks (Q1).

TrainingManager erbt von LayerManager — der LayerManager IST die gemeinsame
Offload-Basis (Discovery, Master-Aufbau inkl. O-Pin-Fixes, Upload/Drop,
Prefetch-Stream, Budget-Anbindung). Ein separates _OffloadCore waere eine
dritte Klasse ohne eigenen Zustand; Vererbung loest die fruehere
Code-Duplikation in training.py (Discovery, Prefetch, module.to()-Pfade)
vollstaendig auf.

Bewusste Einschraenkung: Der Drop-Offload verwirft GPU-Tensoren ersatzlos —
das ist nur korrekt fuer Gewichte, die sich nicht aendern. Basisgewichte
MUESSEN deshalb frozen sein (requires_grad=False); trainiert werden
ausschliesslich Adapter-Parameter (requires_grad=True), die von der
Master/Drop-Mechanik ausgenommen sind und permanent auf der GPU liegen.
Voll-Finetuning mit Offload = anderes Projekt: braeuchte D2H-Writeback beim
Drop plus Optimizer-State-Offload.

Backward-Richtung: Im Backward laeuft die Layer-Reihenfolge rueckwaerts,
also prefetcht bw_post Layer idx-1, idx-2, ... und offloadet die gerade
abgearbeiteten hoeheren Indizes. Bei gradient_checkpointing feuern die
Forward-Hooks waehrend der Rekomputation erneut (in Rueckwaerts-Reihenfolge
der Layer) — fw_post haelt sich dann heraus (self._phase == "bw"), damit
der Forward-Prefetch nicht gegen den Backward-Prefetch arbeitet.
"""
from __future__ import annotations

from typing import Callable

import torch

from .hooks import LayerManager


class TrainingManager(LayerManager):
    """Master/Drop-Offloading fuer Training (frozen Basis + Adapter)."""

    # Trainierbare Params (Adapter) bekommen keinen Master-Eintrag und werden
    # damit nie gedroppt/hochgeladen — sie bleiben permanent auf der GPU.
    _skip_trainable = True

    def __init__(
        self,
        model: torch.nn.Module,
        layer_prefix: str = "model.layers",
        vram_budget_mb: int = 4096,
        prefetch_depth: int = 1,
        *,
        verbose: bool = False,
        pin_ram_fraction: float = 0.7,
        quantize_transfer: bool | str = False,
        int4_group_size: int = 64,
    ):
        self._phase = "fw"
        super().__init__(
            model,
            layer_prefix=layer_prefix,
            vram_budget_mb=vram_budget_mb,
            prefetch_depth=prefetch_depth,
            verbose=verbose,
            pin_ram_fraction=pin_ram_fraction,
            quantize_transfer=quantize_transfer,
            int4_group_size=int4_group_size,
        )

    # -- hooks: forward + backward ------------------------------------------

    def _register_hooks(self) -> None:
        for name, module in self._layers.items():
            self._hook_handles.extend([
                module.register_forward_pre_hook(self._make_pre_hook(name)),
                module.register_forward_hook(self._make_fw_post(name)),
                module.register_full_backward_pre_hook(self._make_bw_pre(name)),
                module.register_full_backward_hook(self._make_bw_post(name)),
            ])

    def _make_fw_post(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, args, output):
            if self._phase != "fw":
                # Rekompute unter gradient_checkpointing: Prefetch/Offload
                # steuern hier die Backward-Hooks, nicht der Forward-Pfad.
                return
            idx = self._layer_idx[name]
            n = len(self._layer_list)
            # Kein Wrap-around wie im Inference-Pfad: die Schritt-Sequenz ist
            # Forward 0..n-1, dann Backward n-1..0.
            for offset in range(1, self.prefetch_depth + 1):
                nxt = idx + offset
                if nxt < n:
                    self._prefetch_async(self._layer_list[nxt])
            for offset in range(self.prefetch_depth + 1, self.prefetch_depth + 3):
                prev = idx - offset
                if prev >= 0:
                    self._offload(self._layer_list[prev])
        return hook

    def _make_bw_pre(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, grad_output):
            self._phase = "bw"
            self.memory.mark_on_gpu(name, module, self._layer_gpu_bytes.get(name))
            self._ensure_on_gpu(name)
        return hook

    def _make_bw_post(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, grad_input, grad_output):
            idx = self._layer_idx[name]
            n = len(self._layer_list)
            for offset in range(1, self.prefetch_depth + 1):
                prev = idx - offset
                if prev >= 0:
                    self._prefetch_async(self._layer_list[prev])
            for offset in range(self.prefetch_depth + 1, self.prefetch_depth + 3):
                nxt = idx + offset
                if nxt < n:
                    self._offload(self._layer_list[nxt])
            if idx == 0:
                self._phase = "fw"  # Backward abgeschlossen, naechster Schritt
        return hook

    # -- public API ---------------------------------------------------------

    def prepare(self) -> None:
        self._assert_frozen_base()
        super().prepare()
        self._adapters_to_gpu()

    def _assert_frozen_base(self) -> None:
        """Harter Guard: getrackte Basisgewichte muessen frozen sein.

        Trainierbare Params innerhalb der Layer gelten als Adapter und
        muessen klein sein (< 10 % der Layer-Bytes). Ist mehr trainierbar,
        ist die Basis vermutlich nicht eingefroren — dann wuerde der
        Drop-Offload Gradienten-Updates stillschweigend verwerfen.
        """
        for name, mod in self._layers.items():
            total = trainable = 0
            for p in mod.parameters():
                b = p.numel() * p.element_size()
                total += b
                if p.requires_grad:
                    trainable += b
            if total and trainable > 0.1 * total:
                raise ValueError(
                    f"Layer '{name}': {trainable / 1024**2:.0f} MB von "
                    f"{total / 1024**2:.0f} MB sind trainierbar. Der Drop-Offload "
                    f"verwirft GPU-Tensoren ersatzlos — Basisgewichte muessen "
                    f"frozen sein (requires_grad=False), trainiert werden nur "
                    f"kleine Adapter. Vor dem TrainingManager: "
                    f"`for p in model.parameters(): p.requires_grad_(False)` "
                    f"und dann Adapter (LoRA) hinzufuegen."
                )

    def _adapters_to_gpu(self) -> None:
        """Adapter (trainierbar, ohne Master) permanent auf die GPU legen."""
        if not self._cuda_available:
            return
        for mod in self._layers.values():
            for p in mod.parameters():
                if p.requires_grad and p.device.type != "cuda":
                    p.data = p.data.to("cuda")

    def report(self) -> str:
        return self.memory.report()
