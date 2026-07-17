from __future__ import annotations

from typing import Callable

import torch

from .hooks import LayerManager, LayerState
from .memory import MemoryManager


class TrainingManager:
    """Training-aware layer manager with backward-pass device management."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_prefix: str = "model.layers",
        vram_budget_mb: int = 4096,
        prefetch_depth: int = 1,
        *,
        verbose: bool = False,
    ):
        self.model = model
        self.prefetch_depth = max(1, prefetch_depth)
        self.verbose = verbose
        self.memory = MemoryManager(vram_budget_mb)

        from .hooks import _resolve_module

        layers, layer_names = self._discover(model, layer_prefix)
        self._layers = layers
        self._layer_list = layer_names
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        self._cuda_available = torch.cuda.is_available()
        self._prefetch_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if self._cuda_available else None
        )
        self._prefetch_events: dict[str, torch.cuda.Event] = {}
        self._prepared = False

        self._register_hooks()

    @staticmethod
    def _discover(
        model: torch.nn.Module, prefix: str
    ) -> tuple[dict[str, torch.nn.Module], list[str]]:
        from collections import OrderedDict

        from .hooks import _resolve_module

        module = _resolve_module(model, prefix)
        if module is None:
            modules = dict(model.named_modules())
            candidates = [
                (n, len(m))
                for n, m in modules.items()
                if isinstance(m, torch.nn.ModuleList)
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            if candidates:
                prefix = candidates[0][0]
                module = _resolve_module(model, prefix)
        if module is None:
            raise ValueError(f"Could not find transformer layers at '{prefix}'.")

        layers = OrderedDict()
        names = []
        for idx, child in enumerate(module.children()):
            name = f"{prefix}.{idx}"
            layers[name] = child
            names.append(name)
        return layers, names

    def _register_hooks(self) -> None:
        for name, module in self._layers.items():
            fw_pre = module.register_forward_pre_hook(self._make_fw_pre(name))
            fw_post = module.register_forward_hook(self._make_fw_post(name))
            bw_pre = module.register_full_backward_pre_hook(self._make_bw_pre(name))
            bw_post = module.register_full_backward_hook(self._make_bw_post(name))
            self._hook_handles.extend([fw_pre, fw_post, bw_pre, bw_post])

    def _make_fw_pre(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, args):
            if not self._prepared:
                self.prepare()
            self.memory.mark_on_gpu(name, module)
            self._ensure_on_gpu(name)
        return hook

    def _make_fw_post(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, args, output):
            idx = self._layer_idx[name]
            for offset in range(1, self.prefetch_depth + 1):
                nxt = idx + offset
                if nxt < len(self._layer_list):
                    self._prefetch_async(self._layer_list[nxt])
            # Do NOT offload — params stay for backward pass
        return hook

    def _make_bw_pre(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, grad_output):
            # Params may have been evicted during forward; re-fetch
            self._ensure_on_gpu(name)
        return hook

    def _make_bw_post(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, grad_input, grad_output):
            self._offload(name)
        return hook

    def _ensure_on_gpu(self, name: str) -> None:
        if not self._cuda_available:
            return
        # Check if already there by peeking at first param
        params = list(self._layers[name].parameters())
        if params and params[0].device.type == "cuda":
            return
        # Wait for pending prefetch if any
        event = self._prefetch_events.pop(name, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
            return
        self._layers[name].to("cuda", non_blocking=True)
        torch.cuda.current_stream().synchronize()

    def _prefetch_async(self, name: str) -> None:
        if not self._cuda_available:
            return
        params = list(self._layers[name].parameters())
        if params and params[0].device.type == "cuda":
            return
        with torch.cuda.stream(self._prefetch_stream):
            self._layers[name].to("cuda", non_blocking=True)
        self._prefetch_events[name] = self._prefetch_stream.record_event()

    def _offload(self, name: str) -> None:
        params = list(self._layers[name].parameters())
        if not params or params[0].device.type == "cpu":
            return
        self._layers[name].to("cpu")
        self._prefetch_events.pop(name, None)
        self.memory.mark_off_gpu(name)

    def prepare(self) -> None:
        target = "cuda" if self._cuda_available else "cpu"
        self.model.to(target)
        self.memory.clear()
        self._prefetch_events.clear()
        for mod in self._layers.values():
            mod.to("cpu")
        limit = min(self.prefetch_depth + 1, len(self._layer_list))
        for i in range(limit):
            name = self._layer_list[i]
            if i == 0:
                self._ensure_on_gpu(name)
                self.memory.mark_on_gpu(name, self._layers[name])
            else:
                self._prefetch_async(name)
        self._prepared = True

    @property
    def _layer_idx(self) -> dict[str, int]:
        if not hasattr(self, "_layer_idx_cache"):
            self._layer_idx_cache = {n: i for i, n in enumerate(self._layer_list)}
        return self._layer_idx_cache

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    def report(self) -> str:
        return self.memory.report()
