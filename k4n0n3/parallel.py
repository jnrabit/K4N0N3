from __future__ import annotations

import torch

from .hooks import LayerManager


class PipelineParallel:
    """Split transformer layers across multiple GPUs (simple pipeline)."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_prefix: str = "model.layers",
        device_ids: list[int] | None = None,
        vram_budget_mb: int = 4096,
        prefetch_depth: int = 1,
        *,
        verbose: bool = False,
    ):
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus < 2:
            raise RuntimeError("PipelineParallel requires at least 2 GPUs.")
        self.device_ids = device_ids or list(range(n_gpus))
        self.n_gpus = len(self.device_ids)

        from .hooks import _resolve_module

        container = _resolve_module(model, layer_prefix)
        if container is None:
            modules = dict(model.named_modules())
            candidates = [
                (n, len(m))
                for n, m in modules.items()
                if isinstance(m, torch.nn.ModuleList)
            ]
            candidates.sort(key=lambda x: x[1], reverse=True)
            if not candidates:
                raise ValueError(f"Could not find transformer layers.")
            layer_prefix = candidates[0][0]
            container = _resolve_module(model, layer_prefix)

        all_children = list(container.children())
        total_layers = len(all_children)

        self.managers: list[LayerManager] = []
        self._boundaries: list[int] = []

        layers_per_gpu = total_layers // self.n_gpus
        remainder = total_layers % self.n_gpus

        start = 0
        for gpu_idx in range(self.n_gpus):
            device = torch.device(f"cuda:{self.device_ids[gpu_idx]}")
            chunk_size = layers_per_gpu + (1 if gpu_idx < remainder else 0)
            end = start + chunk_size

            wrapper = _build_partial(
                all_children[start:end],
                device=device,
                prefix=layer_prefix,
                start_idx=start,
            )

            mgr = LayerManager(
                wrapper,
                layer_prefix="layers",
                vram_budget_mb=vram_budget_mb,
                prefetch_depth=prefetch_depth,
                verbose=verbose,
            )
            self.managers.append(mgr)
            self._boundaries.append(end)
            start = end

        if verbose:
            print(
                f"[K4N0N3] Pipeline: {total_layers} layers → "
                f"{self.n_gpus} GPUs ({[b for b in self._boundaries]})"
            )

    def prepare(self) -> None:
        for mgr in self.managers:
            mgr.prepare()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, mgr in enumerate(self.managers):
            device = torch.device(f"cuda:{self.device_ids[i]}")
            x = x.to(device)
            for layer in mgr._layers.values():
                x = layer(x)
        return x

    def remove_hooks(self) -> None:
        for mgr in self.managers:
            mgr.remove_hooks()


class _PartialModel(torch.nn.Module):
    """Holds a contiguous slice of transformer layers."""

    def __init__(self, layers: list[torch.nn.Module]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _build_partial(
    layers: list[torch.nn.Module],
    device: torch.device,
    prefix: str,
    start_idx: int,
) -> _PartialModel:
    wrapper = _PartialModel(layers)
    return wrapper.to(device)
