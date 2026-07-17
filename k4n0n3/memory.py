from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

import torch


@dataclass
class MemorySnapshot:
    timestamp: float = 0.0
    used_mb: float = 0.0
    peak_mb: float = 0.0
    budget_mb: float = 0.0
    layers_on_gpu: int = 0


class MemoryManager:
    """Tracks VRAM usage and handles LRU eviction."""

    def __init__(self, vram_budget_mb: int = 4096):
        self.budget_bytes = vram_budget_mb * 1024 * 1024
        self._reserve_bytes: int = 0
        self._on_gpu: OrderedDict[str, int] = OrderedDict()
        self._peak_bytes: int = 0
        self._snapshots: list[MemorySnapshot] = []

    def used_bytes(self) -> int:
        return sum(self._on_gpu.values())

    def available_bytes(self) -> int:
        return max(0, self.budget_bytes - self.used_bytes() - self._reserve_bytes)

    def set_reserve(self, bytes_: int) -> None:
        self._reserve_bytes = bytes_

    def reserve_mb(self) -> float:
        return self._reserve_bytes / (1024 * 1024)

    def usage_ratio(self) -> float:
        if self.budget_bytes == 0:
            return 0.0
        return self.used_bytes() / self.budget_bytes

    def mark_on_gpu(self, name: str, module: torch.nn.Module) -> list[str]:
        """Register a module on GPU. Returns list of evicted layer names (LRU order)."""
        evicted: list[str] = []
        size = _module_param_bytes(module)
        if name in self._on_gpu:
            self._on_gpu.move_to_end(name)
            prev = self._on_gpu[name]
            self._on_gpu[name] = size
            self._peak_bytes = max(self._peak_bytes, self.used_bytes())
            if prev != size:
                self._record_snapshot()
            return evicted
        while self._on_gpu and self.used_bytes() + size > self.budget_bytes - self._reserve_bytes:
            evicted_name, _ = self._on_gpu.popitem(last=False)
            evicted.append(evicted_name)
        self._on_gpu[name] = size
        self._peak_bytes = max(self._peak_bytes, self.used_bytes())
        self._record_snapshot()
        return evicted

    def mark_off_gpu(self, name: str) -> None:
        self._on_gpu.pop(name, None)
        self._record_snapshot()

    def clear(self) -> None:
        self._on_gpu.clear()
        self._peak_bytes = 0
        self._snapshots.clear()

    def _record_snapshot(self) -> None:
        self._snapshots.append(
            MemorySnapshot(
                timestamp=time.perf_counter(),
                used_mb=self.used_bytes() / (1024 * 1024),
                peak_mb=self._peak_bytes / (1024 * 1024),
                budget_mb=self.budget_bytes / (1024 * 1024),
                layers_on_gpu=len(self._on_gpu),
            )
        )

    def report(self) -> str:
        if not self._snapshots:
            return "[MemoryManager] No data."
        first = self._snapshots[0]
        last = self._snapshots[-1]
        peak = max(s.used_mb for s in self._snapshots)
        return (
            f"[MemoryManager] Budget: {last.budget_mb:.0f} MB | "
            f"Peak used: {peak:.1f} MB ({100 * peak / last.budget_mb:.1f}%) | "
            f"Layers on GPU: {last.layers_on_gpu}"
        )


def _module_param_bytes(module: torch.nn.Module) -> int:
    return sum(
        p.numel() * p.element_size()
        for p in module.parameters()
        if p.device.type != "meta"
    )
