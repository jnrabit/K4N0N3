"""LayerManager with pinned-memory master copies — upload=copy, offload=drop."""
from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import torch

from .memory import MemoryManager


class LayerState(Enum):
    ON_CPU = auto()
    PREFETCHING = auto()
    ON_GPU = auto()


@dataclass
class LayerInfo:
    state: LayerState = LayerState.ON_CPU
    size_mb: float = 0.0
    transfer_time_ms: float = 0.0
    compute_time_ms: float = 0.0


class LayerManager:
    """Per-layer device placement via forward hooks with async prefetch.

    Weights live in pinned CPU memory as a canonical master copy.
    "Upload"  = copy from pinned master → GPU (async).
    "Offload" = drop GPU tensor, point .data back to pinned master (no D2H).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_prefix: str = "model.layers",
        vram_budget_mb: int = 4096,
        prefetch_depth: int = 1,
        *,
        verbose: bool = False,
        pin_ram_fraction: float = 0.7,
    ):
        self.model = model
        self.prefetch_depth = max(1, prefetch_depth)
        self.verbose = verbose
        self.memory = MemoryManager(vram_budget_mb)

        self._layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
        self._layer_list: list[str] = []
        self._layer_info: dict[str, LayerInfo] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._prepared = False

        self._cuda_available = torch.cuda.is_available()
        self._prefetch_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream() if self._cuda_available else None
        )
        self._prefetch_events: dict[str, torch.cuda.Event] = {}

        # Pinned master copies: layer_name -> {param_name: pinned_tensor}
        self._cpu_master: dict[str, dict[str, torch.Tensor]] = {}
        # Fast param lookup: layer_name -> {param_name: nn.Parameter}
        self._param_refs: dict[str, dict[str, torch.nn.Parameter]] = {}
        # Per-layer pin status
        self._pinned: dict[str, bool] = {}
        self._pin_ram_fraction = max(0.0, min(1.0, pin_ram_fraction))

        self._discover_layers(layer_prefix)
        self._measure_layer_sizes()
        if self._cuda_available:
            self._build_master_copies()
        self._register_hooks()

    # -- discovery ----------------------------------------------------------

    def _discover_layers(self, prefix: str) -> None:
        module = _resolve_module(self.model, prefix)
        if module is None:
            modules = dict(self.model.named_modules())
            prefix = self._guess_layer_prefix(modules) or prefix
            module = _resolve_module(self.model, prefix)
        if module is None:
            raise ValueError(
                f"Could not find transformer layers at '{prefix}'. "
                f"Specify layer_prefix (e.g. 'model.layers' for Llama, "
                f"'transformer.h' for GPT-2)."
            )
        for idx, child in enumerate(module.children()):
            name = f"{prefix}.{idx}"
            self._layers[name] = child
            self._layer_list.append(name)
            self._layer_info[name] = LayerInfo()
        if self.verbose:
            print(f"[K4N0N3] Discovered {len(self._layers)} layers at '{prefix}'")

    @staticmethod
    def _guess_layer_prefix(modules: dict[str, torch.nn.Module]) -> str | None:
        candidates = []
        for name, mod in modules.items():
            if isinstance(mod, torch.nn.ModuleList):
                candidates.append((name, len(mod)))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    def _measure_layer_sizes(self) -> None:
        for name, mod in self._layers.items():
            nbytes = sum(
                p.numel() * p.element_size()
                for p in mod.parameters()
                if p.device.type != "meta"
            )
            self._layer_info[name].size_mb = nbytes / (1024 * 1024)
        if self.verbose:
            total = sum(li.size_mb for li in self._layer_info.values())
            avg = total / max(len(self._layers), 1)
            budget_layers = self.memory.budget_bytes / (avg * 1024 * 1024) if avg > 0 else 0
            print(
                f"[K4N0N3] Layer sizes: {avg:.1f} MB avg, {total:.1f} MB total "
                f"({len(self._layers)} layers) | Budget ~{budget_layers:.1f} layers"
            )

    # -- G+H: pinned/pageable master copies (partial pinning) ----------------

    def _build_master_copies(self) -> None:
        """Build CPU master copies — pinned where budget allows, pageable (zero-copy) otherwise."""
        avail = _available_ram_bytes()
        pin_budget = int(avail * self._pin_ram_fraction) if avail > 0 else 0
        total_pinned_bytes = 0

        for name in self._layer_list:
            mod = self._layers[name]
            master: dict[str, torch.Tensor] = {}
            pref: dict[str, torch.nn.Parameter] = {}
            layer_bytes = sum(p.numel() * p.element_size() for p in mod.parameters())

            can_pin = layer_bytes <= pin_budget and self._pin_ram_fraction > 0.0

            if can_pin:
                pinned_ok = True
                for pname, param in mod.named_parameters():
                    try:
                        pinned = param.detach().to("cpu").pin_memory()
                    except RuntimeError:
                        pinned_ok = False
                        break
                    param.data = pinned
                    master[pname] = pinned
                    pref[pname] = param
                if pinned_ok:
                    for bname, buf in mod.named_buffers():
                        try:
                            pinned_b = buf.detach().to("cpu").pin_memory()
                        except RuntimeError:
                            pinned_ok = False
                            break
                        mod._buffers[bname] = pinned_b
                        master[bname] = pinned_b
                    if pinned_ok:
                        pin_budget -= layer_bytes
                        total_pinned_bytes += layer_bytes
                        self._pinned[name] = True
                        self._cpu_master[name] = master
                        self._param_refs[name] = pref
                        continue

            # Pageable fallback — G: zero-copy, original IS the master
            self._pinned[name] = False
            for pname, param in mod.named_parameters():
                master[pname] = param.data  # reference, no clone
                pref[pname] = param
            for bname, buf in mod.named_buffers():
                master[bname] = buf
            self._cpu_master[name] = master
            self._param_refs[name] = pref

        if self.verbose:
            n_pinned = sum(1 for v in self._pinned.values() if v)
            n_total = len(self._layer_list)
            total_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
            pinned_mb = sum(
                sum(t.numel() * t.element_size() for t in d.values())
                for n, d in self._cpu_master.items() if self._pinned.get(n, False)
            ) / 1024**2
            print(f"[K4N0N3] Pinned {n_pinned}/{n_total} layers "
                  f"({pinned_mb:.0f}/{total_mb:.0f} MB), "
                  f"pin budget {self._pin_ram_fraction*100:.0f}% RAM")

    # -- hooks --------------------------------------------------------------

    def _register_hooks(self) -> None:
        for name, module in self._layers.items():
            pre = module.register_forward_pre_hook(self._make_pre_hook(name))
            post = module.register_forward_hook(self._make_post_hook(name))
            self._hook_handles.extend([pre, post])

    def _make_pre_hook(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, args):
            if not self._prepared:
                self.prepare()
            t0 = time.perf_counter()
            self.memory.mark_on_gpu(name, module)
            self._ensure_on_gpu(name)
            dt = (time.perf_counter() - t0) * 1000
            self._layer_info[name].transfer_time_ms = dt
            if self.verbose:
                used = self.memory.used_bytes() / 1024**2
                print(f"  [pre]  {name} | {dt:5.1f}ms | {used:5.0f}MB | {_state_summary(self._layer_info)}")
        return hook

    def _make_post_hook(self, name: str) -> Callable:
        def hook(module: torch.nn.Module, args, output):
            idx = self._layer_idx[name]
            n = len(self._layer_list)
            # Wrap-around prefetch for autoregressive generation
            for offset in range(1, self.prefetch_depth + 1):
                nxt = (idx + offset) % n
                self._prefetch_async(self._layer_list[nxt])
            # Offload layers far behind (modulo-safe)
            for offset in range(self.prefetch_depth + 1, self.prefetch_depth + 3):
                prev = (idx - offset) % n
                # Don't offload layers in the prefetch window
                self._offload(self._layer_list[prev])
            if self.verbose:
                print(f"  [post] {name} | {_state_summary(self._layer_info)}")
        return hook

    # -- A2: upload = copy from pinned master -------------------------------

    def _ensure_on_gpu(self, name: str) -> None:
        if not self._cuda_available:
            return
        info = self._layer_info[name]
        if info.state == LayerState.ON_GPU:
            return
        if info.state == LayerState.PREFETCHING:
            event = self._prefetch_events.pop(name, None)
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            # record_stream: tell allocator these tensors are now on main stream
            for pname in self._cpu_master.get(name, {}):
                param = _get_param_by_name(self._layers[name], pname)
                if param is not None and param.device.type == "cuda":
                    param.data.record_stream(torch.cuda.current_stream())
            info.state = LayerState.ON_GPU
            return
        # Cold upload
        _upload_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
        torch.cuda.current_stream().synchronize()
        info.state = LayerState.ON_GPU

    def _prefetch_async(self, name: str) -> None:
        if not self._cuda_available:
            return
        info = self._layer_info[name]
        if info.state != LayerState.ON_CPU:
            return
        # B: reserve budget before copying — evict if needed
        evicted = self.memory.mark_on_gpu(name, self._layers[name])
        for ev_name in evicted:
            if ev_name != name:
                self._offload(ev_name)
        info.state = LayerState.PREFETCHING
        with torch.cuda.stream(self._prefetch_stream):
            _upload_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
        self._prefetch_events[name] = self._prefetch_stream.record_event()

    # -- A3: offload = drop GPU tensor, point back to pinned master ---------

    def _offload(self, name: str) -> None:
        info = self._layer_info[name]
        if info.state == LayerState.ON_CPU:
            return
        # If prefetch still in flight, wait for it before dropping
        if info.state == LayerState.PREFETCHING:
            event = self._prefetch_events.pop(name, None)
            if event is not None:
                event.synchronize()
        # Drop: point .data back to pinned master
        _drop_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
        self._prefetch_events.pop(name, None)
        info.state = LayerState.ON_CPU
        self.memory.mark_off_gpu(name)

    # -- public API ---------------------------------------------------------

    def prepare(self) -> None:
        """Move non-layer modules to GPU, drop layers to CPU, prefetch first batch."""
        if self._prepared and self.memory.used_bytes() >= 0:
            # Already prepared — just verify consistency and prefetch first layers
            for i in range(min(self.prefetch_depth + 1, len(self._layer_list))):
                name = self._layer_list[i]
                if self._layer_info[name].state == LayerState.ON_CPU:
                    self._prefetch_async(name)
            if self.verbose:
                print(f"[K4N0N3] Re-prepare: {_state_summary(self._layer_info)}")
            return

        self.memory.clear()
        self._prefetch_events.clear()

        for name in self._layer_list:
            _drop_layer(self._layers[name], self._cpu_master.get(name, {}), self._param_refs.get(name))
            self._layer_info[name].state = LayerState.ON_CPU

        if self._cuda_available:
            self._move_fixed_to_gpu()

        limit = min(self.prefetch_depth + 1, len(self._layer_list))
        for i in range(limit):
            name = self._layer_list[i]
            if i == 0:
                self._ensure_on_gpu(name)
                self.memory.mark_on_gpu(name, self._layers[name])
            else:
                self._prefetch_async(name)

        self._prepared = True
        if self.verbose:
            print(f"[K4N0N3] Prepared: {_state_summary(self._layer_info)}")

    def _move_fixed_to_gpu(self) -> None:
        """Move non-layer leaf modules to GPU without touching layer params."""
        layer_set = set(self._layers.keys())
        for mod_name, mod in self.model.named_modules():
            # Exact prefix match with dot boundary
            if _matches_any_layer(mod_name, layer_set):
                continue
            if list(mod.parameters(recurse=False)):
                mod.to("cuda", non_blocking=True)

    def offload_all(self) -> None:
        """Drop all tracked layers to CPU. Non-layer modules go to CPU via .to()."""
        for name in self._layer_list:
            self._offload(name)
        # Move remaining (non-layer) modules to CPU
        layer_set = set(self._layers.keys())
        for mod_name, mod in self.model.named_modules():
            if _matches_any_layer(mod_name, layer_set):
                continue
            if list(mod.parameters(recurse=False)):
                mod.to("cpu")
        self._prepared = False

    @property
    def _layer_idx(self) -> dict[str, int]:
        if not hasattr(self, "_layer_idx_cache"):
            self._layer_idx_cache = {n: i for i, n in enumerate(self._layer_list)}
        return self._layer_idx_cache

    def stats(self) -> dict:
        return {
            name: {
                "size_mb": info.size_mb,
                "transfer_ms": info.transfer_time_ms,
                "compute_ms": info.compute_time_ms,
            }
            for name, info in self._layer_info.items()
        }

    def remove_hooks(self) -> None:
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()


# -- A2 helpers: per-parameter upload/drop ----------------------------------


def _upload_layer(module: torch.nn.Module, master: dict[str, torch.Tensor],
                  param_refs: dict[str, torch.nn.Parameter] | None = None) -> None:
    """Copy each param/buffer from pinned master to GPU (non-blocking)."""
    if param_refs is None:
        param_dict = dict(module.named_parameters())
    else:
        param_dict = param_refs
    for pname, pinned in master.items():
        param = param_dict.get(pname)
        if param is not None:
            param.data = pinned.to("cuda", non_blocking=True)
        elif pname in module._buffers:
            module._buffers[pname] = pinned.to("cuda", non_blocking=True)


def _drop_layer(module: torch.nn.Module, master: dict[str, torch.Tensor],
                param_refs: dict[str, torch.nn.Parameter] | None = None) -> None:
    """Point each param/buffer back to its pinned CPU master — GPU tensor freed."""
    if param_refs is None:
        param_dict = dict(module.named_parameters())
    else:
        param_dict = param_refs
    for pname, pinned in master.items():
        param = param_dict.get(pname)
        if param is not None:
            param.data = pinned
        elif pname in module._buffers:
            module._buffers[pname] = pinned


def _get_param_by_name(module: torch.nn.Module, name: str) -> torch.nn.Parameter | None:
    for n, p in module.named_parameters():
        if n == name:
            return p
    return None


# -- helpers --------------------------------------------------------------


def _matches_any_layer(mod_name: str, layer_set: set[str]) -> bool:
    """Check if mod_name is inside any tracked layer (exact prefix with dot)."""
    for ln in layer_set:
        if mod_name == ln or mod_name.startswith(ln + "."):
            return True
    return False


def _state_summary(info: dict[str, LayerInfo]) -> str:
    on_gpu = [n for n, i in info.items() if i.state == LayerState.ON_GPU]
    pref = [n for n, i in info.items() if i.state == LayerState.PREFETCHING]
    cpu = [n for n, i in info.items() if i.state == LayerState.ON_CPU]
    parts = []
    if on_gpu:
        parts.append(f"GPU:{_range_str(on_gpu)}")
    if pref:
        parts.append(f"PREF:{_range_str(pref)}")
    if cpu:
        parts.append(f"CPU:{_range_str(cpu)}")
    return " ".join(parts)


def _range_str(names: list[str]) -> str:
    if len(names) <= 3:
        return ",".join(n.rsplit(".", 1)[-1] for n in names)
    first = names[0].rsplit(".", 1)[-1]
    last = names[-1].rsplit(".", 1)[-1]
    return f"{first}..{last}({len(names)})"


def _available_ram_bytes() -> int:
    """Read MemAvailable from /proc/meminfo (Linux). Returns 0 on failure."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return 0


def _resolve_module(root: torch.nn.Module, dotted: str) -> torch.nn.Module | None:
    parts = dotted.split(".")
    current = root
    for part in parts:
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current
