from __future__ import annotations

import torch


def get_gpu_info(device_idx: int = 0) -> dict[str, float] | None:
    """Return GPU memory info: total, free, used (all in MB)."""
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(device_idx)
    return {
        "total_mb": total / (1024 * 1024),
        "free_mb": free / (1024 * 1024),
        "used_mb": (total - free) / (1024 * 1024),
    }


def auto_vram_budget(safety_factor: float = 0.85, device_idx: int = 0) -> int:
    """Return a safe VRAM budget in MB (default: 85% of free VRAM)."""
    info = get_gpu_info(device_idx)
    if info is None:
        return 4096
    return int(info["free_mb"] * safety_factor)


def estimate_model_size(model: torch.nn.Module) -> float:
    """Estimate total model size in MB (parameters only, CPU memory)."""
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return nbytes / (1024 * 1024)


def list_layers(model: torch.nn.Module, max_depth: int = 4) -> list[str]:
    """List named modules up to a given depth (for debugging model structure)."""
    result = []
    for name, module in model.named_modules():
        depth = name.count(".")
        if depth <= max_depth and name:
            pcount = sum(1 for _ in module.parameters(recurse=False))
            if pcount > 0 or isinstance(module, torch.nn.ModuleList):
                result.append(name)
    return result


def available_ram_mb() -> float:
    """MemAvailable from /proc/meminfo in MB. Fallback: psutil. Returns 0 if unknown."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1024**2
    except ImportError:
        return 0.0
