from __future__ import annotations

import torch


class ManagedTensor:
    """A tensor wrapper that tracks device placement and provides transparent access.

    Unlike the previous implementation, this does NOT subclass ``torch.Tensor``.
    It simply wraps a tensor and exposes the ``.data`` property for access.
    """

    def __init__(self, data: torch.Tensor, target_device: str = "cuda"):
        self._target = torch.device(target_device)
        self._data = data

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self) -> torch.device:
        return self._data.device

    def to(self, device: str | torch.device) -> ManagedTensor:
        if isinstance(device, str):
            device = torch.device(device)
        if self._data.device != device:
            self._data = self._data.to(device, non_blocking=True)
        return self

    def cuda(self) -> ManagedTensor:
        return self.to("cuda")

    def cpu(self) -> ManagedTensor:
        return self.to("cpu")

    def ensure_on(self, device: str | torch.device) -> ManagedTensor:
        return self.to(device)

    def __repr__(self) -> str:
        return f"ManagedTensor(shape={tuple(self._data.shape)}, device={self._data.device})"
