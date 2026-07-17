from __future__ import annotations

import time
from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class LRUCache(Generic[T]):
    def __init__(self, max_size: int = 256, ttl: float | None = None):
        self.max_size = max_size
        self.ttl = ttl
        self._store: OrderedDict[str, tuple[T, float]] = OrderedDict()

    def get(self, key: str) -> T | None:
        if key not in self._store:
            return None
        value, inserted = self._store[key]
        if self.ttl is not None and (time.monotonic() - inserted) > self.ttl:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return value

    def put(self, key: str, value: T) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif len(self._store) >= self.max_size:
            self._store.popitem(last=False)
        self._store[key] = (value, time.monotonic())

    def remove(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: str) -> bool:
        if key not in self._store:
            return False
        _value, inserted = self._store[key]
        if self.ttl is not None and (time.monotonic() - inserted) > self.ttl:
            del self._store[key]
            return False
        return True
