import torch
from k4n0n3.memory import MemoryManager, _module_param_bytes


class DummyModule(torch.nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(size, size))


def test_memory_manager_start():
    mgr = MemoryManager(vram_budget_mb=1)
    assert mgr.used_bytes() == 0
    assert mgr.budget_bytes == 1 * 1024 * 1024


def test_mark_on_gpu():
    mgr = MemoryManager(vram_budget_mb=10)
    m = DummyModule(64)
    mgr.mark_on_gpu("layer_0", m)
    assert mgr.used_bytes() > 0


def test_eviction_when_full():
    mgr = MemoryManager(vram_budget_mb=1)
    m_big = DummyModule(512)
    mgr.mark_on_gpu("big", m_big)
    mgr.mark_on_gpu("small", DummyModule(64))
    assert "big" not in mgr._on_gpu
    assert "small" in mgr._on_gpu


def test_mark_off_gpu():
    mgr = MemoryManager(vram_budget_mb=10)
    m = DummyModule(64)
    mgr.mark_on_gpu("x", m)
    mgr.mark_off_gpu("x")
    assert mgr.used_bytes() == 0


def test_clear():
    mgr = MemoryManager(vram_budget_mb=10)
    mgr.mark_on_gpu("a", DummyModule(32))
    mgr.mark_on_gpu("b", DummyModule(32))
    mgr.clear()
    assert mgr.used_bytes() == 0


def test_module_param_bytes():
    m = DummyModule(10)
    b = _module_param_bytes(m)
    assert b == 10 * 10 * 4


def test_usage_ratio():
    mgr = MemoryManager(vram_budget_mb=1)
    assert mgr.usage_ratio() == 0.0
    mgr.mark_on_gpu("x", DummyModule(100))
    assert mgr.usage_ratio() > 0.0


def test_peak_tracking():
    mgr = MemoryManager(vram_budget_mb=10)
    m = DummyModule(64)
    mgr.mark_on_gpu("a", m)
    peak1 = mgr._peak_bytes
    mgr.mark_on_gpu("b", DummyModule(32))
    mgr.mark_off_gpu("b")
    assert mgr._peak_bytes >= peak1


def test_report():
    mgr = MemoryManager(vram_budget_mb=10)
    mgr.mark_on_gpu("x", DummyModule(32))
    r = mgr.report()
    assert "Budget" in r
    assert "Peak" in r
