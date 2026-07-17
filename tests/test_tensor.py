import torch
from k4n0n3.tensor import ManagedTensor

def test_managed_tensor_creation():
    data = torch.randn(3, 4)
    mt = ManagedTensor(data)
    assert mt.shape == (3, 4)
    assert mt.dtype == torch.float32

def test_managed_tensor_to_cpu_cuda():
    data = torch.randn(2, 2)
    mt = ManagedTensor(data)
    assert str(mt.device) == "cpu"
    if torch.cuda.is_available():
        mt.cuda()
        assert str(mt.device).startswith("cuda")
        mt.cpu()
        assert str(mt.device) == "cpu"

def test_managed_tensor_ensure_on():
    data = torch.randn(2, 2)
    mt = ManagedTensor(data)
    mt.ensure_on("cpu")
    assert str(mt.device) == "cpu"

def test_managed_tensor_repr():
    mt = ManagedTensor(torch.randn(4, 5))
    r = repr(mt)
    assert "ManagedTensor" in r
    assert "4, 5" in r
