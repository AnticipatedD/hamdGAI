import torch
import pytest

import rocm_primitives_demo as demo


@pytest.fixture(autouse=True)
def fake_cuda(monkeypatch):
    """Force torch.cuda.is_available to True and mock device name."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "FakeGPU")
    monkeypatch.setattr(torch.version, "cuda", "12.0")


def test_demo_gemm_runs_and_returns_tensor():
    device = torch.device("cpu")  # inject CPU device for testing
    out = demo.demo_rocblas_gemm(device)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 2


def test_demo_miopen_primitives_latency():
    device = torch.device("cpu")
    latency = demo.demo_miopen_primitives(device)
    assert isinstance(latency, float)
    assert latency >= 0.0


def test_demo_fft_and_reconstruction_error():
    device = torch.device("cpu")
    spectrum, max_err = demo.demo_rocfft(device)
    assert isinstance(spectrum, torch.Tensor)
    assert isinstance(max_err, float)
    assert max_err >= 0.0


def test_demo_decompositions_outputs():
    device = torch.device("cpu")
    chol_diff, top5, qr_diff = demo.demo_rocsolver_decompositions(device)
    assert isinstance(chol_diff, float)
    assert isinstance(top5, torch.Tensor)
    assert top5.shape[0] == 5
    assert isinstance(qr_diff, float)
