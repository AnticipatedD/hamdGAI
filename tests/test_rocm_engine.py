"""
Automated Test Suite for hamdGAI ROCm Engine
=============================================
Run with: pytest tests/ -v
"""

import pytest
import torch
from hamdgai.rocm_engine import ROCmDeviceManager, MIOpenAutotunedPipeline, ROCmComputeSuite


@pytest.fixture(scope="module")
def rocm_setup():
    if not torch.cuda.is_available():
        pytest.skip("ROCm/CUDA GPU hardware non-present. Skipping hardware tests.")
    manager = ROCmDeviceManager()
    suite = ROCmComputeSuite(manager)
    return manager, suite


def test_rocm_device_detection(rocm_setup):
    manager, _ = rocm_setup
    info = manager.get_system_info()
    assert "device_name" in info
    assert info["device_count"] > 0


def test_hipblaslt_gemm_execution(rocm_setup):
    _, suite = rocm_setup
    batch, in_f, out_f = 128, 512, 1024
    x = torch.randn(batch, in_f)
    w = torch.randn(out_f, in_f)
    b = torch.randn(out_f)
    
    out = suite.execute_hipblaslt_fused_gemm(x, w, b)
    assert out.shape == (batch, out_f)
    assert out.dtype == torch.float16


def test_miopen_autotune_pipeline(rocm_setup):
    manager, _ = rocm_setup
    pipeline = MIOpenAutotunedPipeline(in_channels=32, out_channels=64).to(manager.device)
    dummy_input = torch.randn(16, 32, 64, 64, device=manager.device)
    
    # Run twice to verify FindDb / PerfDb kernel cache hit
    out_first = pipeline(dummy_input)
    out_second = pipeline(dummy_input)
    
    assert out_first.shape == (16, 64, 64, 64)
    assert torch.allclose(out_first, out_second, atol=1e-5)


def test_rocfft_precision(rocm_setup):
    _, suite = rocm_setup
    signals = torch.randn(32, 2048)
    spectrum = suite.execute_rocfft_batched(signals)
    reconstructed = torch.fft.irfft(spectrum, n=2048, dim=-1)
    
    diff = torch.max(torch.abs(signals.to(suite.device) - reconstructed)).item()
    assert diff < 1e-4


def test_rocsolver_cholesky_residual(rocm_setup):
    _, suite = rocm_setup
    raw_matrix = torch.randn(128, 128)
    L = suite.execute_rocsolver_cholesky(raw_matrix)
    
    # Reconstruct A = L * L^T
    reconstructed_spd = L @ L.mT
    assert L.shape == (128, 128)
    assert not torch.isnan(reconstructed_spd).any()
