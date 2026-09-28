"""
ROCm Compute Libraries Demonstration
====================================
Demonstrates hardware-accelerated primitives (GEMM, conv, FFT, decompositions)
via PyTorch on AMD ROCm (or CUDA) with CPU-friendly structure for testing.
"""

from __future__ import annotations

import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import structlog

logger = structlog.get_logger()


def check_rocm_environment() -> torch.device:
    """Validate that a CUDA/ROCm device is available and return it."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ROCm/CUDA device unavailable. "
            "Ensure ROCm drivers and a PyTorch ROCm build are installed."
        )
    device = torch.device("cuda:0")
    logger.info("device_info",
                name=torch.cuda.get_device_name(device),
                version=torch.version.cuda or getattr(torch.version, "hip", None))
    return device


def demo_rocblas_gemm(device: torch.device, seed: int = 42) -> torch.Tensor:
    """Level-3 GEMM + bias + GELU (rocBLAS / hipBLASLt path)."""
    torch.manual_seed(seed)
    batch_size, in_features, out_features = 2048, 1024, 4096

    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    w = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    b = torch.randn(out_features, dtype=torch.float16, device=device)

    output = F.linear(x, w, b)
    output = F.gelu(output)
    torch.cuda.synchronize()
    return output


def demo_miopen_primitives(device: torch.device, seed: int = 42) -> float:
    """Conv2d + BatchNorm + ReLU pipeline (MIOpen path). Returns avg latency in ms."""
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True

    model = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
    ).to(device)

    dummy_input = torch.randn(32, 64, 224, 224, device=device)

    # Warm-up
    _ = model(dummy_input)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 10  # average over 100 iters
    return elapsed_ms


def demo_rocfft(device: torch.device, seed: int = 42) -> Tuple[torch.Tensor, float]:
    """1-D real FFT + inverse. Returns spectrum and max reconstruction error."""
    torch.manual_seed(seed)
    batch, signal_len = 64, 1024
    time_signal = torch.randn(batch, signal_len, device=device, dtype=torch.float32)

    freq_spectrum = torch.fft.rfft(time_signal, dim=-1)
    reconstructed = torch.fft.irfft(freq_spectrum, n=signal_len, dim=-1)
    max_error = torch.max(torch.abs(time_signal - reconstructed)).item()
    return freq_spectrum, max_error


def demo_rocsolver_decompositions(
    device: torch.device, seed: int = 42
) -> Tuple[float, torch.Tensor, float]:
    """Cholesky, SVD, QR. Returns (cholesky residual, top-5 singular values, QR residual)."""
    torch.manual_seed(seed)
    matrix_size = 512
    A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float64)

    # Cholesky
    SPD = A @ A.mT + torch.eye(matrix_size, device=device, dtype=torch.float64)
    L = torch.linalg.cholesky(SPD)
    cholesky_diff = torch.dist(L @ L.mT, SPD).item()

    # SVD
    _, S, _ = torch.linalg.svd(A)
    top5 = S[:5].detach().cpu()

    # QR
    Q, R = torch.linalg.qr(A)
    qr_diff = torch.dist(Q @ R, A).item()

    return cholesky_diff, top5, qr_diff


if __name__ == "__main__":
    logger.info("init_suite", message="Initializing ROCm High-Performance Compute Library Test Suite")
    gpu_device = check_rocm_environment()

    gemm_out = demo_rocblas_gemm(gpu_device)
    logger.info("gemm_result", shape=str(gemm_out.shape), dtype=str(gemm_out.dtype))

    latency = demo_miopen_primitives(gpu_device)
    logger.info("miopen_latency", avg_ms=latency)

    spectrum, max_err = demo_rocfft(gpu_device)
    logger.info("fft_result", spectrum_shape=str(spectrum.shape), max_error=max_err)

    chol_diff, top5, qr_diff = demo_rocsolver_decompositions(gpu_device)
    logger.info("decomposition_result",
                cholesky_residual=chol_diff,
                top5_singular_values=top5.numpy().tolist(),
                qr_residual=qr_diff)

    logger.info("suite_complete", message="All ROCm compute library pipelines executed successfully")
