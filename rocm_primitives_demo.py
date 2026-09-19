"""
ROCm Compute Libraries Demonstration
====================================
This module demonstrates hardware-accelerated deep learning and scientific computing primitives
leveraging AMD ROCm compute libraries (rocBLAS/hipBLASLt, MIOpen, rocFFT, and rocSOLVER) via PyTorch.

Repository: https://github.com/AnticipatedD/hamdGAI
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def check_rocm_environment() -> torch.device:
    """Validates ROCm support and returns the GPU device."""
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm/CUDA device unavailable. Ensure ROCm drivers and PyTorch ROCm build are installed.")
    
    device = torch.device("cuda:0")
    print(f"[+] Device Name: {torch.cuda.get_device_name(device)}")
    print(f"[+] ROCm / CUDA Version: {torch.version.cuda or torch.version.hip}")
    return device


# ==============================================================================
# 1. rocBLAS / hipBLASLt — Dense Linear Algebra (Level 1, 2, 3 GEMM & Fused Activation)
# ==============================================================================
def demo_rocblas_gemm(device: torch.device):
    """
    Demonstrates Level 3 GEMM operations accelerated by rocBLAS and fused GEMM + Bias + Activation
    layout routines handled by hipBLASLt on CDNA architectures.
    """
    print("\n--- 1. rocBLAS / hipBLASLt (Dense Linear Algebra) ---")
    
    batch_size, in_features, out_features = 2048, 1024, 4096
    
    # FP16 matrix multiplication offloaded to MFMA matrix engines
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device=device)
    w = torch.randn(out_features, in_features, dtype=torch.float16, device=device)
    b = torch.randn(out_features, dtype=torch.float16, device=device)
    
    # GEMM: Y = Activation(alpha * A * B + beta * C + bias)
    # Executed via PyTorch optimized hipBLASLt dispatcher
    output = F.linear(x, w, b)
    output = F.gelu(output)
    
    torch.cuda.synchronize()
    print(f"[✓] GEMM + GELU Tensor Output Shape: {output.shape}, Dtype: {output.dtype}")


# ==============================================================================
# 2. MIOpen — Deep Learning Primitives & Autotuning Engine
# ==============================================================================
def demo_miopen_primitives(device: torch.device):
    """
    Demonstrates MIOpen primitives (Conv2d + BatchNorm + ReLU) with cuDNN-equivalent autotuning.
    FindDb / PerfDb / Kernel Cache acceleration mechanisms are enabled via cudnn/miopen flags.
    """
    print("\n--- 2. MIOpen (Deep Learning Primitives & Autotuner) ---")
    
    # Enable MIOpen / PyTorch autotuner (FindDb / PerfDb lookup & kernel compilation caching)
    torch.backends.cudnn.benchmark = True
    
    # Construct a Conv2D -> BatchNorm -> ReLU fused pipeline
    model = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True)
    ).to(device)
    
    dummy_input = torch.randn(32, 64, 224, 224, device=device)
    
    # Warm-up pass (triggers FindDb search, kernel synthesis, and PerfDb caching)
    _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Benchmark execution (reuses cached optimized GPU binaries)
    start_time = time.perf_counter()
    for _ in range(100):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start_time) * 10
    
    print(f"[✓] MIOpen Conv2D Pipeline 100-iter Avg Latency: {elapsed_ms:.3f} ms")


# ==============================================================================
# 3. rocFFT — Fast Fourier Transforms
# ==============================================================================
def demo_rocfft(device: torch.device):
    """
    Demonstrates 1D/2D Fast Fourier Transforms (Time Domain <-> Frequency Domain)
    accelerated by rocFFT routines.
    """
    print("\n--- 3. rocFFT (Fast Fourier Transforms) ---")
    
    # Create 1D batch of real-valued time-domain signals
    batch, signal_len = 64, 1024
    time_signal = torch.randn(batch, signal_len, device=device, dtype=torch.float32)
    
    # Compute 1D Real-to-Complex FFT via rocFFT backend
    freq_spectrum = torch.fft.rfft(time_signal, dim=-1)
    
    # Reconstruct signal via Inverse Real FFT
    reconstructed_signal = torch.fft.irfft(freq_spectrum, n=signal_len, dim=-1)
    
    max_error = torch.max(torch.abs(time_signal - reconstructed_signal)).item()
    print(f"[✓] Transformed Signal shape: {freq_spectrum.shape} (Complex)")
    print(f"[✓] Signal Reconstruction Max Abs Error: {max_error:.6e}")


# ==============================================================================
# 4. rocSOLVER — Linear Algebra Decompositions
# ==============================================================================
def demo_rocsolver_decompositions(device: torch.device):
    """
    Demonstrates matrix factorizations (Cholesky, QR, SVD, LU, Eigenvalue)
    offloaded to rocSOLVER backend routines.
    """
    print("\n--- 4. rocSOLVER (Matrix Decompositions) ---")
    
    matrix_size = 512
    A = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float64)
    
    # 1. Symmetric Positive-Definite Matrix for Cholesky Decomposition (A = L * L^T)
    SPD = A @ A.mT + torch.eye(matrix_size, device=device, dtype=torch.float64)
    L = torch.linalg.cholesky(SPD)
    cholesky_diff = torch.dist(L @ L.mT, SPD).item()
    print(f"[✓] Cholesky Decomposition Residual ||L*L^T - SPD||: {cholesky_diff:.6e}")
    
    # 2. Singular Value Decomposition (A = U * S * V^T)
    U, S, Vh = torch.linalg.svd(A)
    print(f"[✓] SVD Complete: Top 5 Singular Values = {S[:5].detach().cpu().numpy()}")
    
    # 3. QR Decomposition (A = Q * R)
    Q, R = torch.linalg.qr(A)
    qr_diff = torch.dist(Q @ R, A).item()
    print(f"[✓] QR Decomposition Residual ||Q*R - A||: {qr_diff:.6e}")


# ==============================================================================
# Main Execution Pipeline
# ==============================================================================
if __name__ == "__main__":
    print("Initializing ROCm High-Performance Compute Library Test Suite...")
    gpu_device = check_rocm_environment()
    
    demo_rocblas_gemm(gpu_device)
    demo_miopen_primitives(gpu_device)
    demo_rocfft(gpu_device)
    demo_rocsolver_decompositions(gpu_device)
    
    print("\n[+] All ROCm compute library pipelines executed successfully.")
