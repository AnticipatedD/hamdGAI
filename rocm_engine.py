"""ROCm Accelerated Compute Engine with CPU fallback capabilities."""

import torch
import structlog

logger = structlog.get_logger()


class ROCmComputeSuite:
    def __init__(self, force_cpu: bool = False):
        if not force_cpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("ROCm compute engine initialized on GPU accelerator")
        else:
            self.device = torch.device("cpu")
            logger.info("ROCm compute engine initialized on CPU fallback target")

    def execute_gemm(self, matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
        """Executes General Matrix Multiplication (GEMM)."""
        tensor_a = torch.tensor(matrix_a, dtype=torch.float32, device=self.device)
        tensor_b = torch.tensor(matrix_b, dtype=torch.float32, device=self.device)
        result = torch.matmul(tensor_a, tensor_b)
        return result.cpu().tolist()

    def execute_fft(self, signal: list[float]) -> list[complex]:
        """Executes 1D Fast Fourier Transform."""
        tensor_signal = torch.tensor(signal, dtype=torch.float32, device=self.device)
        result = torch.fft.fft(tensor_signal)
        return result.cpu().tolist()

    def execute_cholesky(self, symmetric_matrix: list[list[float]]) -> list[list[float]]:
        """Executes Cholesky Decomposition."""
        tensor_matrix = torch.tensor(symmetric_matrix, dtype=torch.float32, device=self.device)
        result = torch.linalg.cholesky(tensor_matrix)
        return result.cpu().tolist()
