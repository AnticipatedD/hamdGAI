import pytest
from rocm_engine import ROCmComputeSuite

def test_rocm_compute_suite_gemm():
    suite = ROCmComputeSuite()
    matrix_a = [[1.0, 2.0], [3.0, 4.0]]
    matrix_b = [[5.0, 6.0], [7.0, 8.0]]
    result = suite.execute_gemm(matrix_a, matrix_b)
    assert result is not None
    assert len(result) == 2
    assert len(result[0]) == 2

def test_rocm_compute_suite_fft():
    suite = ROCmComputeSuite()
    signal = [1.0, 2.0, 3.0, 4.0]
    result = suite.execute_fft(signal)
    assert result is not None
    assert isinstance(result, (list, tuple))
    assert len(result) > 0
