import pytest
from rocm_engine import ROCmComputeSuite


def test_rocm_compute_suite_cpu_fallback_gemm():
    engine = ROCmComputeSuite(force_cpu=True)
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[2.0, 0.0], [1.0, 2.0]]
    res = engine.execute_gemm(a, b)
    assert res == [[4.0, 4.0], [10.0, 8.0]]


def test_rocm_compute_suite_cpu_fallback_fft():
    engine = ROCmComputeSuite(force_cpu=True)
    signal = [1.0, 0.0, 1.0, 0.0]
    res = engine.execute_fft(signal)
    assert len(res) == 4


def test_rocm_compute_suite_cpu_fallback_cholesky():
    engine = ROCmComputeSuite(force_cpu=True)
    matrix = [[4.0, 12.0], [12.0, 37.0]]
    res = engine.execute_cholesky(matrix)
    assert res[0][0] == 2.0
