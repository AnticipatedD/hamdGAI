import pytest
from rocm_engine import ROCmComputeSuite

def test_rocm_compute_suite_gemm():
    suite = ROCmComputeSuite()
    result = suite.execute_gemm()
    assert result is True or result is not None

def test_rocm_compute_suite_fft():
    suite = ROCmComputeSuite()
    assert suite.execute_fft() is not None
