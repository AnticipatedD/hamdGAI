"""Tests for rocm_primitives_demo (CPU / offline safe)."""

from __future__ import annotations

import pytest
import torch

from rocm_primitives_demo import check_rocm_environment


def test_check_rocm_environment_raises_when_no_device(monkeypatch):
    """Must raise RuntimeError when no CUDA/ROCm device is present."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="ROCm/CUDA device unavailable"):
        check_rocm_environment()


def test_check_rocm_environment_returns_device(monkeypatch):
    """When a device is reported available, return a cuda device object."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Mock GPU")
    # torch.version.cuda / hip may be None in pure CPU builds; that is fine
    device = check_rocm_environment()
    assert isinstance(device, torch.device)
    assert device.type == "cuda"
