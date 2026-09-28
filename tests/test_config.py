import pytest
from pydantic import ValidationError
from config import Settings


def test_settings_default_allocations():
    """Ensure Settings loads defaults correctly when only required vars are provided."""
    settings = Settings(rocm_api_key="mock-key-token")
    assert settings.max_steps == 20
    assert settings.top_k == 5
    # require_approval_for should contain expected actions
    for action in ("write", "delete", "purchase"):
        assert action in settings.require_approval_for


def test_settings_validation_errors():
    """Ensure invalid values raise validation errors."""
    with pytest.raises(ValidationError):
        Settings(rocm_api_key="mock-key-token", max_steps=-5)

    with pytest.raises(ValidationError):
        Settings(rocm_api_key="mock-key-token", top_k=0)
