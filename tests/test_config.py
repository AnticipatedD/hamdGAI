import pytest
from pydantic import ValidationError
from config import Settings

def test_settings_default_allocations():
    """Asserts that system runtime config structures mount proper baseline variables maps."""
    settings = Settings(rocm_api_key="mock-key-token")
    assert settings.max_steps == 20
    assert settings.top_k == 5
    assert "write" in settings.require_approval_for
    assert "delete" in settings.require_approval_for

def test_settings_validation_errors():
    """Validates boundary constraints handling logic on illegal variables inputs."""
    with pytest.raises(ValidationError):
        Settings(max_steps=-5)
        
    with pytest.raises(ValidationError):
        Settings(top_k=0)
