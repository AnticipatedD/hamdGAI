from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Manages typed configurations for the hamdGAI agent execution environment."""
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    rocm_engine_url: str = Field(default="http://localhost:8000/v1")
    rocm_api_key: str = Field(default="not-needed")
    rocm_model_name: str = Field(default="Qwen3-Coder-30B-A3B-Instruct")
    
    max_steps: int = Field(default=20)
    top_k: int = Field(default=5)
    require_approval_for: List[str] = Field(default_factory=lambda: ["write", "delete", "purchase"])

    @field_validator("max_steps")
    @classmethod
    def validate_max_steps(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_steps must be a positive integer parameter bounds.")
        return value

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("top_k capacity must be greater than zero allocation limits.")
        return value
