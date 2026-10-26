import os
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    rocm_api_key: str = Field(..., description="ROCm API key requirement")
    azure_subscription_id: str = Field(..., description="Azure Subscription ID")
    max_steps: int = Field(default=10, description="Max agent execution steps")
    top_k: int = Field(default=5, description="Top K search results")

    @field_validator("rocm_api_key", "azure_subscription_id")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Configuration value cannot be empty or blank")
        return v.strip()

    @field_validator("max_steps", "top_k")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Value must be a positive integer greater than 0")
        return v

    class Config:
        env_file = ".env"
        extra = "ignore"
