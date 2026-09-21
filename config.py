import os
from pydantic import Field
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    rocm_api_key: str = Field(..., description="ROCm API key requirement")
    azure_subscription_id: str = Field(..., description="Azure Subscription ID")
    max_steps: int = Field(default=10, description="Max agent execution steps")
    top_k: int = Field(default=5, description="Top K search results")

    class Config:
        env_file = ".env"
        extra = "ignore"
