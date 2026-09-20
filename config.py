"""Environment Configuration Schema for hamdGAI Runtime."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings mapped to environment variables."""

    # -------------------------------------------------------------------------
    # AGENT RUNTIME CONFIGURATION
    # -------------------------------------------------------------------------
    agent_max_steps: int = Field(
        default=10,
        alias="AGENT_MAX_STEPS",
        description="Maximum allowed iteration steps per agent execution run",
    )

    # -------------------------------------------------------------------------
    # ROCM COMPUTE ACCELERATION CONFIGURATION
    # -------------------------------------------------------------------------
    rocm_api_key: str = Field(
        default="rocm_sec_key_123",
        alias="ROCM_API_KEY",
        description="Authentication key for binding AMD ROCm acceleration services",
    )

    # -------------------------------------------------------------------------
    # TELEMETRY & MONITORING CONFIGURATION
    # -------------------------------------------------------------------------
    prometheus_port: int = Field(
        default=9090,
        alias="PROMETHEUS_PORT",
        description="Exposed port for metric tracking and operational monitoring",
    )

    # -------------------------------------------------------------------------
    # VECTOR SEARCH & AZURE AI FOUNDRY INTEGRATION
    # -------------------------------------------------------------------------
    ai_search_conn_id: str = Field(
        default="search_conn_01",
        alias="AI_SEARCH_CONN_ID",
        description="Connection identifier for vector retrieval services",
    )
    azure_subscription_id: str = Field(
        default="00000000-0000-0000-0000-000000000000",
        alias="AZURE_SUBSCRIPTION_ID",
        description="Target cloud deployment subscription identifier",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

# Instantiate global settings object for import across modules
settings = Settings()
