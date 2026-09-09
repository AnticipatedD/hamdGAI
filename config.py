from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model
    openai_api_key: Optional[str] = None
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1

    # Agent control
    max_steps: int = 20
    max_tokens: int = 4096
    cost_budget_usd: float = 1.0

    # RAG
    embedding_model: str = "text-embedding-3-small"
    vector_store_path: str = "./data/vectorstore"
    top_k: int = 5

    # Safety
    require_approval_for: list[str] = ["write", "delete", "purchase"]
    log_level: str = "INFO"
