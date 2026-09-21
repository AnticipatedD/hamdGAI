import pytest
from config import Settings
from rag_pipeline import RAGPipeline

def test_settings_initialization():
    settings = Settings(rocm_api_key="test_key", azure_subscription_id="00000000-0000-0000-0000-000000000000", max_steps=5, top_k=3)
    assert settings.max_steps == 5
    assert settings.top_k == 3

def test_rag_grounded_answer():
    rag = RAGPipeline()
    answer = rag.grounded_answer("How to optimize ROCm?", ["passage1", "passage2"])
    assert "passage" in answer
