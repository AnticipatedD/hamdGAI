import pytest
from rag_pipeline import RAGPipeline

def test_rag_pipeline_retrieval_success():
    pipeline = RAGPipeline()
    results = pipeline.retrieve("ROCm linear algebra")
    assert len(results) > 0
    assert "doc2" in [r["id"] for r in results]


def test_rag_pipeline_grounded_answer():
    pipeline = RAGPipeline()
    res = pipeline.grounded_answer("deep learning GPU")
    assert res["confidence"] > 0.0
    assert len(res["passages"]) > 0


def test_rag_retrieve_empty_validation_guards():
    """Validates that RAG logic blocks emit custom execution warnings on empty targets."""
    pipeline = RAGPipeline()
    with pytest.raises(RAGPipelineError, match="cannot be blank space"):
        pipeline.retrieve("   ")

def test_grounded_answer_confidence_scoring():
    """Asserts calculation outputs match specific logic states for active context spans."""
    pipeline = RAGPipeline()
    
    empty_run = pipeline.grounded_answer("Query", [])
    assert empty_run["confidence"] == "low"
    assert "Context base empty" in empty_run["answer"]
    
    mock_passages = [{"content": "Data blocks details", "source": "manifest.json", "score": 0.9}]
    valid_run = pipeline.grounded_answer("Query", mock_passages)
    assert valid_run["confidence"] == "high"
    assert "manifest.json" in valid_run["references"]
