import pytest
from errors import RAGPipelineError
from rag_pipeline import RAGPipeline

def test_rag_retrieve_dictionary_shapes():
    """Asserts that index data access vectors return predictable structural arrays properties."""
    pipeline = RAGPipeline(top_k=3)
    results = pipeline.retrieve("Test operational validation bounds")
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert "content" in results[0]
    assert "source" in results[0]
    assert "score" in results[0]
    assert results[0]["score"] == 0.945

def test_rag_retrieve_empty_validation_guards():
    """Validates that RAG logic blocks emit custom execution warnings on empty targets."""
    pipeline = RAGPipeline()
    with pytest.raises(RAGPipelineError, match="cannot be blank space"):
        pipeline.retrieve("   ")

def test_grounded_answer_confidence_scoring():
    """Asserts calculation outputs match specific logic states for active context spans."""
    pipeline = RAGPipeline()
    
    # Check baseline empty state tracking capabilities behaviour
    empty_run = pipeline.grounded_answer("Query", [])
    assert empty_run["confidence"] == "low"
    assert "Context base empty" in empty_run["answer"]
    
    # Check active data loading matrices values returns
    mock_passages = [{"content": "Data blocks details", "source": "manifest.json", "score": 0.9}]
    valid_run = pipeline.grounded_answer("Query", mock_passages)
    assert valid_run["confidence"] == "high"
    assert "manifest.json" in valid_run["references"]
