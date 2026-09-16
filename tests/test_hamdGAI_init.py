import pytest
from unittest.mock import MagicMock, patch
from config import Settings
from errors import ToolExecutionError, ModelInferenceError
from hamdGAI_init import AgentControlLoop, Tool

@pytest.fixture
def base_config():
    return Settings(rocm_api_key="mock-test-key", max_steps=3, top_k=2)

@pytest.fixture
def sample_schema():
    return {
        "type": "object",
        "properties": {
            "target_id": {"type": "integer"}
        },
        "required": ["target_id"]
    }

def test_tool_boundary_schema_validation_failure(sample_schema):
    """Asserts Tool.execute returns status 'error' when arguments violate the schema fields."""
    def dummy_handler(target_id: int) -> str:
        return f"ID_{target_id}"
        
    tool = Tool(name="verify_id", description="Sample validation tool", schema=sample_schema, handler=dummy_handler)
    
    # Violate type specifications (string passed instead of required integer)
    response = tool.execute({"target_id": "malicious_string_input"})
    assert response["status"] == "error"
    assert "Schema violation" in response["result"]

def test_tool_boundary_successful_execution(sample_schema):
    """Asserts safe argument packages execute smoothly returning proper wrappers."""
    def dummy_handler(target_id: int) -> str:
        return f"COMPLETED_{target_id}"
        
    tool = Tool(name="verify_id", description="Sample validation tool", schema=sample_schema, handler=dummy_handler)
    response = tool.execute({"target_id": 1024})
    
    assert response["status"] == "success"
    assert response["output"] == "COMPLETED_1024"

@patch('hamdGAI_init.OpenAI')
def test_agent_control_loop_successful_completion(mock_openai_class, base_config):
    """Verifies complete deterministic logic tracks map accurate final string output answers."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"status": "complete", "output": "Pipeline Success Outcome"}'))
    ]
    mock_client.chat.completions.create.return_value = mock_response
    
    agent = AgentControlLoop(config=base_config)
    output = agent.run("Verify infrastructure matrix indices details.")
    
    assert output == "Pipeline Success Outcome"
