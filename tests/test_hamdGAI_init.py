import pytest
from unittest.mock import MagicMock, patch
from config import Settings
from errors import ToolExecutionError, ModelInferenceError
from hamdGAI_init import AgentControlLoop, Tool

@pytest.fixture
def base_config():
    return Settings(rocm_api_key="mock-test-key", max_steps=3, top_k=2)

def test_tool_boundary_parameter_mismatches():
    """Asserts tool handler input errors generate clean exception tracking logs."""
    def sample_handler(target_id: int) -> str:
        return f"ID_{target_id}"
        
    tool = Tool(name="verify_id", description="Sample tool target description", handler=sample_handler)
    
    with pytest.raises(ToolExecutionError, match="violates parameter type maps"):
        tool.execute({"wrong_param_key": "data"})

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

@patch('hamdGAI_init.OpenAI')
def test_agent_control_loop_malformed_json_handling(mock_openai_class, base_config):
    """Asserts parser framework captures bad execution trajectories reliably from completions."""
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{bad_json_string_payload}'))]
    mock_client.chat.completions.create.return_value = mock_response
    
    agent = AgentControlLoop(config=base_config)
    
    with pytest.raises(ModelInferenceError, match="Malformed structural decisions output"):
        agent.run("Test execution loop metrics path tracing.")
