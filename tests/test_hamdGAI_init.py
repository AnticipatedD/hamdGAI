import pytest
from hamdGAI_init import HamdGAIAgent, Tool, AgentState
from config import Settings
from rag_pipeline import RAGPipeline


def test_agent_initialises():
    settings = Settings(max_steps=5)
    tools = [
        Tool("rag_search", "test", lambda q: {"ok": True}, {})
    ]
    agent = HamdGAIAgent(settings, tools)
    assert agent is not None
    assert "rag_search" in agent.tools


def test_control_loop_runs():
    settings = Settings(max_steps=3)
    tools = [
        Tool("rag_search", "test", lambda query: {"passages": ["ok"]}, {})
    ]
    agent = HamdGAIAgent(settings, tools)
    state = agent.run("test objective")
    assert isinstance(state, AgentState)
    assert state.steps >= 1
    assert state.budget_remaining < settings.max_steps


def test_tool_error_handling():
    def failing_tool(**kwargs):
        raise ValueError("simulated failure")

    tools = [Tool("fail", "test", failing_tool, {})]
    agent = HamdGAIAgent(Settings(max_steps=2), tools)
    state = agent.run("force error")
    assert any(o.get("status") == "error" for o in state.observations)
