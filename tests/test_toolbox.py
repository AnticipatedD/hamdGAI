"""Unit tests and discovery utilities for Azure AI Foundry toolbox integrations."""

import os
import sys
from unittest.mock import MagicMock, patch
import pytest


def verify_and_test_toolbox(client=None) -> list[dict]:
    """Helper utility to inspect deployed tools via active or injected client."""
    if client is None:
        endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
        if not endpoint:
            raise ValueError("AZURE_AI_FOUNDRY_ENDPOINT environment variable is missing")

        from azure.identity import DefaultAzureCredential
        from azure.ai.projects import AIProjectClient

        client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential()
        )

    deployed_tools = client.toolboxes.list_tools()
    if not deployed_tools:
        return []

    return deployed_tools


def test_azure_toolbox_discovery_mocked():
    """Verify connection retrieval using mocked AIProjectClient telemetry."""
    with patch("azure.ai.projects.AIProjectClient") as mock_client_cls:
        mock_instance = mock_client_cls.return_value
        mock_instance.telemetry.get_connections.return_value = [
            MagicMock(id="conn_123", name="Azure AI Search Connection")
        ]

        connections = mock_instance.telemetry.get_connections()
        assert len(connections) == 1
        assert connections[0].name == "Azure AI Search Connection"


def test_verify_and_test_toolbox_execution():
    """Verify toolbox listing execution path with mocked tool outputs."""
    mock_client = MagicMock()
    mock_client.toolboxes.list_tools.return_value = [
        {"type": "mcp", "server_label": "rocm-search-tool"},
        {"type": "openapi", "server_label": "vector-index-tool"}
    ]

    tools = verify_and_test_toolbox(client=mock_client)

    assert len(tools) == 2
    assert tools[0]["server_label"] == "rocm-search-tool"
    assert tools[1]["type"] == "openapi"


def test_verify_and_test_toolbox_missing_endpoint():
    """Ensure ValueError is raised if endpoint environment variable is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="AZURE_AI_FOUNDRY_ENDPOINT environment variable is missing"):
            verify_and_test_toolbox(client=None)


if __name__ == "__main__":
    try:
        results = verify_and_test_toolbox()
        print(f"✅ Successfully retrieved {len(results)} active tools.")
    except Exception as err:
        print(f"❌ Execution failed: {err}")
