"""Tests for deploy_toolbox (fully mocked, no network / azd required)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from deploy_toolbox import build_toolbox_manifest, deploy_azure_toolbox, run_command


def test_build_toolbox_manifest_contains_expected_tools():
    manifest = build_toolbox_manifest(ai_search_conn_id="test-conn")
    assert "tools" in manifest
    types = {t["type"] for t in manifest["tools"]}
    assert "web_search" in types
    assert "azure_ai_search" in types
    assert any(t.get("connection_id") == "test-conn" for t in manifest["tools"])


def test_run_command_success(monkeypatch):
    mock_result = MagicMock()
    mock_result.stdout = "azd 1.0.0"
    mock_runner = MagicMock(return_value=mock_result)

    output = run_command(["azd", "version"], "test", runner=mock_runner)
    assert output == "azd 1.0.0"
    mock_runner.assert_called_once()


def test_deploy_azure_toolbox_writes_and_removes_temp_file(tmp_path, monkeypatch):
    """End-to-end happy path with fully mocked subprocess."""
    config_file = tmp_path / "temp_toolbox_config.json"
    mock_runner = MagicMock()
    mock_runner.return_value = MagicMock(stdout="ok")

    # Should not raise and should clean up the temp file
    deploy_azure_toolbox(
        config_path=config_file,
        runner=mock_runner,
        remove_temp=True,
    )

    # File must have been removed
    assert not config_file.exists()

    # azd version + azd ai toolbox create must have been called
    assert mock_runner.call_count == 2
    first_cmd = mock_runner.call_args_list[0][0][0]
    second_cmd = mock_runner.call_args_list[1][0][0]
    assert first_cmd == ["azd", "version"]
    assert second_cmd[0:4] == ["azd", "ai", "toolbox", "create"]
