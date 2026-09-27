"""Deploy a toolbox manifest to Azure AI Foundry via azd."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def run_command(
    command: List[str],
    description: str,
    runner: Callable = subprocess.run,
) -> str:
    """Execute a shell command and return stdout. Exit on failure."""
    print(f"🔄 Executing: {description}...")
    try:
        result = runner(command, check=True, text=True, capture_output=True)
        print(f"✅ Success: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during '{description}':\n{e.stderr}", file=sys.stderr)
        sys.exit(1)


def build_toolbox_manifest(
    ai_search_conn_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the toolbox manifest dictionary."""
    return {
        "description": "Production toolbox bundle matching Azure Foundry support matrix.",
        "tools": [
            {"type": "mcp", "server_label": "production_mcp_server"},
            {"type": "web_search"},
            {
                "type": "azure_ai_search",
                "connection_id": ai_search_conn_id
                or os.environ.get("AI_SEARCH_CONN_ID", "default-search-id"),
            },
            {"type": "code_interpreter"},
            {"type": "file_search"},
            {"type": "openapi", "spec_url": "https://example.com"},
            {"type": "agent_to_agent", "target_agent_id": "target_agent_guid_here"},
            {"type": "browser_automation"},
            {"type": "fabric_iq"},
            {"type": "work_iq"},
        ],
    }


def deploy_azure_toolbox(
    config_path: str | Path = "temp_toolbox_config.json",
    runner: Callable = subprocess.run,
    remove_temp: bool = True,
) -> None:
    """Write manifest and invoke azd. All external calls are injectable for tests."""
    config_path = Path(config_path)

    # 1. Verify azd is present
    run_command(["azd", "version"], "Checking azd installation", runner=runner)

    # 2. Soft check for subscription
    if not os.environ.get("AZURE_SUBSCRIPTION_ID"):
        print("⚠️  Warning: AZURE_SUBSCRIPTION_ID not set.")
        print("   Ensure you are logged in via 'azd auth login'.")

    # 3. Build + write manifest
    manifest = build_toolbox_manifest()
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 4. Deploy
    deploy_cmd = ["azd", "ai", "toolbox", "create", "--from-file", str(config_path)]
    run_command(deploy_cmd, "Deploying toolbox manifest to Azure AI Foundry", runner=runner)

    # 5. Cleanup
    if remove_temp and config_path.exists():
        config_path.unlink()
        print("🧹 Cleaned up temporary deployment configuration files.")


if __name__ == "__main__":
    deploy_azure_toolbox()
