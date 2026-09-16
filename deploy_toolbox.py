# deploy_toolbox.py
import subprocess
import json
import os
import sys

def run_command(command, description):
    """Helper function to execute shell commands cleanly."""
    print(f"🔄 Executing: {description}...")
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"✅ Success: {description}")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during '{description}':\n{e.stderr}", file=sys.stderr)
        sys.exit(1)

def deploy_azure_toolbox():
    # 1. Verify Azure Developer CLI (azd) is installed on the machine
    run_command(["azd", "version"], "Checking azd installation")

    # 2. Check for active Azure authentication context
    # If running in a pipeline, make sure AZURE_CREDENTIALS or AZD login env variables are set
    if not os.environ.get("AZURE_SUBSCRIPTION_ID"):
        print("⚠️ Warning: AZURE_SUBSCRIPTION_ID env variable not detected.")
        print("Ensure you are logged in via 'azd auth login' before running this script.")

    # 3. Define the tool items dynamically inside the Python script
    toolbox_manifest = {
        "description": "Production toolbox bundle matching Azure Foundry support matrix.",
        "tools": [
            {"type": "mcp", "server_label": "production_mcp_server"},
            {"type": "web_search"},
            {"type": "azure_ai_search", "connection_id": os.environ.get("AI_SEARCH_CONN_ID", "default-search-id")},
            {"type": "code_interpreter"},
            {"type": "file_search"},
            {"type": "openapi", "spec_url": "https://example.com"},
            {"type": "agent_to_agent", "target_agent_id": "target_agent_guid_here"},
            {"type": "browser_automation"},
            {"type": "fabric_iq"},
            {"type": "work_iq"}
        ]
    }

    # 4. Write config to a temporary file for azd ingestion
    temp_config_path = "temp_toolbox_config.json"
    with open(temp_config_path, "w") as f:
        json.dump(toolbox_manifest, f, indent=2)
    
    # 5. Execute the imperative deployment deployment command from Image 2
    deploy_cmd = ["azd", "ai", "toolbox", "create", "--from-file", temp_config_path]
    run_command(deploy_cmd, "Deploying toolbox manifest to Azure AI Foundry")

    # 6. Clean up temporary files
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
        print("🧹 Cleaned up temporary deployment configuration files.")

if __name__ == "__main__":
    deploy_azure_toolbox()
