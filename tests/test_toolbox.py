# test_toolbox.py
import os
import sys
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.core.exceptions import HttpResponseError

def verify_and_test_toolbox():
    print("🔍 Initializing AI Project connection...")
    
    # Extract endpoint from environment variables
    # Always double-check your physical endpoint labels match your cloud console values
    endpoint = os.environ.get("AZURE_AI_FOUNDRY_ENDPOINT")
    if not endpoint:
        print("❌ Error: AZURE_AI_FOUNDRY_ENDPOINT environment variable is missing!")
        sys.exit(1)

    try:
        client = AIProjectClient(
            endpoint=endpoint,
            credential=DefaultAzureCredential()
        )
        
        print("📡 Fetching deployed tools from your active toolbox...")
        # Simulates the tools/list request from Image 3
        deployed_tools = client.toolboxes.list_tools() 
        
        # Scenario Check: Empty list handling (Troubleshooting line items 1-4)
        if not deployed_tools or len(deployed_tools) == 0:
            print("\n⚠️ Alert: tools/list returned zero tools!")
            print("💡 Diagnostic Suggestions:")
            print("  - [MCP/A2A]: Verify your 'project_connection_id' or 'remote credentials' are valid.")
            print("  - [OpenAPI]: Validate that your OpenAPI specification JSON is structurally well-formed.")
            print("  - [Filters]: Check if an restrictive 'allowed_tools' filter is accidentally hiding your assets.")
            print("  - [Provisioning]: If recently deployed, wait 10 seconds and retry the query.")
            return

        print(f"\n✅ Successfully retrieved {len(deployed_tools)} active tools:")
        for idx, tool in enumerate(deployed_tools, 1):
            print(f"  {idx}. Type: {tool.get('type')} | Name/Label: {tool.get('server_label', 'N/A')}")

    except HttpResponseError as error:
        print(f"\n❌ API Call Failed with status code: {error.status_code}")
        print("💡 Troubleshoot Guide (from documentation):")
        
        if error.status_code == 401:
            print("  -> [401 Unauthorized]: Token has expired or scope is invalid. Run 'az login' again.")
        elif error.status_code == 400:
            print("  -> [400 Bad Request]: Look for duplicate tool definitions without proper unique 'server_label' fields.")
        elif error.status_code == 500:
            print("  -> [500 Server Error]: Ensure custom components implement the required MCP 'ping' and 'prompts/list' methods.")
        else:
            print(f"  -> Raw message detail: {error.message}")

    except Exception as general_err:
        print(f"\n❌ Unexpected runtime connection error: {str(general_err)}")

if __name__ == "__main__":
    verify_and_test_toolbox()
