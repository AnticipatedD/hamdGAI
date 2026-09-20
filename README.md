# hamdGAI

<div align="center">
  
![AMD](https://img.shields.io/badge/AMD-Skills-ED1C24?logo=amd&logoColor=white)
![ROCm](https://img.shields.io/badge/ROCm-Enabled-green)
![Ryzen AI](https://img.shields.io/badge/Ryzen_AI-Ready-1F6FEB)
![Agent Skills](https://img.shields.io/badge/Agent_Skills-Standard-7B2D8E)
[![Cursor](https://img.shields.io/badge/Cursor-Compatible-000000?logo=cursor&logoColor=white)](https://cursor.com)
[![Claude Code](https://img.shields.io/badge/Claude_Code-Compatible-F07535?logo=claude&logoColor=white)](https://www.anthropic.com/claude-code)

</div>

## How AI Agents Work – Model, Tools, Memory & Control Loop

![How hamdGAI agents work](how-hamdGAI-agents-work-1920x1080-1.jpg.webp)

An AI agent is an engineered loop, not just a smart model. `hamdGAI` implements a modular agent framework paired with hardware-accelerated compute capabilities.

### AI Reference Architecture:

1. **Model** – Interprets objectives and proposes actions.
2. **Instructions** – Role boundaries, operational policies, and stopping criteria.
3. **Tools** – Validated, least-privilege capabilities executed safely by the runtime.
4. **State & Memory** – Active execution state with grounded retrieval memory.
5. **Control Loop** – Observe → Decide → Act → Update → (Continue | Escalate | Stop).

---

## Quick Start

### Installation

Ensure Python 3.11+ is installed, then clone and setup the environment using pinned dependencies:

```bash
git clone [https://github.com/AnticipatedD/hamdGAI.git](https://github.com/AnticipatedD/hamdGAI.git)
cd hamdGAI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
cp .env.example .env
```

# Execution & Testing 
```bash
# Run unit test suite (with CPU fallback validation)
pytest

# Smoke test the core agent runtime loop
python hamdGAI_init.py
```
### Core System Features & Architecture 
# Key Design Principles
- **Explicit Tool Contracts**: Typed input validation, structured exception handling, and full execution provenance.
- **​Controlled Execution Budgets**: Strict step budgets, error thresholds, and escalation pathways.
- ​**Grounded Decision Making**: Clear separation of verified facts from model-generated summaries via RAG pipelines.
- **​Hardware Acceleration**: AMD ROCm GPU PyTorch kernels with seamless CPU fallback capabilities.

  """Environment Configuration Schema for hamdGAI Runtime."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings mapped to environment variables."""

    # -------------------------------------------------------------------------
    # AGENT RUNTIME CONFIGURATION
    # -------------------------------------------------------------------------
    agent_max_steps: int = Field(
        default=10,
        alias="AGENT_MAX_STEPS",
        description="Maximum allowed iteration steps per agent execution run",
    )

    # -------------------------------------------------------------------------
    # ROCM COMPUTE ACCELERATION CONFIGURATION
    # -------------------------------------------------------------------------
    rocm_api_key: str = Field(
        default="rocm_sec_key_123",
        alias="ROCM_API_KEY",
        description="Authentication key for binding AMD ROCm acceleration services",
    )

    # -------------------------------------------------------------------------
    # TELEMETRY & MONITORING CONFIGURATION
    # -------------------------------------------------------------------------
    prometheus_port: int = Field(
        default=9090,
        alias="PROMETHEUS_PORT",
        description="Exposed port for metric tracking and operational monitoring",
    )

    # -------------------------------------------------------------------------
    # VECTOR SEARCH & AZURE AI FOUNDRY INTEGRATION
    # -------------------------------------------------------------------------
    ai_search_conn_id: str = Field(
        default="search_conn_01",
        alias="AI_SEARCH_CONN_ID",
        description="Connection identifier for vector retrieval services",
    )
    azure_subscription_id: str = Field(
        default="00000000-0000-0000-0000-000000000000",
        alias="AZURE_SUBSCRIPTION_ID",
        description="Target cloud deployment subscription identifier",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Instantiate global settings object for import across modules
settings = Settings()

# Repository Structure 
hamdGAI/
├── .github/
│   └── workflows/
│       └── ci.yml             # Integrated lint, terraform, and pytest pipeline
├── terraform/
│   └── main.tf                # Agent cluster IaC modules
├── tests/
│   ├── test_rag_pipeline.py   # RAG retrieval and vector matching tests
│   ├── test_rocm_engine_cpu.py# ROCm compute engine CPU fallback unit tests
│   └── test_toolbox.py        # Mocked Azure AI Project client integration tests
├── config.py                  # Pydantic Settings schema
├── hamdGAI_init.py            # Main entry point & control loop
├── rag_pipeline.py            # Grounded in-memory vector search pipeline
├── rocm_engine.py             # ROCm tensor compute suite (GEMM, FFT, Cholesky)
├── .env.example               # Environment variable templates
├── pyproject.toml             # Project build configuration
└── requirements-lock.txt      # Pinned dependency lockfile

# License & Citation 
​Copyright © 2026 MD ABUL HOSSAIN. All Rights Reserved.
Distributed under the MIT License. See [LICENSE](license.md) for details.
