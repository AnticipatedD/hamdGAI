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

# hamdGAI: Hybrid Agent Model & ROCm Compute Architecture

Production-grade agent orchestration framework with integrated retrieval-augmented generation (RAG) and AMD ROCm PyTorch compute acceleration.

## Core Capabilities

- **ROCm Compute Acceleration:** Native tensor operations (GEMM, FFT, Cholesky) optimized for AMD GPU hardware with automatic CPU execution fallbacks.
- **Grounded RAG Pipeline:** Term-frequency cosine similarity passage retrieval engine returning structured answers with confidence metrics.
- **Enterprise Configuration:** Typed Pydantic Settings management and structured JSON log streaming via `structlog`.
- **Infrastructure as Code:** Pinned Terraform modules for agent cluster provisioning.

## Environment Variables

| Variable Name | Description | Default / Example |
| :--- | :--- | :--- |
| `AGENT_MAX_STEPS` | Maximum recursive execution loop steps for agents | `10` |
| `ROCM_API_KEY` | Authentication key for ROCm service bindings | `rocm_sec_key_123` |
| `PROMETHEUS_PORT` | Port exposed for operational metrics collection | `9090` |
| `AI_SEARCH_CONN_ID` | Azure AI Search Connection Identifier | `search_conn_01` |
| `AZURE_SUBSCRIPTION_ID` | Azure Subscription GUID for deployment scripts | `00000000-0000-0000-0000-000000000000` |

## Quickstart Guide

### 1. Installation

Ensure Python 3.11+ is installed, then clone and install pinned dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```
# Execution & Testing 
```bash
# Run unit test suite (with CPU fallback validation)
pytest

# Smoke test the core agent runtime loop
python hamdGAI_init.py
```
Running Unit Tests
​Run the complete test suite (includes CPU fallback tests for GPU acceleration routines):
```bash
pytest tests/ -v --cov=.
```
### Core System Features & Architecture 

# Key Design Principles

- **Explicit Tool Contracts**: Typed input validation, structured exception handling, and full execution provenance.
- **​Controlled Execution Budgets**: Strict step budgets, error thresholds, and escalation pathways.
- ​**Grounded Decision Making**: Clear separation of verified facts from model-generated summaries via RAG pipelines.
- **​Hardware Acceleration**: AMD ROCm GPU PyTorch kernels with seamless CPU fallback capabilities.

---

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
