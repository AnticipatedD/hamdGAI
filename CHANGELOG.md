# Changelog

## [1.1.0] - 2026-09-21
### Fixed
- Resolved method signature mismatch in `RAGPipeline.grounded_answer`.
- Cleared hardcoded secret defaults in `config.py`.
- Corrected test suite imports in `tests/test_rocm_engine.py`.

### Added
- Committed `requirements-lock.txt` for reproducible installs.
- Modularized Terraform infrastructure into `./modules/agent_cluster`.
