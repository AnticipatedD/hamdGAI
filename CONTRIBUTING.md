# Contributing to hamdGAI (short version)

1. Fork & create a feature branch.
2. Keep changes small and focused.
3. Add / update tests for any behaviour change.
4. Run `pytest` and `ruff check .` (or the equivalent lint/format you prefer).
5. Open a PR with a clear description of the architectural impact (model / tools / memory / control loop).

We follow the same principles as the Unite.AI article: explicit contracts, least privilege, observable trajectories, and recoverable failures.
