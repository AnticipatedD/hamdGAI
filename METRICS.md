# Metrics & Evaluation

| Metric                    | Target          | Current (placeholder) |
|---------------------------|-----------------|-----------------------|
| Task success rate         | ≥ 85 %          | TBD                   |
| Avg steps per successful run | ≤ 8          | TBD                   |
| Tool error rate           | < 5 %           | TBD                   |
| Escalation rate           | < 15 %          | TBD                   |
| Cost per successful task  | < $0.15         | TBD                   |

Run the evaluation suite with:

```bash
pytest tests/ --cov=hamdGAI --cov-report=term-missing
python -m evals.runner   # once the eval harness is added
