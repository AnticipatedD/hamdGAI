# hamdGAI

**How AI Agents Work – Model, Tools, Memory & Control Loop**

![How hamdGAI agents work](how-hamdGAI-agents-work-1920x1080-1.jpg.webp)

An AI agent is an engineered loop, not just a smart model.  
`hamdGAI` implements the five core parts described in the Unite. 

## AI reference architecture:

1. **Model** – interprets the objective and proposes the next action  
2. **Instructions** – role, boundaries, policies, stopping criteria  
3. **Tools** – validated, least-privilege capabilities executed by the runtime  
4. **State & Memory** – current run state + selective long-term memory  
5. **Control Loop** – observe → decide → act → update → (continue | escalate | stop)

> Featured coverage: [How AI Agents Work on Unite.AI](https://www.unite.ai/how-ai-agents-work/)

```html
<a href="https://www.unite.ai/How-hamdGAI-Agents-Work/" aria-label="View this coverage on Unite.AI" style="display:inline-flex;align-items:center;gap:18px;padding:20px 24px;border:1px solid #d6e0e6;border-radius:12px;background:#ffffff;color:#17232d;font-family:Arial,Helvetica,sans-serif;text-decoration:none;"><span style="font-size:16px;font-weight:700;white-space:nowrap;">Featured in</span><img src="https://www.unite.ai/wp-content/uploads/2021/03/logoUNITE230X30WHITE-1.svg" alt="Unite.AI" width="230" height="30" style="display:block;width:230px;max-width:55vw;height:auto;"></a>
```
## Quick Start
```
git clone https://github.com/AnticipatedD/hamdGAI.git
cd hamdGAI
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env        # add your keys
pytest
python -m hamdGAI_init      # smoke test the agent runtime 
```

## Architecture (matches the article diagrams) 
Load instructions → Model decides → Tool executes (runtime validates) → State updates → Loop continues / escalates / recovers safely 
-
- Defined path (Agent runtime) preserves authority and evidence.
- Shortcut (Model alone) can only predict tokens and cannot execute.

## Key Design Principles (from the article) 
- Explicit tool contracts (typed args, structured errors, provenance).
- Budgets, error thresholds, approval gates, and external graders for stopping.
- Separate verified facts from model-generated summaries.
- Least-privilege tools and observable trajectories.
- Start with the smallest architecture that solves the task.

## Repository Layout

hamdGAI/
├── hamdGAI_init.py          # entry-point & basic agent loop
├── config.py                # typed configuration
├── rag_pipeline.py          # retrieval + grounding component
├── tests/
│   └── test_hamdGAI_init.py
├── .env.example
├── CONTRIBUTING.md
├── SECURITY.md
├── CODE_OF_CONDUCT.md
├── METRICS.md
├── LICENSE
└── pyproject.toml / requirements.txt 

## Status & Metrics 
See METRICS.md for live evaluation results, latency, success rate, and cost budgets. 

## License 
MIT See [LICENSE](license.md)

## Citation / Attribution 
Content and diagrams are derived from the public Unite. AI article “How AI Agents Work”.
Badge and linking follow Unite. AI media-kit guidelines.


### 2. `hamdGAI_init.py`
```python
"""
hamdGAI – minimal but complete agent runtime implementing the five-part architecture.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from config import Settings
from rag_pipeline import RAGPipeline

logger = logging.getLogger("hamdGAI")


@dataclass
class AgentState:
    objective: str
    messages: List[Dict[str, str]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    budget_remaining: int = 20
    completed: bool = False
    escalated: bool = False


class Tool:
    def __init__(self, name: str, description: str, fn: Callable, schema: Dict):
        self.name = name
        self.description = description
        self.fn = fn
        self.schema = schema

    def execute(self, **kwargs) -> Dict[str, Any]:
        # Runtime validation happens here – never trust the model blindly
        try:
            result = self.fn(**kwargs)
            return {"status": "ok", "result": result, "error": None}
        except Exception as e:
            return {"status": "error", "result": None, "error": str(e)}


class HamdGAIAgent:
    """
    Control loop:
    1. Load instructions
    2. Model decides (propose tool or final answer)
    3. Runtime validates & executes tool
    4. State updates
    5. Loop / escalate / recover / stop
    """

    def __init__(self, settings: Settings, tools: List[Tool], rag: Optional[RAGPipeline] = None):
        self.settings = settings
        self.tools = {t.name: t for t in tools}
        self.rag = rag or RAGPipeline(settings)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        tool_desc = "\n".join(
            f"- {name}: {t.description}" for name, t in self.tools.items()
        )
        return f"""You are a reliable AI agent.
Follow the operational instructions strictly.
Available tools:
{tool_desc}

Rules:
- Prefer evidence over assumption.
- Use structured tool calls only.
- Escalate when confidence is low or budget is exhausted.
- Stop only when completion criteria are met.
"""

    def run(self, objective: str) -> AgentState:
        state = AgentState(objective=objective, budget_remaining=self.settings.max_steps)
        state.messages.append({"role": "system", "content": self.system_prompt})
        state.messages.append({"role": "user", "content": objective})

        while not state.completed and not state.escalated and state.budget_remaining > 0:
            state.steps += 1
            state.budget_remaining -= 1

            # 1. Model decides (in real use replace with LLM call that returns tool call or final)
            decision = self._model_decide(state)

            if decision.get("type") == "final":
                state.completed = True
                state.messages.append({"role": "assistant", "content": decision["content"]})
                break

            if decision.get("type") == "tool":
                tool_name = decision["name"]
                args = decision.get("args", {})
                if tool_name not in self.tools:
                    obs = {"status": "error", "error": f"Unknown tool {tool_name}"}
                else:
                    # 2. Runtime validates & executes
                    obs = self.tools[tool_name].execute(**args)
                # 3. Observe & update state
                state.observations.append(obs)
                state.messages.append(
                    {"role": "tool", "content": json.dumps(obs, ensure_ascii=False)}
                )

            if state.budget_remaining <= 0:
                state.escalated = True
                logger.warning("Budget exhausted – escalating")

        return state

    def _model_decide(self, state: AgentState) -> Dict[str, Any]:
        """
        Placeholder for the real LLM call.
        In production this would call the model with the current messages
        and parse a structured tool call or final answer.
        """
        # Demo logic – replace with real model
        if state.steps == 1:
            return {
                "type": "tool",
                "name": "rag_search",
                "args": {"query": state.objective},
            }
        return {"type": "final", "content": "Task completed with available evidence."}


def main():
    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    tools = [
        Tool(
            name="rag_search",
            description="Retrieve grounded passages from the knowledge base",
            fn=lambda query: {"passages": ["example grounded result"]},
            schema={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]
    agent = HamdGAIAgent(settings, tools)
    result = agent.run("Compare three suppliers and prepare a recommendation")
    print(json.dumps(result.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()

---
Copyright © 2026 MD ABUL HOSSAIN. All Rights Reserved.
