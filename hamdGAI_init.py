import json
from typing import Dict, Any, List

from openai import OpenAI
import structlog
import jsonschema

from config import Settings
from errors import ToolExecutionError, ModelInferenceError, BudgetExceededError
from rag_pipeline import RAGPipeline

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger()


class Tool:
    def __init__(self, name: str, description: str, schema: dict, handler: Any):
        self.name = name
        self.description = description
        self.schema = schema
        self.handler = handler

    def execute(self, arguments: dict) -> dict:
        """Validate arguments against the tool schema, then call the handler."""
        try:
            logger.info("Executing tool", tool_name=self.name)
            jsonschema.validate(instance=arguments, schema=self.schema)
            runtime_output = self.handler(**arguments)
            return {"status": "success", "output": runtime_output}

        except jsonschema.ValidationError as ve:
            logger.error("Schema validation failed", tool_name=self.name, error=str(ve))
            return {"status": "error", "result": f"Schema violation: {str(ve)}"}

        except TypeError as te:
            logger.exception("Parameter mismatch at tool boundary", tool_name=self.name)
            raise ToolExecutionError(
                f"Tool payload signature violation: {str(te)}"
            ) from te

        except Exception as e:
            logger.exception("Unexpected tool execution failure", tool_name=self.name)
            raise ToolExecutionError(
                f"Execution failed inside tool: {str(e)}"
            ) from e


class AgentControlLoop:
    def __init__(self, config: Settings):
        self.config = config
        self.client = OpenAI(
            api_key=self.config.rocm_api_key,
            base_url=self.config.rocm_engine_url,
        )
        self.rag = RAGPipeline(top_k=self.config.top_k)
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        logger.info("Registered tool", tool_name=tool.name)

    def _model_decide(self, history: List[dict]) -> dict:
        """Ask the model for the next structured decision."""
        content = None
        try:
            response = self.client.chat.completions.create(
                model=self.config.rocm_model_name,
                messages=history,
                temperature=0.0,
            )
            # Fixed: choices is a list
            content = response.choices[0].message.content
            if not content:
                raise ModelInferenceError(
                    "Model returned an empty response."
                )
            return json.loads(content)

        except json.JSONDecodeError as jde:
            logger.error(
                "Failed to parse JSON from model output",
                raw_output=content,
                error=str(jde),
            )
            raise ModelInferenceError(
                f"Malformed model output: {str(jde)}"
            ) from jde

        except Exception as e:
            logger.exception("Model inference failed")
            raise ModelInferenceError(
                f"Inference error: {str(e)}"
            ) from e

    def run(self, task: str) -> str:
        """Run the agent control loop until completion or budget exhaustion."""
        logger.info("Starting agent run", task=task)

        try:
            passages = self.rag.retrieve(task)
            grounded_data = self.rag.grounded_answer(task, passages)
            system_context = (
                f"Execute objectives using tools. Context: {grounded_data['answer']}"
            )
        except NotImplementedError as nie:
            logger.warning("RAG fallback mode", reason=str(nie))
            system_context = "Execute objectives using tools. RAG is unavailable."

        history = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": task},
        ]

        for step in range(self.config.max_steps):
            logger.info(
                "Agent step",
                current_step=step,
                max_steps=self.config.max_steps,
            )
            decision = self._model_decide(history)

            if decision.get("status") == "complete":
                logger.info("Task completed")
                return str(decision.get("output"))

            if "tool_name" in decision:
                t_name = decision["tool_name"]
                t_args = decision.get("arguments", {})

                if t_name not in self.tools:
                    raise ToolExecutionError(
                        f"Requested tool '{t_name}' is not registered."
                    )

                execution_result = self.tools[t_name].execute(t_args)
                history.append(
                    {"role": "assistant", "content": json.dumps(decision)}
                )
                history.append(
                    {
                        "role": "user",
                        "content": f"Tool execution result: {json.dumps(execution_result)}",
                    }
                )
            else:
                raise ModelInferenceError(
                    "Model returned an invalid decision (missing tool_name or complete status)."
                )

        raise BudgetExceededError(
            "Agent control loop exceeded max_steps without completing the task."
                            )
