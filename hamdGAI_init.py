import json
from typing import Dict, Any, List, Optional
from openai import OpenAI
import structlog
from config import Settings
from errors import ToolExecutionError, ModelInferenceError
from rag_pipeline import RAGPipeline

# Production-grade structured logging telemetry architecture initialization
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

class Tool:
    def __init__(self, name: str, description: str, handler: Any):
        self.name = name
        self.description = description
        self.handler = handler

    def execute(self, arguments: dict) -> Any:
        """Executes operational parameters through tool definitions with strict error handling bounds."""
        try:
            logger.info("Initiating tool boundary execution", tool_name=self.name, args=arguments)
            return self.handler(**arguments)
        except TypeError as te:
            logger.error("Parameter mismatch detected at tool boundary", tool_name=self.name, error=str(te))
            raise ToolExecutionError(f"Tool input payload violates parameter type maps: {str(te)}") from te
        except Exception as e:
            logger.error("Unexpected tool execution failure caught", tool_name=self.name, error=str(e))
            raise ToolExecutionError(f"Execution failed inside tool target system: {str(e)}") from e

class AgentControlLoop:
    def __init__(self, config: Settings):
        self.config = config
        self.client = OpenAI(api_key=self.config.rocm_api_key, base_url=self.config.rocm_engine_url)
        self.rag = RAGPipeline(top_k=self.config.top_k)
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool
        logger.info("Registered active operational agent tool", tool_name=tool.name)

    def _model_decide(self, history: List[dict]) -> dict:
        """Queries the hardware engine backend to extract structured next-step options matrices."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.rocm_model_name,
                messages=history,
                temperature=0.0
            )
            content = response.choices[0].message.content
            if not content:
                raise ModelInferenceError("Model inference engine returned an empty response string payload.")
            return json.loads(content)
        except json.JSONDecodeError as jde:
            logger.error("Failed to parse structural JSON step from model output", raw_output=content)
            raise ModelInferenceError(f"Malformed structural decisions output caught: {str(jde)}") from jde
        except Exception as e:
            logger.error("Model completion service validation failure caught", error=str(e))
            raise ModelInferenceError(f"Inference server interface execution error occurred: {str(e)}") from e

    def run(self, task: str) -> str:
        """Executes full multi-step execution loops tracking objective parameters states."""
        logger.info("Initializing baseline task orchestration thread", target_task=task)
        
        # Inject RAG context baseline variables directly into the tracking loop
        passages = self.rag.retrieve(task)
        grounded_data = self.rag.grounded_answer(task, passages)
        
        history = [
            {"role": "system", "content": f"Execute objectives using tools. Context: {grounded_data['answer']}"},
            {"role": "user", "content": task}
        ]
        
        for step in range(self.config.max_steps):
            logger.info("Processing execution cycle status snapshot", current_step=step, max_bounds=self.config.max_steps)
            decision = self._model_decide(history)
            
            if decision.get("status") == "complete":
                logger.info("Orchestration objective solved successfully")
                return str(decision.get("output"))
                
            if "tool_name" in decision:
                t_name = decision["tool_name"]
                t_args = decision.get("arguments", {})
                
                if t_name not in self.tools:
                    raise ToolExecutionError(f"Requested tool target mapping '{t_name}' not registered.")
                
                result = self.tools[t_name].execute(t_args)
                history.append({"role": "assistant", "content": json.dumps(decision)})
                history.append({"role": "user", "content": f"Tool execution result: {json.dumps(result)}"})
            else:
                raise ModelInferenceError("Invalid state trajectory selected by model engine framework.")
                
        raise TimeoutError("Agent control loop timed out prior to resolving task conditions targets.")
