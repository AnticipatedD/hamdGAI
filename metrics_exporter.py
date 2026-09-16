import time
from typing import Dict, Any
from prometheus_client import start_http_server, Counter, Histogram

# Initialize production-grade Prometheus gauges and counters
AGENT_TASKS_TOTAL = Counter(
    "hamdgai_agent_tasks_total",
    "Total agent tasks processed by the control loop",
    ["status"]
)

AGENT_STEP_LATENCY = Histogram(
    "hamdgai_agent_step_latency_seconds",
    "Latency matrix profiling per execution step loop cycle",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
)

TOOL_EXECUTIONS_TOTAL = Counter(
    "hamdgai_tool_executions_total",
    "Total tracking metrics for tool boundary calls",
    ["tool_name", "status"]
)

TOKEN_CONSUMPTION_TOTAL = Counter(
    "hamdgai_tokens_total",
    "Calculated token allocations metric tracking throughput across local hardware",
    ["token_type"] # prompt, completion
)

class TelemetryCollector:
    @staticmethod
    def record_task(status: str) -> None:
        AGENT_TASKS_TOTAL.labels(status=status).inc()

    @staticmethod
    def get_step_timer() -> Any:
        return AGENT_STEP_LATENCY.time()

    @staticmethod
    def record_tool(name: str, status: str) -> None:
        TOOL_EXECUTIONS_TOTAL.labels(tool_name=name, status=status).inc()

    @staticmethod
    def record_tokens(prompt: int, completion: int) -> None:
        TOKEN_CONSUMPTION_TOTAL.labels(token_type="prompt").inc(prompt)
        TOKEN_CONSUMPTION_TOTAL.labels(token_type="completion").inc(completion)

def start_telemetry_endpoint(port: int = 9090) -> None:
    """Fires up the dedicated scrapable Prometheus endpoint server."""
    start_http_server(port)
