from prometheus_client import REGISTRY
from metrics_exporter import TelemetryCollector


def test_record_task_increments_counter():
    collector = TelemetryCollector()
    collector.record_task("task_a")
    metric = REGISTRY.get_sample_value("hamdgai_tasks_total", {"task": "task_a"})
    assert metric == 1.0


def test_record_tool_increments_counter():
    collector = TelemetryCollector()
    collector.record_tool("tool_x")
    metric = REGISTRY.get_sample_value("hamdgai_tools_total", {"tool": "tool_x"})
    assert metric == 1.0


def test_record_tokens_increments_counter():
    collector = TelemetryCollector()
    collector.record_tokens("model_y", 50)
    metric = REGISTRY.get_sample_value("hamdgai_tokens_total", {"model": "model_y"})
    assert metric == 50.0
