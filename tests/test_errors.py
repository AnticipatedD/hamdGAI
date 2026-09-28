import pytest
from errors import ToolExecutionError, ModelInferenceError, BudgetExceededError


def test_tool_execution_error_inheritance_and_message():
    err = ToolExecutionError("tool failed")
    assert isinstance(err, Exception)
    assert str(err) == "tool failed"


def test_model_inference_error_inheritance_and_message():
    err = ModelInferenceError("inference failed")
    assert isinstance(err, Exception)
    assert str(err) == "inference failed"


def test_budget_exceeded_error_inheritance_and_message():
    err = BudgetExceededError("budget exceeded")
    assert isinstance(err, Exception)
    assert str(err) == "budget exceeded"
