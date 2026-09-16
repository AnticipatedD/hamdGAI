class AgentException(Exception):
    """Base architectural error class for all hamdGAI software execution tracking loops."""
    pass

class ToolExecutionError(AgentException):
    """Raised when an interactive tool execution boundary reports unexpected internal faults."""
    pass

class BudgetExceededError(AgentException):
    """Raised when the agent exceeds maximum step thresholds or pricing limitations."""
    pass

class RAGPipelineError(AgentException):
    """Raised when index access layers fail to map knowledge boundaries properly."""
    pass

class ModelInferenceError(AgentException):
    """Raised when model completion calls return unhealthy status flags or empty choices."""
    pass
