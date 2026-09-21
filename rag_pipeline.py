from typing import List, Optional

class RAGPipeline:
    def __init__(self):
        pass

    def grounded_answer(self, query: str, passages: Optional[List[str]] = None) -> str:
        if passages:
            return f"Grounded Answer for '{query}' using {len(passages)} passages."
        return f"Grounded Answer for '{query}'."
