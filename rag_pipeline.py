"""RAG Pipeline Module with built-in in-memory vector retrieval."""

from typing import List, Dict, Any
import math


class RAGPipeline:
    def __init__(self, documents: List[Dict[str, str]] = None):
        self.documents = documents or [
            {"id": "doc1", "text": "AMD ROCm software stack provides GPU acceleration for deep learning."},
            {"id": "doc2", "text": "rocSOLVER and rocSPARSE provide optimized linear algebra and sparse matrix operations."},
            {"id": "doc3", "text": "Pydantic settings and structlog ensure production-grade agent configuration."},
        ]

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().replace(".", "").replace(",", "").split()

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Retrieves top_k matching passages using term-frequency similarity."""
        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return []

        scored_docs = []
        for doc in self.documents:
            doc_tokens = self._tokenize(doc["text"])
            overlap = sum(1 for token in query_tokens if token in doc_tokens)
            score = overlap / math.sqrt(len(query_tokens) * (len(doc_tokens) + 1e-5))
            scored_docs.append({"doc": doc, "score": score})

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in scored_docs[:top_k]]

    def grounded_answer(self, query: str) -> Dict[str, Any]:
        passages = self.retrieve(query)
        context = " ".join([p["text"] for p in passages])
        return {
            "query": query,
            "passages": passages,
            "confidence": 0.92 if passages else 0.0,
            "context": context
        }
