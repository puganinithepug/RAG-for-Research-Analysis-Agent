from typing import TypedDict
from langchain_core.documents import Document


class RAGState(TypedDict, total=False):
    question: str
    documents: list[Document]
    answer: str
    attempts: int
    supported: bool
    faithfulness_score: float
    faithfulness_reason: str
    evaluation_error: str
