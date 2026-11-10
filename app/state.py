from typing import Annotated, Any, TypedDict, Literal
from langchain_core.documents import Document
from typing_extensions import TypedDict, NotRequired
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from operator import add

class GradeResult(TypedDict):
    round_id: int
    index: int
    relevant: NotRequired[bool]
    status: Literal["ok", "error"]
    error: NotRequired[str]

class RAGState(TypedDict, total=False):
    question: str
    retrieval_query: str
    request_id: str
    messages: Annotated[list[AnyMessage], add_messages]
    documents: list[Document]

#    web_available: bool
#    web_search_attempted: bool
#    web_error: str | None
    abstained: bool

    answer: str
    attempts: int
    supported: bool
    
    grade_results: Annotated[list[GradeResult], add]
    faithfulness_score: float
    faithfulness_reason: str
    evaluation_error: str
