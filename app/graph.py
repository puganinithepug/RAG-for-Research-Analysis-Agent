from langgraph.graph import END, START, StateGraph

from app.config import settings
from nodes.evaluate import evaluate
from nodes.generate import generate
from nodes.grade import grade
from nodes.retrieve import retrieve
from app.state import RAGState


def route_after_grade(state: RAGState) -> str:
    # Generate handles the no-evidence case by returning an abstention.
    return "generate"


def route_after_evaluate(state: RAGState) -> str:
    # The evaluator failed; don't get stuck retrying as if it scored zero.
    if state.get("evaluation_error"):
        return END

    if state.get("supported", False):
        return END

    if state.get("attempts", 0) >= settings.max_attempts:
        return END

    return "retrieve"


def build_graph():
    builder = StateGraph(RAGState)

    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges("grade", route_after_grade)
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges("evaluate", route_after_evaluate)

    return builder.compile()

