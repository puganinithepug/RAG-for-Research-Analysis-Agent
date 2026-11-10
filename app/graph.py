from langgraph.graph import END, START, StateGraph

from app.config import settings
from nodes.evaluate import evaluate
from nodes.generate import generate

#from nodes.grade import grade
from nodes.grade import fan_out_grading, grade_one, collect_grades

from nodes.rewrite import rewrite_question, contextualize_for_retrieval

from nodes.abstain import abstain

from app.state import RAGState

from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy

from nodes.retrieve import retrieve, retrieval_cache_key

cache = InMemoryCache()


def route_after_grade(state: RAGState) -> str:
    if state.get("documents"):
        return "generate"
    
    if state.get("attempts", 0) < settings.max_attempts:
        return "rewrite"

    return "abstain"


def route_after_evaluate(state: RAGState) -> str:
    # The evaluator failed; don't get stuck retrying as if it scored zero.
    if state.get("evaluation_error"):
        return "abstain"

    if state.get("supported", False):
        return END


    if state.get("attempts", 0) < settings.max_attempts:
        return "rewrite"

    return "abstain"


def build_graph(checkpointer=None):
    builder = StateGraph(RAGState)
    
    builder.add_node(
        "retrieve",
        retrieve,
        cache_policy=CachePolicy(
            ttl=300,
            key_func=retrieval_cache_key,
        ),
    )

#    builder.add_node("grade", grade)
#   concurrent grading
    builder.add_node("fan_out_grading", lambda state: {})
    builder.add_node("grade_one", grade_one)
    builder.add_node("collect_grades", collect_grades)

#   other graph nodes
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)
    builder.add_node("rewrite", rewrite_question)
    builder.add_node("abstain", abstain)
    builder.add_node("contextualize", contextualize_for_retrieval)


    builder.add_edge(START, "contextualize")
    builder.add_edge("contextualize", "retrieve")
    builder.add_edge("rewrite", "retrieve")

#    builder.add_edge("retrieve", "grade")
#    builder.add_conditional_edges("grade", route_after_grade)
#   replace with fan-out grading
    builder.add_edge("retrieve", "fan_out_grading")
    builder.add_conditional_edges(
        "fan_out_grading",
        fan_out_grading,
        ["grade_one", "collect_grades"],
    )
    builder.add_edge("grade_one", "collect_grades")
    builder.add_conditional_edges("collect_grades", route_after_grade)

#   rest of the graph edges
    builder.add_edge("generate", "evaluate")
    builder.add_conditional_edges("evaluate", route_after_evaluate)

    return builder.compile(checkpointer=checkpointer, cache=cache)
