from langgraph.graph import END, START, StateGraph

from app.state import RAGState


def test_graph_runs_to_completion():
    def retrieve(state):
        return {"documents": []}

    def grade(state):
        return {}

    def generate(state):
        return {"answer": "No evidence found."}

    def evaluate(state):
        return {"supported": True}

    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("grade", grade)
    builder.add_node("generate", generate)
    builder.add_node("evaluate", evaluate)
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_edge("grade", "generate")
    builder.add_edge("generate", "evaluate")
    builder.add_edge("evaluate", END)

    result = builder.compile().invoke({
        "question": "Example?",
        "documents": [],
        "attempts": 0,
        "supported": False,
    })

    assert result["answer"] == "No evidence found."

