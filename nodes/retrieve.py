from app.config import settings
from app.retrieval import get_vectorstore, requested_k
from app.state import RAGState

def retrieve(state):
    attempt = state.get("attempts", 0) + 1
    base_k = requested_k(state["question"])
    k = min(base_k * attempt, settings.max_k)

    documents = get_vectorstore().similarity_search(
        state["question"],
        k=k,
    )


    return {"documents": documents, "attempts": attempt}
