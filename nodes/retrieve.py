from app.config import settings
from app.retrieval import get_vectorstore, requested_k
from app.state import RAGState


def retrieval_cache_key(state: RAGState) -> str:
    query = state.get("retrieval_query") or state["question"]
    normalized_query = " ".join(query.casefold().split())

    attempt = state.get("attempts", 0) + 1
    k = min(requested_k(normalized_query) * attempt, settings.max_k)

    return f"index-v1:{normalized_query}:k={k}"


def retrieve(state):
    attempt = state.get("attempts", 0) + 1
    query = state.get("retrieval_query") or state["question"]
    base_k = requested_k(query)
#    base_k = requested_k(state["question"])
    k = min(base_k * attempt, settings.max_k)

#    documents = get_vectorstore().similarity_search(
#        state["question"],
#        k=k,
#    )

    documents = get_vectorstore().similarity_search(query, k=k)

    print(f"[retrieve] k={k}, returned={len(documents)}")

    return {"documents": documents, "attempts": attempt}
