from app.prompts import CONTEXTUALIZE_PROMPT
from app.retrieval import get_llm
from app.state import RAGState

def contextualize_for_retrieval(state: RAGState) -> dict:
    question = state["question"]
    messages = state.get("messages", [])

    # The latest message is the current question; use earlier turns as context.
    history = messages[:-1][-8:]

    if not history:
        return {"retrieval_query": question}

    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in history
    )

    prompt = CONTEXTUALIZE_PROMPT.format(
        history=history,
        question=state["question"],
    )

    response = get_llm().invoke(prompt)
    query = str(response.content).strip()

    print(f"[contextualize] original: {state['question']!r}")
    print(f"[contextualize] search query: {query!r}")

    return {"retrieval_query": query or question}


def rewrite_question(state: RAGState) -> dict:
    attempts = state.get("attempts", 0) + 1

    messages = state.get("messages", [])
    prior_messages = messages[:-1] if messages else []
    # The latest message is the current user question; use earlier turns as history.
    
    if not prior_messages:
        return {
            "retrieval_query": state["question"],
            "attempts": attempts,
        }

    history = "\n".join(
        f"{message.type}: {message.content}"
        for message in prior_messages
    )

    prompt = CONTEXTUALIZE_PROMPT.format(
        history=history,
        question=state["question"],
    )
    rewritten = get_llm().invoke(prompt).content.strip()
    retrieval_query = rewritten or state["question"]
    changed = retrieval_query.strip().casefold() != state["question"].strip().casefold()

    print(f"[rewrite] original: {state['question']!r}")
    print(f"[rewrite] search query: {retrieval_query!r}")
    print(f"[rewrite] changed using history: {changed}")


    return {"retrieval_query": retrieval_query, "attempts" : attempts}

