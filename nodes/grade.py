from app.prompts import RELEVANCE_PROMPT
from app.retrieval import get_llm
from app.state import RAGState


def _text(response) -> str:
    return str(response.content).strip().upper()


def grade(state: RAGState) -> dict:
    llm = get_llm()
    relevant = []

    for document in state.get("documents", []):
        prompt = RELEVANCE_PROMPT.format(
            question=state["question"],
            passage=document.page_content,
        )
        if _text(llm.invoke(prompt)).startswith("YES"):
            relevant.append(document)

    return {"documents": relevant}

