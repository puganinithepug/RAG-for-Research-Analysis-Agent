from app.prompts import ANSWER_PROMPT
from app.retrieval import get_llm, temp
from app.state import RAGState

import re

def generate(state: RAGState) -> dict:
    documents = state.get("documents", [])

    if not documents:
        return {
            "answer": "I couldn't find relevant evidence in the indexed documents."
        }

    evidence_parts = []
    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown source")
        page = document.metadata.get("page")

        label = f"[{index}] {source}"
        if page is not None:
            label += f", page {page}"

        evidence_parts.append(f"{label}\n{document.page_content}")

    prompt = ANSWER_PROMPT.format(
        question=state["question"],
        context="\n\n".join(evidence_parts),
    )

    response = get_llm(temp=temp(state["question"])).invoke(prompt)
    answer = response.content.strip()

    cited_indices = {
        int(match) for match in re.findall(r"\[(\d+)\]", answer)
    }

    sources = []
    for index, document in enumerate(documents, start=1):
        if index not in cited_indices:
            continue

        source = document.metadata.get("source", "unknown source")
        page = document.metadata.get("page")
        label = f"[{index}] {source}"
        if page is not None:
            label += f", page {page}"
        sources.append(label)

    if sources:
        answer += "\n\nSources:\n" + "\n".join(sources)

    return {"answer": answer}

