from app.prompts import RELEVANCE_PROMPT
# from app.retrieval import get_llm
from app.grading import get_grading_llm
from app.state import RAGState

# for the concurrent grading
from langgraph.types import Send


def _text(response) -> str:
    return str(response.content).strip().upper()


#def grade(state: RAGState) -> dict:
#    llm = get_llm()
#    relevant = []

#    for document in state.get("documents", []):
#        prompt = RELEVANCE_PROMPT.format(
#            question=state["question"],
#            passage=document.page_content,
#        )
#        response = llm.invoke(prompt)
#        print(f"[grade] {document.metadata.get('source')}")
#        print(f"page={document.metadata.get('page')}: {response.content!r}")

#        if _text(response).startswith("YES"):
#            relevant.append(document)

#    return {"documents": relevant}


def fan_out_grading(state: RAGState):
    documents = state.get("documents", [])

    if not documents:
        return "collect_grades"

    return [
        Send(
            "grade_one",
            {
                "question": state["question"],
                "document": document,
                "index": index,
                "request_id": state["request_id"],
                "round_id": state.get("attempts", 0),
            },
        )
        for index, document in enumerate(documents)
    ]

def grade_one(task: dict) -> dict:
    document = task["document"]

    try: 
        prompt = RELEVANCE_PROMPT.format(
            question=task["question"],
            passage=document.page_content,
        )
        response = get_grading_llm().invoke(prompt)
        print(f"[grade] {document.metadata.get('source')}")
        print(f"page={document.metadata.get('page')}: {response.content!r}")
        relevant = _text(response).startswith("YES")


        return {
            "grade_results": [{
            "request_id": task["request_id"],
            "round_id": task["round_id"],
            "index": task["index"],
            "status": "ok",
            "relevant": _text(response).startswith("YES"),
            }]
        }

    except Exception as exc:
        # A model/API failure is not the same as "not relevant."
        return {
            "grade_results": [{
                "request_id": task["request_id"],
                "round_id": task["round_id"],
                "index": task["index"],
                "status": "error",
                "error": type(exc).__name__,
            }]
        }



def collect_grades(state: RAGState) -> dict:
    request_id = state["request_id"]
    round_id = state.get("attempts", 0)

    current = [
        r for r in state.get("grade_results", [])
        if r.get("request_id") == request_id
        and r.get("round_id") == round_id
    ]

    by_index = {
        r["index"]: r
        for r in current
        if r.get("status") == "ok"
    }

    documents = state.get("documents", [])
    relevant = [
        doc for i, doc in enumerate(documents)
        if by_index.get(i, {}).get("relevant", False)
    ]

    errors = [r for r in current if r.get("status") == "error"]
    return {
        "documents": relevant,
        "grading_errors": errors,
    }

