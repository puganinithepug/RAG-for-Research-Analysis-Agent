from app.state import RAGState

def abstain(state: RAGState) -> dict:
    return {
        "answer": "I couldn’t find enough reliable information to answer your question.",
        "abstained": True,
    }

