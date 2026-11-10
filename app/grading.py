from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.config import settings

import re

COUNT_PATTERN = re.compile(
    r"\b(?:top|use|retrieve|return|show)\s+(\d+)\s+"
    r"(?:documents?|docs?|chunks?|sources?)\b",
    re.IGNORECASE,
)


def requested_k(question: str) -> int:
    match = COUNT_PATTERN.search(question)
    requested = int(match.group(1)) if match else settings.top_k

    # Keep requests within a configured, predictable range.
    return max(1, min(requested, settings.max_k))


def temp(question: str) -> float:
    q = question.lower()
    return 0.0


def get_grading_llm(temp: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=settings.grader_model,
        base_url=settings.ollama_base_url,
        temperature=temp,
    )
