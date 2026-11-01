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



def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_path,
        embedding_function=get_embeddings(),
    )

def temp(question: str) -> float:
    q = question.lower()
    # some way of identifying creative terminology
    # for every creative term x1.5 or some small increment 
    # base line is 0.01
    # if no creative signals leave at 0
    return 0.0


def get_llm(temp: float = 0.0) -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=temp,
    )

