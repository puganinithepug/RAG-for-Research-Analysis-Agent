import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.loaders import load_documents
from app.retrieval import get_vectorstore


def _chunk_id(source: str, index: int, text: str) -> str:
    raw = f"{source}:{index}:{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def ingest() -> int:
    documents = load_documents(settings.documents_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    ids = [
        _chunk_id(chunk.metadata.get("source", ""), index, chunk.page_content)
        for index, chunk in enumerate(chunks)
    ]

    if chunks:
        get_vectorstore().add_documents(chunks, ids=ids)

    return len(chunks)

