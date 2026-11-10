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
        vectorstore = get_vectorstore()
        batch_size = 1000

        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            vectorstore.add_documents(
                chunks[start:end],
                ids=ids[start:end],
            )
            print(f"Indexed {min(end, len(chunks))}/{len(chunks)} chunks")


    return len(chunks)

