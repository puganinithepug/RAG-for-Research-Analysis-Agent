import hashlib
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.loaders import load_documents
from app.retrieval import get_vectorstore


def _chunk_id(source: str, index: int, text: str) -> str:
    raw = f"{source}:{index}:{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def rebuild_index() -> int:
    vectorstore = get_vectorstore()
    vectorstore.delete_collection()
    print(f"Deleted Chroma collection: {settings.collection_name}")
    return ingest()


def ingest(file_path: str | None = None) -> int:
    print(f"[ingest] documents_path={settings.documents_path}")
    documents = load_documents(file_path or settings.documents_path)
    print(f"[ingest] loaded documents/pages={len(documents)}")

    text_stats = Counter()
    for doc in documents:
        source = doc.metadata.get("source", "<missing source>")
        text = doc.page_content or ""
        text_stats[source] += len(text.strip())

    for source, char_count in sorted(text_stats.items()):
        if char_count == 0:
            print(f"[ingest] WARNING: no extracted text: {source}")
        else:
            print(f"[ingest] extracted characters={char_count}: {source}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"[ingest] chunks created={len(chunks)}")

    if not chunks:
        print("[ingest] No chunks created; nothing indexed.")
        return 0


    sources = sorted({
        chunk.metadata.get("source", "<missing source>")
        for chunk in chunks
    })
    print(f"[ingest] distinct chunk sources={len(sources)}")
    for source in sources:
        source_chunks = sum(
            chunk.metadata.get("source", "<missing source>") == source
            for chunk in chunks
        )
        print(f"[ingest] source={source!r}, chunks={source_chunks}")

    ids = [
        _chunk_id(chunk.metadata.get("source", ""), index, chunk.page_content)
        for index, chunk in enumerate(chunks)
    ]

    vectorstore = get_vectorstore()
    batch_size = 1000


    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        vectorstore.add_documents(
            chunks[start:end],
            ids=ids[start:end],
        )
        print(f"Indexed {min(end, len(chunks))}/{len(chunks)} chunks")

    print(f"[ingest] collection count now={vectorstore._collection.count()}")


    return len(chunks)

