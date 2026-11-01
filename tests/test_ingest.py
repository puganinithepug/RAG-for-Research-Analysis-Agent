from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def test_splitter_creates_overlapping_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=30,
        chunk_overlap=5,
    )

    chunks = splitter.split_documents([
        Document(
            page_content="A long sample passage with enough words to split.",
            metadata={"source": "sample.txt"},
        )
    ])

    assert len(chunks) > 1
    assert all(chunk.metadata["source"] == "sample.txt" for chunk in chunks)

