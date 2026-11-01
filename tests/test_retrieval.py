from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings


def test_chroma_can_store_and_retrieve(tmp_path):
    store = Chroma(
        collection_name="test",
        persist_directory=str(tmp_path),
        embedding_function=FakeEmbeddings(size=8),
    )
    store.add_documents([
        Document(page_content="A test document", metadata={"source": "test.txt"})
    ])

    results = store.similarity_search("test", k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "test.txt"

