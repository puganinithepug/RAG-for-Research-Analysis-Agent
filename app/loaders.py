from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_documents(root: str) -> list[Document]:
    documents: list[Document] = []

    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            loaded = TextLoader(str(path), encoding="utf-8").load()

        for document in loaded:
            document.metadata["source"] = str(path)
        documents.extend(loaded)

    return documents

