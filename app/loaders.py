from pathlib import Path
from collections import Counter
import time

from langchain_community.document_loaders import (
        PyPDFLoader, 
        TextLoader, 
        UnstructuredEPubLoader, 
        UnstructuredPDFLoader,
        UnstructuredImageLoader, 
    )
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt", ".epub", ".jpg", ".jpeg", ".png"}

def load_epub(path: Path) -> list[Document]:
    try:
        loaded = UnstructuredEPubLoader(str(path), mode="single").load()

        if any(doc.page_content.strip() for doc in loaded):
            print(f"[loader] EPUB parsed with Unstructured: {path}")
            return loaded

        print(f"[loader] WARNING: Unstructured returned no text; trying EbookLib: {path}")

    except Exception as exc:
        print(
            f"[loader] WARNING: Unstructured failed for {path}: "
            f"{type(exc).__name__}: {exc}; trying EbookLib"
        )

    book = epub.read_epub(str(path))
    documents = []

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        text = soup.get_text(separator=" ", strip=True)

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "chapter": item.get_name(),
                    },
                )
            )

    if not documents:
        raise ValueError("EbookLib found no text in the EPUB")

    print(f"[loader] EPUB parsed with EbookLib: {path} -> {len(documents)} chapter(s)")
    return documents

def load_documents(root: str) -> list[Document]:
    
    root_path = Path(root)

    if root_path.is_file():
        paths = [root_path]
    elif root_path.is_dir():
        paths = sorted(root_path.rglob("*"))
    else:
        raise FileNotFoundError(f"Path does not exist: {root_path}")

    documents: list[Document] = []
    counts = Counter()
    
    print(f"[loader] Scanning: {root_path}")

    for path in paths:
        if not path.is_file():
            continue


        suffix=path.suffix.lower() 
        
        if suffix not in SUPPORTED_EXTENSIONS:
            counts["unsupported"] += 1
            print(f"[loader] SKIP unsupported: {path} (extension={suffix or '<none>'})")
            continue
        
        try:
            if suffix == ".pdf":
                loaded = PyPDFLoader(str(path)).load()

                if not any(doc.page_content.strip() for doc in loaded):
                    print(f"[loader] No text layer; trying OCR: {path}")
                    loaded = UnstructuredPDFLoader(
                        str(path),
                        mode="single",
                        strategy="ocr_only",
                    ).load()

            elif suffix == ".epub":
                loaded = load_epub(path)
            elif suffix in {".jpg", ".jpeg", ".png"}:
                loaded = UnstructuredImageLoader(str(path), mode="single").load()
            else:
                loaded = TextLoader(str(path), encoding="utf-8").load()


            for document in loaded:
                document.metadata["source"] = str(path)
            
            documents.extend(loaded)

            counts["loaded_files"] += 1
            counts["loaded_documents"] += len(loaded)
            print(f"[loader] OK: {path} -> {len(loaded)} page/document(s)")

            if not loaded:
                print(f"[loader] WARNING: loader returned no content for {path}")

        except Exception as exc:
            counts["failed"] += 1
            print(f"[loader] ERROR: {path}: {type(exc).__name__}: {exc}")

    print(
        "[loader] Summary: "
        f"loaded_files={counts['loaded_files']}, "
        f"loaded_pages_or_documents={counts['loaded_documents']}, "
        f"unsupported_files={counts['unsupported']}, "
        f"failed_files={counts['failed']}"
    )


    return documents

