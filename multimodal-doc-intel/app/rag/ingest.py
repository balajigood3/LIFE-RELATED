"""Document ingestion: PDF text + basic image handling."""
from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


def load_pdf_text(path: Path) -> List[Document]:
    if PdfReader is None:
        return [Document(page_content=f"[pypdf not installed] {path.name}", metadata={"source": path.name})]
    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": i + 1, "type": "pdf"}
            ))
    return docs


def load_directory(dir_path: Path) -> List[Document]:
    docs = []
    if not dir_path.exists():
        return docs
    for f in dir_path.rglob("*"):
        if f.suffix.lower() == ".pdf":
            docs.extend(load_pdf_text(f))
        elif f.suffix.lower() in {".txt", ".md"}:
            docs.append(Document(
                page_content=f.read_text(encoding="utf-8", errors="ignore"),
                metadata={"source": f.name, "type": "text"}
            ))
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = 900, overlap: int = 150) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)
