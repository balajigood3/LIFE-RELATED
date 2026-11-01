from __future__ import annotations
from pathlib import Path
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from app.rag.ingest import load_directory, chunk_documents
from app.utils.config import get_settings, DATA_DIR

def get_embeddings():
    s = get_settings()
    return OpenAIEmbeddings(api_key=s.openai_api_key or None, model=s.embedding_model)

def build_vectorstore(force: bool = False) -> Chroma:
    s = get_settings()
    persist = str(DATA_DIR / "chroma_db")
    emb = get_embeddings()
    if not force and Path(persist).exists() and any(Path(persist).iterdir()):
        return Chroma(persist_directory=persist, embedding_function=emb, collection_name="industrial_docs")

    raw = load_directory(DATA_DIR / "sample_docs")
    # Also load any markdown manuals we ship
    chunks = chunk_documents(raw) if raw else []
    if not chunks:
        # seed with a minimal demo document
        chunks = [Document(
            page_content="IEC 60079-14 requires isolation of all live conductors before maintenance in hazardous areas. SF6 pressure low stage-2 requires immediate investigation and blocking of closing circuits.",
            metadata={"source": "demo_seed", "type": "text"}
        )]
    return Chroma.from_documents(chunks, embedding=emb, persist_directory=persist, collection_name="industrial_docs")

def query(store: Chroma, q: str, k: int = 4) -> List[Document]:
    return store.similarity_search(q, k=k)
