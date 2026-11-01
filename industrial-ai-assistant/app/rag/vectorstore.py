"""Vector store management for document retrieval."""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.utils.config import get_settings


def get_embeddings():
    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key or None,
        base_url=settings.openai_base_url if settings.openai_base_url != "https://api.openai.com/v1" else None,
    )


def load_manual_documents() -> List[Document]:
    """Load all markdown manuals as LangChain Documents."""
    settings = get_settings()
    manuals_dir = settings.manuals_dir
    docs = []

    if not manuals_dir.exists():
        return docs

    for md_file in manuals_dir.glob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        docs.append(
            Document(
                page_content=text,
                metadata={"source": md_file.name, "type": "manual"}
            )
        )
    return docs


def build_or_load_vectorstore(force_rebuild: bool = False) -> Chroma:
    """Build a persistent Chroma vector store from manuals."""
    settings = get_settings()
    persist_dir = str(settings.chroma_persist_dir)
    embeddings = get_embeddings()

    if not force_rebuild and Path(persist_dir).exists() and any(Path(persist_dir).iterdir()):
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="industrial_manuals",
        )

    # Build fresh
    raw_docs = load_manual_documents()
    if not raw_docs:
        # Fallback empty store
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="industrial_manuals",
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(raw_docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="industrial_manuals",
    )
    return vectorstore


def similarity_search(query: str, k: int = 4) -> List[Document]:
    vs = build_or_load_vectorstore()
    return vs.similarity_search(query, k=k)
