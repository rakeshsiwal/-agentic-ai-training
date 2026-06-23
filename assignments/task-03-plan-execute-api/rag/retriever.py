"""
Thin wrapper around ChromaDB for semantic retrieval.
Loaded once at import time so the vector store is not re-opened per request.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document

from rag.ingest import CHROMA_DIR, load_vectorstore


@lru_cache(maxsize=1)
def _get_store():
    return load_vectorstore(CHROMA_DIR)


def retrieve(query: str, k: int = 4) -> list[str]:
    """Return up to *k* text chunks relevant to *query*."""
    store = _get_store()
    docs: list[Document] = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]