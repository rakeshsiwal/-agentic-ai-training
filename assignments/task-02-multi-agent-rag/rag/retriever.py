"""
rag/retriever.py — Similarity search wrapper over the persisted ChromaDB collection.

Usage:
    from rag.retriever import retrieve
    chunks = retrieve("What is self-attention?", k=4)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_docs"


def _get_embeddings():
    """Return the same embedding model used during ingestion."""
    use_openai = os.getenv("OPENAI_API_KEY") and os.getenv("USE_OPENAI_EMBEDDINGS", "").lower() == "true"
    if use_openai:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_vector_store() -> Chroma:
    """Load the persisted Chroma collection (cached for the lifetime of the process)."""
    if not CHROMA_DIR.exists():
        raise RuntimeError(
            f"ChromaDB not found at {CHROMA_DIR}. "
            "Run `python -m rag.ingest` first to populate the vector store."
        )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def retrieve(query: str, k: int = 4) -> list[str]:
    """
    Perform a similarity search and return the top-k matching chunks.

    Args:
        query: The natural language question to search for.
        k:     Number of chunks to return.

    Returns:
        A list of plain-text chunk strings, ordered by relevance (most relevant first).
    """
    store = _get_vector_store()
    results = store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


if __name__ == "__main__":
    # Quick smoke test
    test_query = "What is self-attention in transformers?"
    chunks = retrieve(test_query)
    print(f"Query: {test_query}\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {i} ---\n{chunk}\n")
