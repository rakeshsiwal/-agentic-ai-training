"""
rag/ingest.py — Load, chunk, embed, and persist documents to ChromaDB.

Run this once (or whenever documents change):
    python -m rag.ingest

Re-running is idempotent: existing documents are replaced by their new versions
using a deterministic document ID based on file name + chunk index.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
DOCUMENTS_DIR = Path(__file__).parent.parent / "data" / "documents"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "rag_docs"

# ── Embedding model ────────────────────────────────────────────────────────────
# Uses a free local sentence-transformer — no API key required.
# Switch to OpenAIEmbeddings() if you prefer and have an OPENAI_API_KEY set.
def _get_embeddings():
    use_openai = os.getenv("OPENAI_API_KEY") and os.getenv("USE_OPENAI_EMBEDDINGS", "").lower() == "true"
    if use_openai:
        from langchain_openai import OpenAIEmbeddings
        print("[ingest] Using OpenAI embeddings.")
        return OpenAIEmbeddings(model="text-embedding-3-small")
    print("[ingest] Using local HuggingFace embeddings (sentence-transformers/all-MiniLM-L6-v2).")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _doc_id(source: str, chunk_index: int) -> str:
    """Deterministic ID so re-ingestion updates rather than duplicates."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest(
    documents_dir: Path = DOCUMENTS_DIR,
    chroma_dir: Path = CHROMA_DIR,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> Chroma:
    """
    Ingest all .txt and .md files found in `documents_dir`.

    Returns:
        The populated Chroma vector store instance.
    """
    # files = sorted(
    #     p for p in documents_dir.iterdir()
    #     if p.suffix in {".txt", ".md"} and p.is_file()
    # )
    # if not files:
    #     raise FileNotFoundError(f"No .txt or .md files found in {documents_dir}")

    # print(f"[ingest] Found {len(files)} document(s): {[f.name for f in files]}")
    files = sorted(
        p for p in documents_dir.iterdir()
        if p.suffix.lower() in {".txt", ".md", ".pdf"} and p.is_file()
    )

    if not files:
        raise FileNotFoundError(
            f"No .txt, .md, or .pdf files found in {documents_dir}"
        )

    print(f"[ingest] Found {len(files)} document(s): {[f.name for f in files]}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_texts: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    # for file in files:
    #     raw = file.read_text(encoding="utf-8")
    #     chunks = splitter.split_text(raw)
    #     print(f"[ingest]   {file.name} → {len(chunks)} chunk(s)")
    #     for i, chunk in enumerate(chunks):
    #         all_texts.append(chunk)
    #         all_metadatas.append({"source": file.name, "chunk_index": i})
    #         all_ids.append(_doc_id(file.name, i))
    for file in files:

        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            pages = loader.load()

            raw = "\n".join(page.page_content for page in pages)

        else:
            raw = file.read_text(encoding="utf-8")

        chunks = splitter.split_text(raw)

        print(f"[ingest]   {file.name} → {len(chunks)} chunk(s)")

        for i, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_metadatas.append(
                {
                    "source": file.name,
                    "chunk_index": i
                }
            )
            all_ids.append(_doc_id(file.name, i))

    embeddings = _get_embeddings()

    # Chroma.from_texts with explicit IDs will upsert — idempotent.
    chroma_dir.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma.from_texts(
        texts=all_texts,
        embedding=embeddings,
        metadatas=all_metadatas,
        ids=all_ids,
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_dir),
    )
    print(f"[ingest] ✅ Stored {len(all_texts)} chunk(s) in ChromaDB at {chroma_dir}")
    return vector_store


if __name__ == "__main__":
    ingest()
