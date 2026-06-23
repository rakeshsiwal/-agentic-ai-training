"""
Ingest documents from data/documents/ into a ChromaDB vector store.
Run once (or whenever documents change):
    python -m rag.ingest
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "data" / "documents"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "task03_docs"
# EMBED_MODEL = "all-MiniLM-L6-v2"


# def get_embeddings():
#     return HuggingFaceEmbeddings(model_name=EMBED_MODEL)
EMBED_MODEL = "text-embedding-3-small"

def get_embeddings():
    return OpenAIEmbeddings(model=EMBED_MODEL)

def ingest(docs_dir: Path = DOCS_DIR, chroma_dir: Path = CHROMA_DIR) -> Chroma:
    print(f"Loading documents from {docs_dir} …")
    # loader = DirectoryLoader(
    #     str(docs_dir),
    #     glob="**/*.{txt,md}",
    #     loader_cls=TextLoader,
    #     loader_kwargs={"encoding": "utf-8"},
    #     show_progress=True,
    # )
    loader = DirectoryLoader(
        str(docs_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  Loaded {len(docs)} document(s)")

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)
    print(f"  Split into {len(chunks)} chunk(s)")

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_dir),
    )
    print(f"  Persisted to {chroma_dir}")
    return vectorstore


def load_vectorstore(chroma_dir: Path = CHROMA_DIR) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(chroma_dir),
    )


if __name__ == "__main__":
    ingest()