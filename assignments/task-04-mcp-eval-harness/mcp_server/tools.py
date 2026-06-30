"""
Tool implementations for the MCP server.
"""

from __future__ import annotations

import ast
import math
import operator
from datetime import date
from pathlib import Path

_VECTOR_STORE = None


def _get_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is not None:
        return _VECTOR_STORE

    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    persist_dir = Path(__file__).parent.parent / "data" / "chroma_db"
    persist_dir.mkdir(parents=True, exist_ok=True)

    # Suppress tokenizer parallelism warning on Windows
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    _VECTOR_STORE = Chroma(
        collection_name="task04_docs",
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    if _VECTOR_STORE._collection.count() == 0:
        _ingest_documents(_VECTOR_STORE)

    return _VECTOR_STORE


def _ingest_documents(store) -> None:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    docs_dir = Path(__file__).parent.parent / "data" / "documents"
    if not docs_dir.exists():
        print(f"[ingest] documents dir not found: {docs_dir}", flush=True)
        return

    raw_docs = []
    for pattern in ("*.txt", "*.md"):
        for filepath in docs_dir.glob(pattern):
            try:
                loader = TextLoader(str(filepath), encoding="utf-8")
                raw_docs.extend(loader.load())
            except Exception as e:
                print(f"[ingest] skipping {filepath}: {e}", flush=True)

    if not raw_docs:
        print(f"[ingest] no documents found in {docs_dir}", flush=True)
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)
    for chunk in chunks:
        chunk.metadata["source"] = Path(chunk.metadata.get("source", "unknown")).name

    store.add_documents(chunks)
    print(f"[ingest] added {len(chunks)} chunks from {len(raw_docs)} files", flush=True)


def retrieve(query: str, k: int = 4) -> list[str]:
    store = _get_vector_store()
    results = store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_SAFE_NAMES = {
    "abs": abs, "round": round, "sqrt": math.sqrt,
    "floor": math.floor, "ceil": math.ceil,
    "log": math.log, "log10": math.log10,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


def _safe_eval_node(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        return _SAFE_NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval_node(node.left), _safe_eval_node(node.right))
    if isinstance(node, ast.Call):
        fn = node.func.id if isinstance(node.func, ast.Name) else None
        if fn in _SAFE_NAMES and callable(_SAFE_NAMES[fn]):
            return _SAFE_NAMES[fn](*[_safe_eval_node(a) for a in node.args])
    raise ValueError(f"Unsupported: {type(node).__name__}")


def calculate(expression: str) -> str:
    try:
        result = _safe_eval_node(ast.parse(expression.strip(), mode="eval"))
        return str(round(result, 10))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:
        return f"Error: {exc}"


def get_current_date() -> str:
    return date.today().isoformat()