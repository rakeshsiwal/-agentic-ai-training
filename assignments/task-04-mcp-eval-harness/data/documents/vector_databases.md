# Vector Databases and Embedding Search

Vector databases are purpose-built storage systems for high-dimensional numerical vectors, commonly called embeddings. Unlike relational databases that match rows by exact field values, vector databases find records based on geometric proximity in embedding space.

## What Are Embeddings?

An embedding is a dense numerical representation of data (text, images, audio) produced by a neural network. Similar items are mapped to nearby points in the high-dimensional space. For example, sentences with similar meaning produce embeddings that are close together when measured by cosine similarity.

## How Similarity Search Works

Given a query embedding, a vector database performs approximate nearest-neighbor (ANN) search to find the top-k most similar stored vectors. Common algorithms include HNSW (Hierarchical Navigable Small World graphs) and IVF (Inverted File Index). These algorithms trade a small accuracy loss for massive speed improvements over brute-force comparison.

## ChromaDB

ChromaDB is an open-source vector database optimized for embedding-based search. It supports persistence to disk, filtering by metadata, and integration with LangChain via the `langchain-chroma` package. Documents are added with `.add()` or `.from_documents()` and retrieved with `.similarity_search()`.

## Key Metrics

- **Cosine similarity**: measures the angle between two vectors; 1.0 = identical direction, 0 = orthogonal, -1 = opposite. Most suitable for text embeddings.
- **Euclidean distance**: measures straight-line distance; lower = more similar.
- **Recall@k**: fraction of true top-k results captured by the ANN index, a standard accuracy metric.

## RAG (Retrieval-Augmented Generation)

Vector databases are the backbone of RAG pipelines. When a user asks a question, the query is embedded and the most relevant document chunks are retrieved from the vector store. These chunks are appended to the LLM prompt as context, grounding the model's answer in factual, up-to-date information rather than relying solely on parametric memory.
