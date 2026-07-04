"""
vectorstore.py
--------------
RAG Pipeline — FAISS Vector Store

Purpose : Build, persist, load, and query a FAISS vector index from
          chunked knowledge base documents.
Input   : Chunked Document objects + embedding model
Output  : FAISS vectorstore / retrieved Document objects
"""

import warnings
from pathlib import Path
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Suppress the langchain-community sunset deprecation warning.
# FAISS is still the recommended vector store; no standalone package exists yet.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.vectorstores import FAISS


def build_vector_store(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    store_dir: Path,
) -> FAISS:
    """
    Build a FAISS index from document chunks and persist it to disk.

    Args:
        chunks: List of chunked Document objects to index.
        embeddings: Initialized HuggingFaceEmbeddings instance.
        store_dir: Directory path where the FAISS index will be saved.

    Returns:
        Populated FAISS vectorstore instance.

    Raises:
        ValueError: If the chunks list is empty.
    """
    if not chunks:
        raise ValueError("[VectorStore] Cannot build index from empty chunk list.")

    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    print(f"[VectorStore] Building FAISS index from {len(chunks)} chunk(s)...")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(store_dir))

    print(f"[VectorStore] [OK] FAISS index built and saved to: {store_dir}")
    return vectorstore


def load_vector_store(
    store_dir: Path,
    embeddings: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Load an existing FAISS index from disk.

    Args:
        store_dir: Directory containing the saved FAISS index.
        embeddings: Initialized HuggingFaceEmbeddings instance (must match build-time model).

    Returns:
        Loaded FAISS vectorstore instance.

    Raises:
        FileNotFoundError: If the vector store directory or index files are missing.
    """
    store_dir = Path(store_dir)
    index_file = store_dir / "index.faiss"

    if not store_dir.exists() or not index_file.exists():
        raise FileNotFoundError(
            f"[VectorStore] No FAISS index found at: {store_dir}\n"
            f"  → Run with --rebuild-index to build the index first."
        )

    print(f"[VectorStore] Loading FAISS index from: {store_dir}")
    vectorstore = FAISS.load_local(
        str(store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"[VectorStore] [OK] FAISS index loaded successfully")
    return vectorstore


def retrieve_documents(
    vectorstore: FAISS,
    query: str,
    top_k: int = 5,
) -> list[Document]:
    """
    Perform semantic similarity search against the FAISS index.

    Args:
        vectorstore: Loaded FAISS vectorstore instance.
        query: The query string to search against.
        top_k: Number of most-similar documents to return.

    Returns:
        List of top_k most relevant Document objects.
        Returns empty list if retrieval fails (logged, not raised).
    """
    if not query.strip():
        print("[VectorStore] Warning: Empty query string provided to retrieval.")
        return []

    try:
        results = vectorstore.similarity_search(query, k=top_k)
        print(f"[VectorStore] [OK] Retrieved {len(results)} relevant chunk(s) for query")
        return results
    except Exception as exc:
        print(f"[VectorStore] Warning: Retrieval failed: {exc}")
        return []


def get_or_build_vector_store(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    store_dir: Path,
    rebuild: bool = False,
) -> FAISS:
    """
    Convenience function: load existing index or build a new one.

    Args:
        chunks: Chunks to use if building (ignored if loading existing).
        embeddings: Initialized embedding model.
        store_dir: Vector store directory.
        rebuild: If True, always rebuild even if an index already exists.

    Returns:
        FAISS vectorstore instance.
    """
    index_file = Path(store_dir) / "index.faiss"

    if rebuild or not index_file.exists():
        return build_vector_store(chunks, embeddings, store_dir)

    return load_vector_store(store_dir, embeddings)
