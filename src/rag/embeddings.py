"""
embeddings.py
-------------
RAG Pipeline — Embeddings

Purpose : Provide a Sentence Transformer embedding wrapper compatible
          with LangChain's FAISS integration.
Input   : List of text strings
Output  : Embedding vectors (via LangChain interface)
"""

from langchain_huggingface import HuggingFaceEmbeddings


def build_embeddings(model_name: str = "all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    """
    Initialize the Sentence Transformer embedding model.

    Uses the HuggingFaceEmbeddings wrapper which is fully compatible
    with LangChain's FAISS vectorstore API.

    The model is downloaded automatically on first use and cached locally
    in the HuggingFace cache directory (~/.cache/huggingface/).

    Args:
        model_name: Name of the Sentence Transformer model to use.
                    Default: 'all-MiniLM-L6-v2' (fast, 384-dim, good quality).

    Returns:
        HuggingFaceEmbeddings instance ready for use with FAISS.
    """
    print(f"[Embeddings] Loading embedding model: {model_name}")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[Embeddings] [OK] Embedding model ready: {model_name}")
    return embeddings
