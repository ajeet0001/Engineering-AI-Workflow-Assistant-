"""
chunking.py
-----------
RAG Pipeline — Text Chunking

Purpose : Split loaded knowledge base documents into smaller overlapping
          chunks suitable for embedding and semantic retrieval.
Input   : List of LangChain Document objects
Output  : List of chunked LangChain Document objects
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """
    Split documents into overlapping text chunks using recursive character splitting.

    Uses a hierarchy of separators (paragraphs → newlines → sentences → words)
    to create semantically meaningful chunks.

    Args:
        documents: List of full Document objects from the loader.
        chunk_size: Maximum character count per chunk.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        List of chunked Document objects, each retaining the parent metadata
        plus a 'chunk_index' field.

    Raises:
        ValueError: If the documents list is empty.
    """
    if not documents:
        raise ValueError("[Chunking] No documents provided for chunking.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(
        f"[Chunking] [OK] Split {len(documents)} document(s) into "
        f"{len(chunks)} chunk(s) "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )

    return chunks
