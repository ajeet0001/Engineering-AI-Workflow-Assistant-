"""
loader.py
---------
RAG Pipeline — Document Loader

Purpose : Discover and load all .txt files from the engineering knowledge base
          directory into LangChain Document objects ready for chunking.
Input   : Path to the knowledge base directory (from config)
Output  : List of LangChain Document objects with metadata
"""

from pathlib import Path
from langchain_core.documents import Document


def load_knowledge_base(knowledge_base_dir: Path) -> list[Document]:
    """
    Load all .txt files from the knowledge base directory.

    Each file becomes a Document with page_content (full text) and
    metadata containing the source filename and title (first line).

    Args:
        knowledge_base_dir: Path to the directory containing .txt knowledge files.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the knowledge base directory does not exist.
        ValueError: If the directory contains no .txt files.
    """
    kb_path = Path(knowledge_base_dir)

    if not kb_path.exists():
        raise FileNotFoundError(
            f"[RAG Loader] Knowledge base directory not found: {kb_path}\n"
            f"  → Create the directory and add .txt files to it."
        )

    txt_files = sorted(kb_path.glob("*.txt"))

    if not txt_files:
        raise ValueError(
            f"[RAG Loader] No .txt files found in: {kb_path}\n"
            f"  → Add engineering knowledge documents (.txt format) to this directory."
        )

    documents: list[Document] = []

    for file_path in txt_files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[RAG Loader] Warning: Could not read {file_path.name}: {exc}")
            continue

        if not text.strip():
            print(f"[RAG Loader] Warning: Skipping empty file: {file_path.name}")
            continue

        # Extract title from first non-empty line for metadata
        first_line = next(
            (line.strip() for line in text.splitlines() if line.strip()), ""
        )
        title = first_line.replace("TITLE:", "").strip() if "TITLE:" in first_line else file_path.stem

        doc = Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "title": title,
                "file_path": str(file_path),
            },
        )
        documents.append(doc)

    print(f"[RAG Loader] [OK] Loaded {len(documents)} knowledge base document(s)")
    return documents
