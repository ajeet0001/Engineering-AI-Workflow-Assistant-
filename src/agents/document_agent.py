"""
document_agent.py
-----------------
Agent 1 — Document Processing Agent

Purpose : Load a PDF report, extract text page-by-page using PyMuPDF,
          clean the raw text, and return a structured DocumentOutput.
Input   : Path to a PDF file (str or Path)
Output  : DocumentOutput Pydantic model containing document name,
          combined cleaned text, and page count.
"""

from pathlib import Path
from pydantic import BaseModel, Field
import fitz  # PyMuPDF

from src.utils.text_cleaner import clean_text


# ── Output Model ────────────────────────────────────────────────────────────

class DocumentOutput(BaseModel):
    """Structured output from the Document Processing Agent."""

    document_name: str = Field(description="Filename of the processed PDF")
    text: str = Field(description="Full cleaned text extracted from the PDF")
    pages: int = Field(description="Total number of pages in the document")


# ── Core Functions ───────────────────────────────────────────────────────────

def validate_pdf_path(pdf_path: Path) -> None:
    """
    Validate that the given path exists and is a PDF file.

    Args:
        pdf_path: Path to the file to validate.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"[Document Agent] PDF not found: {pdf_path}\n"
            f"  → Place your PDF report in:  data/reports/"
        )
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"[Document Agent] Expected a .pdf file, got: {pdf_path.suffix}"
        )


def extract_text_from_pdf(pdf_path: Path) -> tuple[str, int]:
    """
    Extract raw text from all pages of a PDF using PyMuPDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A tuple of (combined_raw_text, page_count).

    Raises:
        RuntimeError: If the PDF cannot be opened or is empty.
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise RuntimeError(
            f"[Document Agent] Failed to open PDF: {pdf_path}\n  Reason: {exc}"
        ) from exc

    page_count = len(doc)
    if page_count == 0:
        raise RuntimeError(
            f"[Document Agent] The PDF has no pages: {pdf_path}"
        )

    page_texts: list[str] = []
    for page_num in range(page_count):
        page = doc[page_num]
        raw = page.get_text("text")  # plain text extraction
        if raw.strip():
            page_texts.append(raw)

    doc.close()

    combined = "\n\n".join(page_texts)

    if not combined.strip():
        raise RuntimeError(
            f"[Document Agent] No text could be extracted from: {pdf_path}\n"
            "  → The PDF may be image-only (scanned). OCR is not supported."
        )

    return combined, page_count


def process_document(pdf_path: str | Path) -> DocumentOutput:
    """
    Full document processing pipeline: validate → extract → clean → structure.

    Args:
        pdf_path: Path to the PDF file (str or Path).

    Returns:
        DocumentOutput with cleaned text, document name, and page count.
    """
    path = Path(pdf_path)
    validate_pdf_path(path)

    raw_text, page_count = extract_text_from_pdf(path)
    cleaned_text = clean_text(raw_text)

    return DocumentOutput(
        document_name=path.name,
        text=cleaned_text,
        pages=page_count,
    )


# ── LangGraph Node ───────────────────────────────────────────────────────────

def document_agent_node(state: dict) -> dict:
    """
    LangGraph node for the Document Processing Agent.

    Reads 'pdf_path' from the graph state, processes the PDF,
    and writes 'document_output' back to the state.

    Args:
        state: LangGraph state dictionary. Must contain 'pdf_path'.

    Returns:
        Updated state dictionary with 'document_output' key added.
    """
    pdf_path: str = state["pdf_path"]
    print(f"\n[Document Agent] Processing: {pdf_path}")

    output = process_document(pdf_path)

    print(
        f"[Document Agent] [OK] Extracted {output.pages} page(s), "
        f"{len(output.text):,} characters from '{output.document_name}'"
    )

    return {"document_output": output}
