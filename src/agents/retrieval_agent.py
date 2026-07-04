"""
retrieval_agent.py
------------------
Agent 3 — RAG Retrieval Agent

Purpose : Given the generated report summary, perform semantic retrieval
          against the FAISS vector store to find the most relevant
          engineering knowledge base passages.
Input   : SummaryOutput from the Summary Agent + FAISS vectorstore (via graph state)
Output  : RetrievalOutput Pydantic model
"""

from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.agents.summary_agent import SummaryOutput
from src.rag.vectorstore import retrieve_documents


# ── Output Model ────────────────────────────────────────────────────────────

class RetrievedChunk(BaseModel):
    """Represents a single retrieved knowledge base chunk."""

    content: str = Field(description="The retrieved text content")
    source: str = Field(description="Source filename of the chunk")
    title: str = Field(description="Document title")
    chunk_index: int = Field(description="Index of this chunk within the full document")


class RetrievalOutput(BaseModel):
    """Structured output from the RAG Retrieval Agent."""

    query: str = Field(description="The query used for retrieval")
    retrieved_chunks: list[RetrievedChunk] = Field(
        description="Top-k retrieved knowledge base chunks"
    )
    context_text: str = Field(
        description="All retrieved chunks combined as a single context string"
    )


# ── Core Functions ───────────────────────────────────────────────────────────

def build_retrieval_query(summary: SummaryOutput) -> str:
    """
    Construct a targeted retrieval query from the report summary.

    Combines key findings, risks, and failed tests to create a rich
    query that captures the most important aspects of the report for retrieval.

    Args:
        summary: SummaryOutput from the Summary Agent.

    Returns:
        A string query for semantic retrieval.
    """
    query_parts: list[str] = []

    query_parts.append(summary.executive_summary)

    if summary.key_findings:
        findings_text = ". ".join(summary.key_findings[:3])  # Top 3 findings
        query_parts.append(f"Key findings: {findings_text}")

    if summary.failed_tests:
        failed_text = ". ".join(summary.failed_tests[:2])
        query_parts.append(f"Failed tests: {failed_text}")

    if summary.risks:
        risks_text = ". ".join(summary.risks[:2])
        query_parts.append(f"Risks: {risks_text}")

    return " ".join(query_parts)


def format_retrieved_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks into a single readable context string for the LLM.

    Args:
        chunks: List of retrieved knowledge base chunks.

    Returns:
        Formatted multi-section string.
    """
    if not chunks:
        return "No relevant knowledge base documents were retrieved."

    sections: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        section = (
            f"[Source {i}: {chunk.title} ({chunk.source})]\n"
            f"{chunk.content}"
        )
        sections.append(section)

    return "\n\n---\n\n".join(sections)


def perform_retrieval(
    summary: SummaryOutput,
    vectorstore: FAISS,
    top_k: int = 5,
) -> RetrievalOutput:
    """
    Perform semantic retrieval using the report summary as the query.

    Args:
        summary: SummaryOutput from the Summary Agent.
        vectorstore: Loaded FAISS vectorstore instance.
        top_k: Number of results to retrieve.

    Returns:
        RetrievalOutput with retrieved chunks and combined context text.
    """
    query = build_retrieval_query(summary)

    raw_documents: list[Document] = retrieve_documents(vectorstore, query, top_k=top_k)

    retrieved_chunks: list[RetrievedChunk] = []
    for doc in raw_documents:
        chunk = RetrievedChunk(
            content=doc.page_content,
            source=doc.metadata.get("source", "unknown"),
            title=doc.metadata.get("title", "Unknown Document"),
            chunk_index=doc.metadata.get("chunk_index", 0),
        )
        retrieved_chunks.append(chunk)

    context_text = format_retrieved_context(retrieved_chunks)

    return RetrievalOutput(
        query=query,
        retrieved_chunks=retrieved_chunks,
        context_text=context_text,
    )


# ── LangGraph Node ───────────────────────────────────────────────────────────

def retrieval_agent_node(state: dict) -> dict:
    """
    LangGraph node for the RAG Retrieval Agent.

    Reads 'summary_output' and 'vectorstore' from the graph state,
    performs semantic retrieval, and writes 'retrieval_output' back.

    Args:
        state: LangGraph state dictionary. Must contain 'summary_output'
               and 'vectorstore'.

    Returns:
        Updated state dictionary with 'retrieval_output' key added.
    """
    summary_output: SummaryOutput = state["summary_output"]
    vectorstore: FAISS = state["vectorstore"]
    top_k: int = state.get("top_k", 5)

    print(f"\n[Retrieval Agent] Performing semantic retrieval (top_k={top_k})...")

    retrieval_output = perform_retrieval(summary_output, vectorstore, top_k=top_k)

    print(
        f"[Retrieval Agent] [OK] Retrieved {len(retrieval_output.retrieved_chunks)} "
        f"chunk(s) from knowledge base"
    )

    return {"retrieval_output": retrieval_output}
