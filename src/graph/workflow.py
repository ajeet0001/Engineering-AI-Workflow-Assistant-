"""
workflow.py
-----------
LangGraph Workflow — Engineering AI Pipeline

Purpose : Define the LangGraph state graph that connects all four agents
          in sequence: Document → Summary → Retrieval → Recommendation.
Input   : Initial state with 'pdf_path', 'vectorstore', and 'top_k'
Output  : Final state with all four agent outputs populated
"""

from typing import TypedDict, Any
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END

from src.agents.document_agent import DocumentOutput, document_agent_node
from src.agents.summary_agent import SummaryOutput, summary_agent_node
from src.agents.retrieval_agent import RetrievalOutput, retrieval_agent_node
from src.agents.recommendation_agent import RecommendationOutput, recommendation_agent_node


# ── Graph State ──────────────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """
    Shared state passed between all nodes in the LangGraph workflow.

    Fields are populated incrementally as each agent node runs.

    Attributes:
        pdf_path        : Path to the input PDF report (required at start).
        vectorstore     : Pre-loaded FAISS vectorstore (required at start).
        top_k           : Number of documents to retrieve (optional, default 5).
        document_output : Output from the Document Processing Agent.
        summary_output  : Output from the Summary Agent.
        retrieval_output: Output from the RAG Retrieval Agent.
        recommendation_output: Output from the Recommendation Agent.
    """
    # Inputs
    pdf_path: str
    vectorstore: Any          # FAISS — typed as Any to avoid TypedDict issues
    top_k: int

    # Agent outputs (populated sequentially)
    document_output: DocumentOutput
    summary_output: SummaryOutput
    retrieval_output: RetrievalOutput
    recommendation_output: RecommendationOutput


# ── Graph Construction ───────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """
    Construct and compile the LangGraph workflow.

    The graph executes agents in a strict linear sequence:
        document_agent → summary_agent → retrieval_agent → recommendation_agent

    Returns:
        Compiled LangGraph runnable (StateGraph).
    """
    graph = StateGraph(GraphState)

    # Register agent nodes
    graph.add_node("document_agent", document_agent_node)
    graph.add_node("summary_agent", summary_agent_node)
    graph.add_node("retrieval_agent", retrieval_agent_node)
    graph.add_node("recommendation_agent", recommendation_agent_node)

    # Define execution order (linear pipeline)
    graph.set_entry_point("document_agent")
    graph.add_edge("document_agent", "summary_agent")
    graph.add_edge("summary_agent", "retrieval_agent")
    graph.add_edge("retrieval_agent", "recommendation_agent")
    graph.add_edge("recommendation_agent", END)

    compiled = graph.compile()
    print("[Workflow] [OK] LangGraph workflow compiled successfully")
    return compiled


def run_workflow(
    pdf_path: str,
    vectorstore: FAISS,
    top_k: int = 5,
) -> GraphState:
    """
    Execute the full LangGraph workflow for a given PDF report.

    Args:
        pdf_path: Absolute or relative path to the input PDF.
        vectorstore: Pre-loaded FAISS vectorstore instance.
        top_k: Number of documents to retrieve from the knowledge base.

    Returns:
        Final GraphState with all four agent outputs populated.

    Raises:
        RuntimeError: If any agent in the pipeline raises an exception.
    """
    workflow = build_workflow()

    initial_state: GraphState = {
        "pdf_path": pdf_path,
        "vectorstore": vectorstore,
        "top_k": top_k,
    }

    print("\n" + "=" * 60)
    print("  Engineering AI Workflow — Starting Pipeline")
    print("=" * 60)

    final_state = workflow.invoke(initial_state)

    print("\n" + "=" * 60)
    print("  Pipeline Complete [OK]")
    print("=" * 60 + "\n")

    return final_state
