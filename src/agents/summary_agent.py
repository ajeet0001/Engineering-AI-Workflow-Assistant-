"""
summary_agent.py
----------------
Agent 2 — Summary Agent

Purpose : Call the Gemini LLM with the extracted document text and
          produce a structured summary including executive overview,
          key findings, failed tests, risks, and observations.
Input   : DocumentOutput from the Document Processing Agent (via graph state)
Output  : SummaryOutput Pydantic model
"""

import json
import time
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.agents.document_agent import DocumentOutput
from src.prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT, SUMMARY_HUMAN_PROMPT


# ── Output Model ────────────────────────────────────────────────────────────

class SummaryOutput(BaseModel):
    """Structured output from the Summary Agent."""

    executive_summary: str = Field(description="High-level overview of the report")
    key_findings: list[str] = Field(description="Most important findings from the report")
    failed_tests: list[str] = Field(description="Any tests or checks that failed")
    risks: list[str] = Field(description="Identified risks or concerns")
    observations: list[str] = Field(description="Notable observations from the report")


# ── Core Functions ───────────────────────────────────────────────────────────

def build_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize and return the Gemini LLM client.

    Returns:
        Configured ChatGoogleGenerativeAI instance.
    """
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=0.2,  # Low temperature for consistent structured output
    )


def truncate_text(text: str, max_chars: int = 30000) -> str:
    """
    Truncate text to fit within LLM context limits.

    Args:
        text: Full document text.
        max_chars: Maximum character count to send to the LLM.

    Returns:
        Truncated text with a notice appended if truncation occurred.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    return truncated + "\n\n[NOTE: Document truncated to fit context window]"


def parse_summary_response(raw_response: str) -> SummaryOutput:
    """
    Parse the LLM's JSON response into a SummaryOutput model.

    Args:
        raw_response: Raw string from the LLM (expected to be JSON).

    Returns:
        SummaryOutput Pydantic model.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON.
    """
    # Strip any accidental markdown fences the model might add
    clean = raw_response.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        # Remove first and last fence lines
        clean = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[Summary Agent] LLM returned non-JSON response.\n"
            f"  Raw response (first 500 chars): {raw_response[:500]}\n"
            f"  JSON error: {exc}"
        ) from exc

    return SummaryOutput(
        executive_summary=data.get("executive_summary", ""),
        key_findings=data.get("key_findings", []),
        failed_tests=data.get("failed_tests", []),
        risks=data.get("risks", []),
        observations=data.get("observations", []),
    )


def generate_summary(document_output: DocumentOutput) -> SummaryOutput:
    """
    Generate a structured summary from a processed document.

    Args:
        document_output: Output from the Document Processing Agent.

    Returns:
        SummaryOutput with all structured fields populated.

    Raises:
        RuntimeError: If the LLM call fails.
    """
    llm = build_llm()

    document_text = truncate_text(document_output.text)

    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(
            content=SUMMARY_HUMAN_PROMPT.format(document_text=document_text)
        ),
    ]

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages)
            raw_content = response.content
            break  # Success — exit retry loop
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_rate_limit and attempt < max_retries:
                wait = 35 * attempt  # 35s, 70s
                print(
                    f"[Summary Agent] Rate limit hit (attempt {attempt}/{max_retries}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"[Summary Agent] LLM API call failed after {attempt} attempt(s): {exc}"
            ) from exc

    return parse_summary_response(raw_content)


# ── LangGraph Node ───────────────────────────────────────────────────────────

def summary_agent_node(state: dict) -> dict:
    """
    LangGraph node for the Summary Agent.

    Reads 'document_output' from the graph state, calls the LLM,
    and writes 'summary_output' back to the state.

    Args:
        state: LangGraph state dictionary. Must contain 'document_output'.

    Returns:
        Updated state dictionary with 'summary_output' key added.
    """
    document_output: DocumentOutput = state["document_output"]
    print(f"\n[Summary Agent] Generating summary for: {document_output.document_name}")

    summary = generate_summary(document_output)

    print(
        f"[Summary Agent] [OK] Summary generated — "
        f"{len(summary.key_findings)} findings, "
        f"{len(summary.risks)} risks, "
        f"{len(summary.failed_tests)} failed tests"
    )

    return {"summary_output": summary}
