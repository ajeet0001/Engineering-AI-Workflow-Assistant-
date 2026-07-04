"""
recommendation_agent.py
-----------------------
Agent 4 — Recommendation Agent

Purpose : Combine the structured report summary and retrieved engineering
          knowledge to generate grounded, actionable recommendations.
          The LLM is strictly constrained to use only the provided context.
Input   : SummaryOutput + RetrievalOutput (via graph state)
Output  : RecommendationOutput Pydantic model
"""

import json
import time
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings
from src.agents.summary_agent import SummaryOutput, build_llm
from src.agents.retrieval_agent import RetrievalOutput
from src.prompts.recommendation_prompt import (
    RECOMMENDATION_SYSTEM_PROMPT,
    RECOMMENDATION_HUMAN_PROMPT,
)


# ── Output Model ────────────────────────────────────────────────────────────

class RecommendationOutput(BaseModel):
    """Structured output from the Recommendation Agent."""

    root_cause: str = Field(description="Most likely root cause based on the evidence")
    next_actions: list[str] = Field(description="Immediate next actions to take")
    recommended_tests: list[str] = Field(description="Tests recommended to verify or diagnose")
    preventive_measures: list[str] = Field(description="Actions to prevent recurrence")
    confidence_level: str = Field(description="High, Medium, or Low confidence")
    confidence_rationale: str = Field(description="Explanation for the confidence level")


# ── Core Functions ───────────────────────────────────────────────────────────

def format_summary_for_prompt(summary: SummaryOutput) -> str:
    """
    Convert a SummaryOutput into a readable string for the recommendation prompt.

    Args:
        summary: SummaryOutput from the Summary Agent.

    Returns:
        Formatted string representation of the summary.
    """
    lines = [
        f"Executive Summary: {summary.executive_summary}",
        "",
        "Key Findings:",
        *[f"  - {f}" for f in summary.key_findings],
    ]

    if summary.failed_tests:
        lines += ["", "Failed Tests:"] + [f"  - {t}" for t in summary.failed_tests]

    if summary.risks:
        lines += ["", "Risks:"] + [f"  - {r}" for r in summary.risks]

    if summary.observations:
        lines += ["", "Observations:"] + [f"  - {o}" for o in summary.observations]

    return "\n".join(lines)


def parse_recommendation_response(raw_response: str) -> RecommendationOutput:
    """
    Parse the LLM's JSON response into a RecommendationOutput model.

    Args:
        raw_response: Raw string from the LLM (expected to be JSON).

    Returns:
        RecommendationOutput Pydantic model.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON.
    """
    clean = raw_response.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[Recommendation Agent] LLM returned non-JSON response.\n"
            f"  Raw response (first 500 chars): {raw_response[:500]}\n"
            f"  JSON error: {exc}"
        ) from exc

    return RecommendationOutput(
        root_cause=data.get("root_cause", "Unable to determine root cause from available information."),
        next_actions=data.get("next_actions", []),
        recommended_tests=data.get("recommended_tests", []),
        preventive_measures=data.get("preventive_measures", []),
        confidence_level=data.get("confidence_level", "Low"),
        confidence_rationale=data.get("confidence_rationale", ""),
    )


def generate_recommendations(
    summary: SummaryOutput,
    retrieval: RetrievalOutput,
) -> RecommendationOutput:
    """
    Generate structured engineering recommendations using LLM + retrieved context.

    Args:
        summary: Structured report summary from the Summary Agent.
        retrieval: Retrieved knowledge base chunks from the Retrieval Agent.

    Returns:
        RecommendationOutput with actionable recommendations.

    Raises:
        RuntimeError: If the LLM call fails.
    """
    llm = build_llm()

    summary_text = format_summary_for_prompt(summary)
    context_text = retrieval.context_text

    messages = [
        SystemMessage(content=RECOMMENDATION_SYSTEM_PROMPT),
        HumanMessage(
            content=RECOMMENDATION_HUMAN_PROMPT.format(
                summary=summary_text,
                context=context_text,
            )
        ),
    ]

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = llm.invoke(messages)
            raw_content = response.content
            break  # Success
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            if is_rate_limit and attempt < max_retries:
                wait = 35 * attempt
                print(
                    f"[Recommendation Agent] Rate limit hit (attempt {attempt}/{max_retries}). "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)
                continue
            raise RuntimeError(
                f"[Recommendation Agent] LLM API call failed after {attempt} attempt(s): {exc}"
            ) from exc

    return parse_recommendation_response(raw_content)


# ── LangGraph Node ───────────────────────────────────────────────────────────

def recommendation_agent_node(state: dict) -> dict:
    """
    LangGraph node for the Recommendation Agent.

    Reads 'summary_output' and 'retrieval_output' from the graph state,
    generates recommendations, and writes 'recommendation_output' back.

    Args:
        state: LangGraph state dictionary. Must contain 'summary_output'
               and 'retrieval_output'.

    Returns:
        Updated state dictionary with 'recommendation_output' key added.
    """
    summary_output: SummaryOutput = state["summary_output"]
    retrieval_output: RetrievalOutput = state["retrieval_output"]

    print(f"\n[Recommendation Agent] Generating recommendations...")

    recommendations = generate_recommendations(summary_output, retrieval_output)

    print(
        f"[Recommendation Agent] [OK] Generated recommendations — "
        f"Confidence: {recommendations.confidence_level}, "
        f"{len(recommendations.next_actions)} actions, "
        f"{len(recommendations.recommended_tests)} tests"
    )

    return {"recommendation_output": recommendations}
