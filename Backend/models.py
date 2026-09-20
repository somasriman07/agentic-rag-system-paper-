"""
Backend/models.py
-----------------
Pydantic data models shared across the pipeline.

Each model is a strict schema that LangChain's `with_structured_output()` uses
to parse LLM JSON responses into typed Python objects.  Keeping all schemas in
one place makes it easy to audit what the LLM is expected to return and to
extend or version the schemas without hunting through multiple files.

Models defined here:
  - BtwRouteDecision      — /btw side-channel: does the query need a web search?
  - DualRouterDecision    — main router: intent route + model tier selection
  - RouterDecision        — (legacy) intent-only routing decision
  - ModelTierDecision     — (legacy) model-tier-only decision
  - RelevancyDecision     — retrieval quality gate: are the chunks relevant?
  - SupersedingPaper      — a single paper that supersedes a claim
  - ClaimVerificationResult — full output of the claim-verification node
"""

from typing import Literal
from pydantic import BaseModel, Field


# ── /btw side-channel ─────────────────────────────────────────────────────────

class BtwRouteDecision(BaseModel):
    """Decides whether a /btw query needs a real-time web search."""

    needs_web_search: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the answer requires real-time web data (recent events, "
            "current prices, breaking news); 'no' if general LLM knowledge suffices."
        )
    )

    @property
    def needs_web_search_bool(self) -> bool:
        """Return True if a web search is required."""
        return self.needs_web_search.strip().lower() == "yes"


# ── Main pipeline router ──────────────────────────────────────────────────────

class DualRouterDecision(BaseModel):
    """Combined intent + model-tier routing decision for the main RAG pipeline.

    The router LLM fills this schema in a single call, choosing:
      - *route*       — where to send the query (retrieval, claim verification, or direct)
      - *model_tier*  — which LLM tier should generate the final answer
      - *reason*      — a one-sentence human-readable rationale
    """

    route: Literal["retrieve", "verify_claim", "direct_answer"] = Field(
        description=(
            "'retrieve' for questions about uploaded papers, research methods, "
            "benchmarks, or any document-grounded query; "
            "'verify_claim' for checking whether a specific scientific claim has been "
            "superseded by newer literature; "
            "'direct_answer' for conversational greetings or pure general-knowledge questions."
        )
    )
    model_tier: Literal["gemini", "qwen"] = Field(
        description=(
            "'gemini' (Frontier API) for multi-paper comparative synthesis, theoretical "
            "derivations, complex algorithmic trade-offs, or high-ambiguity questions; "
            "'qwen' (Open-Weight GPU) for specific factual lookups, single-paper section "
            "queries, parameter/metric extraction, definitions, or straightforward summaries."
        )
    )
    reason: str = Field(
        description="One-sentence rationale explaining the model tier and route selection."
    )


# ── Legacy single-dimension routing models ───────────────────────────────────
# Retained for compatibility with any evaluation or utility scripts that import
# these directly.  New code should prefer DualRouterDecision.

class RouterDecision(BaseModel):
    """Intent-only routing decision (legacy — prefer DualRouterDecision)."""

    route: Literal["retrieve", "verify_claim", "direct_answer"]


class ModelTierDecision(BaseModel):
    """Model-tier-only routing decision (legacy — prefer DualRouterDecision)."""

    model_tier: Literal["gemini", "qwen"] = Field(
        description=(
            "'gemini' for complex multi-paper synthesis, deep mathematical/theoretical "
            "analysis, high ambiguity, or broad research reasoning; "
            "'qwen' for direct paper lookup, factual extraction, specific questions, "
            "or standard summaries."
        )
    )
    reason: str = Field(
        description="One-sentence explanation of why this model tier was selected."
    )


# ── Retrieval quality gate ────────────────────────────────────────────────────

class RelevancyDecision(BaseModel):
    """LLM judgement on whether retrieved chunks are relevant to the query."""

    is_relevant: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the retrieved chunks meaningfully address the question; "
            "'no' if the content is off-topic or insufficient."
        )
    )
    reason: str = Field(
        description="Brief explanation of the relevancy decision."
    )

    @property
    def is_relevant_bool(self) -> bool:
        """Return True if the retrieved chunks are judged relevant."""
        return self.is_relevant.strip().lower() == "yes"


# ── Claim verification ────────────────────────────────────────────────────────

class SupersedingPaper(BaseModel):
    """A single paper identified as superseding or challenging a research claim."""

    title: str = Field(description="Full title of the paper.")
    url: str = Field(description="Direct link to the paper (prefer arxiv.org links).")
    summary: str = Field(
        description="One sentence explaining how this paper supersedes or updates the claim."
    )


class ClaimVerificationResult(BaseModel):
    """Full output of the claim-verification node.

    The LLM populates this after searching the web and arXiv for papers that
    challenge or supersede the user's claim.
    """

    is_superseded: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the claim has been superseded or significantly challenged "
            "by more recent work; 'no' if the claim still holds."
        )
    )
    verdict_summary: str = Field(
        description="1–2 sentence verdict suitable for direct display to the user."
    )
    superseding_papers: list[SupersedingPaper] = Field(
        default_factory=list,
        description="Up to 3 papers that supersede or update the claim.",
    )

    @property
    def is_superseded_bool(self) -> bool:
        """Return True if the claim has been superseded."""
        return self.is_superseded.strip().lower() == "yes"
