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
  - DualRouterDecision    — main router: intent route selection
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
    """Intent routing decision for the main RAG pipeline.

    The router LLM fills this schema, choosing:
      - *route*  — where to send the query
      - *reason* — a one-sentence human-readable rationale
    """

    route: Literal["retrieve", "verify_claim", "direct_answer"] = Field(
        description=(
            "Choose exactly one of the three routes based on the user's intent:\n\n"
            "'retrieve' — the user is asking about content from an uploaded research paper. "
            "Use this for questions about methodologies, architectures, results, datasets, "
            "algorithms, authors, figures, tables, or any specific information that would "
            "require searching the document knowledge base. Also use this when the user "
            "mentions a paper title, arXiv ID, or asks you to summarise an uploaded document.\n\n"
            "'verify_claim' — the user wants to check whether a specific scientific claim, "
            "finding, or result is still current or has been superseded by newer research. "
            "Trigger words include: verify, is it still valid, has this been superseded, "
            "is this outdated, what does recent research say about, check if this is true.\n\n"
            "'direct_answer' — the user is making conversation, asking a general knowledge "
            "question not related to any uploaded paper, or asking about the assistant itself. "
            "Use this for greetings, introductions, thank-yous, and questions that can be "
            "answered entirely from world knowledge without any document retrieval."
        )
    )
    reason: str = Field(
        description="One-sentence rationale explaining the route selection."
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
