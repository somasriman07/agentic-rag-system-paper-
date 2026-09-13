from typing import Literal

from pydantic import BaseModel, Field


class BtwRouteDecision(BaseModel):
    needs_web_search: Literal["yes", "no"] = Field(
        description="yes if answering requires a real-time web search, no if general knowledge is sufficient"
    )

    @property
    def needs_web_search_bool(self) -> bool:
        return self.needs_web_search.strip().lower() == "yes"


class RouterDecision(BaseModel):
    route: Literal["retrieve", "verify_claim", "direct_answer"]


class RelevancyDecision(BaseModel):
    is_relevant: Literal["yes", "no"] = Field(
        description="yes if the chunks contain information that meaningfully addresses the question, no if off-topic"
    )
    reason: str = Field(description="Brief reason for the relevancy decision")

    @property
    def is_relevant_bool(self) -> bool:
        return self.is_relevant.strip().lower() == "yes"


class SupersedingPaper(BaseModel):
    title: str
    url: str
    summary: str


class ClaimVerificationResult(BaseModel):
    is_superseded: Literal["yes", "no"] = Field(
        description="yes if the claim has been superseded or challenged by recent work, no otherwise"
    )
    verdict_summary: str = Field(description="1-2 sentence verdict summary suitable for display")
    superseding_papers: list[SupersedingPaper] = Field(
        default_factory=list,
        description="List of up to 3 papers that supersede or update the claim"
    )

    @property
    def is_superseded_bool(self) -> bool:
        return self.is_superseded.strip().lower() == "yes"