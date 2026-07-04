"""Structured-output schema for PageIndex reasoning retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SectionSelection(BaseModel):
    """The sections the model judges relevant to a query."""

    section_ids: list[int] = Field(
        description="Ids of the outline sections most relevant to the query."
    )
    reason: str = Field(description="Brief justification for the choice.")


class RerankSelection(BaseModel):
    """Candidate passage ids reordered most-relevant-first for a query."""

    ranked_ids: list[int] = Field(
        description=(
            "Candidate passage ids ordered from most to least relevant to the "
            "query. Include only ids that actually help answer the query; drop "
            "irrelevant ones."
        )
    )


__all__ = ["SectionSelection", "RerankSelection"]
