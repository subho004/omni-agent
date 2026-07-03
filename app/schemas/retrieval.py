"""Structured-output schema for PageIndex reasoning retrieval."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SectionSelection(BaseModel):
    """The sections the model judges relevant to a query."""

    section_ids: list[int] = Field(
        description="Ids of the outline sections most relevant to the query."
    )
    reason: str = Field(description="Brief justification for the choice.")


__all__ = ["SectionSelection"]
