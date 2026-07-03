"""Pydantic DTOs for planning, evaluation, and orchestrated research.

The `PlanStep`, `Plan`, and `EvalDecision` models double as Gemini
structured-output schemas (docs/implementation-plan.md Phases 7-9).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One node the planner/evaluator proposes."""

    step_number: int = Field(description="Unique step number within the plan.")
    title: str = Field(description="Short title of the step.")
    description: str = Field(
        description="What the sub-agent should do to complete this step."
    )
    depends_on: list[int] = Field(
        description="Step numbers this step needs completed first; empty if none."
    )


class Plan(BaseModel):
    """A full research plan as a set of dependent steps."""

    steps: list[PlanStep]


class EvalDecision(BaseModel):
    """Evaluator verdict after a round of steps completes."""

    done: bool = Field(
        description="True if the plan's results are enough to answer the query."
    )
    reason: str = Field(description="Brief justification for the decision.")
    new_steps: list[PlanStep] = Field(
        description="Additional steps to run if more work is needed; else empty."
    )


class StepReflection(BaseModel):
    """A sub-agent's self-critique of its own result for one step."""

    sufficient: bool = Field(
        description=(
            "True only if the result fully and concretely completes the step "
            "(actual data/findings, not vague pointers or a partial answer)."
        )
    )
    reason: str = Field(
        description="Brief judgment: what is done and what, if anything, is missing."
    )
    next_actions: str = Field(
        description=(
            "If not sufficient, concrete next actions to improve it — which "
            "tools to use and what to fetch/verify. Empty when sufficient."
        )
    )


class RevisePlan(BaseModel):
    """Evaluator/planner verdict when revising a run with new instructions."""

    keep_step_numbers: list[int] = Field(
        description=(
            "Step numbers of existing completed steps whose results are still "
            "valid and should be reused; empty to discard all prior work."
        )
    )
    new_steps: list[PlanStep] = Field(
        description=(
            "Additional steps to run for the revised goal, numbered above the "
            "existing steps; depends_on may reference kept or new steps."
        )
    )


class ReviseRequest(BaseModel):
    instruction: str = Field(
        min_length=1,
        description="New instruction that redefines the goal mid-run.",
    )
    model: str | None = None
    thinking_level: str | None = None


class PlanNodeResponse(BaseModel):
    step_number: int
    title: str
    description: str
    status: str
    depends_on: list[int]
    result: str


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model: str | None = None
    thinking_level: str | None = None


class ResearchResponse(BaseModel):
    session_id: UUID
    answer: str
    plan: list[PlanNodeResponse]
    iterations: int
    input_tokens: int
    output_tokens: int
