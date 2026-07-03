"""Planning brain: decompose, evaluate, and synthesize (LLM reasoning).

Separated from execution/scheduling (OrchestratorService): this module only
turns queries and intermediate results into structured decisions via Gemini
constrained decoding (docs/implementation-plan.md Phases 7-9).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from google.genai import types

from app.schemas.research import EvalDecision, Plan, RevisePlan, StepReflection
from app.services.agent_loop import build_user_content
from app.services.llm_client import LlmClient
from app.tools.registry import ALL_TOOLS

_CAPABILITIES = "\n".join(f"- {t.name}: {t.description}" for t in ALL_TOOLS)

_PLANNER_SYSTEM = (
    "You are a resourceful research planner. Break the user's request into "
    "concrete steps that sub-agents execute with these tools:\n"
    f"{_CAPABILITIES}\n\n"
    "Principles:\n"
    "- gemini_search and web_search are a fast BASE/DISCOVERY layer: use them "
    "to orient, but then plan concrete steps that actually RETRIEVE and VERIFY "
    "the requested information with the other tools. Never stop at 'here is "
    "where to look' — plan to obtain the actual data.\n"
    "- If content is dynamic/JS-heavy or a simple fetch may be blocked, plan to "
    "use crawl_url or browser_use; if a site exposes a data/JSON/API endpoint, "
    "plan to download_file it; always have a fallback source in mind.\n"
    "- Number steps from 1; use depends_on to order dependent steps; each "
    "description is a self-contained instruction to one sub-agent. Keep the "
    "plan focused (1-5 steps) but complete enough to truly answer the request.\n"
    "- MAXIMIZE PARALLELISM: steps that do not need each other's output MUST "
    "have empty depends_on so they run at the same time. Only add a depends_on "
    "edge when a step genuinely needs a prior step's result. Prefer several "
    "independent steps over one big sequential step.\n"
    "- You do not have to foresee every split now: a sub-agent whose step turns "
    "out to be several INDEPENDENT sub-problems can call spawn_subagents to run "
    "them as parallel child agents. So a step may be stated at a slightly higher "
    "level when its internal breakdown will only become clear at run time."
)

_EVAL_SYSTEM = (
    "You are a rigorous, persistent research evaluator. Judge whether the "
    "gathered results ACTUALLY satisfy the user's goal with concrete "
    "information — not merely point to where the answer might be.\n\n"
    "Tools sub-agents can still use:\n"
    f"{_CAPABILITIES}\n\n"
    "Treat the goal as NOT done (done=false) and add new steps that try a "
    "DIFFERENT tool or angle when any of these hold:\n"
    "- a step failed, was blocked, or a tool replied it 'could not' do "
    "something;\n"
    "- results only give links, say 'check the official site', or describe "
    "where the data lives instead of providing it;\n"
    "- the answer is generic while the user asked for specific or current data.\n\n"
    "When you add steps, name the specific tool/approach and why it differs "
    "from what already failed (e.g. crawl_url blocked -> browser_use with "
    "explicit navigation; page is dynamic -> fetch its JSON/API endpoint via "
    "download_file; one source failed -> an alternative source found via "
    "web_search/gemini_search). Number new steps above the existing ones and "
    "use depends_on where needed.\n\n"
    "You can reshape the plan, not only extend it:\n"
    "- remove_step_numbers: drop PENDING steps that are now unnecessary, "
    "redundant, or misguided given what the results already show (never list a "
    "step that already completed);\n"
    "- new_dependencies: inject an edge into an existing PENDING step so it "
    "waits on a newly-added prerequisite or is reordered — this is how you "
    "insert a sub-step BETWEEN existing steps instead of only appending at the "
    "end. Keep independent steps free of dependencies so they still run in "
    "parallel.\n\n"
    "Only set done=true when the user's actual intent is met with concrete "
    "results, OR you have genuinely exhausted several DISTINCT tool-based "
    "approaches across rounds. Prefer trying another approach over giving up — "
    "do not take a single tool's 'no' as the final answer."
)

_REVISE_SYSTEM = (
    "You are a research re-planner. The user stopped a run and gave a new "
    "instruction that redefines the goal. Given the original query, the steps "
    "already completed (with results), and the new instruction, decide which "
    "completed steps are still useful (keep_step_numbers — reuse their "
    "results) and what new steps are needed. Number new steps ABOVE the "
    "existing ones; keep the plan minimal and only add what the new "
    "instruction requires."
)

_REFLECT_SYSTEM = (
    "You are a demanding reviewer of a sub-agent's work on ONE step of a "
    "research task. Given the step and the sub-agent's result, decide whether "
    "the result is sufficient: it must contain the ACTUAL requested information "
    "(concrete facts, numbers, quotes, extracted data) with sources — not a "
    "plan, a vague pointer ('check the official site'), a refusal, or a partial "
    "answer.\n\n"
    "Tools the sub-agent can still use:\n"
    f"{_CAPABILITIES}\n\n"
    "If NOT sufficient, set sufficient=false and give concrete next_actions: "
    "name the specific tool(s) and exactly what to fetch, extract, or verify "
    "(e.g. 'parse_document on artifact X then bm25_search for Y', 'crawl_url "
    "blocked -> browser_use to navigate and read the table', 'fetch the JSON "
    "endpoint via download_file'). Only set sufficient=true when the step is "
    "genuinely and concretely answered; don't demand more once it is."
)

_SYNTH_SYSTEM = (
    "You are a research writer. Using the gathered results, write a clear, "
    "thorough answer to the user's query in GitHub-Flavored Markdown.\n\n"
    "Formatting — you have full freedom; match the format to the request:\n"
    "- headings, bold/italic, and bullet or numbered lists for structure;\n"
    "- Markdown tables for tabular data, comparisons, or metrics;\n"
    "- fenced code blocks for code, commands, or JSON;\n"
    "- Mermaid diagrams inside ```mermaid fences for flows, timelines, "
    "architectures, sequences, or relationship/org charts;\n"
    "- blockquotes and inline links for citations.\n"
    "If the user asked for a table, diagram, chart, or a specific format, "
    "produce exactly that.\n\n"
    "Substance:\n"
    "- Lead with the actual information found — present concrete data, numbers, "
    "and findings directly. Do NOT open with caveats about what was blocked or "
    "restricted; the reader wants the answer, not the obstacles.\n"
    "- Cite only real sources (URLs or document names that appear in the "
    "results). The '[Step N: … — status]' lines are internal plan labels: "
    "never present them as sources or mention step numbers.\n"
    "- If some data is genuinely missing, still give the best answer possible "
    "from what was gathered, and note any gap briefly at the END — not the start."
)


class PlannerService:
    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    async def create_plan(self, query: str) -> tuple[Plan, int, int]:
        parsed, in_tok, out_tok = await self._llm.generate_structured(
            prompt=f"User request:\n{query}",
            system_instruction=_PLANNER_SYSTEM,
            response_schema=Plan,
        )
        plan = parsed if isinstance(parsed, Plan) else Plan(steps=[])
        return plan, in_tok, out_tok

    async def evaluate(
        self, query: str, results_digest: str
    ) -> tuple[EvalDecision, int, int]:
        prompt = (
            f"Original query:\n{query}\n\n"
            f"Results gathered so far:\n{results_digest}"
        )
        parsed, in_tok, out_tok = await self._llm.generate_structured(
            prompt=prompt,
            system_instruction=_EVAL_SYSTEM,
            response_schema=EvalDecision,
        )
        decision = (
            parsed
            if isinstance(parsed, EvalDecision)
            else EvalDecision(done=True, reason="No decision returned", new_steps=[])
        )
        return decision, in_tok, out_tok

    async def revise(
        self,
        original_query: str,
        completed_digest: str,
        existing_summary: str,
        new_instruction: str,
    ) -> tuple[RevisePlan, int, int]:
        prompt = (
            f"Original query:\n{original_query}\n\n"
            f"Existing steps:\n{existing_summary}\n\n"
            f"Completed results so far:\n{completed_digest}\n\n"
            f"New instruction:\n{new_instruction}"
        )
        parsed, in_tok, out_tok = await self._llm.generate_structured(
            prompt=prompt,
            system_instruction=_REVISE_SYSTEM,
            response_schema=RevisePlan,
        )
        revised = (
            parsed
            if isinstance(parsed, RevisePlan)
            else RevisePlan(keep_step_numbers=[], new_steps=[])
        )
        return revised, in_tok, out_tok

    async def reflect(
        self, goal: str, step_title: str, step_desc: str, result: str
    ) -> tuple[StepReflection, int, int]:
        """Self-critique one sub-agent result: sufficient, or what to do next."""

        prompt = (
            f"Overall goal:\n{goal}\n\n"
            f"Step:\n{step_title}\n{step_desc}\n\n"
            f"Sub-agent result:\n{result}"
        )
        parsed, in_tok, out_tok = await self._llm.generate_structured(
            prompt=prompt,
            system_instruction=_REFLECT_SYSTEM,
            response_schema=StepReflection,
        )
        reflection = (
            parsed
            if isinstance(parsed, StepReflection)
            else StepReflection(sufficient=True, reason="No verdict", next_actions="")
        )
        return reflection, in_tok, out_tok

    def _synth_content(
        self, query: str, digest: str, history: str
    ) -> types.Content:
        preamble = f"{history}\n\n" if history else ""
        return build_user_content(
            f"{preamble}Current query:\n{query}\n\nGathered results:\n{digest}"
        )

    async def synthesize(
        self, query: str, results_digest: str, history: str = ""
    ) -> tuple[str, int, int]:
        result = await self._llm.generate(
            contents=[self._synth_content(query, results_digest, history)],
            system_instruction=_SYNTH_SYSTEM,
        )
        return result.text, result.input_tokens, result.output_tokens

    async def synthesize_stream(
        self,
        query: str,
        results_digest: str,
        history: str,
        usage_sink: dict[str, int],
    ) -> AsyncIterator[str]:
        """Stream the final answer; falls back to a single call if the LLM
        client does not support streaming (e.g. test fakes)."""

        content = self._synth_content(query, results_digest, history)
        if hasattr(self._llm, "generate_stream"):
            async for chunk in self._llm.generate_stream(
                [content], _SYNTH_SYSTEM, usage_sink
            ):
                yield chunk
        else:
            result = await self._llm.generate(
                contents=[content], system_instruction=_SYNTH_SYSTEM
            )
            usage_sink["in"] = result.input_tokens
            usage_sink["out"] = result.output_tokens
            yield result.text


__all__ = ["PlannerService"]
