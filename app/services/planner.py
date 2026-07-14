"""Planning brain: decompose, evaluate, and synthesize (LLM reasoning).

Separated from execution/scheduling (OrchestratorService): this module only
turns queries and intermediate results into structured decisions via Gemini
constrained decoding (docs/implementation-plan.md Phases 7-9).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable

from google.genai import types

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.research import EvalDecision, Plan, RevisePlan, StepReflection
from app.services.agent_loop import build_model_content, build_user_content
from app.services.llm_client import LlmClient, is_retryable_llm_error
from app.tools.registry import ALL_TOOLS

logger = get_logger(__name__)

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
    "level when its internal breakdown will only become clear at run time.\n"
    "- CHASE REFERENCED SOURCES: a source often points at ANOTHER document you "
    "must then obtain (a citation, a linked filing/PDF, 'see the annual report', "
    "a download button). Plan to follow such references, not just read the first "
    "page: use crawl_url and read its returned 'links' to find the exact href, "
    "then crawl_url that (article/HTML) or download_file + parse_document it "
    "(PDF/DOCX/data). Reference-following is often iterative and its depth is "
    "unknown up front — state a step like 'obtain and extract the referenced X' "
    "and let the sub-agent follow the chain until it holds the actual source."
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
    "- results merely REFERENCE another document/filing/dataset (or link to it) "
    "that was never actually fetched and extracted — the referenced source must "
    "be obtained, so add a step to crawl_url its link (use the page's returned "
    "'links') or download_file + parse_document it, following the chain until the "
    "primary source is in hand;\n"
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
    "endpoint via download_file'). A result that only CITES or LINKS a referenced "
    "document without having fetched and extracted it is NOT sufficient — the "
    "next_action is to obtain that source: crawl_url its link (from the page's "
    "returned 'links') or download_file + parse_document it, and follow further "
    "references the same way until the primary source is in hand. Only set "
    "sufficient=true when the step is genuinely and concretely answered; don't "
    "demand more once it is."
)

_SYNTH_SYSTEM = (
    "You are a research writer. Using the gathered results, write a clear, "
    "thorough answer to the user's query. ALWAYS respond in well-structured "
    "GitHub-Flavored Markdown — never a bare wall of plain text.\n\n"
    "Formatting palette — the frontend renders all of GFM plus Mermaid, KaTeX "
    "math, and HTML <details>. Reach for whichever elements fit the content; a "
    "substantial answer should use several:\n"
    "- headings (## / ###) to organize sections, and short paragraphs for prose;\n"
    "- **bold**, *italic*, ~~strikethrough~~, and `inline code` for emphasis and "
    "identifiers;\n"
    "- bullet lists, numbered lists, and task lists (`- [ ]` / `- [x]`) for "
    "steps, checklists, or requirement coverage;\n"
    "- Markdown tables for tabular data, comparisons, specs, or metrics;\n"
    "- fenced code blocks (```lang) for code, commands, config, or JSON;\n"
    "- Mermaid diagrams inside ```mermaid fences for flows, timelines, "
    "architectures, sequences, relationship/org charts, and charts (pie, "
    "xychart-beta, quadrantChart, gantt);\n"
    "- inline math `$…$` and block math `$$…$$` (KaTeX) for any formula, "
    "equation, or quantitative relationship;\n"
    "- blockquotes for callouts, and inline links for citations;\n"
    "- horizontal rules (---) to separate major sections;\n"
    "- collapsible `<details><summary>…</summary>…</details>` blocks to tuck "
    "away long tables, raw data, derivations, or supplementary detail without "
    "cluttering the main flow.\n"
    "If the user asked for a table, diagram, chart, math, or a specific format, "
    "produce exactly that.\n\n"
    "Substance — be thorough and lossless:\n"
    "- Lead with the actual information found — present concrete data, numbers, "
    "and findings directly. Do NOT open with caveats about what was blocked or "
    "restricted; the reader wants the answer, not the obstacles.\n"
    "- The extraction often surfaces important details the user did not "
    "explicitly ask for but would want. Preserve those: report the most "
    "relevant facts, figures, caveats, and context in full rather than "
    "over-summarizing. Prefer a longer, well-organized answer over dropping "
    "detail — use headings and <details> sections to keep length navigable.\n"
    "- Cite only real sources (URLs or document names that appear in the "
    "results). The '[Step N: … — status]' lines are internal plan labels: "
    "never present them as sources or mention step numbers.\n"
    "- If some data is genuinely missing, still give the best answer possible "
    "from what was gathered, and note any gap briefly at the END — not the start."
)

_CONTINUE_SYNTH = (
    "Your previous message was cut off because it reached the output length "
    "limit — it is NOT finished. Continue the SAME Markdown answer from exactly "
    "where you stopped, even if that means resuming mid-sentence, mid-list, or "
    "mid-table row. Do NOT repeat, re-introduce, or re-summarize anything you "
    "already wrote, and do NOT add a lead-in like 'continuing' or restate the "
    "heading — your output will be concatenated directly onto the previous text, "
    "so it must join seamlessly. Keep going until the answer is genuinely "
    "complete, then stop."
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
        on_new_part: Callable[[int], Awaitable[None]] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the final answer, transparently spanning multiple model calls.

        A single call is capped at the model's ``max_output_tokens``, so a long
        answer would otherwise be silently truncated. When a call stops because
        it hit that cap (``truncated``), synthesis continues in another streamed
        call that resumes exactly where the last one stopped; every part is
        yielded in order so the caller/UI concatenates them into one answer.
        ``usage_sink`` accumulates token usage across all parts. ``on_new_part``
        (if given) is awaited with the 1-based part number as each part begins.

        Falls back to a single non-streaming call if the LLM client does not
        support streaming (e.g. test fakes).
        """

        base = self._synth_content(query, results_digest, history)
        if not hasattr(self._llm, "generate_stream"):
            result = await self._llm.generate(
                contents=[base], system_instruction=_SYNTH_SYSTEM
            )
            usage_sink["in"] = result.input_tokens
            usage_sink["out"] = result.output_tokens
            yield result.text
            return

        max_parts = settings.max_synthesis_parts
        max_stream_retries = max(settings.llm_max_retries, 1)
        total_in = total_out = 0
        written = ""
        part = 0
        stream_retries = 0
        while True:
            part += 1
            if on_new_part is not None:
                await on_new_part(part)
            # First part sends only the base prompt; each continuation replays
            # the answer-so-far as a model turn, then asks to resume from it.
            contents = [base]
            if written:
                contents.append(build_model_content(written))
                contents.append(build_user_content(_CONTINUE_SYNTH))

            part_sink: dict[str, int] = {}
            before = len(written)
            try:
                async for chunk in self._llm.generate_stream(
                    contents, _SYNTH_SYSTEM, part_sink
                ):
                    written += chunk
                    yield chunk
            except Exception as exc:  # noqa: BLE001 - re-raised if not transient
                if not is_retryable_llm_error(exc):
                    raise
                stream_retries += 1
                if stream_retries > max_stream_retries:
                    raise
                logger.warning(
                    "Stream interrupted mid-answer (%s); resuming from %d "
                    "chars written (retry %d/%d)",
                    exc,
                    len(written),
                    stream_retries,
                    max_stream_retries,
                )
                # Retry this part from where the answer left off; doesn't
                # count against max_synthesis_parts (that budget is for
                # legitimate output-cap continuations, not connection drops).
                part -= 1
                continue
            total_in += part_sink.get("in", 0)
            total_out += part_sink.get("out", 0)

            # A clean part resets the retry budget for the next one.
            stream_retries = 0

            # Stop when the model finished naturally, produced nothing new (no
            # point re-asking), or we've chained the configured maximum parts.
            if not part_sink.get("truncated") or len(written) == before:
                break
            if max_parts > 0 and part >= max_parts:
                logger.warning(
                    "synthesis stopped at max_synthesis_parts=%d; answer may be "
                    "truncated",
                    max_parts,
                )
                break

        usage_sink["in"] = total_in
        usage_sink["out"] = total_out


__all__ = ["PlannerService"]
