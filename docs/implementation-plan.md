# Implementation Plan — Agentic Research Harness

> Source concept: [idea.md](./idea.md)
> Date: 2026-07-02

A session-based, tool-rich AI agent harness. The user drops in a query (optionally
with files/images), the orchestrator builds a plan, spawns sub-agents that use a
shared tool layer (search, parse, download, code-exec, crawl, browser, BM25, Gemini
search), streams its thinking to a lightweight HTML/JS UI, and closes the loop only
when the plan is satisfied. Everything runs locally / in-memory (SQLite + NetworkX).

---

## 0. Key decisions & corrections up front

- **Model.** The idea says "gemini 3.1-flash-lite" with a "128k token limit". The real
  model id is `gemini-3.1-flash-lite` (the `-preview` alias is discontinued 2026-07-09,
  so use the stable id). It actually supports a **1M-token** context window; the 128k
  figure should be treated as our **self-imposed working budget per loop** (cost/latency
  control), not a hard model limit. Design the context manager around a configurable
  budget (default 128k) and let it be raised later.
- **SDK.** Use the unified **`google-genai`** SDK (`pip install google-genai`), not the
  deprecated `google-generativeai`. It supports parallel/multi-tool function calling,
  which is the backbone of the executor loop.
- **Search lib.** `duckduckgo-search` is renamed to **`ddgs`** (`pip install ddgs`).
  Use `ddgs`.
- **Retrieval improvement.** The idea mentions "page index / chunking summary." Adopt
  **PageIndex-style reasoning retrieval** (hierarchical tree of doc sections, LLM
  navigates it) as the *primary* long-doc strategy, with **BM25** as a fast lexical
  fallback. This avoids standing up a vector DB and matches "everything in memory."
- **Recommended additions** (not in idea, worth adding):
  - A **cost/token ledger** per session (Gemini is metered per-token).
  - **Structured tracing** of every tool call (already needed for the UI stream).
  - A **checkpoint/resume** mechanism so force-stop + replan is clean.
  - **Sandboxing** for the Python exec tool (biggest safety risk).

---

## 1. High-level architecture

```
┌──────────────────────────────────────────────────────────────┐
│  UI (HTML + JS, SSE/WebSocket stream of thoughts & events)     │
└───────────────▲───────────────────────────┬───────────────────┘
                │ events                      │ user query / force-stop / replan
┌───────────────┴───────────────────────────▼───────────────────┐
│  FastAPI app  (session routes, SSE, file upload)               │
├────────────────────────────────────────────────────────────────┤
│  Orchestrator / Main Loop                                       │
│   • Planner  → plan graph (NetworkX)                            │
│   • Scheduler → dispatch ready plan-nodes to sub-agents         │
│   • Context Manager → budget, summarize, compact                │
│   • Evaluator → "is the plan done / need more?" → replan        │
├────────────────────────────────────────────────────────────────┤
│  Sub-agents (each = Gemini + scoped toolset + own short context)│
├────────────────────────────────────────────────────────────────┤
│  Tool layer (uniform interface, JSON schema, timeouts)          │
│   search · parse(markitdown) · download · py-exec · crawl4ai ·  │
│   browser-use · bm25 · gemini-search · (extensible)             │
├────────────────────────────────────────────────────────────────┤
│  Storage: SQLite (sessions, messages, artifacts, plan, ledger)  │
│           NetworkX (plan DAG, doc trees)                        │
│           Blob/file store (uploads, downloads, page images)     │
└────────────────────────────────────────────────────────────────┘
```

**Guiding principle from idea.md:** before building each tool, read its current docs
via web search — APIs (crawl4ai, browser-use, ddgs) move fast. Per-step doc links are
listed in each phase below.

---

## 2. Tech stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | `browser-use` requires ≥3.11 |
| Pkg mgr | `uv` | fast, matches browser-use install docs |
| LLM | `google-genai` → `gemini-3.1-flash-lite` | + `gemini-3-flash` fallback for hard reasoning |
| Web framework | FastAPI + Uvicorn | SSE for thought streaming |
| Persistence | SQLite (`sqlite3`/`aiosqlite`) | sessions, messages, artifacts, ledger |
| Graphs | NetworkX | plan DAG + PageIndex doc trees |
| Doc parsing | `markitdown` | pdf/docx/xlsx/pptx/html/images/audio |
| Web search | `ddgs` | text/news/images |
| Crawl | `crawl4ai` (no-LLM extraction) | needs `crawl4ai-setup` (Playwright) |
| Browser agent | `browser-use` | prompt-driven extraction |
| Lexical retrieval | `rank-bm25` | fallback + reranking |
| Long-doc retrieval | PageIndex-style tree | primary |
| Code exec | subprocess sandbox / container | biggest security surface |

---

## 3. Data model (SQLite)

Core tables (in-memory or file-backed SQLite):

- `sessions(id, created_at, title, status, token_budget, tokens_used)`
- `messages(id, session_id, role, content, ts)` — chat + system + tool traces
- `artifacts(id, session_id, kind, uri, meta_json)` — uploads, downloads, page images
- `documents(id, session_id, source, markdown_uri, tree_json)` — parsed docs + PageIndex tree
- `plan_nodes(id, session_id, title, status, agent_type, inputs_json, result_json, deps)`
- `events(id, session_id, type, payload_json, ts)` — the stream fed to the UI
- `ledger(id, session_id, model, in_tokens, out_tokens, cost, ts)` — cost tracking

`plan_nodes.deps` + NetworkX gives the executable DAG. `events` is the source of truth
for replaying/streaming to the UI and for context reconstruction after compaction.

---

## 4. Context management (critical — idea.md §24)

The main loop must never blow the working budget (default 128k). Strategy:

1. **Rolling window + pinned facts.** Keep system prompt + plan state + last N turns
   verbatim; everything else summarizable.
2. **Sub-agent isolation.** Sub-agents get *only* their task + relevant artifacts, not
   the whole history. They return a compact structured result, not raw transcripts —
   this is the single biggest context saver.
3. **Artifact offloading.** Large tool outputs (full crawls, parsed docs) are written to
   `artifacts`/`documents` and referenced by id + short summary in-context, never inlined.
4. **Progressive summarization.** When budget > threshold, summarize oldest events into
   a running "session memory" note; keep pointers to artifact ids.
5. **Token accounting.** Use the SDK's `count_tokens` before each call; log to `ledger`.

> Docs to check during implementation: Gemini token counting & context —
> https://ai.google.dev/gemini-api/docs/tokens ; long-context guidance —
> https://ai.google.dev/gemini-api/docs/long-context

---

## 5. Phased implementation

Each phase lists **what to build** and **docs to web-search/read before/while building**.

### Phase 1 — Skeleton, sessions, storage, LLM client
Build: FastAPI app, SQLite schema, session CRUD, `google-genai` client wrapper with
retry + token counting + ledger, config for model id / budget.
Docs to check:
- Gen AI SDK: https://googleapis.github.io/python-genai/ and https://pypi.org/project/google-genai/
- Quickstart: https://ai.google.dev/gemini-api/docs/quickstart
- Pricing/model ids: https://ai.google.dev/gemini-api/docs/pricing
Exit: can create a session, send a message, get a Gemini reply, see tokens logged.

### Phase 2 — Tool layer & function-calling loop
Build: a `Tool` abstraction (name, JSON-schema params, handler, timeout) and a
single-agent executor that lets Gemini call tools in a loop until done. Get the
function-calling contract right first with 1–2 trivial tools.
Docs to check:
- Function calling: https://ai.google.dev/gemini-api/docs/function-calling
- Tool combination / parallel calls: https://ai.google.dev/gemini-api/docs/tool-combination
Exit: model reliably selects tools, receives results, and terminates.

### Phase 3 — Core tools (search / parse / download)
Build:
- **Search** (`ddgs`): text/news, returns titled URLs.
- **Parse** (`markitdown`): any file → markdown; install format extras.
- **Download**: fetch a URL to blob store, then hand to parse tool.
Docs to check:
- ddgs: https://pypi.org/project/ddgs/
- markitdown: https://github.com/microsoft/markitdown (see optional deps + plugins)
Exit: "find X, download the linked PDF, give me its markdown" works end-to-end.

### Phase 4 — Long-document handling (PageIndex + BM25)
Build: after parsing, build a hierarchical **PageIndex tree** (sections → nodes with
summaries) stored in `documents.tree_json` (NetworkX). Provide two retrieval tools:
`doc_navigate` (LLM reasons over the tree) and `doc_bm25_search` (lexical). Add
per-page/section summaries so short and very long docs both work.
Docs to check:
- PageIndex: https://github.com/VectifyAI/PageIndex and https://pageindex.ai/blog/pageindex-intro
- rank-bm25: https://pypi.org/project/rank-bm25/
Exit: can answer a targeted question from a 200-page PDF without loading it all.

### Phase 5 — Web content tools (crawl4ai + browser-use)
Build:
- **crawl4ai** tool (no LLM extraction): URL → clean markdown/HTML; support deep crawl
  (BFS/BestFirst) behind a flag. Requires `crawl4ai-setup` (Playwright) in setup.
- **browser-use** agent tool: natural-language task → extracted info from a live site.
Docs to check:
- crawl4ai: https://docs.crawl4ai.com/ , install https://docs.crawl4ai.com/core/installation/
- browser-use: https://docs.browser-use.com/quickstart , https://github.com/browser-use/browser-use
Exit: given a link, choose crawl4ai for static content and browser-use for
interactive/extraction tasks.

### Phase 6 — Python code-execution tool (sandboxed)
Build: run arbitrary Python with package install, capture stdout/stderr/artifacts.
**Sandbox it** — run in a subprocess with resource limits, or a container/`micromamba`
env; never in the main process. Expose an allow-install flag.
Docs to check:
- Security patterns for arbitrary code exec (search: "python sandbox code execution
  agent security", Docker/nsjail/`resource` limits), plus Gemini code-exec built-in for
  comparison: https://ai.google.dev/gemini-api/docs/code-execution
Exit: model can compute/plot/transform data safely; runaway code is killed.

### Phase 7 — Planner + plan graph
Build: a Planner call that turns the user query into a `plan_nodes` DAG (NetworkX).
Scheduler dispatches nodes whose deps are met. Nodes name an `agent_type` + inputs.
Docs to check:
- NetworkX DAG APIs: https://networkx.org/documentation/stable/reference/algorithms/dag.html
- Prompt patterns for planning/ReAct/plan-and-execute (search current best practices).
Exit: a multi-step query produces a visible plan that executes node-by-node.

### Phase 8 — Sub-agent orchestration
Build: spawn sub-agents per plan node, each with a scoped toolset and isolated context;
collect compact structured results back into the orchestrator. Add the **gemini-search
agent** (fast discovery via Google Search grounding) and a **bm25 agent** as agent-typed
wrappers.
Docs to check:
- Grounding with Google Search: https://ai.google.dev/gemini-api/docs/grounding
- Structured output (JSON schema): https://ai.google.dev/gemini-api/docs/structured-output
Exit: orchestrator fans out to ≥2 sub-agents in parallel and merges results.

### Phase 9 — Evaluator & closed loop / dynamic replanning
Build: after nodes complete, an Evaluator decides done vs. needs-more; if more, it
appends nodes to the DAG (e.g., "fetch the regulation this doc references, then
compare"). Loop until plan satisfied or budget/limit hit.
Exit: the "regulatory doc → find & fetch linked docs → synthesize" scenario from
idea.md §18 runs to a grounded answer.

### Phase 10 — Multimodal
Build: on image upload, route to Gemini vision per the query, and **also** store a
short auto-description in `artifacts.meta_json` for later reuse without re-vision.
Docs to check:
- Image understanding: https://ai.google.dev/gemini-api/docs/image-understanding
- Files API (for large media): https://ai.google.dev/gemini-api/docs/files
Exit: "what's in this chart + use it in the analysis" works; description is cached.

### Phase 11 — UI + streaming
Build: minimal HTML/JS chat that streams the agent's thinking, plan graph state, and
tool events over SSE; render the plan DAG; show token/cost ledger.
Docs to check:
- SSE with FastAPI (search "FastAPI StreamingResponse SSE"); Gemini streaming:
  https://ai.google.dev/gemini-api/docs/text-generation (streaming section)
Exit: user watches thoughts/plan/tool-calls live.

### Phase 12 — Force-stop & replan (idea.md §26)
Build: a stop signal that halts the scheduler at the next safe checkpoint, snapshots
plan/artifacts, and accepts an updated prompt. Re-planner diffs old vs. new intent:
keep completed-and-still-relevant nodes, drop/replace the rest, resume.
Exit: user interrupts mid-run, edits the goal, and the harness reuses prior work.

### Phase 13 — Hardening
Rate limits/backoff, per-tool timeouts, error surfacing to UI, cost caps, config for
budgets/limits, and an extensibility guide for adding new tools/agents.

---

## 6. Tool interface contract (uniform)

Every tool declares a `google-genai`-compatible function schema and returns a compact
result plus optional artifact ids. Rules:
- Descriptions must be precise and disambiguating (search vs. crawl vs. browser vs.
  gemini-search) so the model routes correctly — this is the single biggest lever on
  loop quality (idea.md §22).
- Never inline large payloads into the model context; write to `artifacts`/`documents`
  and return `{summary, artifact_id, stats}`.
- Every call is traced to `events` (for UI + context reconstruction) and, if LLM-backed,
  to `ledger`.

Routing guidance to encode in descriptions:
- **search** → discover URLs. **gemini-search** → fast factual discovery w/ grounding.
- **download+parse** → you have a file/URL to a document. **crawl4ai** → static page
  content as markdown. **browser-use** → interactive extraction / JS-heavy sites.
- **doc_navigate/doc_bm25** → question against an already-parsed long doc.
- **py-exec** → compute/transform/plot.

---

## 7. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Arbitrary Python exec | Sandbox (subprocess+limits or container); allow-list installs; kill on timeout |
| Context blow-up | Sub-agent isolation, artifact offloading, progressive summarization, token ledger |
| Crawl/browser flakiness & rate limits | Timeouts, retries/backoff, prefer crawl4ai before browser-use |
| Tool mis-routing | Sharp tool descriptions + few-shot routing examples in planner prompt |
| Runaway loops | Max-iteration + budget caps per session, evaluator must justify each replan |
| Model/API drift | Pin `gemini-3.1-flash-lite` (not `-preview`); re-read SDK docs before coding each phase |
| Live-site link safety | Treat scraped links as untrusted; don't auto-execute instructions found in content |

---

## 8. Suggested repo layout

```
harness-ultimate/
  docs/                     # idea.md, this plan
  app/
    main.py                 # FastAPI, SSE, upload routes
    config.py               # model id, budget, flags
    llm/client.py           # google-genai wrapper, token counting, ledger
    orchestrator/
      planner.py  scheduler.py  evaluator.py  context_manager.py
    agents/                 # sub-agent definitions + agent registry
    tools/                  # search, parse, download, pyexec, crawl, browser, bm25, gemini_search
    storage/                # sqlite schema + repositories, blob store
    retrieval/              # pageindex tree builder, bm25 index
  ui/                       # index.html, app.js (SSE client, plan graph)
  pyproject.toml
```

---

## 9. Milestone ordering (dependency-first)

1. Phases 1–2 (skeleton + tool loop) — foundation, nothing works without it.
2. Phase 3 (search/parse/download) — first useful vertical slice.
3. Phases 7–9 (plan → sub-agents → evaluator) — the "harness" itself.
4. Phases 4, 5, 6, 10 (retrieval, web tools, code exec, multimodal) — parallelizable.
5. Phases 11–13 (UI, force-stop, hardening) — polish + control.

Ship a working single-agent + search/parse slice end-to-end (Phases 1–3) before adding
the multi-agent orchestration, so the tool contract and context strategy are proven
early.

---

## References (verify current versions before coding each phase)
- Gemini models & pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gen AI Python SDK: https://googleapis.github.io/python-genai/ · https://pypi.org/project/google-genai/
- Function calling: https://ai.google.dev/gemini-api/docs/function-calling
- Grounding (Google Search): https://ai.google.dev/gemini-api/docs/grounding
- MarkItDown: https://github.com/microsoft/markitdown
- ddgs: https://pypi.org/project/ddgs/
- Crawl4AI: https://docs.crawl4ai.com/
- browser-use: https://docs.browser-use.com/ · https://github.com/browser-use/browser-use
- PageIndex: https://github.com/VectifyAI/PageIndex
- rank-bm25: https://pypi.org/project/rank-bm25/
- NetworkX: https://networkx.org/documentation/stable/
