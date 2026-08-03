"""
Forge research graph: search -> fetch top N results -> synthesize.

Deliberately a deterministic sequence, NOT a router-driven multi-step
chain. This exists because of a real, repeated failure observed live:
after a plain web_search call, the router was asked (via a
step_context steering hint) to decide the next step itself --
answer from the snippets, or fetch a specific result for more detail.
Two different hint designs were tried (a prose instruction, then an
explicit worked JSON example) and both failed the same way: the
model just repeated the identical web_search call instead of
following either option, tripping the loop guard. Disabling
LLAMA_CPP_CACHE_PROMPT and reproducing the exact same failure ruled
out a KV-cache bug (the same architectural issue already root-caused
for this model in v3.8) -- this is a genuine limit of this model
class at "recognize you already did X, do something else now," not a
fixable prompt or infra problem.

The fix is architectural, not another prompt rewrite: remove the
decision from the router's hands entirely. This graph runs
search -> fetch -> synthesize as one fixed sequence every time, the
same pattern already proven for graphs/review.py
(read_file -> run_tests -> llm_review) -- the router only ever makes
ONE decision (call "research"), never a mid-flow judgment call about
what a partial result means.

Nodes:
  search_node      -- calls web_search.search(), stores raw results
  fetch_node        -- fetches the top RESEARCH_FETCH_TOP_N result URLs
                        via web_fetch.run(), each capped at
                        RESEARCH_FETCH_CHARS_PER_RESULT chars; a
                        failed individual fetch is skipped, not fatal
  synthesize_node   -- single LLM call combining the query, search
                        snippets, and fetched excerpts into one
                        answer

Edges:
  search_node    -> fetch_node       (if results found)
  search_node    -> error            (search failed or no results)
  fetch_node     -> synthesize_node  (always -- a fetch failure on
                                      one URL doesn't block synthesis
                                      from the remaining sources)

Usage (Python):
  from forge.graphs.research import run
  print(run("actualités jeu vidéo"))
"""

from forge.config import RESEARCH_FETCH_CHARS_PER_RESULT, RESEARCH_FETCH_TOP_N
from forge.context_info import today_line
from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.text_cleaning import strip_think_blocks, try_unwrap_router_json
from forge.tools import web_fetch, web_search
from forge.types import AgentState

_MAX_SYNTHESIS_OUTPUT_CHARS = 4000

# Same reasoning as graphs/review.py: this model needs the exact JSON
# shape it must NOT produce shown explicitly, a bare "no JSON"
# instruction was proven insufficient there.
_PROMPT_LEAK_MARKERS = [
    "Respond in plain text",
    "GOOD ANSWER:",
    "NEVER DO THIS",
]

_SYNTHESIS_PROMPT = """/no_think
{today_line}
You are answering a question using web search results and fetched
page content gathered for you below. Write a clear, natural answer
in plain text -- summarize and synthesize, don't just repeat the
raw material. Write in the same language as the question. Cite which
source a specific claim comes from only if it matters; otherwise just
answer naturally. Use today's date above to judge what "recent" or
"upcoming" means -- don't assume the search results are from your own
training period.

Question: {query}

--- search results ---
{search_block}
--- end of search results ---

--- fetched page excerpts ---
{fetch_block}
--- end of fetched excerpts ---

Respond in plain text ONLY. Do NOT wrap your answer in JSON, and do
NOT return a {{"tool":...,"content":...}} object -- that format is
for a different system (a routing decision) and never applies here.

GOOD ANSWER: Plusieurs sorties majeures sont attendues cette année,
dont X et Y d'après les dernières actualités. Le marché reste
dynamique avec une hausse des ventes rapportée par plusieurs sources.
NEVER DO THIS: {{"tool":"chat","content":"..."}}

Now write your own answer to the question above, in the same plain
format as GOOD ANSWER -- not the NEVER DO THIS shape. Be concise.
"""


def _search_node(state: AgentState) -> AgentState:
    query = state.context.get("query", state.user_input.strip())
    try:
        results = web_search.search(query)
    except web_search.SearchError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] search failed: {e}"
        return state

    if not results:
        state.ok = False
        state.error = "no results"
        state.final_output = f"[no results] for query: {query!r}"
        return state

    state.context["results"] = results
    log.event("research.search", query=query[:120], results=len(results))
    return state


def _fetch_node(state: AgentState) -> AgentState:
    results = state.context.get("results", [])
    fetched = []
    for r in results[:RESEARCH_FETCH_TOP_N]:
        url = (r.get("url") or "").strip()
        if not url:
            continue
        output = web_fetch.run(url)
        if output.startswith("[error]"):
            log.warning("research.fetch: skipping %s (%s)", url, output)
            continue
        if len(output) > RESEARCH_FETCH_CHARS_PER_RESULT:
            output = output[:RESEARCH_FETCH_CHARS_PER_RESULT].rstrip() + "…"
        fetched.append({"url": url, "content": output})

    state.context["fetched"] = fetched
    log.event(
        "research.fetch", attempted=len(results[:RESEARCH_FETCH_TOP_N]), ok=len(fetched)
    )
    return state


def _clean_synthesis_response(raw: str) -> str:
    """Same reasoning as graphs/review.py's _clean_review_response
    (see forge/text_cleaning.py): this prompt asks for plain text, and
    reusing the router's JSON-first parser on it has already proven
    to misfire in practice for this exact class of prompt. This
    cleaner was originally written WITHOUT the conditional-unwrap step
    below -- an omission that let the exact same bug resurface live on
    research's very first real run (a fully substantive, multi-
    paragraph answer wrapped in {"tool":"chat","content":"..."} and
    shown to the user as raw JSON). Both callers now share one
    implementation specifically so this can't drift again."""
    cleaned = strip_think_blocks(raw)

    unwrapped = try_unwrap_router_json(cleaned, source="research")
    if unwrapped is not None:
        cleaned = unwrapped

    if any(marker in cleaned for marker in _PROMPT_LEAK_MARKERS):
        log.warning("research: model echoed prompt instructions instead of answering")
        return "[error] Le modèle n'a pas généré de réponse exploitable. Réessayez."

    if not cleaned:
        return "[error] Le modèle n'a pas généré de réponse. Réessayez."

    if len(cleaned) > _MAX_SYNTHESIS_OUTPUT_CHARS:
        cleaned = cleaned[:_MAX_SYNTHESIS_OUTPUT_CHARS].rstrip() + "…"

    return cleaned


def _synthesize_node(state: AgentState) -> AgentState:
    query = state.context.get("query", state.user_input.strip())
    results = state.context.get("results", [])
    fetched = state.context.get("fetched", [])

    search_lines = []
    for i, r in enumerate(results, 1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("content") or "").strip()
        search_lines.append(f"{i}. {title} -- {snippet}")
    search_block = "\n".join(search_lines) or "(no search snippets)"

    fetch_lines = []
    for f in fetched:
        fetch_lines.append(f"[{f['url']}]\n{f['content']}")
    fetch_block = "\n\n".join(fetch_lines) or "(no pages fetched successfully)"

    prompt = _SYNTHESIS_PROMPT.format(
        today_line=today_line(),
        query=query,
        search_block=search_block,
        fetch_block=fetch_block,
    )

    log.event("research.llm_call", query=query[:120], prompt_chars=len(prompt))
    try:
        raw = call_llm(prompt)
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    log.event("research.raw_output", raw=raw)
    state.final_output = _clean_synthesis_response(raw)
    state.final_tool = "research"
    log.event("research.done", chars=len(state.final_output))
    return state


def _error_node(state: AgentState) -> AgentState:
    log.warning("research graph: %s", state.error)
    state.ok = True  # surface as message, not crash
    return state


def build() -> Graph:
    g = Graph("research", max_steps=6)
    g.add_node("search", _search_node)
    g.add_node("fetch", _fetch_node)
    g.add_node("synthesize", _synthesize_node)
    g.add_node("error", _error_node)

    g.add_edge("search", "fetch", condition=lambda s: s.ok)
    g.add_edge("search", "error", condition=lambda s: not s.ok)
    g.add_edge("fetch", "synthesize")  # always -- see module docstring

    return g


def run(query: str) -> str:
    """Search, fetch the top results, and synthesize one answer."""
    state = build().run(query, initial_context={"query": query})
    return state.final_output or ""
