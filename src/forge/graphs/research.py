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

import difflib
import re

from forge import lang, non_answer, subtrace, turn
from forge.config import (
    ENFORCE_ANSWER_LANGUAGE,
    RESEARCH_FETCH_CHARS_PER_RESULT,
    RESEARCH_FETCH_TOP_N,
)
from forge.context_info import today_line
from forge.errors import ProviderError
from forge.graph import Graph
from forge.graphs import sysadmin
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

# The "/no_think" prefix below is NOT dead, however dead it looks.
# Qwen3.5 dropped the /think soft switch, and the router GBNF grammar
# (applied to every call, not just routing) already makes a reasoning
# block impossible -- so on paper the token buys nothing. Measured on
# 2026-08-16 with bench/no_think_ab.py, removing it made this model
# return the GOOD ANSWER example below instead of a real answer, twice,
# deterministically. Whatever it does at position 0 is not what its
# name says. Run that harness before touching it.
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
        state.final_output = f"{non_answer.NO_RESULTS_PREFIX}for query: {query!r}"
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

    # Language named in LAST position, and only when forge.lang is
    # sure -- same treatment recall got in the v3.12 dettes batch, for
    # the same reason: this prompt body is English prose, and it pulls
    # a French answer toward English all on its own. Appended rather
    # than templated in, so "last" cannot drift as the template grows.
    language_line = lang.line_for(query)

    log.event("research.llm_call", query=query[:120], prompt_chars=len(prompt))
    try:
        # No grammar, so _grammar_for() supplies the ROUTER's. That is
        # deliberate -- see the header of forge/prose_grammar.py.
        raw = call_llm(prompt + language_line)
        log.event("research.raw_output", raw=raw)
        answer = _clean_synthesis_response(raw)
        # The deterministic half. Naming the language in the prompt is
        # still a wording fix, and wording fixes have lost seven times
        # on this codebase. The retry re-sends the same prompt with a
        # different final line, so the KV prefix survives and only the
        # tail is recomputed.
        answer = lang.enforce(
            query,
            answer,
            retry=lambda line: _clean_synthesis_response(call_llm(prompt + line)),
            enabled=ENFORCE_ANSWER_LANGUAGE,
        )
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    state.final_output = answer
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
    g.add_node("error", _error_node, answers=False)

    g.add_edge("search", "fetch", condition=lambda s: s.ok)
    g.add_edge("search", "error", condition=lambda s: not s.ok)
    g.add_edge("fetch", "synthesize")  # always -- see module docstring

    return g


#: Appended when the question named something that is running here.
#:
#: Same shape and the same reason as sysadmin's _RUNNING_FOOTER: a
#: fact established before the answer, held below it, where it cannot
#: be reinterpreted by whatever the model decided to write.
#: Asterisks and never underscores for emphasis. The web UI's
#: inlineMarkdown implements `**bold**` and `*em*` and nothing else --
#: measured by running that function, not by reading it -- so
#: `_like this_` reaches the screen with its underscores showing. It
#: had, in the local-container footer below, since that footer was
#: written.
#:
#: Adding an underscore rule to the renderer is the wrong repair here:
#: this assistant's answers are full of `file_path`, `MAX_STEPS` and
#: `RECALL_MAX_DISTANCE`, and emphasis on `_` would eat identifiers.
#:
#: Where the answer came from, appended in code.
#:
#: The synthesis prompt has always said "cite which source a specific
#: claim comes from only if it matters", which is a rule asked of the
#: model about a set this module can enumerate exactly -- the URLs it
#: opened are sitting in state.context. Thirteen times on this codebase
#: a rule written in a prompt has been followed most of the time and
#: silently broken the rest, and a citation is worse than most: a
#: plausible URL a model produced is indistinguishable from one it read,
#: and checking it costs the reader the trip. So the model is not asked.
#:
#: The distinction between the two lists is the whole point of having
#: two. RESEARCH_FETCH_TOP_N pages are actually opened and their text
#: goes into the prompt; every other search result contributes one
#: snippet and nothing else. Calling the second kind a source would
#: overstate what was read, and that is exactly the overstatement a
#: sources block is supposed to prevent.
_SOURCES_READ = "\n\n---\n*Sources lues :*\n"
_SOURCES_SNIPPET_ONLY = (
    "\n\n---\n*Aucune page n'a pu être ouverte. Réponse fondée sur les "
    "extraits de recherche renvoyés par :*\n"
)
_ALSO_SEEN = "\n*(+ {n} autre(s) résultat(s) vus en extrait seulement.)*"

#: A title long enough to wrap twice is a page title, not a label.
_MAX_TITLE_CHARS = 90


def _link(title: str, url: str) -> str:
    """
    One source line, in a markdown the web UI will actually render.

    inlineMarkdown parses a markdown link with a label that cannot
    contain a closing bracket and a URL that cannot contain a closing
    parenthesis, so a bracket in the title or a parenthesis in the URL
    silently produces a broken link rather than an error. Both are
    ordinary in the wild -- Wikipedia puts parentheses in paths -- so
    the bare URL is the fallback, which stays readable everywhere
    including the REPL and the CLI.
    """
    title = " ".join((title or "").split()).replace("[", "").replace("]", "")
    if len(title) > _MAX_TITLE_CHARS:
        title = title[:_MAX_TITLE_CHARS].rstrip() + "…"
    if not title or ")" in url:
        return url
    return f"[{title}]({url})"


def _sources_footer(results: list[dict], fetched: list[dict]) -> str:
    """
    The block naming what the answer was built from, or "" when there
    is nothing honest to say.
    """
    titles = {(r.get("url") or "").strip(): r.get("title") or "" for r in results}

    if fetched:
        lines = [
            f"{i}. {_link(titles.get(f['url'], ''), f['url'])}"
            for i, f in enumerate(fetched, 1)
        ]
        block = _SOURCES_READ + "\n".join(lines)
        # Counted against the pages READ, not against the ones the
        # graph tried to read: a fetch that failed contributed exactly
        # what an unvisited result did, which is its snippet.
        others = len(results) - len(fetched)
        if others > 0:
            block += _ALSO_SEEN.format(n=others)
        return block

    if results:
        shown = [
            r for r in results[:RESEARCH_FETCH_TOP_N] if (r.get("url") or "").strip()
        ]
        if not shown:
            return ""
        lines = [
            f"{i}. {_link(r.get('title') or '', r['url'].strip())}"
            for i, r in enumerate(shown, 1)
        ]
        block = _SOURCES_SNIPPET_ONLY + "\n".join(lines)
        others = len(results) - len(shown)
        if others > 0:
            block += _ALSO_SEEN.format(n=others)
        return block

    return ""


_LOCAL_FOOTER = (
    "\n\n---\n*À noter : `{container}` est un conteneur qui tourne sur cette "
    "machine. Cette réponse vient du web, qui n'en sait rien — demande-moi "
    "ses logs si c'était la question.*"
)

#: How close a word has to be to a container name to be called one.
#:
#: High, because the cost is asymmetric in an unusual direction here:
#: this footer never changes an answer, it adds a line under one, so a
#: near-miss costs a sentence and a miss costs nothing at all. 0.85
#: catches `searxn` for `searxng` -- the real question from
#: 2026-08-21 -- and not `qwen` for `forge`.
_NEAR = 0.85


def names_something_local(text: str, containers: list[str]) -> str | None:
    """
    The running container this question is about, if it is about one.

    WHY THIS IS HERE AT ALL. Both router prompts already carry the
    boundary, and research's carries it with this exact example:
    "A question about one of the user's own services or containers
    ('pourquoi searxng a redémarré') is 'sysadmin', never here."
    Measured over every routing in traces.jsonl on 2026-09-12: 18
    research calls, 2 of them local questions, and one of the two is
    that sentence almost verbatim -- `Pourquoi searxng a redémarré ?`,
    routed to the web. The other is `pourquoi searxn plante ?`, the
    same question with a typo. No sysadmin routing went the other way.

    WHAT THIS DOES NOT DO: it does not re-route. A question naming a
    container is usually about the container, but `forge` is also a
    French word and a project name, and hijacking a web question to
    read a container's logs would answer something nobody asked. So
    the correction is additive -- the web answer stands, with one line
    under it naming what is running here.
    """
    if not containers:
        return None
    words = {w for w in re.split(r"[^\w.-]+", text.lower()) if len(w) > 3}
    for container in containers:
        if container.lower() in words:
            return container
    for container in containers:
        if difflib.get_close_matches(container.lower(), words, n=1, cutoff=_NEAR):
            return container
    return None


def run(query: str) -> str:
    """Search, fetch the top results, and synthesize one answer."""
    state = build().run(query, initial_context={"query": query})
    results = state.context.get("results", [])
    fetched = state.context.get("fetched", [])
    subtrace.publish(
        subtrace.from_state(
            state,
            {
                "search": lambda: f"{len(results)} résultat(s) pour « {query} »",
                "fetch": lambda: f"{len(fetched)} page(s) récupérée(s)",
                "synthesize": lambda: (
                    f"synthèse générée ({len(state.final_output or '')} caractères)"
                ),
            },
        )
    )
    answer = state.final_output or ""
    # The turn rather than the query: the router's restatement is what
    # dropped the container name in the case this exists for.
    asked = turn.get_input() or query
    local = names_something_local(asked, sysadmin.running_containers())
    if local and answer:
        log.event("research.named_a_local_container", container=local)
        answer += _LOCAL_FOOTER.format(container=local)
    # Last, and only on an answer: a sources block under an
    # "[error] search failed" line would be citing sources for a
    # sentence that cites nothing. state.ok is not enough on its own --
    # _error_node sets it True on purpose, to surface the failure as a
    # message rather than a crash.
    if answer and not non_answer.is_non_answer(answer):
        sources = _sources_footer(results, fetched)
        if sources:
            log.event("research.sources", read=len(fetched), results=len(results))
            answer += sources
    return answer
