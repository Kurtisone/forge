"""
Forge recall graph: search memory -> synthesize one natural answer.

Deliberately a deterministic sequence, NOT a router-driven multi-step
chain -- the exact same fix already applied to web_search, for the
exact same observed failure. Before this graph existed, a "recall"
question routed through tools/memory.py's "recall" action with
"done": false, and the router prompt (router/prompt.py) carried a
steering hint asking it to answer from the result on the next step
instead of calling memory again. In real usage that hint was
unreliable in two different ways, in order:

  1. The model just repeated the identical memory:recall call instead
     of switching to chat -- tripping the loop guard. This is the same
     model-class limit already root-caused for web_search in v3.10
     (confirmed there with LLAMA_CPP_CACHE_PROMPT=false, ruling out a
     KV-cache bug): a genuine limit at "recognize you already did X,
     do something else now," not a fixable prompt or infra problem.
  2. When it stopped, orchestrator.py's loop guard has a memory-
     specific fallback (predating this graph) that returned the raw
     "- [kind] ..." bullet list as-is instead of surfacing an error --
     strictly better than an error, but still not an answer to
     "Tu peux me lister mon matériel ?".

The fix is the one already proven for research.py: remove the
decision from the router's hands. This graph runs
recall -> synthesize as one fixed sequence every time; the router
makes exactly ONE decision (call "recall"), never a mid-flow judgment
call about what a raw memory hit means.

Nodes:
  recall_node      -- calls tools/memory.search(), stores raw hits
  synthesize_node  -- single LLM call turning the (ranked, clipped)
                       hits into one natural-language sentence

Edges:
  recall_node -> synthesize_node  (if hits found)
  recall_node -> error            (recall failed or nothing found)

Usage (Python):
  from forge.graphs.recall import run
  print(run("Tu peux me lister mon matériel ?"))
"""

from forge import rag
from forge.config import RECALL_MAX_ANSWER_CHARS
from forge.errors import ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.text_cleaning import strip_think_blocks, try_unwrap_router_json
from forge.tools import memory as memory_tool
from forge.types import AgentState

# Same reasoning as graphs/review.py and graphs/research.py: this
# model needs the exact JSON shape it must NOT produce shown
# explicitly, a bare "no JSON" instruction was proven insufficient.
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
You are answering a question using entries retrieved from memory,
listed below. Write ONE short, natural sentence in plain text that
answers the question -- do not just copy the bullet list verbatim,
and don't pad it with anything the entries don't say. Write in the
same language as the question. If none of the entries actually answer
the question, say plainly that you don't have that information yet.

Question: {query}

--- memory entries ---
{entries_block}
--- end of memory entries ---

Respond in plain text ONLY. Do NOT wrap your answer in JSON, and do
NOT return a {{"tool":...,"content":...}} object -- that format is
for a different system (a routing decision) and never applies here.

GOOD ANSWER: Tu as un Steam Deck et un Dell R710.
NEVER DO THIS: {{"tool":"chat","content":"..."}}

Now write your own answer to the question above, in the same plain
format as GOOD ANSWER -- not the NEVER DO THIS shape. Be concise.
"""


def _recall_node(state: AgentState) -> AgentState:
    query = state.context.get("query", state.user_input.strip())
    try:
        results = memory_tool.search(query)
    except rag.EmbeddingError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] recall failed: {e}"
        return state

    if not results:
        state.ok = False
        state.error = "no results"
        state.final_output = f"[no memory] for query: {query!r}"
        return state

    state.context["results"] = results
    log.event("recall.search", query=query[:120], results=len(results))
    return state


def _clean_synthesis_response(raw: str) -> str:
    """Same reasoning as research.py's _clean_synthesis_response and
    review.py's _clean_review_response (see forge/text_cleaning.py):
    this prompt asks for plain text, and reusing the router's
    JSON-first parser on it has already proven to misfire in practice
    for this exact class of prompt."""
    cleaned = strip_think_blocks(raw)

    unwrapped = try_unwrap_router_json(cleaned, source="recall")
    if unwrapped is not None:
        cleaned = unwrapped

    if any(marker in cleaned for marker in _PROMPT_LEAK_MARKERS):
        log.warning("recall: model echoed prompt instructions instead of answering")
        return "[error] Le modèle n'a pas généré de réponse exploitable. Réessayez."

    if not cleaned:
        return "[error] Le modèle n'a pas généré de réponse. Réessayez."

    if len(cleaned) > RECALL_MAX_ANSWER_CHARS:
        cleaned = cleaned[:RECALL_MAX_ANSWER_CHARS].rstrip() + "…"

    return cleaned


def _synthesize_node(state: AgentState) -> AgentState:
    query = state.context.get("query", state.user_input.strip())
    results = state.context.get("results", [])

    entries_block = memory_tool.format_results(results)

    prompt = _SYNTHESIS_PROMPT.format(query=query, entries_block=entries_block)

    log.event("recall.llm_call", query=query[:120], prompt_chars=len(prompt))
    try:
        raw = call_llm(prompt)
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    log.event("recall.raw_output", raw=raw)
    state.final_output = _clean_synthesis_response(raw)
    state.final_tool = "recall"
    log.event("recall.done", chars=len(state.final_output))
    return state


def _error_node(state: AgentState) -> AgentState:
    log.warning("recall graph: %s", state.error)
    state.ok = True  # surface as message, not crash
    return state


def build() -> Graph:
    g = Graph("recall", max_steps=4)
    g.add_node("recall", _recall_node)
    g.add_node("synthesize", _synthesize_node)
    g.add_node("error", _error_node)

    g.add_edge("recall", "synthesize", condition=lambda s: s.ok)
    g.add_edge("recall", "error", condition=lambda s: not s.ok)

    return g


def run(query: str) -> str:
    """Search memory and synthesize one natural answer."""
    state = build().run(query, initial_context={"query": query})
    return state.final_output or ""
