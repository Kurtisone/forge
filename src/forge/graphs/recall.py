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

from forge import lang, prose_grammar, rag
from forge.config import ENFORCE_ANSWER_LANGUAGE, RECALL_MAX_ANSWER_CHARS
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

# _PROMPT_LEAK_MARKERS above catches the model echoing the instruction
# markers. This catches the harder failure: copying the GOOD ANSWER
# *content* verbatim, with no marker in sight, so it comes back looking
# exactly like a real answer. graphs/sysadmin.py hit this in production
# on 2026-08-11; recall hit it on 2026-08-16, surfaced by the
# /no_think experiment (the prefix stayed -- see the comment above the
# prompt), on a question ("quel port utilise le serveur ?") that had
# nothing to do with the example.
#
# Unlike sysadmin, the old example is NOT kept here as a permanent net.
# It named this box's real hardware, so a legitimate recall over
# entries about that hardware would reproduce it word for word and trip
# the check. A placeholder has to be fictional to be detectable, which
# is the point of the rewrite below.
# This prompt asks for ONE SHORT SENTENCE, so the shared unwrap
# minimums (8 words / 40 chars, calibrated on review's multi-sentence
# syntheses) reject correct answers here: "Le serveur utilise le port
# 8080." is 6 words, 32 chars, and was reaching the user as raw JSON.
# Low enough to let a real short sentence through, high enough to still
# reject the degenerate echo actually observed -- the NEVER DO THIS
# example's own content, the three characters "...".
_MIN_UNWRAP_WORDS = 3
_MIN_UNWRAP_CHARS = 15

_EXAMPLE_LEAK_FRAGMENTS = [
    "exemple-hôte",
    "modèle-fictif",
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
and don't pad it with anything the entries don't say. If none of the
entries actually answer the question, say plainly that you don't have
that information yet.

Question: {query}

--- memory entries ---
{entries_block}
--- end of memory entries ---

Respond in plain text ONLY. Do NOT wrap your answer in JSON, and do
NOT return a {{"tool":...,"content":...}} object -- that format is
for a different system (a routing decision) and never applies here.

GOOD ANSWER (an example of FORM AND TONE only -- these names are
fictional placeholders, never real memory entries, and copying any of
them into your own answer is always wrong whatever the entries above
say): Tu as un serveur exemple-hôte et un onduleur modèle-fictif.
NEVER DO THIS: {{"tool":"chat","content":"..."}}

Now write your own answer using ONLY what actually appears in the
memory entries above -- the words "exemple-hôte" and "modèle-fictif"
must never appear in your answer. Same plain format as GOOD ANSWER,
not the NEVER DO THIS shape. Be concise.
{language_line}"""


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
    # What the RAG actually returned, not just how many.
    #
    # On 2026-08-19 a recall answer welded two unrelated memories into
    # one invented causality ("corriger le cache KV en réparant la
    # pagination du journal"). From the log of the day it was not
    # possible to tell which of two very different bugs that was: the
    # retrieval handing the synthesis two entries with nothing in
    # common, or the synthesis inventing a link between two entries
    # that were legitimately related. Those want opposite fixes.
    #
    # rag.search has always selected v.distance and nothing has ever
    # read it. It is the number that separates "the second hit was a
    # close match" from "the second hit was the least bad of five",
    # which is the question here.
    #
    # Observation only, deliberately. A distance cutoff is the obvious
    # next move and would be step three of Primitive -> Observable ->
    # Optimisable taken without step two: nobody can pick that
    # threshold from first principles, and picking it off no
    # measurement is how COMPACTION_THRESHOLD got its first value.
    log.event(
        "recall.search",
        query=query[:120],
        results=len(results),
        entries=[
            {
                "id": r.get("id"),
                "kind": r.get("kind"),
                "distance": round(d, 4)
                if isinstance(d := r.get("distance"), float)
                else d,
                "head": (r.get("content") or "")[:80],
            }
            for r in results
        ],
    )
    return state


def _clean_synthesis_response(raw: str) -> str:
    """Same reasoning as research.py's _clean_synthesis_response and
    review.py's _clean_review_response (see forge/text_cleaning.py):
    this prompt asks for plain text, and reusing the router's
    JSON-first parser on it has already proven to misfire in practice
    for this exact class of prompt."""
    cleaned = strip_think_blocks(raw)

    unwrapped = try_unwrap_router_json(
        cleaned,
        source="recall",
        min_words=_MIN_UNWRAP_WORDS,
        min_chars=_MIN_UNWRAP_CHARS,
    )
    if unwrapped is not None:
        cleaned = unwrapped

    if any(marker in cleaned for marker in _PROMPT_LEAK_MARKERS):
        log.warning("recall: model echoed prompt instructions instead of answering")
        return "[error] Le modèle n'a pas généré de réponse exploitable. Réessayez."

    if any(fragment in cleaned for fragment in _EXAMPLE_LEAK_FRAGMENTS):
        log.warning(
            "recall: model copied the GOOD ANSWER example verbatim "
            "instead of answering from the memory entries"
        )
        return "[error] Le modèle a recopié un exemple au lieu de répondre. Réessayez."

    if not cleaned:
        return "[error] Le modèle n'a pas généré de réponse. Réessayez."

    if len(cleaned) > RECALL_MAX_ANSWER_CHARS:
        cleaned = cleaned[:RECALL_MAX_ANSWER_CHARS].rstrip() + "…"

    return cleaned


def _build_prompt(query: str, entries_block: str, language_line: str = "") -> str:
    return _SYNTHESIS_PROMPT.format(
        query=query, entries_block=entries_block, language_line=language_line
    )


def _synthesize_node(state: AgentState) -> AgentState:
    query = state.context.get("query", state.user_input.strip())
    results = state.context.get("results", [])

    entries_block = memory_tool.format_results(results)

    # Wording half and deterministic half both live in forge.lang now:
    # review, research and sysadmin need the identical instruction, and
    # four copies of a string that has to stay identical is how a fix
    # drifts.
    language_line = lang.line_for(query)
    prompt = _build_prompt(query, entries_block, language_line)

    log.event(
        "recall.llm_call",
        query=query[:120],
        prompt_chars=len(prompt),
        language=(lang.name(lang.detect(query)) or "unknown"),
    )
    try:
        # Without a grammar, providers/llama_cpp._grammar_for() falls
        # back to the ROUTER's -- so the sampler admitted only
        # {"tool":...,"content":...} and the four paragraphs of this
        # prompt asking for plain text were asking for something the
        # decoder could not produce. See forge/prose_grammar.py.
        raw = call_llm(prompt, grammar=prose_grammar.SENTENCE)
        log.event("recall.raw_output", raw=raw)
        answer = _clean_synthesis_response(raw)

        answer = lang.enforce(
            query,
            answer,
            retry=lambda line: _clean_synthesis_response(
                call_llm(
                    _build_prompt(query, entries_block, line),
                    grammar=prose_grammar.SENTENCE,
                )
            ),
            enabled=ENFORCE_ANSWER_LANGUAGE,
        )
    except ProviderError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] LLM unavailable: {e}"
        return state

    state.final_output = answer
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
