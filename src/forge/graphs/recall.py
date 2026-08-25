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

from forge import expansion, lang, non_answer, outcome, rag, subtrace
from forge.config import (
    ENFORCE_ANSWER_LANGUAGE,
    RECALL_CALIBRATED_FOR,
    RECALL_EXPANSION,
    RECALL_LEXICAL,
    RECALL_LEXICAL_TOP_K,
    RECALL_MAX_ANSWER_CHARS,
)
from forge.config import RECALL_MAX_DISTANCE as _CONFIGURED_CUTOFF
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


def cutoff_for(
    configured: float | None, calibrated_for: str | None, current: str
) -> tuple[float | None, str | None]:
    """
    The cutoff to actually use, and what to say about it.

    A distance threshold is a measurement, not a constant: it is only
    valid for the embedding configuration it was measured against.
    Three cases, and the middle one is the reason this exists.

      no cutoff configured   nothing to check, nothing to say.

      tag matches            use it, silently.

      tag missing            use it, and warn. Values predate this
                             mechanism, and breaking a deployment that
                             works to make a point about provenance
                             would be its own kind of wrong. The
                             warning carries the fingerprint to write
                             back.

      tag does not match     do NOT use it. The number was measured in
                             another regime, so it is not too high or
                             too low, it is unrelated -- and of the two
                             ways to be wrong here, only one is
                             visible. A cutoff set too high lets a bad
                             answer through and a bad answer gets
                             argued with. Set too low it produces "je
                             n'ai rien en mémoire" while the answer is
                             sitting in the store, and that gets
                             believed. Off is the recoverable
                             direction.

    Measured on 2026-08-23, this is not hypothetical: .env.example
    shipped 0.95 from a run on raw queries while the default
    configuration prefixed every query with the embedding model's
    instruction. In the new regime the best miss came back at 0.9495 --
    under the cutoff, so the filter validated in real use no longer
    cut it, and nothing anywhere said so.
    """
    if configured is None:
        return None, None
    if calibrated_for is None:
        return configured, (
            f"RECALL_MAX_DISTANCE={configured} carries no calibration tag, so "
            "nothing can tell whether it was measured against the embedding "
            f"configuration now in force ({current}). If it was, write it as "
            f"RECALL_MAX_DISTANCE={configured}@{current} and this stops. If it "
            "was not, the cutoff is filtering on a number that means something "
            "else -- re-measure with bench/recall_distance.py."
        )
    if calibrated_for != current:
        return None, (
            f"RECALL_MAX_DISTANCE={configured} was calibrated against "
            f"{calibrated_for} and the embedding configuration is now "
            f"{current}: distances are not comparable across the two, so the "
            "cutoff is OFF and every hit is being passed to synthesis. "
            "Re-measure with bench/recall_distance.py and set "
            f"RECALL_MAX_DISTANCE=<new value>@{current}."
        )
    return configured, None


# Resolved once, at import, like every other configuration fact -- see
# the tripwires in tools/test.py for the same reasoning: a fact about
# the configuration belongs in the startup log, not repeated into
# every run. _drop_distant reads the module global below, which is
# also what the tests substitute.
RECALL_MAX_DISTANCE, _calibration_note = cutoff_for(
    _CONFIGURED_CUTOFF, RECALL_CALIBRATED_FOR, rag.query_fingerprint()
)
if _calibration_note:
    log.warning("%s", _calibration_note)

# Same class of statement as the calibration note above: a fact about
# the configuration, said once at startup rather than repeated into
# every run.
if RECALL_EXPANSION not in expansion.MODES:
    log.warning(
        "RECALL_EXPANSION=%r is not one of %s -- expansion is off",
        RECALL_EXPANSION,
        ", ".join(expansion.MODES),
    )
elif RECALL_EXPANSION != "off" and RECALL_MAX_DISTANCE is None:
    log.warning(
        "RECALL_EXPANSION=%s is set but no distance cutoff is in force, so it "
        "can never run: the rescue pass only fires when the cutoff drops "
        "everything, and with no cutoff nothing is ever dropped. Measure one "
        "with bench/recall_distance.py and set RECALL_MAX_DISTANCE, or expect "
        "this setting to do nothing at all.",
        RECALL_EXPANSION,
    )


# Same class as the two notes above: a fact about the configuration,
# said once at startup rather than rediscovered from a log.
if RECALL_LEXICAL and RECALL_EXPANSION != "off":
    log.warning(
        "RECALL_LEXICAL and RECALL_EXPANSION=%s are both on. The rescue pass "
        "only fires when the cutoff drops EVERYTHING, and a word match is "
        "never dropped by the cutoff -- so on any question the lexical "
        "channel answers, the expansion pass no longer runs. That is the "
        "cheaper order (no model call), but if you are measuring the "
        "expansion, you are no longer measuring it.",
        RECALL_EXPANSION,
    )


def _recall_node(state: AgentState) -> AgentState:
    query = state.context.get("query", state.user_input.strip())
    try:
        results = memory_tool.search(
            query, lexical=RECALL_LEXICAL, lexical_top_k=RECALL_LEXICAL_TOP_K
        )
    except rag.EmbeddingError as e:
        state.ok = False
        state.error = str(e)
        state.final_output = f"[error] recall failed: {e}"
        return state

    if not results:
        state.ok = False
        state.error = "no results"
        state.final_output = f"{non_answer.NO_MEMORY_PREFIX}for query: {query!r}"
        return state

    kept = _drop_distant(results, query)
    if not kept:
        kept = _rescue(query)
        state.context["expanded"] = bool(kept)
    if not kept:
        # Not an error: the store was reachable, it simply holds
        # nothing close enough to the question. Saying that is the
        # entire value of the cutoff -- the failure it replaces is a
        # fluent sentence built out of the five least-bad rows.
        state.ok = False
        state.error = "no results above the distance cutoff"
        state.final_output = non_answer.NOTHING_CLOSE_ENOUGH
        return state
    results = kept

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
                # Which channel admitted this row. Without it the line
                # below cannot be read at all once there are two: a row
                # with no distance is either a word match or a bug, and
                # a row above the cutoff that survived is either a
                # lexical hit or a filter that stopped working.
                "channel": r.get("channel", "vector"),
                "distance": round(d, 4)
                if isinstance(d := r.get("distance"), float)
                else d,
                "head": (r.get("content") or "")[:80],
            }
            for r in results
        ],
    )
    return state


def _rescue(query: str) -> list[dict]:
    """
    Ask the same question in other words, once the cutoff has dropped
    everything.

    HERE AND NOWHERE ELSE, and that placement is the design. Measured
    on the real store on 2026-08-24, both questions about the same
    hardware:

        "Quel processeur a mon NiPoGi ?"     #308 at rank 1, 0.7289
        "Tu peux me lister mon matériel ?"   nothing within the cutoff

    The second one is not a threshold problem -- the entry never came
    back to be filtered. So this runs on that path only, and three
    things follow from that.

    It is free on every question that already works: a recall whose
    first pass keeps something never reaches this function, never
    builds a variant, never spends the model call.

    It cannot make an answer worse. The alternative outcome on this
    path is "je n'ai rien d'assez proche" -- there is no good answer
    being displaced, only a refusal.

    And it changes no distance the threshold was calibrated against.
    The first pass is untouched, every variant goes through the same
    _as_query wrapper, and rag.search_many keeps real query-to-entry
    distances rather than a statistic over several. RECALL_MAX_DISTANCE
    stays valid, its @tag stays valid, and nothing here needs
    re-measuring before it can be used.

    What it CAN do is let a miss in: a rephrasing that happens to sit
    nearer some unrelated entry than the question did. That is the
    risk, it is not hypothetical, and it is why every rescue is logged
    with the id, the distance AND the variant that produced it --
    bench/recall_expansion.py counts them on a copy of the real store
    before this is worth turning on.

    IT DOES NOT SEARCH ARCHIVED CONVERSATION, and that is the whole
    difference between a rescue that works on this store and one that
    cannot be made safe. Measured 2026-08-24: rephrased as "processeur
    mémoire disque", the question about hardware put #307 -- the fact
    that answers it -- at 0.9435, and #167 at 0.8777. #167 is an
    archived exchange where someone asked to display a Containerfile
    and got the file back. It is long, it is dense with technical
    nouns, and it beats a one-line fact on almost any technical
    question. With it in scope there is no threshold that admits
    0.9435 and refuses 0.8777; the harness said so in those words.

    Archived transcript is not excluded because it is worthless -- the
    first pass still searches all of it. It is excluded HERE because
    this pass has already loosened the query, and loosening the query
    while keeping the noisiest half of the store in scope is what
    manufactures the intruder. A second, looser attempt gets the
    tighter corpus: whatever this returns, someone chose to write it
    down.
    """
    if RECALL_EXPANSION == "off":
        return []

    variants = expansion.variants(query, RECALL_EXPANSION)
    if not variants:
        return []

    log.event(
        "recall.expansion",
        mode=RECALL_EXPANSION,
        query=query[:120],
        variants=variants,
    )
    try:
        results = memory_tool.search_many(variants, exclude_kind=rag.ARCHIVED_KIND)
    except rag.EmbeddingError as e:
        # The first search reached the server, so this is a failure
        # between the two. Not worth an error the user reads: the
        # answer without it is the answer they were getting anyway.
        log.warning("recall: the rescue search failed (%s)", e)
        return []

    kept = _drop_distant(results, query)
    if kept:
        log.event(
            "recall.rescued",
            query=query[:120],
            kept=len(kept),
            entries=[
                {
                    "id": r.get("id"),
                    "distance": round(d, 4)
                    if isinstance(d := r.get("distance"), float)
                    else d,
                    "matched_query": r.get("matched_query"),
                }
                for r in kept
            ],
        )
    return kept


def _drop_distant(results: list[dict], query: str) -> list[dict]:
    """
    Drop hits further than RECALL_MAX_DISTANCE, if a cutoff is set.

    Inert unless configured, and inert as well when the configured
    value was calibrated against another embedding configuration --
    see cutoff_for above. See config.py for why the threshold has no
    default: every distance measured so far comes
    from a query with no good answer in the store, and a cutoff picked
    from negatives alone silences real hits at no visible cost. The
    mechanism ships now so that bench/recall_distance.py has something
    to calibrate; the number is a separate decision, taken from a
    measurement that includes a positive control.

    A row with no distance at all is KEPT. The value comes from
    rag.search's SELECT, and a filter that treats "not reported" as
    "too far" would empty the list on any caller that builds hits
    another way -- failing closed on retrieval means answering "I have
    nothing" while holding the answer.

    THIS IS THE VECTOR CHANNEL'S RULE AND ONLY ITS ROWS, since v3.17.
    RECALL_MAX_DISTANCE is a number measured against one embedding
    configuration, on distances produced by rag.search, and tagged
    with the fingerprint of that configuration. It says nothing
    whatsoever about a row the lexical channel admitted -- that row
    was chosen because the question and the entry share words rare
    enough to mean something, which is a different question with a
    different answer. Applying the cutoff to it would not be strict,
    it would be a category error, and it would delete exactly the
    entries the second channel exists to reach: #307 and #313 are far
    in vector space, which is why no rephrasing ever found them.

    So a row tagged `lexical` or `both` passes through untouched. A
    `both` row keeps its distance for the log and for ordering, and
    that distance being above the cutoff no longer removes it -- one
    channel admitting it is enough, which is what a union means.
    """
    if RECALL_MAX_DISTANCE is None:
        return results

    kept, dropped = [], []
    for r in results:
        d = r.get("distance")
        too_far = isinstance(d, (int, float)) and d > RECALL_MAX_DISTANCE
        judged_here = r.get("channel") in (None, "vector")
        (dropped if too_far and judged_here else kept).append(r)

    if dropped:
        log.event(
            "recall.dropped",
            query=query[:120],
            cutoff=RECALL_MAX_DISTANCE,
            kept=len(kept),
            dropped=[
                {"id": r.get("id"), "distance": r.get("distance")} for r in dropped
            ],
        )
    return kept


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
        # No grammar, so _grammar_for() supplies the ROUTER's -- and
        # that is deliberate. See the header of forge/prose_grammar.py:
        # sampling this prompt as free prose was measured on the Deck
        # and made recall return '<answer>' and nothing else.
        raw = call_llm(prompt)
        log.event("recall.raw_output", raw=raw)
        answer = _clean_synthesis_response(raw)

        answer = lang.enforce(
            query,
            answer,
            retry=lambda line: _clean_synthesis_response(
                call_llm(_build_prompt(query, entries_block, line))
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
    results = state.context.get("results", [])

    # EVERY recall, not just the failed ones. What this function
    # returns was rebuilt from entries the store already holds, so
    # indexing the exchange writes a second, worse copy of material
    # that is already in there -- worse because the copy carries the
    # QUESTION, and an entry containing the question outranks the
    # entry containing the answer for anyone who asks it again.
    #
    # Measured on the Deck on 2026-08-23, after #272 was forgotten:
    # "Tu peux me lister mon matériel ?" came back with #138 at rank 1,
    # 0.7891 -- and #138 is itself an archived recall, whose reply was
    # already partial the day it was written. Forge was reciting a
    # stale snapshot of itself. Left alone, every recall adds one.
    #
    # The cost, stated plainly: a good synthesised answer is not kept.
    # It is re-derivable from the entries it was built from, which are
    # still there. The stale copy is not worth the convenience.
    reason = (
        f"recall: {state.error}" if state.error else "recall: rebuilt from the store"
    )
    outcome.do_not_index(reason)
    subtrace.publish(
        subtrace.from_state(
            state,
            {
                "recall": lambda: (
                    f"{len(results)} entrée(s) retenue(s)"
                    + (" après reformulation" if state.context.get("expanded") else "")
                    if results
                    else "aucune entrée assez proche"
                ),
                "synthesize": lambda: (
                    f"réponse générée ({len(state.final_output or '')} caractères)"
                ),
            },
        )
    )
    return state.final_output or ""
