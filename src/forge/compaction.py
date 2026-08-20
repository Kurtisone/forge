"""
Context compaction (v3.9): keeps forge.memory's rolling history within
a manageable size without simply dropping the oldest exchanges. v3.8's
sliding-window FIFO (MEMORY_MAX_HISTORY) fights KV-cache reuse once it
kicks in -- see the comment on MEMORY_MAX_HISTORY in config.py -- and
throws away context outright besides.

Two responsibilities are deliberately kept separate:
  - WHEN to compact: maybe_compact() checks COMPACTION_THRESHOLD and
    COMPACTION_TOKEN_THRESHOLD.
  - HOW to compact: a strategy function decides what happens to the
    evicted messages.

Both strategies share the same signature (list[dict] messages -> a
single summary dict), so COMPACTION_STRATEGY is a one-line config
change, not a rewrite. Pinned messages (the "tiroir") are always
excluded from what gets compacted -- see forge.memory for how
"pinned" is set.

Sizing is asked of router/prompt.py rather than computed here, which
puts a low-level module's import above a high-level one. That is on
purpose: what compaction BUYS is measured in the rendered prompt, and
the renderer is the only thing that knows the truncation rules. A local
copy of that arithmetic would be a second definition of the same fact,
free to drift from the one that matters.
"""

from forge import prose_grammar, rag
from forge.config import (
    COMPACTION_ENABLED,
    COMPACTION_KEEP_RECENT,
    COMPACTION_STRATEGY,
    COMPACTION_THRESHOLD,
    COMPACTION_TOKEN_TARGET,
    COMPACTION_TOKEN_THRESHOLD,
)
from forge.logger import log
from forge.router.prompt import estimate_history_tokens


class CompactionError(Exception):
    """Raised when the active strategy fails outright (never swallowed silently)."""


def maybe_compact(history: list[dict], force: bool = False) -> list[dict]:
    """
    Called after every turn is persisted (see forge.memory._apply_retention),
    and directly for a manual/forced compaction (e.g. an API or REPL
    command). Returns history unchanged if compaction is disabled, not
    yet needed, or nothing is eligible; otherwise returns a new,
    shorter history list with the oldest non-pinned messages replaced
    by a single summary message.
    """
    if not COMPACTION_ENABLED and not force:
        return history

    rendered = estimate_history_tokens(history)
    over_messages = len(history) > COMPACTION_THRESHOLD
    over_tokens = rendered > COMPACTION_TOKEN_THRESHOLD
    if not force and not over_messages and not over_tokens:
        return history

    pinned = [m for m in history if m.get("pinned")]
    unpinned = [m for m in history if not m.get("pinned")]

    # How many messages this pass will evict, decided independently by
    # each trigger and then taken at the larger of the two -- a pass
    # that satisfies one trigger while leaving the other still over is
    # a pass that runs again next turn.
    split = 0
    if over_messages or force:
        split = max(split, len(unpinned) - COMPACTION_KEEP_RECENT)
    if over_tokens:
        split = max(split, _split_for_token_target(pinned, unpinned))

    if split <= 0:
        # Nothing eligible to compact -- e.g. pinned messages alone
        # pushed the history past a threshold.
        return history

    to_compact, to_keep = unpinned[:split], unpinned[split:]

    summary_message = _run_strategy(to_compact)
    compacted = [*pinned, summary_message, *to_keep]

    # Everything needed to answer "why did this fire?" from one line.
    #
    # This exists because a compaction was seen firing on 2026-08-19
    # with the message threshold nowhere near reached, and the log of
    # the day could not settle it: it named a trigger but neither the
    # message count nor the thresholds in force. The two likeliest
    # explanations -- the token trigger doing its job invisibly, or a
    # deployed .env holding a threshold nobody remembers setting --
    # look identical when the only numbers logged are the ones the code
    # computed.
    #
    # The second is not hypothetical here. LLAMA_CPP_CACHE_PROMPT sat
    # at false in the container for weeks, costing about 75% of every
    # run, precisely because the effective value was never printed
    # anywhere. Log what was measured AND what it was measured against.
    #
    # rendered_after says what the pass bought, which is the number
    # that predicts whether it runs again next turn -- landing just
    # under the threshold is the per-turn eviction MEMORY_HARD_CAP_SLACK
    # exists to record, and it has no symptom other than latency.
    triggers = [
        name
        for name, fired in (
            ("messages", over_messages),
            ("tokens", over_tokens),
            ("forced", force),
        )
        if fired
    ]
    log.event(
        "compaction.run",
        strategy=COMPACTION_STRATEGY,
        # A list, not the first match: a pass at 100 messages AND 7000
        # tokens used to be labelled "tokens", hiding the fact that the
        # count threshold had been crossed too.
        trigger="+".join(triggers) or "none",
        messages=len(history),
        rendered_tokens=rendered,
        rendered_after=estimate_history_tokens(compacted),
        compacted=len(to_compact),
        kept=len(to_keep),
        pinned=len(pinned),
        threshold_messages=COMPACTION_THRESHOLD,
        threshold_tokens=COMPACTION_TOKEN_THRESHOLD,
        target_tokens=COMPACTION_TOKEN_TARGET,
        keep_recent=COMPACTION_KEEP_RECENT,
    )

    # Pinned messages are the "tiroir": kept as a distinct block ahead
    # of the summary rather than re-threaded back into strict
    # chronological order.
    return compacted


def _split_for_token_target(pinned: list[dict], unpinned: list[dict]) -> int:
    """
    How many of the oldest unpinned messages must go for the rendered
    history to fit COMPACTION_TOKEN_TARGET. Zero when nothing needs to.

    Two things this does differently from the message-count path.

    It aims at the TARGET, not the threshold. KEEP_RECENT counts
    messages while the budget counts tokens, so one pass frees an
    amount nobody can predict -- twenty short exchanges free almost
    nothing. Landing just under the threshold means compacting again
    next turn, which is the per-turn eviction MEMORY_HARD_CAP_SLACK
    exists to record: it reintroduces the sliding window v3.8 removed
    and destroys KV-cache reuse, with no symptom other than latency.

    And it may go BELOW COMPACTION_KEEP_RECENT, which is a floor for a
    count-driven pass and cannot be one here. Twenty 4000-char pastes
    are twenty messages: a floor of twenty would forbid touching any of
    them, making the budget unreachable in exactly the case it exists
    for. One exchange is always kept, so the model is never left
    answering with no recent context at all.

    The search runs from the smallest eviction upward and stops at the
    first that fits, so a pass never compacts more than it must.
    Pinned messages count toward the total -- they are part of the
    rendered block, and the budget is about the block -- but are never
    evicted, so a pinned set alone over the target simply means the
    target is unreachable and the caller gets the largest split
    available rather than nothing.
    """
    keep_at_least = 2  # one exchange
    if estimate_history_tokens([*pinned, *unpinned]) <= COMPACTION_TOKEN_TARGET:
        return 0

    for split in range(1, max(1, len(unpinned) - keep_at_least) + 1):
        if (
            estimate_history_tokens([*pinned, *unpinned[split:]])
            <= COMPACTION_TOKEN_TARGET
        ):
            return split
    return max(0, len(unpinned) - keep_at_least)


def _run_strategy(messages: list[dict]) -> dict:
    if COMPACTION_STRATEGY == "llm_summary":
        return _strategy_llm_summary(messages)
    return _strategy_rag_pointer(messages)


def _strategy_rag_pointer(messages: list[dict]) -> dict:
    """
    Default strategy: push the compacted block into vector memory
    verbatim, as one 'history_summary' RAG entry (searchable later via
    !recall / /search), and replace it in the rolling history with a
    short pointer. Cheap -- no LLM call -- but only as faithful as
    what's already indexed, since nothing is reworded.
    """
    joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    try:
        conn = rag.get_connection()
        try:
            entry_id = rag.remember(
                conn, kind="history_summary", content=joined, project=None
            )
        finally:
            conn.close()
    except rag.EmbeddingError as e:
        log.error("compaction: embedding server unreachable: %s", e)
        raise CompactionError(str(e)) from e

    return {
        "id": messages[0]["id"],
        "role": "system",
        "content": (
            f"[{len(messages)} messages précédents compactés -- "
            f"voir mémoire vectorielle #{entry_id}, cherchable via !recall]"
        ),
        "pinned": False,
    }


def _strategy_llm_summary(messages: list[dict]) -> dict:
    """
    Alternative strategy: ask the chat LLM to condense the block into
    a short prose summary, kept inline in history (not only in RAG).
    More faithful than the pointer strategy, costs one LLM call per
    compaction.
    """
    from forge.errors import ProviderError
    from forge.llm import call_llm

    joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
    prompt = (
        "Résume cet échange en conservant les décisions prises, l'état "
        "du travail en cours, et les fichiers/projets mentionnés. "
        "Sois concis (5 à 8 lignes), en français, sans préambule.\n\n"
        f"{joined}"
    )

    try:
        # A summary is prose too. Under the router grammar this
        # strategy could only ever produce a routing decision, which
        # would then be pasted into the history as the compacted
        # block -- and COMPACTION_STRATEGY is one config line away.
        summary = call_llm(prompt, grammar=prose_grammar.PROSE)
    except ProviderError as e:
        log.error("compaction: llm_summary strategy failed: %s", e)
        raise CompactionError(str(e)) from e

    return {
        "id": messages[0]["id"],
        "role": "system",
        "content": f"[Résumé de {len(messages)} messages précédents] {summary}",
        "pinned": False,
    }
