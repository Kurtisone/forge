"""
Context compaction (v3.9): keeps forge.memory's rolling history within
a manageable size without simply dropping the oldest exchanges. v3.8's
sliding-window FIFO (MEMORY_MAX_HISTORY) fights KV-cache reuse once it
kicks in -- see the comment on MEMORY_MAX_HISTORY in config.py -- and
throws away context outright besides.

Two responsibilities are deliberately kept separate:
  - WHEN to compact: maybe_compact() checks COMPACTION_THRESHOLD.
  - HOW to compact: a strategy function decides what happens to the
    evicted messages.

Both strategies share the same signature (list[dict] messages -> a
single summary dict), so COMPACTION_STRATEGY is a one-line config
change, not a rewrite. Pinned messages (the "tiroir") are always
excluded from what gets compacted -- see forge.memory for how
"pinned" is set.
"""

from forge import rag
from forge.config import (
    COMPACTION_ENABLED,
    COMPACTION_KEEP_RECENT,
    COMPACTION_STRATEGY,
    COMPACTION_THRESHOLD,
)
from forge.logger import log


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
    if not force and len(history) <= COMPACTION_THRESHOLD:
        return history

    pinned = [m for m in history if m.get("pinned")]
    unpinned = [m for m in history if not m.get("pinned")]

    if len(unpinned) <= COMPACTION_KEEP_RECENT:
        # Nothing eligible to compact -- e.g. pinned messages alone
        # pushed len(history) past the threshold.
        return history

    split = len(unpinned) - COMPACTION_KEEP_RECENT
    to_compact, to_keep = unpinned[:split], unpinned[split:]

    summary_message = _run_strategy(to_compact)
    log.event(
        "compaction.run",
        strategy=COMPACTION_STRATEGY,
        compacted=len(to_compact),
        kept=len(to_keep),
        pinned=len(pinned),
    )

    # Pinned messages are the "tiroir": kept as a distinct block ahead
    # of the summary rather than re-threaded back into strict
    # chronological order, which the id field still lets you recover
    # if needed.
    return [*pinned, summary_message, *to_keep]


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
        summary = call_llm(prompt)
    except ProviderError as e:
        log.error("compaction: llm_summary strategy failed: %s", e)
        raise CompactionError(str(e)) from e

    return {
        "role": "system",
        "content": f"[Résumé de {len(messages)} messages précédents] {summary}",
        "pinned": False,
    }
