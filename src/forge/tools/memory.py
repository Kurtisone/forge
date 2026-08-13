"""
Autonomous memory tool, dispatchable by the router (v3.7).

Same forge.rag backend as the REPL commands (!remember/!recall) and
the HTTP endpoints (/remember, /search) -- this is a third entry
point into the same storage, called in-process. The difference is
who initiates it: here it's the model itself, deciding from the
conversation that something is worth storing or worth looking up,
rather than a human typing an explicit command.

Interface (consistent with all Forge tools):
    run(content: str) -> str

content is a JSON instruction:
    {"action": "remember", "kind": "decision"|"todo"|"fact", "content": "...", "project": "..."}
    {"action": "recall", "query": "...", "top_k": 5, "kind": "...", "project": "..."}

"project" is always optional. "top_k"/"kind"/"project" on recall are
also optional (top_k defaults to 5). "kind" on remember defaults to
"fact" if omitted or empty -- a casual mention ("I have a Steam Deck")
isn't a decision or a todo, and a small local model asked to route a
plain statement won't reliably invent a kind for it either.

Recall output is formatted for a prompt, not for a log: entries are
ranked so deliberately-recorded ones come before archived transcript,
and each is clipped to MEMORY_RECALL_MAX_CHARS. What this tool returns
is pasted straight into the next routing decision, so its size is paid
for twice -- in context window and in prefill time.

The router no longer dispatches a bare "recall" action itself for a
natural-language question: chaining recall into a synthesis step via
"done": false reliably failed live with the local model, the exact
same failure already root-caused for web_search (see
graphs/research.py's docstring) -- a repeated identical call instead
of following the steering hint. graphs/recall.py (dispatchable as the
"recall" tool) now runs recall -> synthesize as one deterministic
sequence instead. This module's "recall" action stays available
directly -- graphs/recall.py calls search() below rather than
duplicating the RAG query, and a raw bullet list is still occasionally
the right answer (e.g. from Python/tests, or a future non-chat caller
that wants the list itself, not a sentence about it).

To activate this tool add it to ENABLED_TOOLS in .env.local:
    ENABLED_TOOLS=chat,code,memory
"""

import json

from forge import rag
from forge.config import MEMORY_RECALL_MAX_CHARS
from forge.logger import log

_VALID_KINDS = ("decision", "todo", "fact")

# Kinds this tool can retrieve but never writes: compaction.py stores
# evicted history under "history_summary" by calling rag.remember()
# directly, bypassing _VALID_KINDS. Recall has to know about it anyway
# -- it comes back in search results and dominates them by sheer size.
_ARCHIVE_KINDS = ("history_summary",)


def _remember(instruction: dict) -> str:
    kind = instruction.get("kind", "").strip().lower() or "fact"
    text = instruction.get("content", "").strip()
    project = instruction.get("project") or None

    if kind not in _VALID_KINDS:
        return "[error] 'remember' requires kind to be 'decision', 'todo', or 'fact'"
    if not text:
        return "[error] 'remember' requires a non-empty 'content' field"

    conn = rag.get_connection()
    try:
        try:
            entry_id = rag.remember(conn, kind=kind, content=text, project=project)
        except rag.EmbeddingError as e:
            log.error("memory tool: remember failed: %s", e)
            return f"[error] remember failed: embedding server unreachable ({e})"
    finally:
        conn.close()

    log.event("memory.remember", entry_id=entry_id, kind=kind, project=project)
    return f"Remembered (#{entry_id})."


def search(
    query: str,
    top_k: int = 5,
    kind: str | None = None,
    project: str | None = None,
) -> list[dict]:
    """
    Query the RAG store and return raw hits as a list of
    {"kind", "content", "project", ...} dicts, ranked and clipped by
    nothing -- that formatting belongs to a caller. Raises
    rag.EmbeddingError on failure. This is the structured form used by
    graphs/recall.py; _recall() below formats the same data as a
    display string for direct chat/router dispatch (same split as
    web_search.search() / run()).
    """
    conn = rag.get_connection()
    try:
        return rag.search(conn, query=query, top_k=top_k, kind=kind, project=project)
    finally:
        conn.close()


def format_results(results: list[dict]) -> str:
    """
    Format raw search() hits as the ranked, clipped bullet list this
    tool has always returned for a direct "recall" action -- and that
    graphs/recall.py also feeds into its synthesis prompt, so the two
    callers can't drift into two different notions of what a memory
    hit looks like.
    """
    lines = []
    for r in _rank(results):
        proj = f"/{r['project']}" if r["project"] else ""
        lines.append(f"- [{r['kind']}{proj}] {_clip(r['content'])}")
    return "\n".join(lines)


def _recall(instruction: dict) -> str:
    query = instruction.get("query", "").strip()
    if not query:
        return "[error] 'recall' requires a non-empty 'query' field"

    top_k = instruction.get("top_k", 5)
    kind = instruction.get("kind") or None
    project = instruction.get("project") or None

    try:
        results = search(query, top_k=top_k, kind=kind, project=project)
    except rag.EmbeddingError as e:
        log.error("memory tool: recall failed: %s", e)
        return f"[error] recall failed: embedding server unreachable ({e})"

    log.event("memory.recall", query=query, hits=len(results))

    if not results:
        return "No matching memory found."

    return format_results(results)


def _rank(results: list[dict]) -> list[dict]:
    """
    Put deliberately-recorded entries ahead of archived transcript.

    rag.search() orders purely by vector distance, which treats a
    one-line fact and a whole compacted conversation as equal
    candidates. They aren't: a "fact" was written because someone
    decided it was worth keeping, while a "history_summary" is bulk
    archive that compaction dumped in verbatim. Asked "list my
    hardware", the search returned both and the two useful lines
    landed at the bottom of several thousand characters of unrelated
    chat.

    This is a stable sort, so distance order still decides within each
    group -- only the two groups are separated.
    """
    return sorted(results, key=lambda r: r["kind"] in _ARCHIVE_KINDS)


def _clip(content: str) -> str:
    if len(content) <= MEMORY_RECALL_MAX_CHARS:
        return content
    return content[:MEMORY_RECALL_MAX_CHARS].rstrip() + " […]"


def run(content: str) -> str:
    """
    Execute a memory operation described by a JSON instruction.

    Expected shapes:
        {"action": "remember", "kind": "decision", "content": "...", "project": "forge"}
        {"action": "recall", "query": "...", "top_k": 5}
    """
    try:
        instruction = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return f"[error] memory tool expects JSON, got: {content[:80]!r}"

    action = instruction.get("action", "").strip().lower()

    if action == "remember":
        return _remember(instruction)
    if action == "recall":
        return _recall(instruction)
    return f"[error] unknown action {action!r} (use: remember / recall)"
