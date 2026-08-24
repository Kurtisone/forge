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

import difflib
import json
import re

from forge import rag
from forge.config import MEMORY_RECALL_MAX_CHARS
from forge.kernel.capability import LOCAL_READONLY
from forge.logger import log
from forge.tool_payload import loads_payload

# Writes to RAG_DB_FILE, not to WORKSPACE_DIR, and the embedding
# call goes to EMBEDDING_URL on the host -- neither is Internet
# access nor an LLM generation.
REQUIREMENTS = LOCAL_READONLY


_VALID_KINDS = ("decision", "todo", "fact")

# Kinds this tool can retrieve but never writes: compaction.py stores
# evicted history under "history_summary" by calling rag.remember()
# directly, bypassing _VALID_KINDS. Recall has to know about it anyway
# -- it comes back in search results and dominates them by sheer size.
_ARCHIVE_KINDS = ("history_summary",)

# How much of the store to read when building the vocabulary. Bounded
# because this runs on every write and the store grows without limit:
# 500 entries is well past what any spelling suggestion needs, and the
# words that matter here are recent by construction.
_VOCABULARY_ENTRIES = 500


# A fact stored as a comma-separated list of fragments is a fact
# written in the one shape this store retrieves WORST. Measured
# 2026-08-24: the deterministic keyword rewrites lost against the
# full question on four retrieval tests out of four, because the
# embedding model is instruction-tuned on natural language and a
# keyword bag is off-distribution for it. Then the router wrote three
# new facts in exactly that shape.
#
#     stored   NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, …
#     lost     "processeur", "Ryzen" -- the two words that made #307
#              findable by a question about a processor
#
# The signature is short fragments: the average comma-separated piece
# of a written fact runs three or four words, a telegram runs one or
# two.
_TERSE_FRAGMENT_WORDS = 2.5
_MIN_FRAGMENTS = 3


def _is_telegraphic(text: str) -> bool:
    """
    Does this read as a keyword list rather than a written fact?

    Deliberately crude, and deliberately advisory. A genuine
    enumeration -- a list of service names, a list of ports -- trips
    it too, and that is acceptable because nothing here refuses
    anything: losing a fact entirely is worse than storing a terse
    one. The value is that the degradation becomes VISIBLE at the
    moment it happens, instead of surfacing three weeks later in a
    bench run as a question that cannot be answered.
    """
    fragments = [f.strip() for f in text.split(",")]
    fragments = [f for f in fragments if f]
    if len(fragments) < _MIN_FRAGMENTS:
        return False
    average = sum(len(f.split()) for f in fragments) / len(fragments)
    return average < _TERSE_FRAGMENT_WORDS


_TERSE_NOTE = (
    "\n\n[note] Enregistré tel quel, mais rédigé comme une liste de "
    "mots-clés. Ce magasin retrouve mal cette forme : une phrase "
    "contenant les mots que tu emploierais pour la chercher (« processeur "
    "», « mémoire », « conteneurs ») est retrouvée là où une énumération "
    "ne l'est pas."
)


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
        except rag.DegenerateEntry as e:
            # Not a crash and not a silent drop: the model asked to
            # store something that asserts nothing, and the useful
            # response is to say what was wrong with it so the next
            # call carries the value that went missing.
            log.warning("memory tool: refused a degenerate entry: %s", e)
            return f"[error] {e}"
        except rag.EmbeddingError as e:
            log.error("memory tool: remember failed: %s", e)
            return f"[error] remember failed: embedding server unreachable ({e})"

        total = rag.count_entries(conn)["by_kind"].get(kind, 1)
        odd = _unfamiliar_words(conn, text, entry_id)
    finally:
        conn.close()

    terse = _is_telegraphic(text)
    log.event(
        "memory.remember",
        entry_id=entry_id,
        kind=kind,
        project=project,
        unfamiliar=len(odd),
        terse=terse,
    )
    if terse:
        log.warning(
            "memory tool: entry %s was written as a keyword list (%r) -- the "
            "shape this store retrieves worst. Stored anyway; a fact nobody "
            "can find still beats a fact nobody wrote.",
            entry_id,
            text[:80],
        )

    confirmation = _confirmation(entry_id, kind, text, project, total, odd)
    return confirmation + _TERSE_NOTE if terse else confirmation


def _confirmation(
    entry_id: int,
    kind: str,
    text: str,
    project: str | None,
    total: int,
    odd: list[tuple[str, str]],
) -> str:
    """
    Say what was stored, not that something was.

    "Remembered (#305)." is a receipt for a transaction nobody can
    check. The entry that provoked this went in as "NiPoGi AM06PRO,
    pocresseur 5500U, 32Go de RAM" and the typo was only found days
    later, by reading the store with a debugging tool. Echoing the
    stored text puts it in front of the person who wrote it while
    !forget is still one line away.

    Same lesson as files:write, which used to answer with a byte count
    until a created file had to be opened by hand to see what was in
    it (v3.11). A write that reports only that it happened hides what
    happened.
    """
    where = f"/{project}" if project else ""
    lines = [f"Noté (#{entry_id}, {kind}{where}) :", f"  {text}"]
    lines.append(
        f"\n{total} entrée{'s' if total > 1 else ''} de type {kind} en mémoire."
    )
    if odd:
        lines.append(_spelling_note(odd))
        lines.append(f"Si c'est une faute : `!forget {entry_id}` puis réécris-la.")
    return "\n".join(lines)


# A word shorter than this is not worth checking: "SSD", "RAM", "Go",
# "PC" are the vocabulary, not the typos, and difflib on three letters
# matches almost anything.
_MIN_WORD = 5

# How close a word has to be to an existing one to be worth
# mentioning. 0.85 on difflib's ratio is roughly "one or two
# characters out of eight" -- deliberately tight, because the cost of
# a false positive is a distracting line in every confirmation, and
# this feature is only ever a suggestion.
_CLOSE_ENOUGH = 0.85

_WORD_RE = re.compile(rf"[^\W\d_]{{{_MIN_WORD},}}", re.UNICODE)


def _unfamiliar_words(conn, text: str, entry_id: int) -> list[tuple[str, str]]:
    """
    Words in `text` that appear nowhere else in the store but sit one
    or two characters from a word that does.

    The dictionary is THE STORE ITSELF, and that is the whole design.
    A French spellchecker on this corpus is a machine for breaking
    identifiers: NiPoGi, sqlite-vec, busctl, aardvark-dns, GBNF are
    precisely the tokens that carry the information, and a general
    dictionary corrects them towards common words. Vocabulary drawn
    from what has already been written knows those words because they
    were already used, and it gets sharper with every entry instead of
    needing a maintained allow-list.

    It only ever SUGGESTS. Silently rewriting a memory entry is the
    one place in Forge where being approximately right is worse than
    being wrong -- nobody re-reads an entry, so it comes back weeks
    later as a fact with no trace that it was altered. Everywhere else
    a mistake is visible: a bad file, a red test, a diagnosis the logs
    contradict.

    Returns pairs of (written, closest word already in the store).
    """
    words = {w.lower() for w in _WORD_RE.findall(text)}
    if not words:
        return []

    # The entry being confirmed is already committed by the time this
    # runs, so it has to be excluded by id -- otherwise every word in
    # it is "already in the store", which it is, because we just put it
    # there.
    known: set[str] = set()
    for entry in rag.list_entries(conn, limit=_VOCABULARY_ENTRIES):
        if entry["id"] == entry_id:
            continue
        known.update(w.lower() for w in _WORD_RE.findall(entry["content"]))

    found = []
    for word in sorted(words):
        # A word the store already uses is familiar, full stop, and no
        # neighbour of it is worth mentioning. The first version
        # subtracted the whole new text from the vocabulary before
        # searching -- meant to stop a word matching itself, it also
        # deleted the evidence that the word was fine. "J'utilise
        # aardvark-dns pour la résolution DNS" was flagged twice on
        # `utilise` and `résolution`, two words written a dozen times
        # in that store, each matched against a near neighbour only
        # because the exact hit had just been removed.
        if word in known:
            continue
        near = difflib.get_close_matches(word, known, n=1, cutoff=_CLOSE_ENOUGH)
        if near:
            found.append((word, near[0]))
    return found


def _spelling_note(odd: list[tuple[str, str]]) -> str:
    pairs = ", ".join(f"« {written} » (proche de « {near} »)" for written, near in odd)
    return f"\nJamais vu ailleurs en mémoire : {pairs}."


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


def search_many(
    queries: list[str],
    top_k: int = 5,
    kind: str | None = None,
    project: str | None = None,
    exclude_kind: str | None = None,
) -> list[dict]:
    """
    search(), asked several ways at once -- see rag.search_many for
    what the merge keeps and why.

    Here for the same reason search() is: graphs/recall.py goes
    through this module rather than reaching into forge.rag itself, so
    that both ways of reading the store cross the same boundary. One
    connection for the whole batch, not one per query.
    """
    conn = rag.get_connection()
    try:
        return rag.search_many(
            conn,
            queries=queries,
            top_k=top_k,
            kind=kind,
            project=project,
            exclude_kind=exclude_kind,
        )
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
        instruction = loads_payload(content, "memory")
    except (json.JSONDecodeError, TypeError):
        return f"[error] memory tool expects JSON, got: {content[:80]!r}"

    # Same guard files/review/sysadmin already had, and the reason
    # they had it: valid JSON is not necessarily an object. A bare
    # '"recall"' or a list parses fine, then .get() raises
    # AttributeError out of a tool that is supposed to return its
    # errors as text.
    if not isinstance(instruction, dict):
        return f"[error] memory tool expects a JSON object, got: {content[:80]!r}"

    action = instruction.get("action", "").strip().lower()

    if action == "remember":
        return _remember(instruction)
    if action == "recall":
        return _recall(instruction)
    return f"[error] unknown action {action!r} (use: remember / recall)"
