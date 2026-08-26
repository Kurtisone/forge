"""
Vector memory (RAG) for decisions and TODOs — v3.7.

Distinct from forge.memory (JSON rolling history + facts): this
module is a separate concern, backed by SQLite-vec, and talks to a
dedicated embedding-only llama.cpp instance (EMBEDDING_URL) instead
of the chat model at LLAMA_CPP_URL.

Storage: a single SQLite file (RAG_DB_FILE, default
data/forge_rag.db) with two tables --
  memory_entries : the actual decision/todo rows
  memory_vectors : a sqlite-vec vec0 virtual table, linked to
                   memory_entries by rowid (sqlite-vec's own
                   convention -- no foreign key is possible on a
                   virtual table).
"""

import hashlib
import math
import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import requests
import sqlite_vec

from forge.config import (
    EMBEDDING_DIM,
    EMBEDDING_MAX_CHARS,
    EMBEDDING_MAX_CHUNKS,
    EMBEDDING_QUERY_INSTRUCT,
    EMBEDDING_TIMEOUT,
    EMBEDDING_URL,
    RAG_DB_FILE,
    RECALL_LEXICAL_EXCLUDE_ARCHIVED,
    RECALL_LEXICAL_MAX_DF,
)
from forge.logger import log

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    project TEXT,
    created_at TEXT NOT NULL,
    status TEXT DEFAULT 'active'
);
"""

_VEC_SCHEMA = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
    embedding FLOAT[{EMBEDDING_DIM}]
);
"""

# --- Lexical index (v3.17) -------------------------------------------------
# An FTS5 index over the SAME rows, so a question can also be answered
# through the words it shares with an entry instead of only through
# the vectors.
#
# WHY A SECOND INDEX EXISTS AT ALL. Measured on the real store on
# 2026-08-24: #307 (the hardware fact) and #313 ("Steam Deck, SteamOS,
# conteneurs Podman") are both in there, and no natural-language
# question reaches either one through the vector channel -- not with
# the query instruction, not with any of the rephrasings the expansion
# pass produced over six rounds of measurement. An entry that is not
# written in natural language is not retrievable by a natural-language
# query; that is a property of an instruction-tuned embedding model,
# not a threshold someone set wrong. The words are sitting in the
# entry. Nothing was ever looking for them.
#
# EXTERNAL CONTENT, NOT A COPY. content='memory_entries' means the
# index stores terms and points back at the original rows, so there is
# exactly one copy of every entry's text and no way for the two to
# disagree about what an entry says.
#
# remove_diacritics 2 because "matériel" has to be findable by someone
# who types "materiel", and version 2 folds the whole Unicode range
# rather than the Latin-1 subset version 1 covers. Changing this line
# later means rebuilding the index: the terms already stored were
# folded by the old rule, and nothing will say so.
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memory_entries',
    content_rowid='id',
    tokenize="unicode61 remove_diacritics 2"
);
"""

# TRIGGERS AND NOT CALLS IN _insert()/forget(), deliberately. Four
# paths write this store (the memory tool, the REPL, POST /remember,
# compaction) and three delete from it (forget, deploy/rag_resplit.py,
# and a human with sqlite3 open -- which is how this store gets
# repaired today). A trigger covers all of them, and covers the next
# one nobody has written yet. The vector table is kept in sync by hand
# and that is exactly why memory_vectors has needed a comment since
# v3.7 warning that deleting one and not the other leaves a memory
# that is invisible and answering.
_FTS_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS memory_fts_insert
AFTER INSERT ON memory_entries BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS memory_fts_delete
AFTER DELETE ON memory_entries BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS memory_fts_update
AFTER UPDATE ON memory_entries BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.id, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
END;
"""


def _path() -> Path:
    return Path(RAG_DB_FILE)


def get_connection() -> sqlite3.Connection:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(_SCHEMA)
    conn.execute(_VEC_SCHEMA)
    conn.commit()
    _ensure_fts(conn)
    return conn


def _ensure_fts(conn: sqlite3.Connection) -> bool:
    """
    Create the lexical index if it is missing, backfill it once, and
    say whether this store has a usable lexical channel.

    THE BACKFILL IS THE MIGRATION, which is why this ships without a
    script. FTS5's 'rebuild' command reads memory_entries and indexes
    every row, so a store written before this existed becomes
    searchable on the first connection after the upgrade -- no
    deploy/ step to forget, no window during which half the store is
    findable and the other half is not.

    It runs only when the table was ABSENT, checked against
    sqlite_master before the CREATE. Afterwards there is no way to
    tell a table created a second ago from one that has been indexed
    for weeks, and rebuilding on every connection would re-index the
    whole store to open it.

    FTS5 IS AN OPTIONAL SQLITE MODULE and a Forge that cannot open its
    memory is worse than a Forge with one retrieval channel. A build
    without it gets a warning and the vector channel exactly as it
    was, rather than an exception on the path every single write and
    read goes through.

    sqlite_vec is already loaded by the caller, and that ordering is
    not incidental: any schema statement on a database carrying a vec0
    table needs the extension present, whatever the statement itself
    is about.
    """
    existed = (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
        ).fetchone()
        is not None
    )
    try:
        conn.execute(_FTS_SCHEMA)
        conn.executescript(_FTS_TRIGGERS)
        if not existed:
            conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            indexed = conn.execute("SELECT count(*) FROM memory_entries").fetchone()[0]
            if indexed:
                log.event("rag.fts_backfilled", entries=indexed)
        conn.commit()
    except sqlite3.OperationalError as e:
        log.warning(
            "rag: no lexical index on this store (%s) -- this SQLite build has "
            "no FTS5, so recall keeps the vector channel alone and any entry "
            "that is only reachable by its words stays unreachable",
            e,
        )
        return False
    return True


def has_lexical_index(conn: sqlite3.Connection) -> bool:
    """Whether the lexical channel can run against this connection."""
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_fts'"
        ).fetchone()
        is not None
    )


class EmbeddingError(Exception):
    """Raised when the embedding server is unreachable or returns bad data."""


def _split_for_embedding(text: str) -> list[str]:
    """
    Cut text into chunks of at most EMBEDDING_MAX_CHARS, preferring
    line boundaries so a compacted conversation splits between
    messages rather than mid-word. A single line longer than the limit
    is hard-split, since nothing better is available.
    """
    chunks: list[str] = []
    current = ""

    for line in text.splitlines(keepends=True):
        while len(line) > EMBEDDING_MAX_CHARS:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(line[:EMBEDDING_MAX_CHARS])
            line = line[EMBEDDING_MAX_CHARS:]
        if len(current) + len(line) > EMBEDDING_MAX_CHARS:
            chunks.append(current)
            current = line
        else:
            current += line

    if current:
        chunks.append(current)
    return chunks or [text]


def _embed_one(text: str) -> list[float]:
    try:
        resp = requests.post(
            EMBEDDING_URL, json={"input": text}, timeout=EMBEDDING_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()[0]["embedding"][0]
    except (requests.RequestException, KeyError, IndexError) as e:
        log.error("embedding request failed (%s): %s", EMBEDDING_URL, e)
        raise EmbeddingError(str(e)) from e


def _embed(text: str) -> list[float]:
    """
    Embed arbitrarily long text as a single vector.

    The naive version posted the whole string in one request, which
    worked for every caller except the one that mattered: compaction
    (compaction.py) embeds an entire block of evicted conversation at
    once. llama-server needs the full input in one physical batch, so
    past --ubatch-size it answers 400 Bad Request rather than
    truncating -- which made compaction fail permanently, fall back to
    the drop-oldest hard cap, and take KV-cache reuse down with it.
    Short queries (recall) kept working, so the failure looked like an
    unreachable server instead of an oversized request.

    Chunks are averaged and re-normalised to unit length. llama-server
    L2-normalises what it returns, so a single-chunk vector is already
    unit length; normalising the mean keeps long and short entries
    comparable under the same distance metric.
    """
    chunks = _split_for_embedding(text)

    if len(chunks) == 1:
        return _embed_one(chunks[0])

    if len(chunks) > EMBEDDING_MAX_CHUNKS:
        log.warning(
            "embedding input split into %d chunks, keeping the first %d "
            "(raise EMBEDDING_MAX_CHUNKS, or compact more often)",
            len(chunks),
            EMBEDDING_MAX_CHUNKS,
        )
        chunks = chunks[:EMBEDDING_MAX_CHUNKS]

    log.event("rag.embed_chunked", chunks=len(chunks), chars=len(text))
    vectors = [_embed_one(chunk) for chunk in chunks]

    mean = [sum(values) / len(vectors) for values in zip(*vectors, strict=True)]
    norm = math.sqrt(sum(v * v for v in mean))
    return [v / norm for v in mean] if norm else mean


# A stored entry has to assert something -- at minimum, a subject and
# something said about it.
#
# One word, deliberately, and not a character count. The observed
# failure is `[fact] S'appelle`: a predicate whose object went
# missing, which is exactly a one-word entry. Anything with two words
# has a shape that can be right, and short legitimate entries are
# real -- "use sqlite-vec" and "a todo" both live in this repository's
# own fixtures. A character floor would have rejected those for being
# terse rather than for being empty, which is a different and wrong
# complaint.
_MIN_ENTRY_WORDS = 2


class DegenerateEntry(ValueError):
    """Raised when content is too thin to be worth embedding."""


def forget(conn: sqlite3.Connection, entry_id: int) -> bool:
    """
    Delete one entry and its vector. True if it existed.

    Added the same day list_entries was, and for the same reason: once
    you can see that the store contains `[fact] S'appelle`, the next
    thing you need is to remove it. Until now the only way to correct
    this store was to open the SQLite file by hand -- and a store you
    can only fix by hand is one nobody fixes.

    Both tables, in one transaction. memory_vectors is keyed by rowid
    against memory_entries.id, so deleting one and not the other leaves
    a vector that search can still match and list_entries can no longer
    show -- a memory that is invisible and answering.
    """
    cur = conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
    conn.execute("DELETE FROM memory_vectors WHERE rowid = ?", (entry_id,))
    conn.commit()
    return cur.rowcount > 0


def remember(
    conn: sqlite3.Connection, kind: str, content: str, project: str | None
) -> int:
    # Checked HERE and not only in tools/memory.py, because this is the
    # boundary every writer crosses -- the tool, the REPL, /remember,
    # and compaction. tools/memory.py already refused empty content and
    # `S'appelle` is not empty; it is a predicate whose value went
    # missing, which reads as valid to every check that asks "is there
    # a string".
    #
    # It is not a cosmetic problem. That single entry is what closed
    # the gap in the 2026-08-22 calibration run: it pulled the
    # unanswerable "Comment s'appelle mon chat ?" to 0.9356, nearer
    # than the worst genuine hit at 0.9386, and a dangling verb is a
    # magnet for every question phrased around it.
    if len(content.split()) < _MIN_ENTRY_WORDS:
        raise DegenerateEntry(
            f"refusing to store {content!r}: an entry must assert something, "
            f"and this is a single word. If a value was meant to follow, "
            f"send the whole statement. A dangling fragment matches every "
            f"question phrased around it and answers none of them."
        )

    entry_id = _insert(conn, kind, content, project)
    conn.commit()
    return entry_id


def remember_many(
    conn: sqlite3.Connection, kind: str, contents: list[str], project: str | None
) -> list[int]:
    """
    Store several entries as several ROWS, in one transaction.

    The alternative -- what compaction did until now -- is to join the
    lot into one string and store it as a single entry. That entry then
    gets one vector, and since the joined text is far past
    EMBEDDING_MAX_CHARS, that vector is the AVERAGE of a dozen chunk
    vectors. A mean of a dozen unrelated subjects is close to no
    question in particular, which is the 0.27 of distance measured on
    2026-08-22 between a fact stored alone and the same content buried
    in a compacted block.

    It costs no extra embedding calls: _embed already made one request
    per chunk. The change is that the chunks are kept apart instead of
    being collapsed into their mean.

    A degenerate item is SKIPPED, not raised on. remember() raises
    because a caller asserting a one-word fact should hear about it;
    here the caller is archiving a block it did not write, and failing
    the whole compaction because one evicted message was a single word
    would leave the history uncompacted with no way to recover.

    All or nothing on the embedding server, deliberately: an
    EmbeddingError propagates before the commit, so a block is never
    half-indexed. A partially indexed block is worse than an unindexed
    one -- the pointer written into the history claims a range that
    does not hold what it says it holds.

    Exact duplicates are skipped, both against what is already stored
    and within the batch itself. Compaction blocks overlap -- the real
    store held the same exchange three times at distance 0.8306,
    taking three of the five slots a recall query gets. Nothing is
    lost by storing it once: the content is identical, so the
    surviving row answers every question the copies would have.

    Only here, not in remember(). A human asserting the same fact
    twice is saying something -- they think it was forgotten. An
    archive holding the same exchange twice is redundancy nobody
    chose.
    """
    ids: list[int] = []
    seen: set[str] = set()
    for content in contents:
        if len(content.split()) < _MIN_ENTRY_WORDS:
            log.warning("rag: skipping a degenerate entry in a batch: %r", content)
            continue
        if content in seen or _already_stored(conn, content, project):
            log.event("rag.duplicate_skipped", chars=len(content))
            continue
        seen.add(content)
        ids.append(_insert(conn, kind, content, project))

    conn.commit()
    return ids


def _already_stored(
    conn: sqlite3.Connection, content: str, project: str | None
) -> bool:
    """
    Exact match, within the same project. A near-duplicate is a
    judgement call with a threshold to tune; an identical string is a
    fact.

    Scoped to the project because that is the namespace: the same
    sentence filed under two projects is two statements about two
    things, and deduplicating across them would silently drop one.
    `IS` rather than `=` so a NULL project matches a NULL project,
    which is every entry compaction writes.
    """
    row = conn.execute(
        "SELECT 1 FROM memory_entries WHERE content = ? AND project IS ? LIMIT 1",
        (content, project),
    ).fetchone()
    return row is not None


def _insert(
    conn: sqlite3.Connection, kind: str, content: str, project: str | None
) -> int:
    """Write one row and its vector. Does NOT commit -- the caller owns
    the transaction, which is what lets remember_many be atomic."""
    cur = conn.execute(
        "INSERT INTO memory_entries (kind, content, project, created_at) VALUES (?, ?, ?, ?)",
        (kind, content, project, datetime.now(UTC).isoformat()),
    )
    entry_id = cur.lastrowid

    embedding = _embed(content)
    conn.execute(
        "INSERT INTO memory_vectors (rowid, embedding) VALUES (?, ?)",
        (entry_id, sqlite_vec.serialize_float32(embedding)),
    )
    return entry_id


def _as_query(text: str) -> str:
    """
    Wrap a search string in the embedding model's query instruction.

    Applied HERE and only here, which is the whole design. Documents
    reach the store through remember(), queries reach it through
    search(), and putting the wrapper at the query choke point makes
    the asymmetry structural rather than a rule someone has to
    remember. There is no call path that could accidentally embed a
    stored fact with a question's instruction glued to the front.

    It is also why this needed no migration: every vector already in
    the database was written raw by remember() and stays exactly as
    valid as it was.

    Empty EMBEDDING_QUERY_INSTRUCT disables it -- see config.py for
    the measurement, and for why a model that is not instruction-aware
    wants it off.
    """
    instruct = EMBEDDING_QUERY_INSTRUCT.strip()
    if not instruct:
        return text
    return f"Instruct: {instruct}\nQuery: {text}"


def query_fingerprint() -> str:
    """
    Six hex characters naming the query-side transform in force.

    Distances are only comparable within one embedding
    configuration. A threshold measured before
    EMBEDDING_QUERY_INSTRUCT was set means something else after it is,
    and nothing in the code noticed: on 2026-08-23 the instruction
    shipped on by default while .env.example kept a cutoff measured on
    raw queries, sitting above the best miss of the new regime. The
    only thing that said so was a comment.

    Hashing _as_query("") rather than the setting itself covers the
    whole wrapper -- the instruction AND the "Instruct:/Query:" shape
    around it -- so a change to either is a change of regime.

    LIMIT, and it is not a small one: this fingerprints what Forge
    controls. Swapping the embedding model behind the same
    EMBEDDING_URL changes every distance in the store and leaves this
    string identical. Re-measure after that too; nothing here will
    remind you.
    """
    return hashlib.sha256(_as_query("").encode("utf-8")).hexdigest()[:6]


#: The kind that is archived conversation rather than something
#: anyone chose to write down. Named here because two different
#: callers need to talk about it: format_results ranks it last, and
#: the recall rescue pass refuses to search it at all.
ARCHIVED_KIND = "history_summary"


def search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    kind: str | None = None,
    project: str | None = None,
    exclude_kind: str | None = None,
) -> list[dict]:
    query_embedding = _embed(_as_query(query))

    filters = []
    params: list = [sqlite_vec.serialize_float32(query_embedding), top_k]
    if kind is not None:
        filters.append("e.kind = ?")
        params.append(kind)
    if exclude_kind is not None:
        filters.append("e.kind != ?")
        params.append(exclude_kind)
    if project is not None:
        filters.append("e.project = ?")
        params.append(project)
    filter_sql = (" AND " + " AND ".join(filters)) if filters else ""

    rows = conn.execute(
        f"""
        SELECT e.id, e.kind, e.content, e.project, e.status, e.created_at, v.distance
        FROM memory_vectors v
        JOIN memory_entries e ON e.id = v.rowid
        WHERE v.embedding MATCH ?
          AND k = ?
          {filter_sql}
        ORDER BY v.distance
        """,
        params,
    ).fetchall()

    return [
        {
            "id": r[0],
            "kind": r[1],
            "content": r[2],
            "project": r[3],
            "status": r[4],
            "created_at": r[5],
            "distance": r[6],
        }
        for r in rows
    ]


# Word-ish tokens: letters and digits, no underscores, two characters
# and up. Digits are IN, and on this store that is not a detail --
# "5500U", "32", "AM06PRO" are the most discriminating words a
# hardware question can carry. The two-character floor removes French
# elision debris ("l'ordinateur" tokenises as "l" and "ordinateur");
# everything else that deserves removing is removed by frequency
# below, which is a measurement rather than a rule someone maintains.
_TERM_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)


def _terms(query: str) -> list[str]:
    """The distinct word-ish tokens of a query, lowercased, in order."""
    return list(dict.fromkeys(t.lower() for t in _TERM_RE.findall(query)))


def _document_frequency(conn: sqlite3.Connection, term: str) -> int:
    """
    How many entries contain *term*, counted through FTS5 itself.

    Through MATCH and not through Python, deliberately. The index
    folds case and diacritics with unicode61; any count computed on
    this side would have to reproduce that folding exactly, and would
    drift from it silently the first time the tokenizer line changes.
    Asking the index means the question and the answer are tokenised
    by the same code.

    The term is quoted as a phrase so an FTS5 keyword ("or", "and",
    "not", "near") is read as a word rather than as syntax.
    """
    row = conn.execute(
        "SELECT count(*) FROM memory_fts WHERE memory_fts MATCH ?",
        (f'"{term}"',),
    ).fetchone()
    return row[0] if row else 0


def informative_terms(
    conn: sqlite3.Connection, query: str, max_df: float = RECALL_LEXICAL_MAX_DF
) -> tuple[list[str], list[tuple[str, int]]]:
    """
    The words of *query* worth searching for, and the ones that were
    dropped for being everywhere.

    See RECALL_LEXICAL_MAX_DF in config.py for why the admission rule
    of this channel lives on the query side rather than on the score.

    The consequence worth stating out loud: a question made entirely
    of common words returns NOTHING from this channel. That is the
    correct outcome and it is what keeps the union honest. The
    alternative -- matching on "mon" because it was the only word left
    -- hands synthesis a random slice of the store carrying the
    authority of a lexical hit, which is the failure mode this whole
    line of work has been removing from the vector side since the
    cutoff was introduced.

    A word that appears in NO entry is dropped silently. It cannot
    match anything, so it is not a finding; the words that were
    refused for being too common are the ones worth logging, because
    that is the number you turn the knob against.
    """
    total = conn.execute("SELECT count(*) FROM memory_entries").fetchone()[0]
    if not total:
        return [], []

    # At least one entry, always. On a store of ten rows, 0.2 rounds
    # down to two and a strict fraction would refuse every word of a
    # question about the only entry that answers it.
    ceiling = max(1, int(max_df * total))

    kept: list[str] = []
    too_common: list[tuple[str, int]] = []
    for term in _terms(query):
        df = _document_frequency(conn, term)
        if df == 0:
            continue
        if df <= ceiling:
            kept.append(term)
        else:
            too_common.append((term, df))
    return kept, too_common


def search_lexical(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    kind: str | None = None,
    project: str | None = None,
    exclude_kind: str | None = None,
    max_df: float = RECALL_LEXICAL_MAX_DF,
) -> list[dict]:
    """
    Find entries by the words they share with the question, ranked by
    bm25.

    No embedding call, so this costs nothing on the inference server
    and answers in milliseconds -- the whole channel is a handful of
    SQLite queries against an index that is already there.

    Rows come back with a `score` (bm25, lower is better) and NO
    `distance`, and that absence is load-bearing rather than an
    omission: a row this channel admitted has not been measured
    against RECALL_MAX_DISTANCE and must not be judged by it. See
    search_hybrid, and _drop_distant in graphs/recall.py, which is
    where the two admission rules are kept apart.

    Returns nothing at all, quietly, on a SQLite build without FTS5.
    The channel is an addition to retrieval; losing it is a
    degradation, and turning that into an exception would take recall
    down on a store the vector channel can still read perfectly well.
    """
    terms, too_common = informative_terms(conn, query, max_df=max_df)
    if not terms:
        log.event(
            "rag.lexical_skipped",
            query=query[:120],
            too_common=[{"term": t, "entries": df} for t, df in too_common],
        )
        return []

    filters = []
    params: list = [" OR ".join(f'"{t}"' for t in terms)]
    if kind is not None:
        filters.append("e.kind = ?")
        params.append(kind)
    if exclude_kind is not None:
        filters.append("e.kind != ?")
        params.append(exclude_kind)
    if project is not None:
        filters.append("e.project = ?")
        params.append(project)
    filter_sql = (" AND " + " AND ".join(filters)) if filters else ""
    params.append(top_k)

    try:
        rows = conn.execute(
            f"""
            SELECT e.id, e.kind, e.content, e.project, e.status, e.created_at,
                   bm25(memory_fts) AS score
            FROM memory_fts
            JOIN memory_entries e ON e.id = memory_fts.rowid
            WHERE memory_fts MATCH ?
              {filter_sql}
            ORDER BY score
            LIMIT ?
            """,
            params,
        ).fetchall()
    except sqlite3.OperationalError as e:
        log.warning("rag: lexical search unavailable (%s)", e)
        return []

    results = [
        {
            "id": r[0],
            "kind": r[1],
            "content": r[2],
            "project": r[3],
            "status": r[4],
            "created_at": r[5],
            "score": r[6],
            # Which words found it -- the lexical equivalent of
            # search_many's matched_query, and needed for the same
            # reason: when a lexical hit turns out to be junk, the
            # question is which word dragged it in, and that has to be
            # readable in the log rather than reconstructed.
            "matched_terms": terms,
        }
        for r in rows
    ]

    log.event(
        "rag.lexical",
        query=query[:120],
        terms=terms,
        too_common=[{"term": t, "entries": df} for t, df in too_common],
        results=[
            {"id": r["id"], "score": round(r["score"], 4), "kind": r["kind"]}
            for r in results
        ],
    )
    return results


def _distance_key(row: dict) -> tuple[int, float]:
    """
    Sort key putting rows with no distance LAST rather than first.

    A missing distance is not a distance of zero. rag.search always
    reports one, but _drop_distant in graphs/recall.py already treats
    "not reported" as "keep" rather than "too far", and sorting the
    same row to the front would put an unmeasured entry above every
    measured one.
    """
    d = row.get("distance")
    return (0, float(d)) if isinstance(d, (int, float)) else (1, 0.0)


def _is_closer(candidate: dict, current: dict) -> bool:
    """Does *candidate* beat *current*, missing distances included."""
    return _distance_key(candidate) < _distance_key(current)


def search_many(
    conn: sqlite3.Connection,
    queries: list[str],
    top_k: int = 5,
    kind: str | None = None,
    project: str | None = None,
    exclude_kind: str | None = None,
) -> list[dict]:
    """
    Search once per query, merged on entry id, keeping each entry's
    BEST distance and the query that produced it.

    One embedding call and one SQL query per string, so the caller
    pays for the list it passes. Nothing is written and nothing is
    re-embedded on the store side -- the asymmetry that makes query
    expansion cheap is the same one that made the query instruction
    cheap (see _as_query): documents stay exactly as they were
    stored.

    MIN and not mean. An entry that one phrasing finds at 0.72 and
    another at 1.10 IS at 0.72 from the question, asked the right way;
    averaging the two would punish an entry for the phrasings that
    missed it, which is the opposite of what asking several ways is
    for. It also keeps the numbers on the scale RECALL_MAX_DISTANCE
    was measured against: every distance here is a real query-to-entry
    distance, not a statistic over several.

    Each row carries `matched_query`, the string that found it at that
    distance. Without it a merged list is unattributable -- when a
    rescued answer turns out to be wrong, the question is which
    rephrasing dragged it in, and that has to be readable in the log
    rather than reconstructed.

    `top_k` applies twice, deliberately: each query returns its own
    top_k, and the merge is truncated to top_k again. So an entry
    reaches the caller only if at least one phrasing ranked it in its
    own top_k -- a merge of N queries does not hand synthesis N times
    the material.
    """
    merged: dict[int, dict] = {}
    for query in queries:
        for row in search(
            conn,
            query=query,
            top_k=top_k,
            kind=kind,
            project=project,
            exclude_kind=exclude_kind,
        ):
            row = dict(row, matched_query=query)
            current = merged.get(row["id"])
            if current is None or _is_closer(row, current):
                merged[row["id"]] = row

    return sorted(merged.values(), key=_distance_key)[:top_k]


def _hybrid_key(row: dict) -> tuple[int, float, float]:
    """
    Order a merged list: measured distances first, then bm25.

    Two channels means two scales and no exchange rate between them.
    Inventing one -- normalising both into a single number, weighting
    them, reciprocal-rank fusion -- would put a tunable constant
    between the store and the answer, and every tunable constant in
    this file has cost a measurement campaign to place. There is a
    free ordering available instead: a vector hit that survived the
    cutoff was admitted by a threshold measured against real hits and
    real misses, and a lexical hit was admitted by a rule that says
    nothing about how relevant it is, only that it is not noise. So
    the measured ones go first and the word matches follow, each
    ordered within its own channel by its own number.
    """
    d = row.get("distance")
    if isinstance(d, (int, float)):
        return (0, float(d), 0.0)
    score = row.get("score")
    return (1, 0.0, float(score) if isinstance(score, (int, float)) else 0.0)


def search_hybrid(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    lexical_top_k: int = 3,
    kind: str | None = None,
    project: str | None = None,
    exclude_kind: str | None = None,
    lexical_exclude_archived: bool = RECALL_LEXICAL_EXCLUDE_ARCHIVED,
) -> list[dict]:
    """
    Both channels, unioned on entry id, each with its own budget and
    its own admission rule.

    A UNION AND NOT AN INTERSECTION, and not a re-ranking of one by
    the other either. The two channels fail on different things: the
    vector channel cannot reach an entry that is not written in prose
    (#307, #313), and the lexical channel cannot reach an entry that
    answers the question in other words than the ones asked. An entry
    either channel can find is an entry Forge can answer from; asking
    for both would keep only the questions that were already working.

    EACH CHANNEL KEEPS ITS OWN BUDGET rather than sharing top_k. A
    shared budget sorted by distance would let five vector rows crowd
    out the lexical ones entirely -- and the case this exists for is
    precisely the one where the vector rows are all about to be cut by
    the cutoff, so they would be spending slots on their way to being
    dropped.

    Every row carries `channel`, and the caller needs it. A row this
    function returns has been admitted by a rule, but not by the same
    rule, and only the tag says which: see _drop_distant in
    graphs/recall.py, where the vector cutoff is applied to the rows
    it was measured for and to nothing else.

    A row found by both keeps its distance AND its score. It is the
    strongest kind of hit this store can produce -- the words match
    and the meaning matches -- and it is the row you want to see first
    in a log when a hybrid answer is right.
    """
    merged: dict[int, dict] = {}

    for row in search(
        conn,
        query=query,
        top_k=top_k,
        kind=kind,
        project=project,
        exclude_kind=exclude_kind,
    ):
        merged[row["id"]] = dict(row, channel="vector")

    # The word channel skips archived transcript by default, and only
    # the word channel does. See RECALL_LEXICAL_EXCLUDE_ARCHIVED: an
    # archived unit contains the question verbatim, so matching a
    # question against it is matching a question against a copy of
    # itself. A caller that already excludes something is left alone --
    # one exclusion is all the query has room for, and the caller's is
    # the deliberate one.
    lexical_exclude = exclude_kind
    if lexical_exclude is None and lexical_exclude_archived:
        lexical_exclude = ARCHIVED_KIND

    for row in search_lexical(
        conn,
        query=query,
        top_k=lexical_top_k,
        kind=kind,
        project=project,
        exclude_kind=lexical_exclude,
    ):
        current = merged.get(row["id"])
        if current is None:
            merged[row["id"]] = dict(row, channel="lexical")
        else:
            current.update(
                channel="both",
                score=row["score"],
                matched_terms=row["matched_terms"],
            )

    return sorted(merged.values(), key=_hybrid_key)


def list_entries(
    conn: sqlite3.Connection,
    *,
    kind: str | None = None,
    project: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    List entries as stored, newest first, with no query involved.

    `search` is the only way anything ever read this store, and it is
    semantic: it needs a question and returns whatever is nearest to
    it. There was no way to answer "what is actually in there?" -- and
    on 2026-08-22 that turned into a real dead end. Picking calibration
    questions for bench/recall_distance.py needs three things the store
    can answer and three it cannot, and finding them by asking `recall`
    means guessing at the contents through the exact mechanism whose
    reliability is in question.

    A store you cannot enumerate is a store you cannot debug. This
    reads it directly, ordered by id, no embedding call at all.
    """
    filters, params = [], []
    if kind is not None:
        filters.append("kind = ?")
        params.append(kind)
    if project is not None:
        filters.append("project = ?")
        params.append(project)
    where = (" WHERE " + " AND ".join(filters)) if filters else ""
    params += [limit, offset]

    rows = conn.execute(
        f"""
        SELECT id, kind, content, project, status, created_at
        FROM memory_entries
        {where}
        ORDER BY id DESC
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()

    return [
        {
            "id": r[0],
            "kind": r[1],
            "content": r[2],
            "project": r[3],
            "status": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


def hot_entries(conn: sqlite3.Connection) -> list[dict]:
    """
    Everything anyone deliberately wrote down, oldest first, all of it.

    This is not a search and it takes no query. `search` answers a
    question of PROXIMITY -- what is nearest to this -- and six
    measurement campaigns on this store established that it cannot
    answer a question of COMPLETENESS. "List my hardware" wants a set,
    and every mechanism in this file returns the nearest rows of one.
    So for the entries small enough to fit, the answer is to stop
    searching and look.

    Three deliberate differences from list_entries, which is the
    reader that already exists:

      ascending id      list_entries is newest-first because a human
                        reading a store wants the recent end. This
                        output goes into a prompt, and appending at
                        the end is the only order in which adding an
                        entry leaves the earlier ones byte-identical.

      no LIMIT          list_entries defaults to 50, which is a page
                        size. A page size silently deciding what a
                        completeness answer contains is the failure
                        this exists to remove; the cap that does
                        decide is a token budget, applied by the
                        caller, out loud.

      kind IS NOT       not `!=`. A row whose kind is NULL is not
                        archived transcript, and `kind != 'x'` is
                        false for NULL in SQL, so `!=` would drop it
                        without a word.

    ARCHIVED_KIND is excluded and nothing else is. Not `kind =
    'fact'`: docs/memory.md records that the memory tool defaults a
    missing kind to "fact" rather than failing, so the kind on any row
    the router wrote is a 9B's on-the-fly guess. What is reliable is
    the binary distinction tools/memory._rank already sorts on --
    someone chose to write this down, or compaction dumped it here.
    """
    rows = conn.execute(
        """
        SELECT id, kind, content, project, created_at
        FROM memory_entries
        WHERE kind IS NOT ?
        ORDER BY id ASC
        """,
        (ARCHIVED_KIND,),
    ).fetchall()

    return [
        {
            "id": r[0],
            "kind": r[1],
            "content": r[2],
            "project": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def count_entries(conn: sqlite3.Connection, *, kind: str | None = None) -> dict:
    """
    How many entries there are, broken down by kind.

    The breakdown is the point rather than the total. The recurring
    diagnosis on this store is "it holds compaction pointers and almost
    no facts", and until now that was an inference from whatever
    `search` happened to return. One query settles it.
    """
    where, params = ("WHERE kind = ?", [kind]) if kind else ("", [])
    rows = conn.execute(
        f"SELECT kind, COUNT(*) FROM memory_entries {where} GROUP BY kind",
        params,
    ).fetchall()
    by_kind = {r[0]: r[1] for r in rows}
    return {"total": sum(by_kind.values()), "by_kind": by_kind}
