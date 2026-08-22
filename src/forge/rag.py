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

import math
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import requests
import sqlite_vec

from forge.config import (
    EMBEDDING_DIM,
    EMBEDDING_MAX_CHARS,
    EMBEDDING_MAX_CHUNKS,
    EMBEDDING_TIMEOUT,
    EMBEDDING_URL,
    RAG_DB_FILE,
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
    return conn


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
    conn.commit()
    return entry_id


def search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    kind: str | None = None,
    project: str | None = None,
) -> list[dict]:
    query_embedding = _embed(query)

    filters = []
    params: list = [sqlite_vec.serialize_float32(query_embedding), top_k]
    if kind is not None:
        filters.append("e.kind = ?")
        params.append(kind)
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
