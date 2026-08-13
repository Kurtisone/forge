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


def remember(
    conn: sqlite3.Connection, kind: str, content: str, project: str | None
) -> int:
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
