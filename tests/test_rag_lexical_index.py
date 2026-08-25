"""
The lexical index itself: does it exist, does it hold what the rows
hold, and does it survive a store that predates it.

Nothing here searches. Retrieval is tested in
tests/test_rag_lexical_search.py; this file is about the invariant
those searches rest on -- memory_fts and memory_entries never
disagree about what is in the store.
"""

import sqlite3

import pytest

from forge import rag

FAKE_DIM = 8


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(
        rag, "_VEC_SCHEMA", rag._VEC_SCHEMA.replace("1024", str(FAKE_DIM))
    )
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * FAKE_DIM)
    connection = rag.get_connection()
    yield connection
    connection.close()


def _indexed(conn) -> set[int]:
    return {r[0] for r in conn.execute("SELECT rowid FROM memory_fts").fetchall()}


def test_get_connection_creates_the_lexical_index(conn):
    assert rag.has_lexical_index(conn)


def test_a_stored_entry_is_indexed_by_the_trigger(conn):
    entry_id = rag.remember(
        conn, kind="fact", content="Le NiPoGi a un processeur Ryzen", project=None
    )

    assert entry_id in _indexed(conn)


def test_forgetting_an_entry_removes_it_from_the_index(conn):
    entry_id = rag.remember(
        conn, kind="fact", content="Le NiPoGi a un processeur Ryzen", project=None
    )
    rag.forget(conn, entry_id)

    # The vector table has needed a hand-written DELETE since v3.7
    # precisely because a half-deleted entry stays matchable. The
    # lexical index must not reintroduce that: a forgotten entry that
    # is still findable by its words is the same invisible-and-
    # answering memory under a different index.
    assert entry_id not in _indexed(conn)
    assert not conn.execute(
        "SELECT 1 FROM memory_fts WHERE memory_fts MATCH ?", ('"NiPoGi"',)
    ).fetchall()


def test_a_store_written_before_the_index_existed_is_backfilled(tmp_path, monkeypatch):
    """
    The upgrade path, which is the whole reason this ships without a
    migration script. A store with rows and no memory_fts is what
    every existing deployment looks like on the day it pulls this.
    """
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(
        rag, "_VEC_SCHEMA", rag._VEC_SCHEMA.replace("1024", str(FAKE_DIM))
    )
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * FAKE_DIM)

    first = rag.get_connection()
    entry_id = rag.remember(
        conn=first,
        kind="fact",
        content="Steam Deck, SteamOS, conteneurs Podman",
        project=None,
    )
    # Back to a pre-v3.17 store: the rows, no index, no triggers.
    first.executescript(
        """
        DROP TRIGGER memory_fts_insert;
        DROP TRIGGER memory_fts_delete;
        DROP TRIGGER memory_fts_update;
        DROP TABLE memory_fts;
        """
    )
    first.commit()
    first.close()

    second = rag.get_connection()
    try:
        assert entry_id in _indexed(second)
    finally:
        second.close()


def test_reopening_an_indexed_store_does_not_rebuild_it(conn, tmp_path, monkeypatch):
    """
    'rebuild' on every connection would re-index the whole store to
    open it, and the cost is invisible until the store is large.
    """
    rag.remember(conn, kind="fact", content="Le NiPoGi a 32 Go", project=None)
    conn.close()

    events = []
    monkeypatch.setattr(rag.log, "event", lambda name, **kw: events.append(name))
    again = rag.get_connection()
    try:
        assert "rag.fts_backfilled" not in events
    finally:
        again.close()


def test_a_sqlite_without_fts5_leaves_the_store_usable(monkeypatch):
    """
    FTS5 is an optional module. Losing the lexical channel is a
    degradation; failing here would take every read and write of the
    store down with it, because this runs on the connection path.
    """

    class NoFts5:
        def execute(self, sql, *args):
            if "fts5" in sql:
                raise sqlite3.OperationalError("no such module: fts5")
            return self

        def executescript(self, sql):
            return self

        def fetchone(self):
            return None

        def commit(self):
            pass

    assert rag._ensure_fts(NoFts5()) is False
