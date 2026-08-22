"""
Tests for rag.remember_many.

The point of the batch write is that N units become N ROWS with N
vectors, instead of one row whose single vector is the average of N
chunks. The average is what put a question 0.27 further from the
content answering it than the same content stored on its own
(measured on the Deck, 2026-08-22).
"""

import pytest

from forge import rag


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    yield conn
    conn.close()


def test_each_unit_becomes_its_own_entry(store):
    ids = rag.remember_many(
        store,
        kind="history_summary",
        contents=["user: quel port ?", "user: et l'hôte ?"],
        project=None,
    )

    assert len(ids) == 2
    assert rag.count_entries(store)["by_kind"] == {"history_summary": 2}
    assert [e["content"] for e in rag.list_entries(store)] == [
        "user: et l'hôte ?",
        "user: quel port ?",
    ]


def test_each_entry_gets_its_own_vector(store):
    ids = rag.remember_many(
        store, kind="history_summary", contents=["un bloc", "un autre"], project=None
    )

    rows = store.execute(
        "SELECT rowid FROM memory_vectors WHERE rowid IN (?, ?)", ids
    ).fetchall()
    assert len(rows) == 2


def test_a_degenerate_unit_is_skipped_not_raised(store):
    """
    remember() raises on a one-word entry because the caller is
    asserting a fact and needs to hear that the value went missing.
    Here the caller is archiving a block it did not write, and one
    one-word message must not fail the whole compaction.
    """
    ids = rag.remember_many(
        store,
        kind="history_summary",
        contents=["user: bonjour\nassistant: salut", "ok"],
        project=None,
    )

    assert len(ids) == 1
    assert rag.count_entries(store)["total"] == 1


def test_an_embedding_failure_stores_nothing(store, monkeypatch):
    """
    All or nothing. A half-indexed block is worse than an unindexed
    one: the pointer written into the history claims a range that does
    not hold what it says it holds.
    """
    calls = {"n": 0}

    def _flaky(text):
        calls["n"] += 1
        if calls["n"] == 2:
            raise rag.EmbeddingError("server down")
        return [0.1] * rag.EMBEDDING_DIM

    monkeypatch.setattr(rag, "_embed", _flaky)

    with pytest.raises(rag.EmbeddingError):
        rag.remember_many(
            store,
            kind="history_summary",
            contents=["premier bloc", "second bloc", "troisième bloc"],
            project=None,
        )

    store.rollback()
    assert rag.count_entries(store)["total"] == 0


def test_an_empty_batch_is_not_an_error(store):
    assert (
        rag.remember_many(store, kind="history_summary", contents=[], project=None)
        == []
    )
    assert rag.count_entries(store)["total"] == 0


def test_single_writes_still_commit_on_their_own(store):
    """remember() shares _insert with the batch path now -- it must
    still be a complete write by itself."""
    entry_id = rag.remember(store, kind="fact", content="un vrai fait", project=None)

    assert rag.list_entries(store)[0]["id"] == entry_id
