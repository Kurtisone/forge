"""
Tests for supersession -- the answer to the question the aggregation
tier could not be written without.

When compaction extracts one fact per subject out of several
overlapping ones, the sources are neither deleted (destructive and
irreversible on entries a human typed) nor left alongside (which grows
the hot block and restores the overlap the aggregation removed). They
are LINKED to the entry that now speaks for them, and one reader --
hot_entries -- skips them.

What these pin is the set of properties that make that choice
defensible rather than a soft delete with extra steps: nothing becomes
unreachable, the fold is visible, the fold is reversible, and
forgetting an aggregate releases what it spoke for.
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


def _write(conn, content, kind="fact"):
    return rag.remember(conn, kind=kind, content=content, project=None)


@pytest.fixture
def nipogi(store):
    """The real store's three overlapping NiPoGi entries, plus a fold."""
    sources = [
        _write(
            store, "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM"
        ),
        _write(store, "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, Ansible"),
        _write(store, "Le NiPoGi a 32 Go de RAM"),
    ]
    aggregate = _write(
        store,
        "Le NiPoGi AM06PRO tourne sous Arch avec un processeur Ryzen 5500U, "
        "32 Go de RAM et Ansible",
    )
    return store, sources, aggregate


def test_the_block_loses_the_sources_and_keeps_the_aggregate(nipogi):
    conn, sources, aggregate = nipogi
    assert len(rag.hot_entries(conn)) == 4

    rag.supersede(conn, sources, aggregate)

    ids = [e["id"] for e in rag.hot_entries(conn)]
    assert ids == [aggregate]


def test_nothing_becomes_unreachable(nipogi):
    """
    The property the whole design rests on. An aggregate written by a
    9B may quietly drop a detail; if that dropped the source out of
    retrieval too, the store would answer "je n'ai rien" while the
    text sits in it -- the failure docs/memory.md says gets believed.
    """
    conn, sources, aggregate = nipogi
    rag.supersede(conn, sources, aggregate)

    still_there = {row[0] for row in conn.execute("SELECT id FROM memory_entries")}
    assert set(sources) <= still_there

    # max_df=1.0 so the word channel's own admission rule (which needs
    # a bigger store than four rows to mean anything) does not decide
    # this test. What is under test is scope, not term frequency.
    reachable = {
        r["id"]
        for r in rag.search_lexical(conn, query="32Go Ansible", top_k=10, max_df=1.0)
    }
    assert set(sources) & reachable


def test_the_fold_is_visible_to_the_reader_a_human_uses(nipogi):
    conn, sources, aggregate = nipogi
    rag.supersede(conn, sources, aggregate)

    by_id = {e["id"]: e for e in rag.list_entries(conn, limit=50)}
    assert all(by_id[s]["superseded_by"] == aggregate for s in sources)
    assert by_id[aggregate]["superseded_by"] is None


def test_the_fold_is_reversible(nipogi):
    conn, sources, aggregate = nipogi
    rag.supersede(conn, sources, aggregate)

    assert sorted(rag.unsupersede(conn, aggregate)) == sorted(sources)
    assert len(rag.hot_entries(conn)) == 4


def test_forgetting_the_aggregate_releases_its_sources(nipogi):
    """
    Without this, `!forget` on a bad aggregate is the one way this
    becomes the destructive design it was chosen not to be: the
    sources stay hidden, pointing at a row that no longer exists.
    """
    conn, sources, aggregate = nipogi
    rag.supersede(conn, sources, aggregate)

    assert rag.forget(conn, aggregate) is True

    ids = [e["id"] for e in rag.hot_entries(conn)]
    assert ids == sources


def test_a_source_is_folded_once_and_chains_are_refused(nipogi):
    conn, sources, first = nipogi
    rag.supersede(conn, sources, first)

    second = _write(conn, "Le mini PC NiPoGi et ses 32 Go de RAM")
    assert rag.supersede(conn, sources, second) == []

    by_id = {e["id"]: e for e in rag.list_entries(conn, limit=50)}
    assert all(by_id[s]["superseded_by"] == first for s in sources)


def test_an_aggregate_that_is_itself_superseded_cannot_speak_for_anything(nipogi):
    conn, sources, aggregate = nipogi
    outer = _write(conn, "Un NiPoGi AM06PRO sous Arch")
    rag.supersede(conn, [aggregate], outer)

    assert rag.supersede(conn, sources, aggregate) == []


def test_an_unknown_aggregate_folds_nothing(nipogi):
    conn, sources, _ = nipogi
    assert rag.supersede(conn, sources, 9999) == []
    assert len(rag.hot_entries(conn)) == 4


def test_an_entry_cannot_speak_for_itself(nipogi):
    conn, _, aggregate = nipogi
    assert rag.supersede(conn, [aggregate], aggregate) == []
    assert aggregate in [e["id"] for e in rag.hot_entries(conn)]


def test_a_store_written_before_the_column_existed_opens_and_migrates(
    tmp_path, monkeypatch
):
    """
    The migration is an ALTER and nothing else: NULL means "not
    superseded", which is true of every row already on disk. No
    backfill, no deploy/ step, no window.
    """
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "old.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)

    conn = rag.get_connection()
    entry_id = _write(conn, "Possède un Steam Deck")
    conn.execute("ALTER TABLE memory_entries DROP COLUMN superseded_by")
    conn.commit()
    conn.close()

    conn = rag.get_connection()
    columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_entries)")}
    assert "superseded_by" in columns
    assert [e["id"] for e in rag.hot_entries(conn)] == [entry_id]
    conn.close()
