"""
Tests for rag.hot_entries -- the reader that takes no question.

Six measurement campaigns on the real store (docs/memory.md) ended on
the same sentence: retrieval answers proximity questions, not
completeness questions. "Tu peux me lister mon matériel ?" wants a
set; every mechanism in rag.py returns the nearest rows of one.

Measured on the Deck, 2026-08-26: the whole deliberate store is 11
entries and ~195 tokens. At that size the answer to a completeness
question is not a better search, it is to stop searching.

What these pin is the three ways this reader deliberately differs
from list_entries, because each one is a place a plausible-looking
change would silently decide what a completeness answer contains.
"""

import pytest

from forge import rag


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    for kind, content in [
        ("fact", "Possède un Steam Deck"),
        ("history_summary", "system: [9 messages précédents compactés]"),
        ("decision", "Ne pas épingler les messages avec des emojis"),
        ("fact", "Le NiPoGi a 32 Go de RAM"),
        ("history_summary", "system: [19 messages précédents compactés]"),
    ]:
        rag.remember(conn, kind=kind, content=content, project=None)
    yield conn
    conn.close()


def test_archived_transcript_is_the_only_thing_excluded(store):
    kinds = [e["kind"] for e in rag.hot_entries(store)]

    assert rag.ARCHIVED_KIND not in kinds
    # Not kind == "fact". docs/memory.md: the memory tool defaults a
    # missing kind to "fact" rather than failing, so the kind on a
    # router-written row is a 9B's guess made on the fly. The reliable
    # distinction is the binary one tools/memory._rank already sorts
    # on -- deliberate, or dumped here by compaction.
    assert "decision" in kinds


def test_ascending_id_so_a_new_entry_leaves_the_others_untouched(store):
    before = [e["content"] for e in rag.hot_entries(store)]

    rag.remember(
        store,
        kind="fact",
        content="Le proxy podman écoute sur un socket unix",
        project=None,
    )
    after = [e["content"] for e in rag.hot_entries(store)]

    # The order this block is rendered in is the order it is prefilled
    # in. Ascending id is the only one where adding an entry appends;
    # newest-first (what list_entries does, correctly, for a human
    # reader) rewrites the front of the block on every write.
    assert after[: len(before)] == before
    assert len(after) == len(before) + 1


def test_no_page_size_decides_what_a_completeness_answer_holds(store):
    for n in range(60):
        rag.remember(store, kind="fact", content=f"fait numéro {n}", project=None)

    # list_entries defaults to 50 because a page size is right for a
    # human paging through a store. A page size silently truncating a
    # completeness answer is the failure this reader exists to remove;
    # the cap that does truncate is a token budget, applied by the
    # caller, and it says so.
    assert len(rag.hot_entries(store)) == 63


def test_the_schema_is_what_makes_the_null_case_impossible(store):
    # `kind IS NOT ?` rather than `kind != ?` is null-safe, and today
    # that buys nothing: memory_entries.kind is NOT NULL, so the two
    # spellings are the same query. This test says WHY the defensive
    # one is there rather than leaving it looking like superstition --
    # `!=` is FALSE for NULL in SQL, so the day that constraint is
    # relaxed the obvious spelling starts dropping rows silently, and
    # a completeness answer is the one place a silent drop is worse
    # than an error.
    with pytest.raises(Exception, match="NOT NULL"):
        store.execute(
            "INSERT INTO memory_entries (kind, content, project) VALUES (NULL, ?, NULL)",
            ("une entrée sans kind",),
        )
