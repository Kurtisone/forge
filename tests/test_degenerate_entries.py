"""
Tests for entries that assert nothing, and for removing them.

The real store on 2026-08-22 contained `#3 [fact] S'appelle` -- a
predicate whose object went missing when a remember call lost its
value. Every existing check passed it: tools/memory.py rejects EMPTY
content, and "S'appelle" is not empty.

It was not cosmetic. In the calibration run that day it was the entry
that closed the gap: it pulled the unanswerable "Comment s'appelle mon
chat ?" to 0.9356, NEARER than the worst genuine hit at 0.9386. A
dangling verb is a magnet for every question phrased around it, and it
answers none of them.
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


def test_a_single_word_entry_is_refused(store):
    with pytest.raises(rag.DegenerateEntry) as excinfo:
        rag.remember(store, kind="fact", content="S'appelle", project=None)

    assert "S'appelle" in str(excinfo.value)
    assert rag.count_entries(store)["total"] == 0


def test_short_but_real_entries_are_kept(store):
    """
    The rule is one word, not a character count, on purpose. Terse is
    not the same complaint as empty -- "use sqlite-vec" and "a todo"
    are both legitimate and both live in this repository's fixtures. A
    character floor would have rejected them for the wrong reason.
    """
    for content in ("use sqlite-vec", "a todo", "S'appelle Jean"):
        rag.remember(store, kind="decision", content=content, project=None)

    assert rag.count_entries(store)["total"] == 3


def test_whitespace_does_not_buy_a_second_word(store):
    with pytest.raises(rag.DegenerateEntry):
        rag.remember(store, kind="fact", content="   S'appelle  \n ", project=None)


def test_the_check_sits_at_the_store_boundary_not_only_in_the_tool(store):
    """
    Every writer crosses rag.remember: the memory tool, the REPL,
    POST /remember, and compaction. A check in tools/memory.py alone
    leaves three doors open, and the broken entry came in through one
    of them.
    """
    import json

    from forge.tools import memory as memory_tool

    out = memory_tool.run(
        json.dumps({"action": "remember", "kind": "fact", "content": "S'appelle"})
    )

    assert out.startswith("[error]")
    assert "single word" in out


def test_forgetting_removes_the_entry_and_its_vector(store):
    """
    Both tables or neither. memory_vectors is keyed by rowid against
    memory_entries.id, so deleting one and not the other leaves a
    vector search can still match and list_entries can no longer
    show -- a memory that is invisible and answering.
    """
    entry_id = rag.remember(
        store, kind="fact", content="un fait à oublier", project=None
    )

    assert rag.forget(store, entry_id) is True
    assert rag.count_entries(store)["total"] == 0

    orphans = store.execute(
        "SELECT COUNT(*) FROM memory_vectors WHERE rowid = ?", (entry_id,)
    ).fetchone()[0]
    assert orphans == 0


def test_forgetting_something_absent_says_so(store):
    assert rag.forget(store, 999) is False
