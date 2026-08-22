"""
Tests for reading the vector store without asking it a question.

The dead end, 2026-08-22: choosing calibration questions for
bench/recall_distance.py needs three things the store CAN answer and
three it cannot. Finding them meant asking `recall` -- which runs the
exact retrieval whose reliability is the thing in question. It came
back with five compaction pointers and a refusal, and there was no
other way to look.

`search` was the only reader and it is semantic by construction. A
store you cannot enumerate is a store you cannot debug.
"""

import pytest

from forge import rag


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)
    conn = rag.get_connection()
    for kind, content in [
        ("fact", "Le Steam Deck a 16 Go de RAM."),
        ("decision", "Les commits restent sous le nom de Kurtisone."),
        ("history_summary", "system: [9 messages précédents compactés]"),
        ("history_summary", "system: [19 messages précédents compactés]"),
    ]:
        rag.remember(conn, kind=kind, content=content, project=None)
    yield conn
    conn.close()


def test_listing_returns_everything_newest_first(store):
    entries = rag.list_entries(store)

    assert len(entries) == 4
    assert [e["id"] for e in entries] == sorted(
        (e["id"] for e in entries), reverse=True
    )
    assert "Steam Deck" in entries[-1]["content"]


def test_listing_takes_no_query_and_makes_no_embedding_call(store, monkeypatch):
    """
    The whole point. Enumerating must not route through the mechanism
    being debugged, and must work with the embedding server down.
    """

    def no_embed(text):  # pragma: no cover - must not run
        raise AssertionError("listing called the embedding server")

    monkeypatch.setattr(rag, "_embed", no_embed)

    assert len(rag.list_entries(store)) == 4


def test_listing_filters_by_kind(store):
    summaries = rag.list_entries(store, kind="history_summary")

    assert len(summaries) == 2
    assert all(e["kind"] == "history_summary" for e in summaries)


def test_listing_pages(store):
    first = rag.list_entries(store, limit=2)
    second = rag.list_entries(store, limit=2, offset=2)

    assert len(first) == len(second) == 2
    assert {e["id"] for e in first}.isdisjoint({e["id"] for e in second})


def test_counting_breaks_down_by_kind(store):
    """
    The breakdown matters more than the total: "this store holds
    compaction pointers and almost no facts" has been the standing
    diagnosis for a week, inferred from whatever `search` happened to
    return. One query settles it.
    """
    counts = rag.count_entries(store)

    assert counts["total"] == 4
    assert counts["by_kind"] == {"fact": 1, "decision": 1, "history_summary": 2}


def test_the_endpoint_and_both_front_ends_exist():
    import pathlib

    from forge.api import app

    assert "/memory" in {r.path for r in app.routes}

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "forge"
    assert "!memory" in (root / "main.py").read_text(), "the REPL has no !memory"
    assert "'!memory'" in (root / "static" / "index.html").read_text(), (
        "the web UI has no !memory -- which is where the dead end happened"
    )
