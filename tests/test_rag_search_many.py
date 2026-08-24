"""
Tests for the multi-query merge behind query expansion.

Pure unit tests: rag.search is substituted, so nothing here needs an
embedding server. What is being pinned is the arithmetic of the merge
-- which distance survives, which query gets the credit, and what
happens to a row that has no distance at all.
"""

from forge import rag


def _row(entry_id, distance, content="entry"):
    return {
        "id": entry_id,
        "kind": "fact",
        "content": content,
        "project": None,
        "status": "active",
        "created_at": "2026-08-24T00:00:00+00:00",
        "distance": distance,
    }


def _canned(by_query, monkeypatch):
    """rag.search returning a fixed list per query string."""

    def fake_search(conn, query, top_k=5, kind=None, project=None):
        return by_query.get(query, [])[:top_k]

    monkeypatch.setattr(rag, "search", fake_search)


def test_an_entry_found_twice_keeps_its_best_distance(monkeypatch):
    """
    MIN, not mean. An entry one phrasing finds at 0.72 is at 0.72 from
    the question asked that way; averaging in the phrasings that
    missed it would punish it for them.
    """
    _canned(
        {
            "matériel": [_row(1, 1.02)],
            "processeur mémoire": [_row(1, 0.72)],
        },
        monkeypatch,
    )

    merged = rag.search_many(None, ["matériel", "processeur mémoire"])

    assert [(r["id"], r["distance"]) for r in merged] == [(1, 0.72)]


def test_the_winning_query_is_named_on_the_row(monkeypatch):
    """
    A merged list nobody can attribute is a rescue nobody can debug:
    when a wrong entry comes back, the question is which rephrasing
    dragged it in.
    """
    _canned(
        {
            "matériel": [_row(1, 1.02)],
            "processeur mémoire": [_row(1, 0.72)],
        },
        monkeypatch,
    )

    merged = rag.search_many(None, ["matériel", "processeur mémoire"])

    assert merged[0]["matched_query"] == "processeur mémoire"


def test_rows_come_back_ordered_by_distance(monkeypatch):
    _canned(
        {
            "a": [_row(1, 0.9), _row(2, 0.4)],
            "b": [_row(3, 0.6)],
        },
        monkeypatch,
    )

    merged = rag.search_many(None, ["a", "b"])

    assert [r["id"] for r in merged] == [2, 3, 1]


def test_the_merge_is_capped_at_top_k(monkeypatch):
    """
    top_k applies twice on purpose: asking four ways must not hand
    synthesis four times the material.
    """
    _canned(
        {
            "a": [_row(1, 0.1), _row(2, 0.2)],
            "b": [_row(3, 0.3), _row(4, 0.4)],
        },
        monkeypatch,
    )

    merged = rag.search_many(None, ["a", "b"], top_k=2)

    assert [r["id"] for r in merged] == [1, 2]


def test_a_row_with_no_distance_is_kept_but_sorts_last(monkeypatch):
    """
    A missing distance is not a distance of zero. graphs/recall.py
    already keeps such a row rather than treating it as too far;
    sorting it to the front would rank an unmeasured entry above every
    measured one.
    """
    _canned({"a": [dict(_row(1, None))], "b": [_row(2, 1.4)]}, monkeypatch)

    merged = rag.search_many(None, ["a", "b"])

    assert [r["id"] for r in merged] == [2, 1]


def test_a_measured_distance_beats_a_missing_one_for_the_same_entry(monkeypatch):
    _canned({"a": [dict(_row(1, None))], "b": [_row(1, 1.4)]}, monkeypatch)

    merged = rag.search_many(None, ["a", "b"])

    assert merged[0]["distance"] == 1.4
    assert merged[0]["matched_query"] == "b"


def test_no_queries_means_no_searches(monkeypatch):
    def must_not_run(*a, **kw):  # pragma: no cover - the assertion is that it doesn't
        raise AssertionError("searched with nothing to search for")

    monkeypatch.setattr(rag, "search", must_not_run)

    assert rag.search_many(None, []) == []


def test_the_tool_wrapper_opens_one_connection_for_the_batch(monkeypatch):
    """
    graphs/recall.py goes through tools.memory, not forge.rag. One
    connection for the whole batch, not one per query.
    """
    from forge.tools import memory as memory_tool

    opened = []

    class FakeConn:
        def close(self):
            opened.append("closed")

    monkeypatch.setattr(
        rag, "get_connection", lambda: (opened.append("open"), FakeConn())[1]
    )
    monkeypatch.setattr(
        rag, "search_many", lambda conn, queries, top_k, kind, project: [_row(1, 0.5)]
    )

    assert memory_tool.search_many(["a", "b"])[0]["id"] == 1
    assert opened == ["open", "closed"]
