"""
Tests for forge.tools.memory -- the router-dispatchable memory tool.

rag._embed is monkeypatched, same boundary as test_rag.py/test_main.py.
No network / no real embedding server involved.
"""

import json

import pytest

from forge import rag
from forge.tools import memory as memory_tool


@pytest.fixture(autouse=True)
def _rag_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)


# ── remember ─────────────────────────────────────────────────────────


def test_remember_stores_entry():
    out = memory_tool.run(
        json.dumps({"action": "remember", "kind": "decision", "content": "use podman"})
    )
    assert out == "Remembered (#1)."


def test_remember_with_project():
    out = memory_tool.run(
        json.dumps(
            {
                "action": "remember",
                "kind": "todo",
                "content": "write tests",
                "project": "forge",
            }
        )
    )
    assert "Remembered" in out


def test_remember_accepts_fact_kind():
    out = memory_tool.run(
        json.dumps(
            {"action": "remember", "kind": "fact", "content": "Possède un Steam Deck"}
        )
    )
    assert out == "Remembered (#1)."


def test_remember_defaults_to_fact_when_kind_missing():
    """The real failure this guards against: a casual mention like
    "Mémorise, je possède un Steam Deck" isn't a decision or a todo,
    and a small router model asked to classify a plain statement may
    not supply a kind at all -- this must not hard-fail."""
    out = memory_tool.run(
        json.dumps({"action": "remember", "content": "Possède un Steam Deck"})
    )
    assert out == "Remembered (#1)."


def test_remember_defaults_to_fact_when_kind_empty():
    out = memory_tool.run(
        json.dumps(
            {"action": "remember", "kind": "", "content": "Possède un Steam Deck"}
        )
    )
    assert out == "Remembered (#1)."


def test_remember_rejects_invalid_kind():
    out = memory_tool.run(
        json.dumps({"action": "remember", "kind": "note", "content": "entrée de test"})
    )
    assert "kind" in out
    assert out.startswith("[error]")


def test_remember_rejects_empty_content():
    out = memory_tool.run(
        json.dumps({"action": "remember", "kind": "decision", "content": ""})
    )
    assert out.startswith("[error]")


def test_remember_reports_embedding_failure(monkeypatch):
    def _raise(text):
        raise rag.EmbeddingError("connection refused")

    monkeypatch.setattr(rag, "_embed", _raise)
    out = memory_tool.run(
        json.dumps(
            {"action": "remember", "kind": "decision", "content": "entrée de test"}
        )
    )
    assert "remember failed" in out


# ── recall ────────────────────────────────────────────────────────────


def test_recall_finds_stored_entry():
    memory_tool.run(
        json.dumps(
            {
                "action": "remember",
                "kind": "decision",
                "content": "use podman",
                "project": "forge",
            }
        )
    )
    out = memory_tool.run(json.dumps({"action": "recall", "query": "podman"}))
    assert "use podman" in out
    assert "decision/forge" in out


def test_recall_no_matches():
    out = memory_tool.run(json.dumps({"action": "recall", "query": "anything"}))
    assert out == "No matching memory found."


def test_recall_rejects_empty_query():
    out = memory_tool.run(json.dumps({"action": "recall", "query": "  "}))
    assert out.startswith("[error]")


def test_recall_filters_by_kind():
    memory_tool.run(
        json.dumps(
            {"action": "remember", "kind": "decision", "content": "une décision prise"}
        )
    )
    memory_tool.run(
        json.dumps(
            {"action": "remember", "kind": "todo", "content": "une tâche à faire"}
        )
    )
    out = memory_tool.run(
        json.dumps({"action": "recall", "query": "anything", "kind": "todo"})
    )
    assert "[todo" in out
    assert "[decision" not in out


def test_recall_reports_embedding_failure(monkeypatch):
    def _raise(text):
        raise rag.EmbeddingError("connection refused")

    monkeypatch.setattr(rag, "_embed", _raise)
    out = memory_tool.run(json.dumps({"action": "recall", "query": "x"}))
    assert "recall failed" in out


# ── dispatch / contract ─────────────────────────────────────────────


def test_invalid_json_is_reported_not_raised():
    out = memory_tool.run("not json at all")
    assert out.startswith("[error]")


def test_unknown_action_is_reported():
    out = memory_tool.run(json.dumps({"action": "bogus"}))
    assert out.startswith("[error]")


def test_run_always_returns_non_empty_str():
    """Matches the tool contract enforced in test_tool_contract.py."""
    for payload in [
        json.dumps(
            {"action": "remember", "kind": "decision", "content": "entrée de test"}
        ),
        json.dumps({"action": "recall", "query": "x"}),
        json.dumps({"action": "bogus"}),
        "garbage",
    ]:
        out = memory_tool.run(payload)
        assert isinstance(out, str)
        assert out.strip()


# ── search() (structured form used by graphs/recall.py) ────────────────


def test_search_returns_raw_dicts():
    memory_tool.run(
        json.dumps({"action": "remember", "kind": "fact", "content": "Possède un Deck"})
    )

    results = memory_tool.search("Deck")

    assert isinstance(results, list)
    assert results[0]["kind"] == "fact"
    assert results[0]["content"] == "Possède un Deck"


def test_search_is_unranked_unclipped():
    """
    Ranking (fact before history_summary) and clipping belong to
    format_results(), the formatter -- search() itself hands back
    rag.search()'s raw order and full content, so graphs/recall.py's
    prompt-building and this module's own _recall() can each decide
    how to present it without duplicating the RAG query.
    """
    conn = rag.get_connection()
    rag.remember(
        conn, kind="history_summary", content="archiver les logs", project=None
    )
    rag.remember(conn, kind="fact", content="un fait noté", project=None)
    conn.close()

    results = memory_tool.search("q")

    # Both kinds present, un-clipped content, no reordering applied.
    kinds = {r["kind"] for r in results}
    assert kinds == {"history_summary", "fact"}


def test_search_raises_on_embedding_failure(monkeypatch):
    def raise_error(text):
        raise rag.EmbeddingError("400 Bad Request")

    monkeypatch.setattr(rag, "_embed", raise_error)

    with pytest.raises(rag.EmbeddingError):
        memory_tool.search("q")
