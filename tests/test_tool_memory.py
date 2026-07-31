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
        json.dumps({"action": "remember", "kind": "note", "content": "x"})
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
        json.dumps({"action": "remember", "kind": "decision", "content": "x"})
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
        json.dumps({"action": "remember", "kind": "decision", "content": "a"})
    )
    memory_tool.run(json.dumps({"action": "remember", "kind": "todo", "content": "b"}))
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
        json.dumps({"action": "remember", "kind": "decision", "content": "x"}),
        json.dumps({"action": "recall", "query": "x"}),
        json.dumps({"action": "bogus"}),
        "garbage",
    ]:
        out = memory_tool.run(payload)
        assert isinstance(out, str)
        assert out.strip()
