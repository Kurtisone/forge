"""
Unit tests for forge.compaction. rag.get_connection/rag.remember are
monkeypatched so no real SQLite/embedding server is needed -- these
tests only care about compaction's own eviction/strategy logic.
"""

import pytest

from forge import compaction, rag


def _messages(n: int, pinned_ids: set[int] | None = None) -> list[dict]:
    pinned_ids = pinned_ids or set()
    return [
        {
            "id": i,
            "role": "user" if i % 2 == 0 else "assistant",
            "content": f"msg{i}",
            "pinned": i in pinned_ids,
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _fake_rag(monkeypatch):
    """rag_pointer strategy is the default -- stub its one dependency."""

    class _FakeConn:
        def close(self):
            pass

    monkeypatch.setattr(rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(
        rag, "remember", lambda conn, kind, content, project: 42, raising=False
    )
    # compaction imported `rag` directly, patch the same module object
    monkeypatch.setattr(compaction.rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(
        compaction.rag, "remember", lambda conn, kind, content, project: 42
    )


def test_below_threshold_is_untouched(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 80)
    history = _messages(10)

    assert compaction.maybe_compact(history) == history


def test_compacts_oldest_non_pinned_above_threshold(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 10)
    monkeypatch.setattr(compaction, "COMPACTION_KEEP_RECENT", 4)
    history = _messages(12)

    result = compaction.maybe_compact(history)

    # 12 - 4 = 8 messages compacted into one summary; 4 most recent kept.
    assert len(result) == 5
    assert result[0]["role"] == "system"
    assert "8 messages" in result[0]["content"]
    assert [m["id"] for m in result[1:]] == [8, 9, 10, 11]


def test_pinned_messages_are_never_compacted(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 10)
    monkeypatch.setattr(compaction, "COMPACTION_KEEP_RECENT", 4)
    history = _messages(12, pinned_ids={0, 1})

    result = compaction.maybe_compact(history)

    result_ids = [m["id"] for m in result if m.get("pinned")]
    assert result_ids == [0, 1]
    # pinned messages must survive verbatim, not folded into the summary
    assert result[0]["id"] == 0
    assert result[1]["id"] == 1


def test_nothing_eligible_when_all_pinned(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 5)
    monkeypatch.setattr(compaction, "COMPACTION_KEEP_RECENT", 4)
    history = _messages(10, pinned_ids=set(range(10)))

    assert compaction.maybe_compact(history) == history


def test_disabled_returns_history_unchanged(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_ENABLED", False)
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 1)
    history = _messages(20)

    assert compaction.maybe_compact(history) == history


def test_force_compacts_even_below_threshold(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_ENABLED", False)
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 999)
    monkeypatch.setattr(compaction, "COMPACTION_KEEP_RECENT", 2)
    history = _messages(5)

    result = compaction.maybe_compact(history, force=True)

    assert len(result) == 3
    assert result[0]["role"] == "system"


def test_embedding_failure_raises_compaction_error(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 1)
    monkeypatch.setattr(compaction, "COMPACTION_KEEP_RECENT", 1)

    def _boom(conn, kind, content, project):
        raise rag.EmbeddingError("embedding server down")

    monkeypatch.setattr(compaction.rag, "remember", _boom)
    history = _messages(5)

    with pytest.raises(compaction.CompactionError):
        compaction.maybe_compact(history)


def test_llm_summary_strategy(monkeypatch):
    monkeypatch.setattr(compaction, "COMPACTION_STRATEGY", "llm_summary")
    monkeypatch.setattr(compaction, "COMPACTION_THRESHOLD", 1)
    monkeypatch.setattr(compaction, "COMPACTION_KEEP_RECENT", 1)
    monkeypatch.setattr("forge.llm.call_llm", lambda prompt: "résumé condensé")
    history = _messages(5)

    result = compaction.maybe_compact(history)

    assert result[0]["role"] == "system"
    assert "résumé condensé" in result[0]["content"]
