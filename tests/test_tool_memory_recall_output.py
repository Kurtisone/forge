"""
Tests for what recall actually hands back to the router.

Reproduces the shape of a real run: asked to list hardware, recall
returned 5 hits totalling 7008 characters. Four were "history_summary"
entries -- whole conversation blocks that compaction had archived
verbatim -- and the two lines that answered the question were buried
in them. The router then re-read all of it on its next decision, paid
for twice, in context window and in prefill time.

rag._embed is monkeypatched, same boundary as test_tool_memory.py.
"""

import json

import pytest

from forge import rag
from forge.tools import memory as memory_tool


@pytest.fixture(autouse=True)
def _rag_setup(tmp_path, monkeypatch):
    monkeypatch.setattr(rag, "RAG_DB_FILE", str(tmp_path / "rag.db"))
    monkeypatch.setattr(rag, "_embed", lambda text: [0.1] * rag.EMBEDDING_DIM)


def _store(kind: str, content: str) -> None:
    conn = rag.get_connection()
    try:
        rag.remember(conn, kind=kind, content=content, project=None)
    finally:
        conn.close()


def _recall(**extra) -> str:
    return memory_tool.run(json.dumps({"action": "recall", "query": "q", **extra}))


# ── clipping ─────────────────────────────────────────────────────────


def test_a_long_entry_is_clipped(monkeypatch):
    monkeypatch.setattr(memory_tool, "MEMORY_RECALL_MAX_CHARS", 50)
    _store("history_summary", "user: " + "bavardage " * 200)

    out = _recall()

    assert len(out) < 120
    assert out.endswith("[…]")


def test_a_short_entry_is_untouched(monkeypatch):
    monkeypatch.setattr(memory_tool, "MEMORY_RECALL_MAX_CHARS", 50)
    _store("fact", "Possède un Steam Deck")

    assert _recall() == "- [fact] Possède un Steam Deck"


def test_clipping_bounds_the_whole_answer(monkeypatch):
    """Five archived blocks must not add up to thousands of characters."""
    monkeypatch.setattr(memory_tool, "MEMORY_RECALL_MAX_CHARS", 100)
    for i in range(5):
        _store("history_summary", f"bloc {i}: " + "transcription " * 300)

    out = _recall()

    assert len(out) < 5 * 160


# ── ranking ──────────────────────────────────────────────────────────


def test_facts_come_before_archived_transcript():
    _store("history_summary", "user: on parlait de pommes Fuji")
    _store("history_summary", "user: et encore de pommes Fuji")
    _store("fact", "Possède un Steam Deck")
    _store("fact", "Possède un Dell R710")

    lines = _recall().splitlines()

    assert all("history_summary" not in line for line in lines[:2])
    assert all("history_summary" in line for line in lines[2:])


def test_ranking_is_stable_within_a_group():
    """Distance order still decides among entries of the same class."""
    results = [
        {"kind": "history_summary", "content": "archive proche", "distance": 0.1},
        {"kind": "fact", "content": "fait proche", "distance": 0.2},
        {"kind": "history_summary", "content": "archive lointaine", "distance": 0.3},
        {"kind": "decision", "content": "décision lointaine", "distance": 0.4},
    ]

    ranked = [r["content"] for r in memory_tool._rank(results)]

    # Groups swap, order inside each group is preserved.
    assert ranked == [
        "fait proche",
        "décision lointaine",
        "archive proche",
        "archive lointaine",
    ]


def test_a_single_archived_hit_is_still_returned():
    """Deprioritised is not filtered out -- it's all there is here."""
    _store("history_summary", "user: une vieille conversation")

    assert "history_summary" in _recall()


def test_kind_filter_still_wins():
    """An explicit kind filter is the caller's decision, not ours."""
    _store("fact", "Possède un Steam Deck")
    _store("history_summary", "user: bavardage")

    out = _recall(kind="history_summary")

    assert "history_summary" in out
    assert "Steam Deck" not in out
