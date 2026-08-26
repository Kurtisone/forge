"""
Tests for the condition that decides whether recall answers at all.

This is where the hot tier would have been invisible in exactly the
case it was built for. "Tu peux me lister mon matériel ?" is the
question graphs/recall.py fails on, and the way it fails is here: the
distance cutoff drops every row, _recall_node short-circuits to the
error node, and no LLM call happens at all -- measured on the Deck on
2026-08-22, 4 s instead of 17. A prompt carrying the whole deliberate
store would never have been built, let alone seen.

"Nothing close enough to the question" is not the same as "nothing to
answer from", and until this change the graph could not tell them
apart because it only ever had one source.
"""

import pytest

from forge import hot_memory, rag
from forge.graphs import recall
from forge.types import AgentState


@pytest.fixture
def state():
    return AgentState(
        user_input="Tu peux me lister mon matériel ?",
        context={"query": "Tu peux me lister mon matériel ?"},
        max_steps=4,
    )


def _no_hot_block(monkeypatch):
    monkeypatch.setattr(hot_memory, "RECALL_HOT_FACTS", False)


def _hot_block(monkeypatch, text="\n- [fact] Possède un Steam Deck\n"):
    monkeypatch.setattr(recall.hot_memory, "block", lambda: text)


def test_an_empty_store_still_refuses(monkeypatch, state):
    _no_hot_block(monkeypatch)
    monkeypatch.setattr(recall.memory_tool, "search", lambda *a, **k: [])

    out = recall._recall_node(state)

    assert out.ok is False
    assert out.error == "no results"


def test_the_cutoff_still_refuses_on_its_own(monkeypatch, state):
    _no_hot_block(monkeypatch)
    monkeypatch.setattr(
        recall.memory_tool,
        "search",
        lambda *a, **k: [{"id": 1, "kind": "fact", "content": "x", "distance": 9.9}],
    )
    monkeypatch.setattr(recall, "_drop_distant", lambda results, query: [])
    monkeypatch.setattr(recall, "_rescue", lambda query: [])

    out = recall._recall_node(state)

    assert out.ok is False
    assert out.error == "no results above the distance cutoff"


def test_the_hot_block_carries_the_run_past_the_cutoff(monkeypatch, state):
    _hot_block(monkeypatch)
    monkeypatch.setattr(
        recall.memory_tool,
        "search",
        lambda *a, **k: [{"id": 1, "kind": "fact", "content": "x", "distance": 9.9}],
    )
    monkeypatch.setattr(recall, "_drop_distant", lambda results, query: [])
    monkeypatch.setattr(recall, "_rescue", lambda query: [])

    out = recall._recall_node(state)

    assert out.ok is True
    assert out.context["hot_section"]


def test_the_hot_block_carries_the_run_past_an_empty_search(monkeypatch, state):
    _hot_block(monkeypatch)
    monkeypatch.setattr(recall.memory_tool, "search", lambda *a, **k: [])

    out = recall._recall_node(state)

    assert out.ok is True


def test_an_embedding_outage_is_still_reported(monkeypatch, state):
    _hot_block(monkeypatch)

    def down(*a, **k):
        raise rag.EmbeddingError("connection refused")

    monkeypatch.setattr(recall.memory_tool, "search", down)

    out = recall._recall_node(state)

    # The hot block needs no embedding server and could have answered
    # here. It deliberately does not: an outage is a fact about the
    # deployment, all three entry points of this store fail the same
    # predictable way on purpose, and answering fluently from a
    # partial capability is how an outage lasts a week.
    assert out.ok is False
    assert "connection refused" in out.error
