"""
Tests for the recall distance cutoff.

The cutoff is DISABLED by default and these tests mostly pin that,
because the default is the decision: every distance measured so far
comes from queries with no answer in the store, and a threshold set
from misses alone silences real hits with no visible symptom.

bench/recall_distance.py is what produces the missing half.
"""

import pytest

from forge.graphs import recall


def _hits(*distances):
    return [
        {"id": i, "kind": "note", "content": f"entry {i}", "distance": d}
        for i, d in enumerate(distances)
    ]


def test_no_cutoff_means_no_filtering(monkeypatch):
    monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", None)
    results = _hits(0.1, 0.9, 1.4)

    assert recall._drop_distant(results, "q") == results


def test_a_cutoff_drops_only_what_is_beyond_it(monkeypatch):
    monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.8)

    kept = recall._drop_distant(_hits(0.1, 0.79, 0.8, 0.81, 1.4), "q")

    assert [r["distance"] for r in kept] == [0.1, 0.79, 0.8]


def test_a_hit_with_no_distance_is_kept(monkeypatch):
    """
    Failing closed on retrieval means answering "I have nothing" while
    holding the answer. The distance comes from rag.search's SELECT;
    any caller assembling hits another way must not be silently emptied.
    """
    monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.5)
    results = [{"id": 1, "kind": "note", "content": "x"}]

    assert recall._drop_distant(results, "q") == results


def test_dropping_everything_says_so_instead_of_answering(monkeypatch):
    """
    The point of the cutoff: the failure it replaces is a fluent
    sentence assembled from the five least-bad rows in the store.
    """
    monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.5)
    monkeypatch.setattr(
        recall.memory_tool, "search", lambda q, **kw: _hits(0.9, 0.95, 1.0)
    )

    def no_call(*a, **kw):  # pragma: no cover - must not run
        raise AssertionError("the model was asked to answer from nothing")

    monkeypatch.setattr(recall, "call_llm", no_call)

    from forge.types import AgentState

    state = AgentState(user_input="une question", context={}, max_steps=4)
    state = recall._recall_node(state)

    assert not state.ok
    assert "mémoire" in state.final_output


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_env_value_disables_the_cutoff(value, monkeypatch):
    """
    RECALL_MAX_DISTANCE= in a .env file must mean "off", not crash on
    float(""). .env files acquire empty keys.
    """
    monkeypatch.setenv("RECALL_MAX_DISTANCE", value)
    import importlib

    from forge import config

    importlib.reload(config)
    try:
        assert config.RECALL_MAX_DISTANCE is None
    finally:
        monkeypatch.delenv("RECALL_MAX_DISTANCE", raising=False)
        importlib.reload(config)
