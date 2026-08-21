"""
On 2026-08-19 a recall answer welded two unrelated memories into one
invented causality. The log of the day recorded how many entries came
back and nothing else, so it could not distinguish the retrieval
handing the synthesis two unrelated entries from the synthesis
inventing a link between two related ones -- and those want opposite
fixes.

rag.search has always selected v.distance and nothing has ever read
it. These pin that recall now logs it.
"""

from forge.graphs import recall as recall_mod
from forge.types import AgentState


def _run(monkeypatch, results):
    events = []
    monkeypatch.setattr(
        recall_mod.log, "event", lambda name, **f: events.append((name, f))
    )
    monkeypatch.setattr(recall_mod.memory_tool, "search", lambda q: results)

    state = AgentState(user_input="cache KV", max_steps=4)
    state.context = {"query": "cache KV"}
    recall_mod._recall_node(state)
    return dict(events)["recall.search"]


def test_the_entries_behind_an_answer_are_recorded(monkeypatch):
    fields = _run(
        monkeypatch,
        [
            {
                "id": 12,
                "kind": "fact",
                "content": "cache KV désactivé",
                "distance": 0.31,
            },
            {
                "id": 47,
                "kind": "note",
                "content": "pagination cassée",
                "distance": 0.88,
            },
        ],
    )

    assert fields["results"] == 2
    assert [e["id"] for e in fields["entries"]] == [12, 47]
    # The number that says whether the second hit was a close match or
    # merely the least bad of five.
    assert [e["distance"] for e in fields["entries"]] == [0.31, 0.88]
    assert fields["entries"][0]["kind"] == "fact"


def test_entry_content_is_clipped(monkeypatch):
    """Enough to recognise the entry, not enough to put the whole
    memory store in the log on every recall."""
    fields = _run(
        monkeypatch,
        [{"id": 1, "kind": "fact", "content": "x" * 500, "distance": 0.1}],
    )
    assert len(fields["entries"][0]["head"]) == 80


def test_a_missing_distance_does_not_break_the_log(monkeypatch):
    """Nothing has ever read this field, so nothing has ever kept it
    present. A diagnostic that raises on the shape it is diagnosing is
    worse than no diagnostic."""
    fields = _run(monkeypatch, [{"id": 1, "kind": "fact", "content": "x"}])
    assert fields["entries"][0]["distance"] is None
