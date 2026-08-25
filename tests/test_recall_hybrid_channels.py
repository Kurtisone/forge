"""
Two channels, two admission rules, and the line between them.

RECALL_MAX_DISTANCE=0.88@a5c47b was measured against one embedding
configuration, on distances produced by rag.search. These tests pin
that it is applied to those rows and to nothing else -- because the
entries the lexical channel exists to reach (#307, #313) are FAR in
vector space, and a cutoff applied to them would delete exactly what
the second channel was added to find.
"""

import pytest

from forge.graphs import recall
from forge.types import AgentState


def _vector(distance, entry_id=1):
    return {
        "id": entry_id,
        "kind": "fact",
        "content": f"entrée {entry_id}",
        "distance": distance,
        "channel": "vector",
    }


def _lexical(entry_id=2, score=-4.5):
    return {
        "id": entry_id,
        "kind": "fact",
        "content": "Steam Deck, SteamOS, conteneurs Podman",
        "score": score,
        "matched_terms": ["conteneurs", "podman"],
        "channel": "lexical",
    }


def _both(distance, entry_id=3):
    return {**_vector(distance, entry_id), "channel": "both", "score": -3.2}


@pytest.fixture
def cutoff(monkeypatch):
    monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)


def test_a_word_match_is_not_judged_by_the_vector_cutoff(cutoff):
    kept = recall._drop_distant([_vector(1.4), _lexical()], "q")

    assert [r["channel"] for r in kept] == ["lexical"]


def test_a_row_both_channels_found_survives_a_distance_above_the_cutoff(cutoff):
    """
    A union means one channel admitting a row is enough. Its distance
    is kept for the log and for ordering; it stops being a verdict.
    """
    kept = recall._drop_distant([_both(1.2)], "q")

    assert [r["id"] for r in kept] == [3]
    assert kept[0]["distance"] == 1.2


def test_the_cutoff_still_cuts_vector_rows(cutoff):
    kept = recall._drop_distant([_vector(0.5, 1), _vector(0.95, 2)], "q")

    assert [r["id"] for r in kept] == [1]


def test_an_untagged_row_is_still_judged(cutoff):
    """
    Everything written before v3.17 produces rows with no channel at
    all -- the direct memory:recall action, the bench harnesses, any
    caller assembling hits by hand. Those are vector rows and the
    cutoff is theirs.
    """
    kept = recall._drop_distant([{"id": 1, "kind": "fact", "distance": 1.4}], "q")

    assert kept == []


def test_the_knob_off_asks_for_the_vector_channel_alone(monkeypatch):
    seen = {}

    def _search(query, **kw):
        seen.update(kw)
        return [_vector(0.5)]

    monkeypatch.setattr(recall, "RECALL_LEXICAL", False)
    monkeypatch.setattr(recall.memory_tool, "search", _search)

    recall._recall_node(AgentState(user_input="q", context={}, max_steps=4))

    assert seen["lexical"] is False


def test_the_knob_on_asks_for_both(monkeypatch):
    seen = {}

    def _search(query, **kw):
        seen.update(kw)
        return [_vector(0.5)]

    monkeypatch.setattr(recall, "RECALL_LEXICAL", True)
    monkeypatch.setattr(recall, "RECALL_LEXICAL_TOP_K", 3)
    monkeypatch.setattr(recall.memory_tool, "search", _search)

    recall._recall_node(AgentState(user_input="q", context={}, max_steps=4))

    assert seen["lexical"] is True
    assert seen["lexical_top_k"] == 3


def test_the_retrieval_log_names_the_channel(monkeypatch, cutoff):
    """
    Without it the line cannot be read once there are two channels: a
    row with no distance is either a word match or a bug, and a row
    above the cutoff that survived is either a lexical hit or a filter
    that stopped working.
    """
    events = []
    monkeypatch.setattr(
        recall.log, "event", lambda name, **kw: events.append((name, kw))
    )
    monkeypatch.setattr(recall, "RECALL_LEXICAL", True)
    monkeypatch.setattr(
        recall.memory_tool, "search", lambda q, **kw: [_vector(0.5), _lexical()]
    )

    recall._recall_node(AgentState(user_input="q", context={}, max_steps=4))

    (search_event,) = [kw for name, kw in events if name == "recall.search"]
    assert [e["channel"] for e in search_event["entries"]] == ["vector", "lexical"]


def test_the_default_stays_off():
    """
    A retrieval mechanism nobody has measured on their own store does
    not turn itself on in a deployment nobody measured it in -- the
    argument RECALL_MAX_DISTANCE and RECALL_EXPANSION both ship on.
    """
    from forge import config

    assert config.RECALL_LEXICAL is False


def test_a_row_kept_on_words_alone_does_not_lead_on_its_distance(cutoff):
    """
    Seen on the real store, 2026-08-25: the cutoff dropped four rows
    between 0.9083 and 0.9833, and an unrelated transcript at 0.9796 --
    farther than the row dropped at 0.9083 -- led the survivors,
    because the word channel had also found it. It was archived, so
    memory._rank demoted it before the prompt; a fact in the same
    position would have led with nothing to catch it.
    """
    good = {**_lexical(entry_id=9, score=-8.0), "channel": "lexical"}
    survivor = _both(0.9796, entry_id=61)

    ordered = recall._demote_unmeasured([survivor, good])

    assert [r["id"] for r in ordered] == [9, 61]


def test_a_measured_row_still_leads(cutoff):
    ordered = recall._demote_unmeasured([_lexical(entry_id=9), _both(0.5, 61)])

    assert [r["id"] for r in ordered] == [61, 9]


def test_no_cutoff_leaves_the_order_alone():
    rows = [_both(1.2, 61), _lexical(entry_id=9)]

    assert recall._demote_unmeasured(rows) == rows
