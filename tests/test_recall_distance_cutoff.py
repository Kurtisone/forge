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


def test_the_env_example_value_and_the_config_note_agree():
    """
    The threshold is now stated in two files -- an active line in
    .env.example and the reasoning in config.py -- and two places
    stating the same number is how one of them goes stale. This is the
    same drift guard the pointer builder and its regex have.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    env = (root / ".env.example").read_text()
    config = (root / "src" / "forge" / "config.py").read_text()

    active = re.search(r"^RECALL_MAX_DISTANCE=([\d.]+)(?:@\w+)?$", env, re.MULTILINE)
    assert active, ".env.example no longer sets RECALL_MAX_DISTANCE"

    value = active.group(1)
    assert f"{value} AND NOT THE MIDPOINT" in config or f"{value}," in config, (
        f".env.example sets {value} but config.py does not explain that number"
    )


def test_the_cutoff_is_still_off_when_unset():
    """
    .env.example carries a value now; the code default must not. A
    deployment that has not measured its own store gets no filtering
    rather than someone else's threshold.
    """
    import importlib
    import os

    import forge.config as config_module

    saved = os.environ.pop("RECALL_MAX_DISTANCE", None)
    try:
        assert importlib.reload(config_module).RECALL_MAX_DISTANCE is None
    finally:
        if saved is not None:
            os.environ["RECALL_MAX_DISTANCE"] = saved
        importlib.reload(config_module)


class TestCalibrationRegime:
    """
    A threshold is a measurement, and a measurement belongs to the
    configuration it was taken in. On 2026-08-23 EMBEDDING_QUERY_INSTRUCT
    shipped on by default, every distance in the store moved, and
    .env.example kept a 0.95 measured on raw queries -- above the best
    miss of the new regime, so the filter validated in real use no
    longer cut it. Only a comment said so.
    """

    def test_no_cutoff_configured_says_nothing(self):
        from forge.graphs import recall

        assert recall.cutoff_for(None, None, "a5c47b") == (None, None)

    def test_a_matching_tag_is_used_silently(self):
        from forge.graphs import recall

        assert recall.cutoff_for(0.92, "a5c47b", "a5c47b") == (0.92, None)

    def test_an_untagged_value_still_works_and_says_what_to_write(self):
        from forge.graphs import recall

        cutoff, note = recall.cutoff_for(0.95, None, "a5c47b")
        # Values predate the tag. Breaking a working deployment to make
        # a point about provenance would be its own kind of wrong.
        assert cutoff == 0.95
        assert "0.95@a5c47b" in note

    def test_a_stale_tag_turns_the_cutoff_off(self):
        from forge.graphs import recall

        cutoff, note = recall.cutoff_for(0.95, "e3b0c4", "a5c47b")
        # Off, not adjusted. A number from another regime is not too
        # high or too low, it is unrelated -- and of the two ways to be
        # wrong, "je n'ai rien en mémoire" while holding the answer is
        # the one that gets believed.
        assert cutoff is None
        assert "e3b0c4" in note and "a5c47b" in note

    def test_the_env_example_tag_is_the_regime_it_was_measured_in(self):
        """
        e3b0c4 is the fingerprint of no instruction at all, which is
        how that 0.95 was measured. If this ever equals the fingerprint
        of the shipped configuration, someone has re-tagged a number
        without re-measuring it.
        """
        import hashlib
        import re
        from pathlib import Path

        from forge import rag

        env = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
        tag = re.search(r"^RECALL_MAX_DISTANCE=[\d.]+@(\w+)$", env, re.MULTILINE)
        assert tag, ".env.example no longer tags the threshold with its regime"
        assert tag.group(1) == hashlib.sha256(b"").hexdigest()[:6]
        assert tag.group(1) != rag.query_fingerprint()


class TestQueryFingerprint:
    def test_it_covers_the_wrapper_and_not_just_the_setting(self, monkeypatch):
        """
        The "Instruct:/Query:" shape is part of what the model sees, so
        changing it is a change of regime even at the same setting.
        """
        from forge import rag

        monkeypatch.setattr(rag, "EMBEDDING_QUERY_INSTRUCT", "retrieve the answer")
        with_wrapper = rag.query_fingerprint()
        monkeypatch.setattr(rag, "EMBEDDING_QUERY_INSTRUCT", "")
        assert rag.query_fingerprint() != with_wrapper

    def test_no_instruction_hashes_the_empty_query(self, monkeypatch):
        import hashlib

        from forge import rag

        monkeypatch.setattr(rag, "EMBEDDING_QUERY_INSTRUCT", "")
        assert rag.query_fingerprint() == hashlib.sha256(b"").hexdigest()[:6]
