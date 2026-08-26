"""
Tests for the hot tier: the deliberate store, whole, in the prompt.

The invariant that matters most here is the boring one -- with the
knob off, the synthesis prompt is byte-identical to what it was. Every
mechanism in docs/memory.md ships off and is earned by a harness, and
a mechanism that quietly reshapes the prompt while "off" cannot be
measured against the arm it is supposed to be compared with.
"""

import pytest

from forge import hot_memory, tokens
from forge.graphs import recall


@pytest.fixture
def entries():
    return [
        {"id": 1, "kind": "fact", "project": None, "content": "Possède un Steam Deck"},
        {
            "id": 2,
            "kind": "decision",
            "project": "chat preferences",
            "content": "Ne pas épingler les messages avec des emojis",
        },
        {
            "id": 3,
            "kind": "fact",
            "project": None,
            "content": "Le NiPoGi a 32 Go de RAM",
        },
    ]


def test_off_leaves_the_prompt_byte_for_byte(monkeypatch):
    monkeypatch.setattr(hot_memory, "RECALL_HOT_FACTS", False)

    assert hot_memory.block() == ""

    with_section = recall._build_prompt("q", "- [fact] x", "", hot_memory.block())
    without = recall._build_prompt("q", "- [fact] x", "")
    assert with_section == without


def test_the_block_sits_above_the_question(entries):
    prompt = recall._build_prompt(
        "Tu peux me lister mon matériel ?",
        "- [fact] Possède un Steam Deck",
        "",
        f"\n{hot_memory.render(entries)}\n",
    )

    # Not decoration. Everything after "Question:" changes on every
    # turn, so a block placed below it can never be a shared prefix
    # however stable its own content is.
    assert prompt.index("Le NiPoGi a 32 Go de RAM") < prompt.index("Question:")


def test_rendering_carries_entries_whole_and_in_store_order(entries):
    rendered = hot_memory.render(entries)

    assert rendered.splitlines() == [
        "- [fact] Possède un Steam Deck",
        "- [decision/chat preferences] Ne pas épingler les messages avec des emojis",
        "- [fact] Le NiPoGi a 32 Go de RAM",
    ]


def test_nothing_is_clipped(entries):
    long_entry = {"id": 4, "kind": "fact", "project": None, "content": "x" * 5000}

    rendered = hot_memory.render([long_entry])

    # tools/memory.format_results clips at MEMORY_RECALL_MAX_CHARS,
    # which is right for a ranked list of search hits and wrong here:
    # this tier is defined by carrying what it carries whole.
    assert rendered.count("x") == 5000
    assert "[…]" not in rendered


def test_everything_fits_when_the_budget_is_not_reached(entries):
    kept, dropped = hot_memory.fit(entries, budget=1000)

    assert kept == entries
    assert dropped == 0


def test_overflow_cuts_the_tail_and_leaves_the_survivors_untouched(entries):
    budget = tokens.estimate_tokens(hot_memory.render(entries[:2])) + 2

    kept, dropped = hot_memory.fit(entries, budget)

    assert dropped == 1
    # The surviving lines are the same lines, in the same place. Any
    # rule that reorders (by age, by relevance, by pin) is a ranking
    # under another name and rewrites the block on every write.
    assert hot_memory.render(kept) == "\n".join(
        hot_memory.render(entries).splitlines()[: len(kept)]
    )


def test_truncation_is_visible_inside_the_block(monkeypatch, entries):
    monkeypatch.setattr(hot_memory, "RECALL_HOT_FACTS", True)
    monkeypatch.setattr(hot_memory, "RECALL_HOT_MAX_TOKENS", 12)
    monkeypatch.setattr(hot_memory.rag, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(hot_memory.rag, "hot_entries", lambda conn: entries)

    section = hot_memory.block()

    # A silently short inventory reads complete and is wrong, which is
    # worse than the visibly incomplete answer this replaces. The
    # sentinel is the only thing standing between this mechanism and
    # being able to make Forge worse than not having it.
    assert "TRUNCATED" in section
    assert "incomplete" in section


def test_an_unreadable_store_costs_nothing(monkeypatch):
    monkeypatch.setattr(hot_memory, "RECALL_HOT_FACTS", True)

    def boom():
        raise RuntimeError("disk is on fire")

    monkeypatch.setattr(hot_memory.rag, "get_connection", boom)

    # A synthesis that would have happened without this block still
    # happens. This tier adds an answer where there was none; it must
    # never remove one.
    assert hot_memory.block() == ""


class _FakeConn:
    def close(self):
        pass
