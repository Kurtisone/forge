"""
Tests for how often the hard cap evicts, not just how much.

MEMORY_MAX_HISTORY was raised to 100 in v3.8 because a sliding FIFO
window fights KV-cache reuse: every eviction shifts the router
prompt's prefix and forces llama-server to re-process all of it. The
hard cap trimmed to exactly the cap, which put the next turn straight
back over it, so once history was full it evicted on every single
turn -- the same FIFO, arriving through the safety net. These tests
pin the eviction *rate*, which is the property that matters.
"""

from forge import memory


def _fresh(tmp_path, monkeypatch, cap=10, slack=4):
    monkeypatch.setattr(memory, "MEMORY_FILE", str(tmp_path / "memory.json"))
    monkeypatch.setattr(memory, "MEMORY_MAX_HISTORY", cap)
    monkeypatch.setattr(memory, "MEMORY_HARD_CAP_SLACK", slack)
    monkeypatch.setattr(memory.compaction, "COMPACTION_ENABLED", False)


def _ids(history):
    return [m["id"] for m in history]


def test_trimming_lands_below_the_cap(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch, cap=10, slack=4)

    for i in range(11):
        memory.add_message("user", f"msg{i}")

    assert len(memory.get_history()) == 6  # 10 - 4, not 10


def test_the_prefix_is_stable_between_evictions(tmp_path, monkeypatch):
    """
    The oldest kept message must stay put for several turns in a row.
    If it moves every turn, the cacheable prompt prefix moves with it.
    """
    _fresh(tmp_path, monkeypatch, cap=10, slack=4)

    for i in range(11):
        memory.add_message("user", f"msg{i}")

    oldest_ids = []
    for i in range(11, 15):
        memory.add_message("user", f"msg{i}")
        oldest_ids.append(_ids(memory.get_history())[0])

    assert len(set(oldest_ids)) == 1


def test_eviction_is_rare_rather_than_per_turn(tmp_path, monkeypatch):
    _fresh(tmp_path, monkeypatch, cap=10, slack=4)

    evictions = 0
    previous_oldest = None
    for i in range(40):
        memory.add_message("user", f"msg{i}")
        oldest = _ids(memory.get_history())[0]
        if previous_oldest is not None and oldest != previous_oldest:
            evictions += 1
        previous_oldest = oldest

    # 40 turns, one eviction per SLACK+1 turns once full -- not 30.
    assert evictions <= 8


def test_slack_larger_than_the_cap_still_keeps_something(tmp_path, monkeypatch):
    """A misconfiguration must not empty the history outright."""
    _fresh(tmp_path, monkeypatch, cap=3, slack=99)

    for i in range(10):
        memory.add_message("user", f"msg{i}")

    history = memory.get_history()
    assert len(history) == 1
    assert history[-1]["content"] == "msg9"
