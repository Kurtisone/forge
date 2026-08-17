"""
Compaction on a token budget, alongside the message count.

Neither trigger replaces the other, and the reason is arithmetic. User
entries are capped at 4000 chars by router/prompt.py, assistant entries
at 120. So one exchange costs ~90 rendered tokens in ordinary chat and
~1150 at the ceiling -- a factor of thirteen. At the ceiling the
context window is gone in roughly 24 messages, and a threshold of 80
messages never fires in time. Going the other way, a token budget
cannot see a flood of tiny entries, which costs little in the prompt
and plenty in _format_history and in every persist.

The second thing pinned here is the target. COMPACTION_KEEP_RECENT
counts MESSAGES while the budget is in TOKENS, so how much one pass
frees is unpredictable. Landing just under the threshold means
compacting again next turn -- the per-turn eviction MEMORY_HARD_CAP_SLACK
exists to record, which reintroduces v3.8's sliding window and destroys
KV-cache reuse with no symptom other than latency.
"""

import pytest

import forge.compaction as comp


@pytest.fixture(autouse=True)
def _no_rag(monkeypatch):
    """rag_pointer would reach for an embedding server."""
    monkeypatch.setattr(
        comp,
        "_run_strategy",
        lambda messages: {"role": "system", "content": f"[summary of {len(messages)}]"},
    )


def _msgs(n, content="court", role_cycle=("user", "assistant")):
    return [
        {"role": role_cycle[i % 2], "content": content, "pinned": False}
        for i in range(n)
    ]


def _paste(n):
    """Exchanges at the ceiling: a 4000-char user entry is what
    router/prompt.py lets through untruncated."""
    out = []
    for i in range(n):
        out.append({"role": "user", "content": "x" * 4000, "pinned": False})
        out.append({"role": "assistant", "content": "y" * 500, "pinned": False})
    return out


def test_a_short_history_is_left_alone():
    h = _msgs(10)
    assert comp.maybe_compact(h) is h


def test_pastes_trigger_compaction_long_before_the_message_threshold(monkeypatch):
    """The gap this exists for. Well under 80 messages, and already
    over the token budget."""
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_THRESHOLD", 6000)
    h = _paste(8)  # 16 messages
    assert len(h) < comp.COMPACTION_THRESHOLD
    assert comp.estimate_history_tokens(h) > 6000
    assert len(comp.maybe_compact(h)) < len(h)


def test_many_tiny_messages_still_trigger_on_the_count(monkeypatch):
    """The reverse gap: cheap in the prompt, expensive everywhere else.
    A token budget alone would never notice."""
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_THRESHOLD", 100_000)
    h = _msgs(comp.COMPACTION_THRESHOLD + 2, content="ok")
    assert comp.estimate_history_tokens(h) < 100_000
    assert len(comp.maybe_compact(h)) < len(h)


def test_a_pass_reaches_the_target_not_merely_the_threshold(monkeypatch):
    """Landing just under the threshold means compacting again next
    turn. Aiming at the target buys turns."""
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_THRESHOLD", 6000)
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_TARGET", 3000)
    out = comp.maybe_compact(_paste(10))
    assert comp.estimate_history_tokens(out) <= 6000


def test_keep_recent_is_a_floor_not_a_quota(monkeypatch):
    """Twenty huge exchanges kept intact would leave the history over
    budget. KEEP_RECENT must yield to the budget, not the reverse."""
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_THRESHOLD", 6000)
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_TARGET", 3000)
    monkeypatch.setattr(comp, "COMPACTION_KEEP_RECENT", 20)
    out = comp.maybe_compact(_paste(15))  # 30 messages, all large
    kept = [m for m in out if m.get("role") != "system"]
    assert len(kept) < 20


def test_pinned_messages_are_never_compacted(monkeypatch):
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_THRESHOLD", 6000)
    h = _paste(8)
    h[0]["pinned"] = True
    out = comp.maybe_compact(h)
    assert sum(1 for m in out if m.get("pinned")) == 1


def test_a_pass_always_frees_something_even_when_the_target_is_unreachable(
    monkeypatch,
):
    """Pinned messages alone can exceed the target. Refusing to help at
    all would be worse than helping partially."""
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_THRESHOLD", 100)
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_TARGET", 1)
    monkeypatch.setattr(comp, "COMPACTION_KEEP_RECENT", 2)
    h = _paste(5)
    h[0]["pinned"] = True
    out = comp.maybe_compact(h)
    assert len(out) < len(h)
    assert out  # never empties the history


def test_nothing_eligible_is_a_no_op(monkeypatch):
    """Pinned messages alone over the threshold: there is nothing this
    pass is allowed to touch."""
    monkeypatch.setattr(comp, "COMPACTION_TOKEN_THRESHOLD", 10)
    h = _paste(3)
    for m in h:
        m["pinned"] = True
    assert comp.maybe_compact(h) is h
