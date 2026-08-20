"""
A compaction was seen firing on 2026-08-19 with the message threshold
nowhere near reached, and the log of the day could not settle why: it
named one trigger and neither the message count nor the thresholds in
force. These pin the line that can settle it.

Not tests of compaction behaviour -- test_compaction.py and
test_compaction_tokens.py own that. These are tests that the
instrument reports what happened, which is a separate promise and the
one that failed here.
"""

import forge.compaction as compaction_mod
from forge import api, compaction


def _messages(n, chars=10):
    return [
        {"id": i, "role": "user" if i % 2 == 0 else "assistant", "content": "x" * chars}
        for i in range(n)
    ]


def _capture(monkeypatch):
    events = []

    def fake_event(name, **fields):
        events.append((name, fields))

    monkeypatch.setattr(compaction_mod.log, "event", fake_event)
    return events


def _no_strategy(monkeypatch):
    """Compaction's strategies reach the embedding server or the LLM.
    Neither is the subject here."""
    monkeypatch.setattr(
        compaction_mod,
        "_run_strategy",
        lambda msgs: {
            "id": msgs[0]["id"],
            "role": "system",
            "content": "[…]",
            "pinned": False,
        },
    )


def test_the_event_carries_the_thresholds_it_was_measured_against(monkeypatch):
    _no_strategy(monkeypatch)
    events = _capture(monkeypatch)
    monkeypatch.setattr(compaction_mod, "COMPACTION_THRESHOLD", 4)
    monkeypatch.setattr(compaction_mod, "COMPACTION_KEEP_RECENT", 2)

    compaction.maybe_compact(_messages(10))

    name, fields = events[-1]
    assert name == "compaction.run"
    # The measured side and what it was measured against, together.
    assert fields["messages"] == 10
    assert fields["threshold_messages"] == 4
    assert fields["threshold_tokens"] == compaction_mod.COMPACTION_TOKEN_THRESHOLD
    assert fields["target_tokens"] == compaction_mod.COMPACTION_TOKEN_TARGET
    assert fields["keep_recent"] == 2


def test_both_triggers_are_named_when_both_fired(monkeypatch):
    """The old label was a first-match chain, so a pass that crossed
    the message threshold AND the token budget was reported as
    "tokens" -- and the count trigger became invisible in exactly the
    case where knowing about it matters."""
    _no_strategy(monkeypatch)
    events = _capture(monkeypatch)
    monkeypatch.setattr(compaction_mod, "COMPACTION_THRESHOLD", 4)
    monkeypatch.setattr(compaction_mod, "COMPACTION_TOKEN_THRESHOLD", 1)
    monkeypatch.setattr(compaction_mod, "COMPACTION_KEEP_RECENT", 2)

    compaction.maybe_compact(_messages(10))

    trigger = events[-1][1]["trigger"]
    assert "messages" in trigger and "tokens" in trigger


def test_a_forced_pass_says_so(monkeypatch):
    _no_strategy(monkeypatch)
    events = _capture(monkeypatch)
    monkeypatch.setattr(compaction_mod, "COMPACTION_THRESHOLD", 1000)
    monkeypatch.setattr(compaction_mod, "COMPACTION_TOKEN_THRESHOLD", 10**9)
    monkeypatch.setattr(compaction_mod, "COMPACTION_KEEP_RECENT", 2)

    compaction.maybe_compact(_messages(10), force=True)

    assert events[-1][1]["trigger"] == "forced"


def test_the_event_says_what_the_pass_bought(monkeypatch):
    """rendered_after is the number that predicts whether this runs
    again next turn. Landing just under the threshold reintroduces the
    per-turn eviction v3.8 removed, and its only symptom is latency."""
    _no_strategy(monkeypatch)
    events = _capture(monkeypatch)
    monkeypatch.setattr(compaction_mod, "COMPACTION_THRESHOLD", 4)
    monkeypatch.setattr(compaction_mod, "COMPACTION_KEEP_RECENT", 2)

    compaction.maybe_compact(_messages(20, chars=400))

    fields = events[-1][1]
    assert fields["rendered_after"] < fields["rendered_tokens"]


def test_nothing_is_logged_when_nothing_is_compacted(monkeypatch):
    _no_strategy(monkeypatch)
    events = _capture(monkeypatch)
    monkeypatch.setattr(compaction_mod, "COMPACTION_THRESHOLD", 1000)
    monkeypatch.setattr(compaction_mod, "COMPACTION_TOKEN_THRESHOLD", 10**9)

    compaction.maybe_compact(_messages(10))

    assert not [e for e in events if e[0] == "compaction.run"]


def test_startup_prints_the_settings_whose_wrong_value_only_costs_latency(monkeypatch):
    """LLAMA_CPP_CACHE_PROMPT was false in the container for weeks
    while the default said true. The reason it survived that long is
    that the effective value was printed nowhere."""
    events = []
    monkeypatch.setattr(api.log, "event", lambda name, **f: events.append((name, f)))

    api.log_effective_settings()

    name, fields = events[-1]
    assert name == "config.effective"
    for key in (
        "cache_prompt",
        "use_grammar",
        "compaction_threshold",
        "compaction_token_threshold",
        "compaction_strategy",
    ):
        assert key in fields
