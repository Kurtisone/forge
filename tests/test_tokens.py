"""
The local estimator and the drift check that keeps it honest.

The calibration samples below are real: each pairs the length of a
prompt Forge actually sent with the prompt_tokens llama-server reported
for that same call, measured on the Deck on 2026-08-16. They are the
evidence CHARS_PER_TOKEN rests on, so they belong in a test rather than
only in a docstring -- if a future change breaks the ratio for the
prompts Forge really sends, this is what says so.
"""

import forge.llm as llm_mod
from forge.tokens import CHARS_PER_TOKEN, estimate_messages, estimate_tokens

# (label, chars, tokens actually reported)
CALIBRATION = [
    ("router prompt", 10927, 2727),
    ("router prompt", 11011, 2751),
    ("router prompt", 10922, 2722),
    ("router prompt", 10912, 2721),
    ("sysadmin synthesis", 1638, 409),
    ("review synthesis", 1259, 314),
    ("research synthesis", 1384, 345),
    ("recall fixture", 1326, 319),
    ("recall in production", 2977, 810),
    # Added 2026-08-16 from the first live run of this instrumentation.
    ("router prompt, 12-msg history", 11944, 3014),
    ("router prompt, 14-msg history", 12130, 3069),
]


def test_the_estimate_never_understates_a_measured_prompt():
    """The whole point of picking the low end of the observed range.
    Understating is what overflows a context window; overstating just
    compacts a little early."""
    for label, chars, actual in CALIBRATION:
        estimated = estimate_tokens("x" * chars)
        assert estimated >= actual, f"{label}: {estimated} < {actual}"


def test_the_estimate_stays_within_a_useful_margin():
    """Biased high, but not so high it becomes useless -- an estimator
    that doubles every number would pass the test above and still be
    worthless."""
    for label, chars, actual in CALIBRATION:
        estimated = estimate_tokens("x" * chars)
        assert estimated <= actual * 1.20, f"{label}: {estimated} > {actual} + 20%"


def test_the_ratio_sits_below_every_measured_sample():
    """Documents WHY the constant is not the ~4.0 mean: one real prompt
    came in at 3.675 chars/token. Strictly below, not equal -- landing
    on the observed minimum leaves no room for the next sample, and a
    first draft that rounded 3.675 up to a tidy 3.7 understated that
    very prompt."""
    observed = [chars / actual for _, chars, actual in CALIBRATION]
    assert CHARS_PER_TOKEN < min(observed)


def test_empty_and_tiny_inputs():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1


def test_messages_count_their_framing():
    """Two messages cost more than their concatenated text: each one
    carries a role prefix and a separator."""
    msgs = [
        {"role": "user", "content": "bonjour"},
        {"role": "assistant", "content": "salut"},
    ]
    assert estimate_messages(msgs) > estimate_tokens("bonjoursalut")


def test_malformed_entries_do_not_raise():
    """This runs on the persistence path. A crash here would cost the
    user the turn they just took."""
    assert estimate_messages([{"role": "user"}, {}, "not a dict", None]) >= 0


class _Usage:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens


def _drift_events(monkeypatch):
    seen = []
    monkeypatch.setattr(
        llm_mod.log, "event", lambda name, **kw: seen.append((name, kw))
    )
    return seen


def test_no_drift_logged_when_the_estimate_is_close(monkeypatch):
    seen = _drift_events(monkeypatch)
    llm_mod._check_estimate("x" * 10927, 2727)
    assert not [e for e in seen if e[0] == "tokens.estimate_drift"]


def test_drift_logged_when_the_estimate_is_far_off(monkeypatch):
    seen = _drift_events(monkeypatch)
    llm_mod._check_estimate("x" * 10927, 1000)
    drift = [kw for name, kw in seen if name == "tokens.estimate_drift"]
    assert len(drift) == 1
    assert drift[0]["actual"] == 1000
    assert drift[0]["error_pct"] > 0


def test_a_silent_backend_is_not_a_drift_signal(monkeypatch):
    """prompt_tokens is None whenever the provider reports no counts --
    that is the case the estimator exists to cover, not an anomaly to
    warn about."""
    seen = _drift_events(monkeypatch)
    llm_mod._check_estimate("x" * 500, None)
    llm_mod._check_estimate("x" * 500, 0)
    assert not seen
