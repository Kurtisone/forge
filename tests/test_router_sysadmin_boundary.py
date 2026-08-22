"""
Tests for the research/sysadmin boundary in the router prompt.

Defect (3) of the v3.11 list, observed live: "Pourquoi searxng a
redémarré ?" routed to `research`, which then searched the public web
for the state of a container running on the Steam Deck.

It is not a silly mistake. `research` describes itself as the default
for "an actual answer about something current/live", and a past-tense
question about an event that just happened is exactly that shape. What
the prompt never said is that `research` cannot see this machine.

Same failure mode as the review/files ambiguity in v3.10, and treated
the same way: name the boundary in BOTH descriptions (the router reads
them all, so a one-sided disclaimer leaves the other tool still
advertising itself for the case) and pin the surface form that lost.

These assert the PROMPT, not the routing -- routing needs the model.
The real check is the live run.
"""

from forge.router.prompt import _TOOL_EXAMPLES, TOOL_DESCRIPTIONS


def test_research_says_it_cannot_see_this_machine():
    text = TOOL_DESCRIPTIONS["research"]

    assert "PUBLIC WEB" in text
    assert "sysadmin" in text


def test_sysadmin_claims_past_tense_questions():
    """
    The half that matters most: without it, `sysadmin` still describes
    itself only in terms of things that are broken right now
    ("X plante", "X ne répond plus"), and a question about last night
    reads as belonging to whoever claims recent events.
    """
    text = TOOL_DESCRIPTIONS["sysadmin"]

    assert "past-tense" in text
    assert "research" in text


def test_the_phrase_that_misrouted_is_pinned_as_an_example():
    prompts = [user for user, _ in _TOOL_EXAMPLES["sysadmin"]]

    assert any("a redémarré" in p for p in prompts), (
        "the exact phrasing observed misrouting to research must stay "
        "in the examples -- it is the regression, not an illustration"
    )


def test_both_descriptions_name_each_other():
    """
    A boundary stated on one side only is how review/files drifted:
    each description was individually correct and together they still
    left the case unclaimed.
    """
    assert "sysadmin" in TOOL_DESCRIPTIONS["research"]
    assert "research" in TOOL_DESCRIPTIONS["sysadmin"]
