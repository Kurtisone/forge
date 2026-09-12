"""
The leak markers must be text the prompt actually contains.

`_PROMPT_LEAK_MARKERS` is how the router recognises its own
instructions coming back at it as an answer. It is a closed set of
literals typed by hand next to a prompt that is edited independently,
which is the arrangement that fails silently: reword the prompt and the
marker stops matching, and nothing anywhere reports it. Found on
2026-09-12 with "No explanation or text outside the JSON" still first
in the list long after the prompt started saying "NEVER add text
outside the JSON".

This closes the direction a test can close -- a marker that no longer
exists. The other direction, prompt text that no marker covers, is
open by construction: every sentence in the prompt is a candidate, and
the one that leaked in the field (the search-chaining instruction) was
registered only after a model produced it.
"""

import pytest

from forge.router.parser import _PROMPT_LEAK_MARKERS
from forge.router.prompt import build_router_prompt
from forge.tools import registry


@pytest.fixture(scope="module")
def prompts():
    """Every prompt variant a marker could plausibly live in."""
    registry.load_tools()
    tools = sorted(registry.available_tools())
    return {
        "bare": build_router_prompt(
            "salut", history=[], step_context=[], available_tools=tools
        ),
        "with history": build_router_prompt(
            "salut",
            history=[
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ],
            step_context=[],
            available_tools=tools,
        ),
        "after a web_search step": build_router_prompt(
            "salut",
            history=[],
            step_context=[
                {
                    "role": "assistant",
                    "content": "[web_search] 1. X - https://x - snippet",
                }
            ],
            available_tools=tools,
        ),
    }


@pytest.mark.parametrize("marker", _PROMPT_LEAK_MARKERS)
def test_every_marker_appears_in_a_prompt_we_build(marker, prompts):
    where = [name for name, text in prompts.items() if marker in text]
    assert where, (
        f"leak marker {marker!r} appears in no prompt this code builds, "
        "so it can never fire. Either the prompt was reworded and the "
        "marker was left behind, or the marker was never right."
    )


def test_the_markers_are_not_words_a_real_answer_could_contain(prompts):
    """
    A marker is allowed to be silent-by-accident, never
    loud-by-accident: matching it discards the model's whole reply.
    Short or generic strings are the failure mode -- "User:" was
    rejected for this reason when the history block was reshaped.
    """
    for marker in _PROMPT_LEAK_MARKERS:
        assert len(marker) >= 13, (
            f"{marker!r} is short enough to appear in a real answer, and "
            "a false positive here costs the user their whole reply"
        )
