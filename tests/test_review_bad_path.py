"""
A file_path that is not a path at all.

Seen live on 2026-08-17. Asked "Analyse et donne moi ton avis sur cette
routine" over a pasted workout, the router chose review and put the
ENTIRE pasted text -- 640 characters, newlines and all -- into
file_path. Three separate things then went wrong, and each is pinned
below:

  1. Path.exists() RAISED instead of returning False. pathlib ignores
     ENOENT/ENOTDIR/EBADF/ELOOP and lets every other OSError through,
     so ENAMETOOLONG escaped, and _read_file_node blew up before
     reaching its own error handling.
  2. _error_node flipped state.ok and returned, trusting the failed
     node to have set final_output. A node that RAISES never does.
  3. The user got "tool 'review' returned empty output" -- the least
     informative message the system can produce, from the node whose
     entire job is to produce an informative one.

The lot-3 path grounding guard could not have caught it, and that is
worth stating plainly: it checks PROVENANCE, and this string did appear
in the conversation -- it WAS the conversation. Grounding says where a
string came from, not whether it is a path. The checks are
complementary.
"""

import json

import pytest

from forge.graphs import review as review_graph
from forge.tools import review as review_tool
from forge.types import AgentState

PASTED = "LOWER BODY - Beginner, no equipment\n\nknee circles\nsquat large\n" * 10


def _run(**payload):
    return review_tool.run(json.dumps(payload))


def test_a_pasted_document_is_refused_before_touching_the_disk():
    out = _run(file_path=PASTED)
    assert out.startswith("[error]")
    assert "line break" in out


def test_the_refusal_says_what_the_field_is_for():
    """A bare 'invalid path' leaves the router to guess, and it guesses
    by repeating itself -- it sent the same payload twice in the live
    incident."""
    out = _run(file_path=PASTED)
    assert "contents" in out


def test_a_single_overlong_line_is_refused_too():
    """No line break, still not a path: 255 bytes is the filename limit
    on ext4 and btrfs."""
    out = _run(file_path="a" * 300)
    assert out.startswith("[error]")
    assert "300" in out


def test_an_empty_path_is_refused():
    assert _run(file_path="   ").startswith("[error]")


def test_an_ordinary_path_still_resolves():
    """The check must not become the bug. A normal relative filename
    reaches the graph, which then reports it missing rather than
    malformed."""
    out = _run(file_path="notes.md")
    assert "must be the NAME" not in out


@pytest.mark.parametrize("bad", [PASTED, "b" * 300])
def test_the_node_never_raises_on_a_malformed_path(bad):
    """Defence in depth: even if something reaches the node directly,
    it returns an error instead of exploding. A node that raises loses
    its message."""
    state = AgentState(user_input="x", max_steps=3)
    state.context["file_path"] = "/tmp/" + bad
    out = review_graph._read_file_node(state)
    assert out.ok is False
    assert out.final_output.startswith("[error]")


def test_the_error_node_always_produces_output():
    """The failure that made the live incident unreadable: a raising
    node leaves final_output empty, and the error node used to pass it
    straight through."""
    state = AgentState(user_input="x", max_steps=3)
    state.ok = False
    state.error = "something blew up"
    out = review_graph._error_node(state)
    assert out.final_output
    assert "something blew up" in out.final_output


def test_the_error_node_keeps_a_message_a_node_already_wrote():
    state = AgentState(user_input="x", max_steps=3)
    state.ok = False
    state.error = "raw"
    state.final_output = "[error] File not found: notes.md"
    assert review_graph._error_node(state).final_output.endswith("notes.md")
