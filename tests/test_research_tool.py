"""Tests for forge.tools.research (dispatchable wrapper around
forge.graphs.research)."""

from unittest.mock import patch

import forge.tools.research as research_tool_mod


def test_research_tool_dispatches_to_graph():
    with patch.object(
        research_tool_mod, "research_run", return_value="Synthesized answer."
    ) as mock_run:
        r = research_tool_mod.run("actualité jeu vidéo")

    assert r == "Synthesized answer."
    mock_run.assert_called_once_with("actualité jeu vidéo")


def test_research_tool_strips_whitespace():
    with patch.object(research_tool_mod, "research_run", return_value="ok") as mock_run:
        research_tool_mod.run("  query with spaces  ")

    mock_run.assert_called_once_with("query with spaces")


def test_research_tool_empty_query():
    r = research_tool_mod.run("   ")
    assert "[error]" in r
