"""Tests for forge.tools.recall (dispatchable wrapper around
forge.graphs.recall)."""

from unittest.mock import patch

import forge.tools.recall as recall_tool_mod


def test_recall_tool_dispatches_to_graph():
    with patch.object(
        recall_tool_mod, "recall_run", return_value="Tu as un Steam Deck."
    ) as mock_run:
        r = recall_tool_mod.run("Tu peux me lister mon matériel ?")

    assert r == "Tu as un Steam Deck."
    mock_run.assert_called_once_with("Tu peux me lister mon matériel ?")


def test_recall_tool_strips_whitespace():
    with patch.object(recall_tool_mod, "recall_run", return_value="ok") as mock_run:
        recall_tool_mod.run("  query with spaces  ")

    mock_run.assert_called_once_with("query with spaces")


def test_recall_tool_empty_query():
    r = recall_tool_mod.run("   ")
    assert "[error]" in r
