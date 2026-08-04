"""Tests for forge.tools.sysadmin (dispatchable wrapper around
forge.graphs.sysadmin)."""

from unittest.mock import patch

import forge.tools.sysadmin as sysadmin_tool_mod


def test_sysadmin_tool_dispatches_to_graph_with_both_fields():
    with patch.object(
        sysadmin_tool_mod, "sysadmin_run", return_value="Diagnosis."
    ) as mock_run:
        r = sysadmin_tool_mod.run(
            '{"target_hint":"searxng","question":"pourquoi ça plante ?"}'
        )

    assert r == "Diagnosis."
    mock_run.assert_called_once_with("searxng", "pourquoi ça plante ?")


def test_sysadmin_tool_target_hint_only():
    with patch.object(
        sysadmin_tool_mod, "sysadmin_run", return_value="ok"
    ) as mock_run:
        sysadmin_tool_mod.run('{"target_hint":"forge"}')

    mock_run.assert_called_once_with("forge", None)


def test_sysadmin_tool_question_only():
    with patch.object(
        sysadmin_tool_mod, "sysadmin_run", return_value="ok"
    ) as mock_run:
        sysadmin_tool_mod.run('{"question":"pourquoi le système est lent ?"}')

    mock_run.assert_called_once_with(None, "pourquoi le système est lent ?")


def test_sysadmin_tool_empty_content_falls_through_with_no_fields():
    """No content is still a valid request ('regarde ce qui ne va pas'
    with nothing named) -- falls through to the graph with both
    fields empty rather than being rejected as invalid."""
    with patch.object(
        sysadmin_tool_mod, "sysadmin_run", return_value="ok"
    ) as mock_run:
        sysadmin_tool_mod.run("   ")

    mock_run.assert_called_once_with(None, None)


def test_sysadmin_tool_invalid_json_returns_error():
    r = sysadmin_tool_mod.run("not json")
    assert "[error]" in r


def test_sysadmin_tool_json_array_returns_error():
    r = sysadmin_tool_mod.run("[1, 2, 3]")
    assert "[error]" in r


def test_sysadmin_tool_empty_string_fields_treated_as_absent():
    with patch.object(
        sysadmin_tool_mod, "sysadmin_run", return_value="ok"
    ) as mock_run:
        sysadmin_tool_mod.run('{"target_hint":"","question":""}')

    mock_run.assert_called_once_with(None, None)
