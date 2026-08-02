"""Tests for forge.tools.review (the dispatchable wrapper around
forge.graphs.review, reachable from plain chat via the router)."""

import json
from unittest.mock import patch

import forge.tools.review as review_tool_mod


def test_review_tool_dispatches_to_graph(tmp_path):
    src = tmp_path / "f.py"
    src.write_text("x = 1")

    with patch.object(
        review_tool_mod, "review_run", return_value="Looks fine."
    ) as mock_run:
        content = json.dumps({"file_path": str(src)})
        r = review_tool_mod.run(content)

    assert r == "Looks fine."
    mock_run.assert_called_once_with(
        str(src), question="Que peut-on améliorer ?", test_path=None
    )


def test_review_tool_passes_through_question_and_test_path():
    with patch.object(review_tool_mod, "review_run", return_value="ok") as mock_run:
        content = json.dumps(
            {
                "file_path": "src/forge/graph.py",
                "question": "Focus on edge cases",
                "test_path": "tests/test_graph.py",
            }
        )
        review_tool_mod.run(content)

    mock_run.assert_called_once_with(
        "src/forge/graph.py",
        question="Focus on edge cases",
        test_path="tests/test_graph.py",
    )


def test_review_tool_missing_file_path():
    r = review_tool_mod.run(json.dumps({"question": "x"}))
    assert "[error]" in r
    assert "file_path" in r


def test_review_tool_invalid_json():
    r = review_tool_mod.run("not json")
    assert "[error]" in r


def test_review_tool_json_not_an_object():
    r = review_tool_mod.run(json.dumps(["src/forge/graph.py"]))
    assert "[error]" in r
