"""Tests for forge.tools.review (the dispatchable wrapper around
forge.graphs.review, reachable from plain chat via the router).

file_path/test_path confinement to WORKSPACE_DIR is tested here
specifically -- graphs.review.run() itself has no such confinement
(by design, see module docstring in tools/review.py), so this
boundary only exists at this dispatch layer.
"""

import json
from unittest.mock import patch

import forge.config as cfg
import forge.tools.review as review_tool_mod


def test_review_tool_dispatches_to_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(review_tool_mod, "WORKSPACE_DIR", str(tmp_path))

    (tmp_path / "f.py").write_text("x = 1")

    with patch.object(
        review_tool_mod, "review_run", return_value="Looks fine."
    ) as mock_run:
        content = json.dumps({"file_path": "f.py"})
        r = review_tool_mod.run(content)

    assert r == "Looks fine."
    mock_run.assert_called_once_with(
        str((tmp_path / "f.py").resolve()),
        question="Que peut-on améliorer ?",
        test_path=None,
    )


def test_review_tool_passes_through_question_and_test_path(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(review_tool_mod, "WORKSPACE_DIR", str(tmp_path))

    (tmp_path / "graph.py").write_text("x = 1")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_graph.py").write_text("def test_x(): pass")

    with patch.object(review_tool_mod, "review_run", return_value="ok") as mock_run:
        content = json.dumps(
            {
                "file_path": "graph.py",
                "question": "Focus on edge cases",
                "test_path": "tests/test_graph.py",
            }
        )
        review_tool_mod.run(content)

    mock_run.assert_called_once_with(
        str((tmp_path / "graph.py").resolve()),
        question="Focus on edge cases",
        test_path="tests/test_graph.py",
    )


def test_review_tool_leading_slash_means_workspace_root(tmp_path, monkeypatch):
    """Regression test: a router-emitted '/hello.go' used to be
    resolved against the actual filesystem root (no confinement at
    all pre-fix), producing a confusing 'File not found: /hello.go'
    even though the file existed at the workspace root. A leading
    slash must mean 'workspace root', same file as without it."""
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(review_tool_mod, "WORKSPACE_DIR", str(tmp_path))

    (tmp_path / "hello.go").write_text("package main")

    with patch.object(review_tool_mod, "review_run", return_value="ok") as mock_run:
        review_tool_mod.run(json.dumps({"file_path": "/hello.go"}))

    mock_run.assert_called_once_with(
        str((tmp_path / "hello.go").resolve()),
        question="Que peut-on améliorer ?",
        test_path=None,
    )


def test_review_tool_blocks_traversal_outside_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(review_tool_mod, "WORKSPACE_DIR", str(tmp_path))

    r = review_tool_mod.run(json.dumps({"file_path": "../../etc/passwd"}))
    assert "[error]" in r
    assert "escapes workspace" in r


def test_review_tool_blocks_test_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(review_tool_mod, "WORKSPACE_DIR", str(tmp_path))

    (tmp_path / "f.py").write_text("x = 1")
    r = review_tool_mod.run(
        json.dumps({"file_path": "f.py", "test_path": "../../etc/passwd"})
    )
    assert "[error]" in r
    assert "escapes workspace" in r


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
