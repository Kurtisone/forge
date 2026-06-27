"""
Tests for forge.tools.files, forge.graphs.review, and forge.cli replay.
"""

import json
import os
import tempfile

import forge.config as cfg
import forge.tools.files as files_mod
import forge.graphs.review as review_mod
from forge.graphs.review import build as build_review
from forge.graph import Graph


# -------------------------------------------------------------------
# files tool
# -------------------------------------------------------------------

def test_files_write_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(json.dumps({"action": "write", "path": "hello.txt", "content": "Bonjour !"}))
    assert "[ok]" in r

    r = files_mod.run(json.dumps({"action": "read", "path": "hello.txt"}))
    assert r == "Bonjour !"


def test_files_list(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(json.dumps({"action": "write", "path": "a.txt", "content": "x"}))
    r = files_mod.run(json.dumps({"action": "list", "path": "."}))
    assert "a.txt" in r


def test_files_traversal_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(json.dumps({"action": "read", "path": "../../etc/passwd"}))
    assert "[error]" in r
    assert "permission" in r.lower()


def test_files_read_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(json.dumps({"action": "read", "path": "missing.txt"}))
    assert "[error]" in r


def test_files_invalid_json(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run("not json")
    assert "[error]" in r


def test_files_unknown_action(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(json.dumps({"action": "delete", "path": "x"}))
    assert "[error]" in r


# -------------------------------------------------------------------
# review graph
# -------------------------------------------------------------------

def test_review_reads_file_and_calls_llm(tmp_path, monkeypatch):
    test_file = tmp_path / "sample.py"
    test_file.write_text("def add(a, b):\n    return a + b\n")

    monkeypatch.setattr(review_mod, "call_llm", lambda p: "Simple and readable.")

    state = build_review().run(
        str(test_file),
        initial_context={"file_path": str(test_file), "question": "Improve?"},
    )
    assert state.ok
    assert "Simple" in state.final_output
    assert state.steps_taken == 2
    assert "file_content" in state.context


def test_review_missing_file_is_graceful(monkeypatch):
    monkeypatch.setattr(review_mod, "call_llm", lambda p: "ok")

    state = build_review().run(
        "/nonexistent/file.py",
        initial_context={"file_path": "/nonexistent/file.py"},
    )
    assert state.ok            # error node recovered
    assert "[error]" in state.final_output


def test_review_provider_failure(tmp_path, monkeypatch):
    from forge.errors import ProviderError
    test_file = tmp_path / "f.py"
    test_file.write_text("x = 1")

    monkeypatch.setattr(
        review_mod, "call_llm",
        lambda p: (_ for _ in ()).throw(ProviderError("down"))
    )
    state = build_review().run(
        str(test_file),
        initial_context={"file_path": str(test_file)},
    )
    assert not state.ok
    assert "LLM unavailable" in state.final_output


# -------------------------------------------------------------------
# initial_context flows through graph
# -------------------------------------------------------------------

def test_initial_context_reaches_node():
    def ctx_node(state):
        state.final_output = state.context.get("greeting", "missing")
        return state

    g = Graph("ctx_test")
    g.add_node("n", ctx_node)
    s = g.run("hello", initial_context={"greeting": "Bonjour !"})
    assert s.final_output == "Bonjour !"
