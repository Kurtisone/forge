"""
Tests for forge.tools.files, forge.graphs.review, and forge.cli replay.
"""

import json

import forge.config as cfg
import forge.graphs.review as review_mod
import forge.tools.files as files_mod
from forge.graph import Graph
from forge.graphs.review import build as build_review

# -------------------------------------------------------------------
# files tool
# -------------------------------------------------------------------


def test_files_write_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(
        json.dumps({"action": "write", "path": "hello.txt", "content": "Bonjour !"})
    )
    assert "[ok]" in r

    r = files_mod.run(json.dumps({"action": "read", "path": "hello.txt"}))
    assert r == "Bonjour !"


def test_files_write_over_existing_returns_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(
        json.dumps({"action": "write", "path": "f.txt", "content": "line1\nline2\n"})
    )
    r = files_mod.run(
        json.dumps({"action": "write", "path": "f.txt", "content": "line1\nline3\n"})
    )

    assert "```diff" in r
    assert "-line2" in r
    assert "+line3" in r
    # the file on disk must actually be updated, not just diffed
    assert (tmp_path / "f.txt").read_text() == "line1\nline3\n"


def test_files_write_diff_keeps_lines_separate_without_trailing_newline(
    tmp_path, monkeypatch
):
    """
    Regression test: a file with no trailing newline (e.g. a
    single-line script -- the real case this was caught on) made
    unified_diff's last line come out with no line ending either,
    which then joined directly onto the next line with no separator
    ("-old+new" glued together in the rendered diff). Each diff line
    must end up on its own line regardless of the source file's
    trailing-newline situation.
    """
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(
        json.dumps(
            {"action": "write", "path": "hello.py", "content": "print('Hello World')"}
        )
    )
    r = files_mod.run(
        json.dumps(
            {"action": "write", "path": "hello.py", "content": "print('Bienvenue')"}
        )
    )

    assert "-print('Hello World')\n" in r
    assert "+print('Bienvenue')" in r
    assert "World')+print" not in r  # the exact glued-together bug


def test_files_write_identical_content_reports_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    files_mod.run(json.dumps({"action": "write", "path": "f.txt", "content": "same"}))
    r = files_mod.run(
        json.dumps({"action": "write", "path": "f.txt", "content": "same"})
    )

    assert "inchangé" in r
    assert "```diff" not in r


def test_files_write_new_path_has_no_diff(tmp_path, monkeypatch):
    """A brand-new file has nothing to diff against -- plain confirmation."""
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))

    r = files_mod.run(
        json.dumps({"action": "write", "path": "new.txt", "content": "content"})
    )

    assert "[ok] written" in r
    assert "```diff" not in r


def test_files_write_over_oversized_existing_file_skips_diff(tmp_path, monkeypatch):
    """Diffing is skipped (not attempted) against a huge existing file,
    consistent with the read tool's own size guard -- the write itself
    must still succeed."""
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_mod, "_MAX_READ_BYTES", 10)

    (tmp_path / "big.txt").write_text("x" * 100)
    r = files_mod.run(
        json.dumps({"action": "write", "path": "big.txt", "content": "short"})
    )

    assert "[ok] written" in r
    assert "```diff" not in r
    assert (tmp_path / "big.txt").read_text() == "short"


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
    assert state.ok  # error node recovered
    assert "[error]" in state.final_output


def test_review_provider_failure(tmp_path, monkeypatch):
    from forge.errors import ProviderError

    test_file = tmp_path / "f.py"
    test_file.write_text("x = 1")

    monkeypatch.setattr(
        review_mod, "call_llm", lambda p: (_ for _ in ()).throw(ProviderError("down"))
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
