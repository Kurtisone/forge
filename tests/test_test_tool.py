"""Tests for forge.tools.test (dedicated pytest/ruff runner)."""

import forge.config as cfg
import forge.tools.test as test_mod


def test_test_tool_allowed_runner(tmp_path, monkeypatch):
    # test.py imports TEST_ALLOWED_COMMANDS / TEST_TIMEOUT by value at
    # import time, so the module's own attributes must be patched too
    # (same caveat as test_shell_git.py for shell.py).
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(cfg, "TEST_TIMEOUT", 30)
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(test_mod, "TEST_TIMEOUT", 30)

    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    r = test_mod.run("pytest test_sample.py")
    assert "1 passed" in r


def test_test_tool_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(cfg, "TEST_TIMEOUT", 30)
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(test_mod, "TEST_TIMEOUT", 30)

    (tmp_path / "test_sample.py").write_text("def test_fail():\n    assert 1 == 2\n")
    r = test_mod.run("pytest test_sample.py")
    assert "1 failed" in r


def test_test_tool_blocked_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest"})

    r = test_mod.run("rm -rf /")
    assert "not in the allowlist" in r


def test_test_tool_empty_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))

    r = test_mod.run("")
    assert "[error]" in r


def test_test_tool_shell_allowlist_is_independent(tmp_path, monkeypatch):
    # A command allowed in the general shell tool must NOT be allowed
    # here unless it's also in TEST_ALLOWED_COMMANDS -- the two
    # allowlists are deliberately independent.
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_ALLOWED_COMMANDS", {"cat"})
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest"})

    r = test_mod.run("cat somefile.txt")
    assert "not in the allowlist" in r
