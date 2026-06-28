"""Tests for forge.tools.shell and forge.tools.git."""

import forge.config as cfg
import forge.tools.shell as shell_mod
import forge.tools.git as git_mod


# ── shell ──────────────────────────────────────────────────────────

def test_shell_allowed_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_ALLOWED_COMMANDS", {"echo"})
    monkeypatch.setattr(cfg, "SHELL_TIMEOUT", 10)
    monkeypatch.setattr(shell_mod, "WORKSPACE_DIR", str(tmp_path))
    r = shell_mod.run("echo hello")
    assert "hello" in r


def test_shell_blocked_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_ALLOWED_COMMANDS", {"echo"})
    monkeypatch.setattr(shell_mod, "WORKSPACE_DIR", str(tmp_path))
    r = shell_mod.run("rm -rf /")
    assert "not in the allowlist" in r


def test_shell_empty_command(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_ALLOWED_COMMANDS", {"echo"})
    monkeypatch.setattr(shell_mod, "WORKSPACE_DIR", str(tmp_path))
    r = shell_mod.run("")
    assert "[error]" in r


def test_shell_python(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_ALLOWED_COMMANDS", {"python3"})
    monkeypatch.setattr(cfg, "SHELL_TIMEOUT", 10)
    monkeypatch.setattr(shell_mod, "WORKSPACE_DIR", str(tmp_path))
    r = shell_mod.run('python3 -c "print(1+1)"')
    assert "2" in r


# ── git ────────────────────────────────────────────────────────────

def test_git_blocked_subcommand():
    r = git_mod.run("push origin main")
    assert "not allowed" in r


def test_git_blocked_commit():
    r = git_mod.run("commit -m test")
    assert "not allowed" in r


def test_git_empty_command():
    r = git_mod.run("")
    assert "[error]" in r


def test_git_status_runs():
    # Just check it doesn't crash — output depends on the environment
    r = git_mod.run("status")
    assert isinstance(r, str) and len(r) > 0


def test_git_log_runs():
    r = git_mod.run("log")
    assert isinstance(r, str) and len(r) > 0
