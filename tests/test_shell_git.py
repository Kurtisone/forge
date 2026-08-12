"""Tests for forge.tools.shell and forge.tools.git."""

from pathlib import Path

import forge.config as cfg
import forge.tools.git as git_mod
import forge.tools.shell as shell_mod

# ── shell ──────────────────────────────────────────────────────────


def test_shell_allowed_command(tmp_path, monkeypatch):
    # shell.py imports SHELL_ALLOWED_COMMANDS / SHELL_TIMEOUT by value at
    # import time (`from forge.config import ...`), so patching forge.config
    # alone has no effect on the already-bound names in shell_mod — the
    # module's own attributes must be patched too, same as WORKSPACE_DIR
    # below.
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_ALLOWED_COMMANDS", {"echo"})
    monkeypatch.setattr(cfg, "SHELL_TIMEOUT", 10)
    monkeypatch.setattr(shell_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(shell_mod, "SHELL_ALLOWED_COMMANDS", {"echo"})
    monkeypatch.setattr(shell_mod, "SHELL_TIMEOUT", 10)
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


def test_shell_python_is_not_allowed_by_default():
    """
    python3 used to ship in the default allowlist, which made the whole
    allowlist decorative -- `python3 -c "..."` runs anything. The
    default is now interpreter-free (audit C-2).
    """
    assert "python3" not in cfg.SHELL_ALLOWED_COMMANDS
    assert "pip" not in cfg.SHELL_ALLOWED_COMMANDS
    assert "find" not in cfg.SHELL_ALLOWED_COMMANDS


def test_shell_python_still_works_when_explicitly_allowed(tmp_path, monkeypatch):
    """
    Removing it from the default must not remove the capability: an
    operator who puts python3 back gets python3. The point of C-2 is
    that this becomes a written-down choice, not the out-of-the-box
    state.
    """
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SHELL_TIMEOUT", 10)
    monkeypatch.setattr(shell_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(shell_mod, "SHELL_ALLOWED_COMMANDS", {"python3"})
    monkeypatch.setattr(shell_mod, "SHELL_TIMEOUT", 10)
    r = shell_mod.run('python3 -c "print(1+1)"')
    assert "2" in r


def test_allowlist_defeating_set_covers_the_obvious_interpreters():
    for name in ("python3", "pip", "find", "sh", "bash", "xargs", "env"):
        assert name in cfg._SHELL_ALLOWLIST_DEFEATING


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


def test_git_stash_is_rejected():
    """
    `git stash` (bare) is `git stash push`: it removes uncommitted
    changes from the working tree. It sat in a list labelled
    "read-only" (audit E-3).
    """
    r = git_mod.run("stash")
    assert "not allowed" in r


def test_git_stash_clear_is_rejected():
    r = git_mod.run("stash clear")
    assert "not allowed" in r


def test_git_rejects_output_argument():
    r = git_mod.run("log --output=/tmp/forge-git-should-not-exist")
    assert "[error]" in r
    assert not Path("/tmp/forge-git-should-not-exist").exists()


def test_git_rejects_exec_argument():
    r = git_mod.run("log --exec=/bin/sh")
    assert "[error]" in r


def test_git_subprocess_env_carries_no_host_secrets(monkeypatch):
    """
    git.py used to pass no env at all, so the subprocess inherited
    everything the API process held -- API_TOKEN included.
    """
    assert "API_TOKEN" not in git_mod._ENV
    assert "OPENROUTER_API_KEY" not in git_mod._ENV
    assert git_mod._ENV["GIT_TERMINAL_PROMPT"] == "0"
