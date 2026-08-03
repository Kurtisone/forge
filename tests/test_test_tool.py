"""Tests for forge.tools.test (dedicated pytest/ruff runner)."""

import shutil

import forge.config as cfg
import forge.tools.test as test_mod


def test_test_tool_resolves_runner_via_real_path_not_hardcoded_one(
    tmp_path, monkeypatch
):
    """
    Regression test for a real CI failure: GitHub Actions installs
    pytest/ruff via actions/setup-python into its own hostedtoolcache,
    not into /usr/local/bin, /usr/bin, or /bin -- the subprocess's
    own hardcoded PATH (deliberately minimal, no leaked host env).
    Before this fix, the tool looked the runner up using that
    restricted PATH and failed with "executable not found" anywhere
    the real installation wasn't in one of those three directories,
    even though the runner genuinely was installed and on the actual
    process PATH.
    """
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(cfg, "TEST_TIMEOUT", 30)
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(test_mod, "TEST_TIMEOUT", 30)

    real_pytest = shutil.which("pytest")
    assert real_pytest is not None, "pytest must be installed to run this test"

    # Simulate a real pytest installed somewhere NOT in the
    # subprocess's hardcoded PATH, only on the actual process PATH.
    fake_dir = tmp_path / "not_a_standard_bin_dir"
    fake_dir.mkdir()
    fake_pytest = fake_dir / "pytest"
    fake_pytest.symlink_to(real_pytest)
    monkeypatch.setenv("PATH", str(fake_dir))

    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n")
    r = test_mod.run("pytest test_sample.py")
    assert "1 passed" in r


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
