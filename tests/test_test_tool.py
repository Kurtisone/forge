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


# ── audit E-1: argument confinement ──────────────────────────────────
#
# These bound WHERE the code the runner executes can come from. They
# do not make this tool safe -- running pytest is running workspace
# code, by design. See tools/test.py's docstring and SECURITY.md.


def _workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest", "ruff"})
    monkeypatch.setattr(cfg, "TEST_TIMEOUT", 30)
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest", "ruff"})
    monkeypatch.setattr(test_mod, "TEST_TIMEOUT", 30)


def test_absolute_path_argument_is_refused(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    r = test_mod.run("ruff check /etc")
    assert "points outside the workspace" in r


def test_parent_traversal_argument_is_refused(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    r = test_mod.run("pytest ../../tests")
    assert "points outside the workspace" in r


def test_a_flag_value_that_looks_absolute_is_still_refused(tmp_path, monkeypatch):
    """
    `-p /somewhere/plugin` is how pytest is told to import a plugin
    module. The flag itself is skipped, its value is not -- that value
    is exactly the path this check exists for.
    """
    _workspace(tmp_path, monkeypatch)
    r = test_mod.run("pytest -p /tmp/evil_plugin")
    assert "points outside the workspace" in r


def test_flags_are_not_treated_as_paths(tmp_path, monkeypatch):
    """A refusal on "--tb=short" would make the tool useless."""
    _workspace(tmp_path, monkeypatch)
    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert True\n")
    r = test_mod.run("pytest --tb=short -q test_sample.py")
    assert "1 passed" in r


def test_a_subdirectory_argument_is_allowed(tmp_path, monkeypatch):
    _workspace(tmp_path, monkeypatch)
    sub = tmp_path / "suite"
    sub.mkdir()
    (sub / "test_sample.py").write_text("def test_ok():\n    assert True\n")
    r = test_mod.run("pytest suite")
    assert "1 passed" in r


def test_traversal_that_comes_back_inside_is_allowed(tmp_path, monkeypatch):
    """
    ".." is not banned as a string -- what matters is where the path
    lands. Rejecting the substring would be a different rule, and a
    worse one: it refuses legitimate paths while a symlink still
    walks straight past it.
    """
    _workspace(tmp_path, monkeypatch)
    sub = tmp_path / "suite"
    sub.mkdir()
    (sub / "test_sample.py").write_text("def test_ok():\n    assert True\n")
    r = test_mod.run("pytest suite/../suite/test_sample.py")
    assert "1 passed" in r


def test_enabling_files_and_test_together_warns_at_import(caplog):
    """
    Audit E-1. The two tools together are equivalent to `shell`, and
    nothing in either tool's own allowlist says so. The warning is the
    only place that configuration fact is stated at runtime, so it is
    worth a test even though shell.py's equivalent tripwire has none.

    Logged at import rather than per run, deliberately: a fact about
    how the instance is configured belongs in the startup log, not
    repeated into output people learn to scroll past.
    """
    import importlib

    original = cfg.ENABLED_TOOLS
    try:
        cfg.ENABLED_TOOLS = {"chat", "code", "files", "test"}
        with caplog.at_level("WARNING"):
            importlib.reload(test_mod)
        assert "equivalent to 'shell'" in caplog.text

        caplog.clear()
        cfg.ENABLED_TOOLS = {"chat", "code", "test"}
        with caplog.at_level("WARNING"):
            importlib.reload(test_mod)
        assert "equivalent to 'shell'" not in caplog.text
    finally:
        cfg.ENABLED_TOOLS = original
        importlib.reload(test_mod)


# ── the runner is not optional, and saying so usefully ───────────────
#
# Observed live 2026-08-19: "Lance les tests dans
# tests/test_inexistant_xyz.py" routed to
# {"tool":"test","content":"tests/test_inexistant_xyz.py"} -- the path
# with no runner in front of it. The tool answered that
# "tests/test_inexistant_xyz.py" was not an allowed RUNNER and
# suggested adding it to TEST_ALLOWED_COMMANDS, which is advice that
# cannot work. The router had nothing to go on: "test" had no
# description and no example in router/prompt.py and fell through to
# the generic "content is the input this tool expects."


def test_a_bare_path_is_named_as_the_mistake_it_is(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest", "ruff"})
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest", "ruff"})

    r = test_mod.run("tests/test_inexistant_xyz.py")

    assert "is a path, not a test runner" in r
    # Both intents, because the runner is what distinguishes them --
    # which is also why this is never repaired by guessing one.
    assert "pytest tests/test_inexistant_xyz.py" in r
    assert "ruff check tests/test_inexistant_xyz.py" in r
    # The advice that cannot work must not be the one given.
    assert "TEST_ALLOWED_COMMANDS" not in r


def test_a_forbidden_binary_still_gets_the_allowlist_message(tmp_path, monkeypatch):
    """The other mistake, whose fix really is editing the allowlist."""
    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(test_mod, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest"})

    r = test_mod.run("mypy src")

    assert "not in the allowlist" in r
    assert "TEST_ALLOWED_COMMANDS" in r


def test_the_router_prompt_teaches_the_runner_first_shape():
    """
    The half of this fix that is upstream of the tool: "test" used to
    have no description and no example at all.
    """
    from forge.router.prompt import build_router_prompt

    prompt = build_router_prompt(
        "lance les tests", history=[], available_tools=["test"]
    )

    assert "content is the input this tool expects" not in prompt
    assert '"pytest tests/test_graph.py"' in prompt
    assert '"ruff check src/forge/graph.py"' in prompt


def test_a_missing_runner_is_reported_as_a_deployment_gap(monkeypatch):
    """
    The shipped image installs requirements.txt only, and pytest/ruff
    live in requirements-dev.txt -- so in the container this tool
    cannot run at all. Two real runs on 2026-08-19 came back with the
    old "executable not found (not on PATH)" wording and it was read
    as the path-grounding guard refusing them. It was the deployment.
    """
    monkeypatch.setattr(test_mod.shutil, "which", lambda name: None)

    out = test_mod.run("pytest tests/")

    assert out.startswith("[error]")
    assert "not installed" in out
    assert "requirements-dev.txt" in out
    # Must not read as a refusal: nothing was blocked here.
    assert "not allowed" not in out
    assert "allowlist" not in out


def test_the_startup_tripwire_names_the_runners_that_are_absent(monkeypatch):
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest", "ruff", "sh"})
    monkeypatch.setattr(
        test_mod.shutil, "which", lambda name: None if name != "sh" else "/bin/sh"
    )

    assert test_mod._missing_runners() == ["pytest", "ruff"]


def test_the_tripwire_stays_quiet_when_the_runners_are_there(monkeypatch):
    monkeypatch.setattr(test_mod, "TEST_ALLOWED_COMMANDS", {"pytest"})
    monkeypatch.setattr(test_mod.shutil, "which", lambda name: "/usr/bin/pytest")

    assert test_mod._missing_runners() == []
