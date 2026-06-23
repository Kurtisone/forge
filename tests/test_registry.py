"""
Tests for forge.tools.registry: discovery, lookup, and the
no-silent-failure guarantee (a broken tool module must be logged,
never make load_tools() crash or vanish without a trace).
"""

import sys

import forge.tools as tools_pkg
import forge.tools.registry as registry


def test_real_tools_are_discovered():
    registry.load_tools()
    assert "chat" in registry.available_tools()
    assert "code" in registry.available_tools()


def test_get_tool_returns_none_for_unknown():
    registry.load_tools()
    assert registry.get_tool("does_not_exist") is None


def test_get_tool_returns_callable_handler():
    registry.load_tools()
    handler = registry.get_tool("chat")
    assert callable(handler)
    assert handler("hello") == "hello"


def test_registry_excludes_itself():
    registry.load_tools()
    assert "registry" not in registry.available_tools()


def test_broken_tool_module_does_not_crash_load(tmp_path, monkeypatch):
    """
    A tool module that fails to import must be skipped and logged,
    not crash load_tools() and not silently vanish (the original bug
    this registry was written to fix: bare except/continue).
    """
    (tmp_path / "tool_ok_case.py").write_text(
        "def run(content):\n    return content.upper()\n"
    )
    (tmp_path / "tool_broken_case.py").write_text(
        "raise ImportError('simulated failure')\n"
    )
    (tmp_path / "tool_no_run_case.py").write_text("X = 1\n")

    monkeypatch.setattr(tools_pkg, "__path__", [str(tmp_path)])
    monkeypatch.setattr(registry, "ENABLED_TOOLS", {"tool_ok_case"})
    for name in ("tool_ok_case", "tool_broken_case", "tool_no_run_case"):
        sys.modules.pop(f"forge.tools.{name}", None)

    try:
        registry.load_tools()  # must not raise

        assert "tool_ok_case" in registry.available_tools()
        assert registry.get_tool("tool_ok_case")("hi") == "HI"

        # broken module: skipped, not present, no crash
        assert "tool_broken_case" not in registry.available_tools()

        # module with no run(): skipped too
        assert "tool_no_run_case" not in registry.available_tools()
    finally:
        for name in ("tool_ok_case", "tool_broken_case", "tool_no_run_case"):
            sys.modules.pop(f"forge.tools.{name}", None)
        registry.load_tools()  # restore the real tool set for later tests


def test_tool_with_run_but_not_enabled_is_skipped(tmp_path, monkeypatch):
    """
    Having run() is necessary but not sufficient: a module not listed
    in ENABLED_TOOLS must stay undispatchable even though it has a
    perfectly valid run() function. This is the guard that protects
    files.py / git.py / shell.py from becoming silently reachable the
    moment someone fills them in.
    """
    (tmp_path / "tool_not_enabled_case.py").write_text(
        "def run(content):\n    return 'should never be reachable'\n"
    )

    monkeypatch.setattr(tools_pkg, "__path__", [str(tmp_path)])
    monkeypatch.setattr(registry, "ENABLED_TOOLS", {"chat", "code"})  # not this tool
    sys.modules.pop("forge.tools.tool_not_enabled_case", None)

    try:
        registry.load_tools()
        assert "tool_not_enabled_case" not in registry.available_tools()
        assert registry.get_tool("tool_not_enabled_case") is None
    finally:
        sys.modules.pop("forge.tools.tool_not_enabled_case", None)
        registry.load_tools()
