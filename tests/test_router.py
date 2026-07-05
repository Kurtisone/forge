"""
Tests for the router becoming tool-aware (v3.5): the prompt only
describes tools that are actually enabled+loaded, and the parser only
accepts a router-picked tool if it's in that same set.

Before this: files/shell/git were reachable only via an explicit
Graph (POST /run), never from a normal chat turn, even with
ENABLED_TOOLS listing them -- the router's own prompt and validation
hardcoded exactly {"chat", "code"}. Nothing about ENABLED_TOOLS itself
changes here: a tool still has to be explicitly opted into that list
to be reachable either way. What changes is that the router can now
actually offer an already-opted-in tool during a conversation.
"""

import forge.tools.registry as registry_mod
from forge.router.parser import _validate_json_obj, parse_router_output
from forge.router.prompt import build_router_prompt


# ── parser: dynamic tool validation ─────────────────────────────────

def test_unlisted_tool_falls_back_to_chat(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code"])
    decision = _validate_json_obj({"tool": "shell", "content": "ls"}, "raw")
    assert decision.tool == "chat"


def test_enabled_tool_is_accepted(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code", "shell"])
    decision = _validate_json_obj({"tool": "shell", "content": "ls -la"}, "raw")
    assert decision.tool == "shell"
    assert decision.content == "ls -la"


def test_chat_and_code_always_valid_even_if_registry_excludes_them(monkeypatch):
    """_VALID_TOOLS is a floor: a misconfigured ENABLED_TOOLS that
    somehow excludes chat/code must not make the router itself unable
    to fall back to chat."""
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["shell"])
    decision = _validate_json_obj({"tool": "code", "content": "print(1)"}, "raw")
    assert decision.tool == "code"


def test_parse_router_output_end_to_end_with_files_enabled(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code", "files"])
    raw = '{"tool":"files","content":"{\\"action\\":\\"read\\",\\"path\\":\\"x.py\\"}"}'
    decision = parse_router_output(raw)
    assert decision.tool == "files"


def test_parse_router_output_end_to_end_with_files_disabled(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code"])
    raw = '{"tool":"files","content":"{\\"action\\":\\"read\\",\\"path\\":\\"x.py\\"}"}'
    decision = parse_router_output(raw)
    assert decision.tool == "chat"


# ── prompt: dynamic tool descriptions ────────────────────────────────

def test_prompt_with_only_chat_and_code_omits_other_tools():
    prompt = build_router_prompt("hi", available_tools=["chat", "code"])
    assert '"chat" or "code"' in prompt
    assert "shell" not in prompt
    assert "files" not in prompt
    assert "git" not in prompt


def test_prompt_includes_shell_when_enabled():
    prompt = build_router_prompt("run ls", available_tools=["chat", "code", "shell"])
    assert '"shell"' in prompt
    assert "single shell command" in prompt
    assert "ls -la" in prompt  # the shell example


def test_prompt_includes_files_description_when_enabled():
    prompt = build_router_prompt("read a file", available_tools=["chat", "code", "files"])
    assert "action" in prompt
    assert '"read"' in prompt or "read" in prompt


def test_prompt_includes_git_description_when_enabled():
    prompt = build_router_prompt("git log", available_tools=["chat", "code", "git"])
    assert "git subcommand" in prompt
    assert "Read-only" in prompt


def test_prompt_defaults_to_registry_when_available_tools_not_passed(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "code", "git"])
    prompt = build_router_prompt("hi")
    assert '"git"' in prompt


def test_prompt_falls_back_to_default_pair_if_registry_returns_empty(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: [])
    prompt = build_router_prompt("hi")
    assert '"chat" or "code"' in prompt


def test_prompt_still_fills_in_user_input_and_history():
    prompt = build_router_prompt(
        "what's next?",
        history=[{"role": "user", "content": "earlier message"}],
        available_tools=["chat", "code"],
    )
    assert "what's next?" in prompt
    assert "earlier message" in prompt


def test_unknown_tool_without_a_description_gets_generic_wording():
    """A custom tool the operator wrote, with no entry in
    TOOL_DESCRIPTIONS, must still produce a valid (if generic) prompt
    section instead of crashing or silently omitting it."""
    prompt = build_router_prompt("do the thing", available_tools=["chat", "custom_tool"])
    assert '"custom_tool"' in prompt
    assert "content is the input this tool expects" in prompt
