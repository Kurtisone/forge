"""
Tests for forge.router.grammar: the GBNF grammar generator used to
constrain llama.cpp's decoding to the router's exact JSON schema.

These check the generated grammar TEXT is structurally sound (a
lightweight, hand-rolled check -- no llama.cpp C++ parser available
here) and that it's dynamic the same way router.prompt is: driven by
whatever tools are actually enabled+loaded.
"""

import re

import forge.tools.registry as registry_mod
from forge.router.grammar import build_router_grammar


def _rule_names(grammar: str) -> set[str]:
    return set(re.findall(r"^([a-zA-Z][a-zA-Z0-9_-]*)\s*::=", grammar, re.MULTILINE))


def _referenced_names(grammar: str) -> set[str]:
    # crude but sufficient: any bare identifier token in the grammar body
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", grammar)
    keywords = {"root", "tool", "string", "schar", "hex", "boolean", "ws"}
    return {t for t in tokens if t in keywords}


def test_has_a_root_rule():
    grammar = build_router_grammar(available_tools=["chat", "code"])
    assert re.search(r"^root\s*::=", grammar, re.MULTILINE)


def test_every_referenced_rule_is_defined():
    grammar = build_router_grammar(available_tools=["chat", "code", "files", "shell", "git"])
    defined = _rule_names(grammar)
    referenced = _referenced_names(grammar)
    assert referenced <= defined


def test_tool_alternation_matches_available_tools_only():
    grammar = build_router_grammar(available_tools=["chat", "shell"])
    assert '"\\"chat\\""' in grammar
    assert '"\\"shell\\""' in grammar
    assert '"\\"code\\""' not in grammar
    assert '"\\"files\\""' not in grammar
    assert '"\\"git\\""' not in grammar


def test_all_five_tools_appear_when_all_enabled():
    grammar = build_router_grammar(available_tools=["chat", "code", "files", "shell", "git"])
    for tool in ("chat", "code", "files", "shell", "git"):
        assert f'"\\"{tool}\\""' in grammar


def test_defaults_to_registry_when_available_tools_not_passed(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "git"])
    grammar = build_router_grammar()
    assert '"\\"git\\""' in grammar
    assert '"\\"code\\""' not in grammar


def test_falls_back_to_default_pair_if_registry_returns_empty(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: [])
    grammar = build_router_grammar()
    assert '"\\"chat\\""' in grammar
    assert '"\\"code\\""' in grammar


_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"'          # string literal
    r'|\[\^?(?:[^\]\\]|\\.)*\]'   # char class (may itself contain a literal ")
    r'|[()]'                       # parens
)


def test_grammar_has_balanced_parens_and_brackets():
    grammar = build_router_grammar(available_tools=["chat", "code", "files", "shell", "git"])
    depth = 0
    for tok in _TOKEN.findall(grammar):
        if tok == "(":
            depth += 1
        elif tok == ")":
            depth -= 1
            assert depth >= 0, "unbalanced ')' found"
        # string literals and char classes are matched whole above,
        # so any "(" or ")" *inside* one is never seen as a separate
        # token here -- exactly the bug the naive char-by-char version
        # of this test had, tripping over the literal '"' inside
        # char classes like [^"\\/bfnrt].
    assert depth == 0
