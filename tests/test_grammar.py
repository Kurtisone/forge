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
from forge.gbnf import check_rule_names, referenced_names, rule_names
from forge.router.grammar import build_router_grammar


def test_rule_names_use_only_characters_llama_cpp_accepts():
    """
    Regression test for a grammar that was structurally valid, passed
    every other test in this file, and was rejected outright by
    llama-server.

    Underscored rule names (`payload_call`) lex as the rule `payload`
    followed by garbage, so the server answers 400 on every completion
    and the router stops working altogether -- it doesn't degrade, it
    dies. Nothing else here could catch it: the generated text was
    well-formed by every structural measure, and there is no llama.cpp
    parser available in this environment to reject it. So the lexer's
    own rule gets encoded instead.
    """
    grammar = build_router_grammar(
        available_tools=["chat", "code", "files", "memory", "review", "sysadmin"]
    )
    assert check_rule_names(grammar) == []


def test_has_a_root_rule():
    grammar = build_router_grammar(available_tools=["chat", "code"])
    assert re.search(r"^root\s*::=", grammar, re.MULTILINE)


def test_every_referenced_rule_is_defined():
    grammar = build_router_grammar(
        available_tools=["chat", "code", "files", "shell", "git"]
    )
    defined = rule_names(grammar)
    referenced = referenced_names(grammar)
    assert referenced <= defined


def test_tool_alternation_matches_available_tools_only():
    grammar = build_router_grammar(available_tools=["chat", "shell"])
    assert '"\\"chat\\""' in grammar
    assert '"\\"shell\\""' in grammar
    assert '"\\"code\\""' not in grammar
    assert '"\\"files\\""' not in grammar
    assert '"\\"git\\""' not in grammar


def test_all_five_tools_appear_when_all_enabled():
    grammar = build_router_grammar(
        available_tools=["chat", "code", "files", "shell", "git"]
    )
    for tool in ("chat", "code", "files", "shell", "git"):
        assert f'"\\"{tool}\\""' in grammar


def test_defaults_to_registry_when_available_tools_not_passed(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", lambda: ["chat", "git"])
    grammar = build_router_grammar()
    assert '"\\"git\\""' in grammar
    assert '"\\"code\\""' not in grammar


def test_falls_back_to_default_pair_if_registry_returns_empty(monkeypatch):
    monkeypatch.setattr(registry_mod, "available_tools", list)
    grammar = build_router_grammar()
    assert '"\\"chat\\""' in grammar
    assert '"\\"code\\""' in grammar


def _rule_body(grammar: str, name: str) -> str:
    match = re.search(rf"^{name}\s*::=(.*)$", grammar, re.MULTILINE)
    assert match, f"rule {name!r} not defined"
    return match.group(1)


def test_json_payload_tools_can_only_emit_an_object():
    """
    The heart of the fix, and its second attempt: offering
    "content ::= string | object" to every tool left the escaped-string
    shape reachable, and a 9B's prior for it beat the worked examples
    on the first real files:write (it came back as an escaped string
    and died on an unescaped quote in `import "fmt"`).

    Escaping inside the string shape is invisible to the grammar -- to
    schar the whole payload is just characters -- so it can only be
    forbidden, not validated. GBNF can condition on it because "tool"
    is pinned before "content".
    """
    grammar = build_router_grammar(
        available_tools=["chat", "code", "files", "memory", "review", "sysadmin"]
    )
    payload_body = _rule_body(grammar, "payload-call")
    assert '"\\"content\\"" ws ":" ws object' in payload_body
    assert "string" not in payload_body

    for tool in ("files", "memory", "review", "sysadmin"):
        assert f'"\\"{tool}\\""' in _rule_body(grammar, "payload-tool")
        assert f'"\\"{tool}\\""' not in _rule_body(grammar, "text-tool")


def test_prose_tools_still_emit_a_plain_string():
    """chat and code must NOT be pushed into the object shape -- their
    content is prose or source, not a payload."""
    grammar = build_router_grammar(available_tools=["chat", "code", "files"])
    text_body = _rule_body(grammar, "text-call")
    assert '"\\"content\\"" ws ":" ws string' in text_body
    for tool in ("chat", "code"):
        assert f'"\\"{tool}\\""' in _rule_body(grammar, "text-tool")


def test_a_branch_with_no_tools_is_omitted():
    """An empty GBNF alternation is unsatisfiable, and a tool set with
    only one kind is a legal ENABLED_TOOLS value."""
    only_payload = build_router_grammar(available_tools=["files"])
    assert _rule_body(only_payload, "root").strip() == "payload-call"
    assert "text-tool" not in only_payload

    only_text = build_router_grammar(available_tools=["chat", "code"])
    assert _rule_body(only_text, "root").strip() == "text-call"
    assert "payload-tool" not in only_text


def test_the_payload_object_can_hold_any_json_value():
    """A payload field may legitimately be a number or a list; the
    model shouldn't be steered into stringifying it."""
    grammar = build_router_grammar(available_tools=["files"])
    value_body = _rule_body(grammar, "value")
    for kind in ("string", "object", "array", "number", "boolean"):
        assert kind in value_body


def test_done_stays_available_to_both_branches():
    """The read(done:false) -> write flow from v3.9 runs through the
    payload branch; losing "done" there would break it silently."""
    grammar = build_router_grammar(available_tools=["chat", "files"])
    assert "done?" in _rule_body(grammar, "payload-call")
    assert "done?" in _rule_body(grammar, "text-call")
    assert re.search(r'^done\s*::=.*"\\"done\\""', grammar, re.MULTILINE)


_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"'  # string literal
    r"|\[\^?(?:[^\]\\]|\\.)*\]"  # char class (may itself contain a literal ")
    r"|[()]"  # parens
)


def test_grammar_has_balanced_parens_and_brackets():
    grammar = build_router_grammar(
        available_tools=["chat", "code", "files", "shell", "git"]
    )
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


def test_delegate_is_routable():
    """
    A tool the grammar forbids is a tool the router can never choose,
    however well the prompt describes it -- the failure mode that made
    prompt/grammar drift expensive in the first place.
    """
    grammar = build_router_grammar(available_tools=["chat", "code", "delegate"])
    assert "delegate" in grammar
    assert check_rule_names(grammar) == []
