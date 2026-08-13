"""
Tests for the two bugs that made files:write through the router fail
for most real file content (found live during the security lot 3
pre-merge tests, both preexisting on main):

1. _all_json_objects counted braces without tracking string context,
   so a valid router object whose nested "content" carried file text
   with unbalanced braces (Go, C, Rust, JS, a Python dict) was thrown
   away and fell through to the plain-text fallback. "hello.py" only
   ever worked because print('...') contains no brace at all.

2. The double escaping a JSON payload nested inside a JSON string
   requires is not something the 9B model holds on multi-line content
   (it emits \\n where \\\\n was needed), so the inner parse died on an
   invalid control character.
"""

import json

import forge.tools.registry as registry_mod
from forge.router.parser import _all_json_objects, parse_router_output

# Real file bodies whose braces do NOT balance on their own, which is
# the normal case for a snippet, not an exotic one.
UNBALANCED_BODIES = {
    "js_arrow_truncated": "const f = () => {\n  return {a: 1};\n",
    "c_extra_close": "if (x) { return 1; }\n}\n",
    "rust_comment_brace": "// closes with } eventually\nfn main() {\n",
}

BALANCED_BODIES = {
    "go": 'package main\n\nfunc main() {\n\tfmt.Println("Hi")\n}\n',
    "python_dict": 'CONF = {"a": 1, "b": {"c": 2}}\n',
}


def _router_output(body: str, path: str = "x.txt") -> str:
    """The legacy shape: a JSON payload re-encoded as a JSON string."""
    inner = json.dumps({"action": "write", "path": path, "content": body})
    return json.dumps({"tool": "files", "content": inner})


# ── 1. string-aware brace scanning ──────────────────────────────────


def test_unbalanced_file_content_no_longer_hides_the_router_object():
    for name, body in UNBALANCED_BODIES.items():
        objects = _all_json_objects(_router_output(body))
        assert len(objects) == 1, f"{name}: router object lost"
        assert objects[0]["tool"] == "files"


def test_balanced_file_content_still_parses():
    for name, body in BALANCED_BODIES.items():
        objects = _all_json_objects(_router_output(body))
        assert len(objects) == 1, f"{name}: router object lost"


def test_write_decision_survives_unbalanced_content(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "available_tools", lambda: ["chat", "code", "files"]
    )
    body = UNBALANCED_BODIES["js_arrow_truncated"]
    decision = parse_router_output(_router_output(body, "f.js"))
    assert decision.tool == "files"
    assert decision.is_fallback is False
    assert json.loads(decision.content)["content"] == body


def test_brace_inside_a_string_never_opens_an_object():
    """A lone brace in prose is not the start of a JSON object."""
    assert _all_json_objects('the model said "use { here"') == []


def test_a_truncated_object_does_not_hide_a_later_complete_one():
    """Depth counting made a later object unreachable once an earlier
    "{" never closed -- its closing brace only brought depth 2 -> 1."""
    text = '{"tool":"chat" truncated... {"tool":"chat","content":"real"}'
    objects = _all_json_objects(text)
    assert [o["content"] for o in objects] == ["real"]


def test_last_object_still_wins_over_echoed_earlier_ones():
    text = (
        '{"tool":"chat","content":"echo from history"}\n'
        '{"tool":"chat","content":"the actual answer"}'
    )
    assert parse_router_output(text).content == "the actual answer"
