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
from forge.router.prompt import build_router_prompt

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


# ── 2. nested object content (no double escaping) ───────────────────


def _files_tool(monkeypatch):
    monkeypatch.setattr(
        registry_mod, "available_tools", lambda: ["chat", "code", "files"]
    )


def test_object_content_is_re_encoded_for_the_tool(monkeypatch):
    """The payload arrives as a real object; the tool contract
    (run(content: str) parsing JSON) is unchanged."""
    _files_tool(monkeypatch)
    body = "def main():\n    print('hi')\n"
    raw = json.dumps(
        {
            "tool": "files",
            "content": {"action": "write", "path": "a.py", "content": body},
        }
    )
    decision = parse_router_output(raw)
    assert decision.tool == "files"
    assert isinstance(decision.content, str)
    assert json.loads(decision.content) == {
        "action": "write",
        "path": "a.py",
        "content": body,
    }


def test_object_content_survives_unbalanced_braces(monkeypatch):
    """Both halves of the fix have to hold at once: the object shape
    is worthless if the scanner still loses the enclosing object."""
    _files_tool(monkeypatch)
    body = UNBALANCED_BODIES["js_arrow_truncated"]
    raw = json.dumps(
        {
            "tool": "files",
            "content": {"action": "write", "path": "f.js", "content": body},
        }
    )
    decision = parse_router_output(raw)
    assert decision.tool == "files"
    assert json.loads(decision.content)["content"] == body


def test_object_content_keeps_non_ascii_readable(monkeypatch):
    _files_tool(monkeypatch)
    raw = json.dumps(
        {
            "tool": "files",
            "content": {"action": "write", "path": "n.txt", "content": "é"},
        },
        ensure_ascii=False,
    )
    assert "é" in parse_router_output(raw).content


def test_done_false_still_works_with_object_content(monkeypatch):
    _files_tool(monkeypatch)
    raw = json.dumps(
        {
            "tool": "files",
            "content": {"action": "read", "path": "hello.go"},
            "done": False,
        }
    )
    decision = parse_router_output(raw)
    assert decision.done is False


def test_empty_object_content_falls_through(monkeypatch):
    """An empty payload is no more usable than an empty string, and
    must not short-circuit the rest of the extraction cascade."""
    _files_tool(monkeypatch)
    text = '{"tool":"files","content":{}}\nBien sûr, voici la réponse.'
    decision = parse_router_output(text)
    assert decision.tool == "chat"
    assert "Bien sûr" in decision.content


def test_string_content_shape_still_accepted(monkeypatch):
    """The object shape is what the prompt now teaches, but the string
    shape has to keep working: grammar-disabled setups, other
    providers, and any model that drifts back to it."""
    _files_tool(monkeypatch)
    decision = parse_router_output(_router_output(BALANCED_BODIES["go"], "hello.go"))
    assert decision.tool == "files"
    assert json.loads(decision.content)["path"] == "hello.go"


# ── 3. the prompt and the parser agree ──────────────────────────────

_JSON_PAYLOAD_TOOLS = ("files", "memory", "review", "sysadmin")

_ALL_TOOLS = [
    "chat",
    "code",
    "files",
    "shell",
    "git",
    "memory",
    "recall",
    "review",
    "research",
    "web_fetch",
    "web_search",
    "sysadmin",
]


def _example_lines(prompt: str) -> list[str]:
    return [
        line.strip()
        for line in prompt.splitlines()
        if line.strip().startswith("{") and '"tool":"' in line
    ]


def test_every_worked_example_parses_into_the_tool_it_names(monkeypatch):
    """The examples ARE the contract for a small local model: whatever
    shape they demonstrate is what it will imitate. If one drifts out
    of what the parser accepts, the model is being taught to fail."""
    monkeypatch.setattr(registry_mod, "available_tools", lambda: list(_ALL_TOOLS))
    prompt = build_router_prompt("x", available_tools=_ALL_TOOLS)
    lines = _example_lines(prompt)
    assert len(lines) >= 15
    for line in lines:
        expected = json.loads(line)["tool"]
        decision = parse_router_output(line)
        assert decision.tool == expected, line
        assert decision.is_fallback is False, line


def test_json_payload_examples_never_re_encode_their_payload(monkeypatch):
    """Guards the regression directly: a payload demonstrated as an
    escaped string teaches the double escaping the model cannot hold
    on multi-line content."""
    prompt = build_router_prompt("x", available_tools=_ALL_TOOLS)
    for line in _example_lines(prompt):
        obj = json.loads(line)
        if obj["tool"] in _JSON_PAYLOAD_TOOLS:
            assert isinstance(obj["content"], dict), line


def test_json_payload_examples_reach_their_tool_as_parseable_json(monkeypatch):
    """The re-encoding has to land as something the tool's own
    json.loads accepts -- str(dict) would have produced Python repr
    with single quotes and broken every one of them."""
    monkeypatch.setattr(registry_mod, "available_tools", lambda: list(_ALL_TOOLS))
    prompt = build_router_prompt("x", available_tools=_ALL_TOOLS)
    for line in _example_lines(prompt):
        obj = json.loads(line)
        if obj["tool"] in _JSON_PAYLOAD_TOOLS:
            assert json.loads(parse_router_output(line).content) == obj["content"]
