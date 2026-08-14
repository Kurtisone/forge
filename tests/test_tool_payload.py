"""
Tests for forge.tool_payload: the shared lenient loader the four
JSON-payload tools use.

The router's object shape (router/parser.py) is what removes the
double escaping in normal operation. This is the fallback for text
the model wrote itself -- grammar disabled, another provider, or
drift back to the escaped-string shape -- where a raw newline lands
inside the inner JSON because the model wrote \\n where \\\\n was
needed.
"""

import json

import pytest

from forge.tool_payload import loads_payload
from forge.tools import files as files_tool
from forge.tools import memory as memory_tool

# What the model actually produces: correct payload, one level of
# escaping short. json.loads raises "Invalid control character".
UNDER_ESCAPED = '{"action":"write","path":"a.py","content":"l1\nl2"}'


def test_well_formed_payload_is_parsed_strictly():
    assert loads_payload('{"action":"read","path":"a.py"}', "files") == {
        "action": "read",
        "path": "a.py",
    }


def test_under_escaped_control_character_is_recovered():
    with pytest.raises(json.JSONDecodeError):
        json.loads(UNDER_ESCAPED)
    payload = loads_payload(UNDER_ESCAPED, "files")
    assert payload["content"] == "l1\nl2"


def test_recovery_is_logged(caplog):
    """A run that needs this still worked, but the model is drifting --
    that has to be visible, not silently absorbed."""
    with caplog.at_level("WARNING"):
        loads_payload(UNDER_ESCAPED, "files")
    assert "unescaped control characters" in caplog.text


def test_a_genuinely_malformed_payload_still_raises():
    """Lenient must not mean permissive: the caller's own error
    message has to keep reaching the user."""
    with pytest.raises(json.JSONDecodeError):
        loads_payload('{"action":"write",', "files")


def test_files_write_survives_under_escaped_content(tmp_path, monkeypatch):
    monkeypatch.setattr(files_tool, "WORKSPACE_DIR", str(tmp_path))
    body = "def main():\n    print('hi')\n"
    instruction = f'{{"action":"write","path":"a.py","content":"{body}"}}'
    result = files_tool.run(instruction)
    assert not result.startswith("[error]")
    assert (tmp_path / "a.py").read_text() == body


def test_files_still_reports_truly_broken_json(tmp_path, monkeypatch):
    monkeypatch.setattr(files_tool, "WORKSPACE_DIR", str(tmp_path))
    assert files_tool.run("not json at all").startswith("[error]")


def test_memory_rejects_valid_json_that_is_not_an_object():
    """Valid JSON is not necessarily an object -- .get() on a string
    used to raise AttributeError out of a tool whose contract is to
    return its errors as text."""
    assert memory_tool.run('"recall"').startswith("[error]")
    assert memory_tool.run("[1, 2]").startswith("[error]")
