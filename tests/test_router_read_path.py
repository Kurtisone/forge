"""
The post-read steering hint must never hand the model a path-shaped
blank to fill in.

The bug this pins down: the hint used to spell the path as the literal
string "<same path as above>", and the model copied those exact bytes
into a real write payload instead of substituting the file it had just
read (bench fixture f01). A placeholder in a payload field is an
invitation, not an instruction -- so the path is interpolated from the
routing decision now, and when no decision supplied one the hint stops
steering toward a write at all.

Note the asymmetry, which is deliberate: "<the FULL file content
above...>" stays a placeholder. Content is something the model must
genuinely produce from what it can see; a path is something Forge
already knows and must therefore never ask for.
"""

import pytest

from forge.orchestrator import _read_path_of
from forge.router.prompt import build_router_prompt
from forge.types import RouterDecision

TOOLS = ["chat", "files", "code"]
READ_RESULT = [{"role": "assistant", "content": "[files] PORT = 8080\nDEBUG = True"}]


def _prompt(**kw):
    return build_router_prompt(
        "Passe DEBUG à False", step_context=READ_RESULT, available_tools=TOOLS, **kw
    )


def test_known_path_is_interpolated_not_placeheld():
    prompt = _prompt(last_read_path="src/app.py")
    assert '"path":"src/app.py"' in prompt
    assert "src/app.py" in prompt


def test_no_path_placeholder_anywhere_in_the_prompt():
    # The exact bytes the model was observed copying, plus the general
    # shape: a "path" JSON key whose value is an angle-bracket slot.
    for prompt in (_prompt(last_read_path="src/app.py"), _prompt()):
        assert "<same path as above>" not in prompt
        assert '"path":"<' not in prompt
        assert '"path": "<' not in prompt


def test_unknown_path_refuses_to_steer_toward_a_write():
    prompt = _prompt()
    assert "No real file path is known" in prompt
    # It must not simultaneously steer toward a write. Scoped to the
    # hint: the static examples earlier in the template legitimately
    # show action:write, and asserting over the whole prompt would
    # only test that those examples still exist.
    assert "CURRENT, real content" not in prompt


def test_content_stays_a_placeholder():
    # Guards the asymmetry: tightening paths must not accidentally
    # remove the content slot, which the model does have to fill.
    assert "<the FULL file content above" in _prompt(last_read_path="src/app.py")


@pytest.mark.parametrize(
    "tool,content,expected",
    [
        ("files", '{"action":"read","path":"src/app.py"}', "src/app.py"),
        ("files", '{"action":"write","path":"src/app.py","content":"x"}', None),
        ("files", '{"action":"read"}', None),
        ("files", '{"action":"read","path":""}', None),
        ("files", '{"action":"read","path":123}', None),
        ("files", "not json at all", None),
        ("review", '{"action":"read","path":"src/app.py"}', None),
        ("chat", "hello", None),
    ],
)
def test_read_path_of_is_narrow(tool, content, expected):
    assert _read_path_of(RouterDecision(tool=tool, content=content)) == expected
