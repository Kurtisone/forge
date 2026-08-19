"""
A write or an edit must never land on a path the model invented.

Two rounds of prompt wording failed to stop this (bench fixtures
c05b/f01b): asked to act on "ce fichier" with no path anywhere in
sight, the model produces a plausible one -- src/forge/main.py,
src/forge/config.py -- with complete confidence. That is the fourth
time on this codebase that a wording fix lost to a deterministic
check, so the guarantee lives here now and the bench fixtures are back
to unscored observation.

The asymmetry matters and is tested both ways: a READ of an invented
path fails loudly and harmlessly, so it still runs. A WRITE succeeds,
leaving a file nobody asked for -- so it is refused before dispatch.

EVERY path in a payload is checked, not just the first one found.
The guard shipped looking at two key spellings, "path" and
"file_path", and review already had a third: "test_path", whose value
graphs/review.py hands to pytest. A grounded file_path next to an
invented test_path went straight through.
"""

import json

import pytest

import forge.orchestrator as orch_mod
from forge.orchestrator import (
    Orchestrator,
    _decision_paths,
    _is_path_key,
    _looks_like_a_path,
    _path_is_grounded,
)
from forge.tool_payload import JSON_PAYLOAD_TOOLS
from forge.types import AgentState, RouterDecision


def _state(user_input="", history=None, last_read_path=None):
    state = AgentState(user_input=user_input, max_steps=2)
    state.history = history or []
    state.last_read_path = last_read_path
    return state


@pytest.mark.parametrize(
    "tool,content,expected",
    [
        (
            "files",
            '{"action":"write","path":"a.py","content":"x"}',
            (("path", "a.py"),),
        ),
        (
            "files",
            '{"action":"edit","path":"a.py","find":"x","replace":"y"}',
            (("path", "a.py"),),
        ),
        ("review", '{"file_path":"a.py"}', (("file_path", "a.py"),)),
        # The one that used to be invisible: two paths, and the guard
        # only ever saw the first.
        (
            "review",
            '{"file_path":"a.py","test_path":"tests/test_a.py"}',
            (("file_path", "a.py"), ("test_path", "tests/test_a.py")),
        ),
        ("files", '{"action":"list"}', ()),
        ("files", "not json", ()),
        ("chat", '{"path":"a.py"}', ()),
        # A path-shaped key with a non-path value is not a path.
        ("review", '{"file_path":"a.py","test_path":""}', (("file_path", "a.py"),)),
        ("review", '{"file_path":"a.py","test_path":42}', (("file_path", "a.py"),)),
        # `test`'s content is a runner command string, not JSON -- it
        # was invisible to this function entirely before this branch.
        ("test", "pytest tests/test_x.py", (("arg1", "tests/test_x.py"),)),
        ("test", "ruff check src/forge/graph.py", (("arg1", "src/forge/graph.py"),)),
        # The runner itself (parts[0]) is never treated as a path.
        ("test", "pytest", ()),
        # Flags are not paths, even when they look path-shaped.
        ("test", "pytest tests/ -k test_shell", (("arg1", "tests/"),)),
        # A bare word with no "/" and no suffix isn't shaped like a
        # path -- same rule tools/test.py itself uses.
        ("test", "ruff check", ()),
        ("test", 'not "valid shlex', ()),
    ],
)
def test_decision_paths_collects_every_path_in_the_payload(tool, content, expected):
    assert _decision_paths(RouterDecision(tool=tool, content=content)) == expected


@pytest.mark.parametrize(
    "key,guarded",
    [
        ("path", True),
        ("file_path", True),
        ("test_path", True),
        ("some_future_path", True),
        ("question", False),
        ("content", False),
        ("pathological", False),
    ],
)
def test_path_keys_are_recognized_by_shape(key, guarded):
    assert _is_path_key(key) is guarded


def test_every_path_key_in_the_router_prompt_is_guarded():
    """
    The invariant that keeps this from rotting again.

    test_path escaped the guard because someone taught the router a
    new path key and nothing here noticed. Teaching the router a
    fourth one now fails this test instead of shipping a hole.
    """
    import re as _re

    from forge.router.prompt import build_router_prompt

    prompt = build_router_prompt(
        "bonjour", history=[], available_tools=sorted(JSON_PAYLOAD_TOOLS)
    )
    keys = set(_re.findall(r'"([A-Za-z_]*path)"\s*:', prompt))
    assert "file_path" in keys and "test_path" in keys, (
        "the prompt no longer shows the payload keys this test reads; "
        f"found {sorted(keys)}"
    )
    unguarded = sorted(k for k in keys if not _is_path_key(k))
    assert not unguarded, (
        f"the router prompt teaches path keys the grounding guard ignores: "
        f"{unguarded}. Add them to _is_path_key in orchestrator.py."
    )


def test_user_message_grounds_a_path():
    assert _path_is_grounded("hello.go", _state(user_input="corrige hello.go"))


def test_history_grounds_a_path():
    state = _state(history=[{"role": "user", "content": "lis src/app.py"}])
    assert _path_is_grounded("src/app.py", state)


def test_a_read_this_run_grounds_the_write_that_follows():
    assert _path_is_grounded("hello.go", _state(last_read_path="hello.go"))


def test_leading_dot_slash_is_normalized_on_both_sides():
    assert _path_is_grounded("./hello.go", _state(last_read_path="hello.go"))


def test_a_basename_is_not_grounded_by_a_longer_path():
    # Two files can share a name. Deciding which one is meant is
    # exactly what the model must not do silently.
    state = _state(history=[{"role": "user", "content": "lis src/app.py"}])
    assert not _path_is_grounded("app.py", state)


def test_tool_output_does_not_ground_a_path():
    # step_context is untrusted: a fetched page naming a file must not
    # authorize writing to it. Same escalation the E-2 guard refuses,
    # arriving by a quieter door.
    state = _state(user_input="résume ça")
    state.step_context = [
        {"role": "assistant", "content": "[web_fetch] see src/app.py"}
    ]
    assert not _path_is_grounded("src/app.py", state)


def test_invented_write_path_is_refused_before_dispatch(monkeypatch, tmp_path):
    import forge.config as cfg
    from forge.tools import files as files_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_tool, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setitem(TOOLS, "files", files_tool.run)

    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {
                "tool": "files",
                "content": json.dumps(
                    {"action": "write", "path": "src/app.py", "content": "nope"}
                ),
            }
        ),
    )

    result = Orchestrator(max_steps=2).run("améliore le fichier")

    assert result.tool == "chat"
    assert "which file" in result.output.lower()
    # The point of guarding before dispatch rather than reporting after.
    assert not (tmp_path / "src" / "app.py").exists()


def test_invented_review_path_is_refused(monkeypatch, tmp_path):
    """
    review is read-only, but it is a terminal analysis of a file the
    model NAMED -- not a discovery step. Running it on an invented path
    buys nothing and costs everything: observed live at 47 seconds to
    reach "file not found". files:read and files:list stay exempt
    because they are how a run legitimately finds out what exists.
    """
    from forge.tools.registry import TOOLS

    # Registered only so the router doesn't fall back to chat before
    # the guard is reached. It is never called: the point of guarding
    # before dispatch is that the tool does not run.
    monkeypatch.setitem(TOOLS, "review", lambda content: pytest.fail("dispatched"))
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {"tool": "review", "content": json.dumps({"file_path": "src/nope.py"})}
        ),
    )

    result = Orchestrator(max_steps=1).run("améliore le fichier")

    assert result.tool == "chat"
    assert "which file" in result.output.lower()


def test_invented_test_path_is_refused(monkeypatch, tmp_path):
    """
    The hole this branch closes: `test`'s content is a runner command
    string, not JSON, so JSON_PAYLOAD_TOOLS never covered it and an
    invented path reached subprocess.run() with zero grounding check.
    Observed live 2026-08-19: "pytest tests/test_inexistant_xyz.py"
    dispatched straight through, saved only by pytest being absent
    from PATH in that environment -- an accident, not a guarantee.
    `test` is already in _MUTATING_TOOLS, so the fix is entirely in
    _decision_paths() seeing this tool at all.
    """
    from forge.tools.registry import TOOLS

    # Registered only so the router doesn't fall back to chat before
    # the guard is reached. It is never called: the point of guarding
    # before dispatch is that the tool does not run.
    monkeypatch.setitem(TOOLS, "test", lambda content: pytest.fail("dispatched"))
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {"tool": "test", "content": "pytest tests/test_inexistant_xyz.py"}
        ),
    )

    result = Orchestrator(max_steps=1).run("lance les tests")

    assert result.tool == "chat"
    assert "which" in result.output.lower()


def test_grounded_test_path_still_runs(monkeypatch):
    """A test path the user actually named must not be blocked."""
    from forge.tools.registry import TOOLS

    monkeypatch.setitem(TOOLS, "test", lambda content: "ok")
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {"tool": "test", "content": "pytest tests/test_router.py"}
        ),
    )

    result = Orchestrator(max_steps=1).run("lance les tests dans tests/test_router.py")

    assert result.tool == "test"
    assert result.output == "ok"


def test_invented_read_path_still_runs(monkeypatch, tmp_path):
    import forge.config as cfg
    from forge.tools import files as files_tool
    from forge.tools.registry import TOOLS

    monkeypatch.setattr(cfg, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setattr(files_tool, "WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setitem(TOOLS, "files", files_tool.run)

    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {
                "tool": "files",
                "content": json.dumps({"action": "read", "path": "nope.py"}),
            }
        ),
    )

    result = Orchestrator(max_steps=1).run("améliore le fichier")

    assert result.tool == "files"
    # Loud and harmless: the tool ran, found nothing, and said so.
    assert "not found" in result.output


def test_grounded_file_path_does_not_cover_an_invented_test_path(monkeypatch):
    """
    The hole this branch closes.

    "Relis src/forge/graph.py et lance ses tests" names one file. The
    model fills in the other from convention, and graphs/review.py
    runs `pytest <that>` -- executing workspace code chosen by a
    string nobody typed. pytest also auto-loads conftest.py from the
    rootdir before collecting, so "the file doesn't exist" is not the
    protection it sounds like.
    """
    from forge.tools.registry import TOOLS

    monkeypatch.setitem(TOOLS, "review", lambda content: pytest.fail("dispatched"))
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {
                "tool": "review",
                "content": json.dumps(
                    {
                        "file_path": "src/forge/graph.py",
                        "test_path": "tests/test_graph.py",
                    }
                ),
            }
        ),
    )

    result = Orchestrator(max_steps=1).run(
        "relis src/forge/graph.py et lance ses tests"
    )

    assert result.tool == "chat"
    # Names the field, so the next turn can answer the question that
    # was actually left open -- which tests, not which file.
    assert "test_path" in result.output


def test_two_grounded_paths_still_run(monkeypatch):
    """The mirror image: name both files and the review goes through."""
    from forge.tools.registry import TOOLS

    monkeypatch.setitem(TOOLS, "review", lambda content: "[review] ok")
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {
                "tool": "review",
                "content": json.dumps(
                    {
                        "file_path": "src/forge/graph.py",
                        "test_path": "tests/test_graph.py",
                    }
                ),
            }
        ),
    )

    result = Orchestrator(max_steps=1).run(
        "relis src/forge/graph.py et lance tests/test_graph.py"
    )

    assert result.tool == "review"
    assert result.output == "[review] ok"


@pytest.mark.parametrize(
    "value,is_path",
    [
        ("src/forge/graph.py", True),
        ("tests/test_graph.py", True),
        ("./a.py", True),
        ("/app/data/workspace/x.md", True),
        # A path may legally contain a space. A space alone proves
        # nothing, so it must not be what this rejects.
        ("mon fichier de notes.md", True),
        # The one observed live: pasted prose sitting in file_path.
        (
            (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
                "sed do eiusmod tempor incididunt..."
            ),
            False,
        ),
        ("Voici un texte, dis-moi ce que tu en penses", False),
        ("ligne un\nligne deux", False),
        ("a" * 256, False),
    ],
)
def test_prose_is_not_a_path(value, is_path):
    assert _looks_like_a_path(value) is is_path


def test_pasted_text_in_file_path_is_refused_even_though_it_is_grounded(monkeypatch):
    """
    Why shape is checked BEFORE provenance.

    Text the user pasted is grounded by construction -- it is a
    substring of their own message -- so the grounding guard can never
    catch it. Observed live 2026-08-19: the run spent 36 seconds
    reaching "File not found: /app/data/workspace/Lorem ipsum dolor
    sit amet, ...".
    """
    from forge.tools.registry import TOOLS

    pasted = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
        "sed do eiusmod tempor incididunt"
    )
    monkeypatch.setitem(TOOLS, "review", lambda content: pytest.fail("dispatched"))
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps(
            {"tool": "review", "content": json.dumps({"file_path": pasted})}
        ),
    )

    result = Orchestrator(max_steps=1).run(
        f"Voici un texte, dis-moi ce que tu en penses : {pasted}"
    )

    assert result.tool == "chat"
    # It is grounded -- that is the whole point of this test.
    assert _path_is_grounded(pasted, _state(user_input=f"Voici : {pasted}"))
    assert "not a file path" in result.output
