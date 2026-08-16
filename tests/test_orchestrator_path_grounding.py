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
"""

import json

import pytest

import forge.orchestrator as orch_mod
from forge.orchestrator import Orchestrator, _decision_path, _path_is_grounded
from forge.types import AgentState, RouterDecision


def _state(user_input="", history=None, last_read_path=None):
    state = AgentState(user_input=user_input, max_steps=2)
    state.history = history or []
    state.last_read_path = last_read_path
    return state


@pytest.mark.parametrize(
    "tool,content,expected",
    [
        ("files", '{"action":"write","path":"a.py","content":"x"}', "a.py"),
        ("files", '{"action":"edit","path":"a.py","find":"x","replace":"y"}', "a.py"),
        ("review", '{"file_path":"a.py"}', "a.py"),
        ("files", '{"action":"list"}', None),
        ("files", "not json", None),
        ("chat", '{"path":"a.py"}', None),
    ],
)
def test_decision_path_reads_both_key_spellings(tool, content, expected):
    assert _decision_path(RouterDecision(tool=tool, content=content)) == expected


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
