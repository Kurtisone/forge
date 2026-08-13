"""
Audit E-2, the half that holds: once a run has pulled in content Forge
doesn't control, no later step of that run may dispatch a mutating
tool.

Deterministic on purpose. The prompt-side framing
(tests/test_prompt_provenance.py) asks the model not to be steered;
this asks nothing of the model at all. Prompt wording has failed three
times on this project -- a refusal here cannot be talked out of.
"""

import json

import pytest

import forge.orchestrator as orch_mod
from forge.orchestrator import Orchestrator, _is_mutating


def _router(*decisions):
    """Replay a fixed sequence of router decisions, one per call."""
    seq = iter(decisions)

    def fake_llm(prompt):
        return json.dumps(next(seq))

    return fake_llm


def _enable(monkeypatch, *tools):
    """
    Register no-op handlers so dispatch is observable without side
    effects. Registered in the real TOOLS dict, not just via get_tool:
    the parser validates the router's chosen tool against
    available_tools() and silently falls back to chat otherwise, so a
    tool that isn't registered never reaches the guard at all.
    """
    from forge.tools.registry import TOOLS

    calls = []

    def make(name):
        def handler(content):
            calls.append((name, content))
            return f"[{name}] ran"

        return handler

    for name in tools:
        monkeypatch.setitem(TOOLS, name, make(name))
    return calls


# ── _is_mutating: the classification itself ──────────────────────────


@pytest.mark.parametrize("tool", ["shell", "test"])
def test_shell_and_test_always_mutate(tool):
    assert _is_mutating(tool, "anything")


@pytest.mark.parametrize("tool", ["chat", "code", "web_fetch", "research", "recall"])
def test_answering_tools_do_not_mutate(tool):
    assert not _is_mutating(tool, "anything")


def test_files_read_and_list_do_not_mutate():
    assert not _is_mutating("files", '{"action":"read","path":"a.py"}')
    assert not _is_mutating("files", '{"action":"list","path":"."}')


def test_files_write_mutates():
    assert _is_mutating("files", '{"action":"write","path":"a.py","content":"x"}')


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "[]",
        '"a string"',
        "{}",
        '{"path":"a.py"}',
        '{"action":"WRITE","path":"a.py"}',
    ],
)
def test_unparseable_or_ambiguous_files_payload_fails_closed(content):
    """
    files.py's own parser is more forgiving than this check. Anything
    that isn't demonstrably a read or a list has to count as a write --
    the gap between the two parsers is exactly where an escalation
    would live.
    """
    assert _is_mutating("files", content)


# ── the guard in a real run ──────────────────────────────────────────


def test_web_fetch_then_files_write_is_refused(monkeypatch):
    calls = _enable(monkeypatch, "web_fetch", "files")
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        _router(
            {"tool": "web_fetch", "content": "https://example.com", "done": False},
            {
                "tool": "files",
                "content": '{"action":"write","path":"pwn.py","content":"x"}',
            },
        ),
    )

    result = Orchestrator(max_steps=3).run("résume cette page")

    assert not result.ok
    assert "escalation guard" in result.error
    # The refusal is the point: the write never reached the tool.
    assert [name for name, _ in calls] == ["web_fetch"]


def test_refusal_names_the_source_and_offers_a_way_forward(monkeypatch):
    _enable(monkeypatch, "research", "shell")
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        _router(
            {"tool": "research", "content": "actualité", "done": False},
            {"tool": "shell", "content": "ls"},
        ),
    )

    result = Orchestrator(max_steps=3).run("cherche puis liste")

    assert "research" in result.output
    assert "separate request" in result.output


def test_web_fetch_then_files_read_is_still_allowed(monkeypatch):
    """
    The guard blocks mutation, not thinking. A read after a fetch is
    not an escalation and must stay reachable.
    """
    calls = _enable(monkeypatch, "web_fetch", "files")
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        _router(
            {"tool": "web_fetch", "content": "https://example.com", "done": False},
            {"tool": "files", "content": '{"action":"read","path":"a.py"}'},
        ),
    )

    result = Orchestrator(max_steps=3).run("compare la page et le fichier")

    assert result.ok
    assert [name for name, _ in calls] == ["web_fetch", "files"]


def test_files_read_then_write_is_untouched(monkeypatch):
    """
    The v3.9 read-then-write flow is the one legitimate multi-step
    chain this project actually uses. files:read is deliberately not
    an ingest tool, and this is what says so.
    """
    calls = _enable(monkeypatch, "files")
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        _router(
            {
                "tool": "files",
                "content": '{"action":"read","path":"hello.go"}',
                "done": False,
            },
            {
                "tool": "files",
                "content": '{"action":"write","path":"hello.go","content":"new"}',
            },
        ),
    )

    result = Orchestrator(max_steps=3).run("remplace Hello par Bienvenue")

    assert result.ok
    assert len(calls) == 2


def test_a_single_mutating_step_is_never_blocked(monkeypatch):
    """
    MAX_STEPS=1 is the default. Nothing about the common case changes:
    with no earlier step there is no external data, so the guard is
    inert.
    """
    calls = _enable(monkeypatch, "shell")
    monkeypatch.setattr(
        orch_mod, "call_llm", _router({"tool": "shell", "content": "ls -la"})
    )

    result = Orchestrator().run("liste les fichiers")

    assert result.ok
    assert calls == [("shell", "ls -la")]


def test_the_taint_does_not_survive_into_the_next_run(monkeypatch):
    """
    Per-run, not per-session: a blocked request is a refusal to do it
    in the same breath as reading a page, not a lockout.
    """
    _enable(monkeypatch, "web_fetch", "shell")
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        _router(
            {"tool": "web_fetch", "content": "https://example.com", "done": False},
            {"tool": "shell", "content": "ls"},
        ),
    )
    assert not Orchestrator(max_steps=3).run("fetch puis ls").ok

    monkeypatch.setattr(
        orch_mod, "call_llm", _router({"tool": "shell", "content": "ls"})
    )
    assert Orchestrator(max_steps=3).run("ls").ok


def test_the_guard_can_be_switched_off_deliberately(monkeypatch):
    calls = _enable(monkeypatch, "web_fetch", "shell")
    monkeypatch.setattr(orch_mod, "ALLOW_MUTATION_AFTER_EXTERNAL_DATA", True)
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        _router(
            {"tool": "web_fetch", "content": "https://example.com", "done": False},
            {"tool": "shell", "content": "ls"},
        ),
    )

    result = Orchestrator(max_steps=3).run("fetch puis ls")

    assert result.ok
    assert [name for name, _ in calls] == ["web_fetch", "shell"]


def test_a_blocked_run_is_not_persisted_to_memory(monkeypatch):
    """
    The refusal text must not become the assistant's remembered answer
    -- same reasoning as every other early exit in run().
    """
    _enable(monkeypatch, "web_fetch", "shell")
    remembered = []
    monkeypatch.setattr(
        orch_mod.Orchestrator,
        "_remember",
        lambda self, u, o: remembered.append((u, o)),
    )
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        _router(
            {"tool": "web_fetch", "content": "https://example.com", "done": False},
            {"tool": "shell", "content": "ls"},
        ),
    )

    Orchestrator(max_steps=3).run("fetch puis ls")

    assert remembered == []
