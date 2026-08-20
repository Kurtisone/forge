"""
web_search runs at most once per run, whatever the query text.

Bench fixture e02 is the case this exists for: results from a first
search are already sitting in step_context, and the router searches
again with a *reformulated* query instead of answering from what it
has. The generic loop guard misses that -- it compares (tool, content)
pairs, and the content differs -- so the run burns a second search and
a step to arrive where it already was.

Two hint phrasings failed to stop the chaining (prose-only, then an
explicit worked JSON example, both with prompt caching disabled to
rule out a cache-reuse bug). "research" exists because of that: it
does search -> fetch -> synthesize inside one graph call, taking the
decision out of the router's hands. This guard is the same move at the
orchestrator level -- the fifth time on this codebase that a wording
fix lost to a deterministic check.

Degrading rather than erroring is deliberate: the run really did
gather results, and handing those back beats an internal guard
message. See also
test_web_search_repeated_call_degrades_gracefully_instead_of_erroring
in test_orchestrator.py, which covers the byte-identical repeat that
this guard now subsumes.
"""

import json

import pytest

import forge.orchestrator as orch_mod
from forge.orchestrator import Orchestrator
from forge.tools.registry import TOOLS

_RESULTS = "[web_search] 1. Qwen3.5 release notes - https://example.org/q - ..."


def _scripted(monkeypatch, decisions):
    """Feed the router one canned decision per step, in order."""
    calls = iter(decisions)
    monkeypatch.setattr(
        orch_mod, "call_llm", lambda p, grammar=None: json.dumps(next(calls))
    )


@pytest.fixture(autouse=True)
def _stub_search(monkeypatch):
    monkeypatch.setitem(TOOLS, "web_search", lambda content: _RESULTS)


def test_a_reformulated_second_search_is_refused(monkeypatch):
    _scripted(
        monkeypatch,
        [
            {"tool": "web_search", "content": "nouveautés Qwen3", "done": False},
            {"tool": "web_search", "content": "Qwen3 release notes", "done": False},
        ],
    )

    result = Orchestrator(max_steps=3).run("Quelles sont les nouveautés de Qwen3 ?")

    assert result.ok
    assert result.error is None
    assert result.output == _RESULTS
    assert "Stopped" not in result.output


def test_the_second_search_never_reaches_the_tool(monkeypatch):
    """The point of the guard is that the search does not run, not
    that its result is discarded afterwards -- a dispatched search
    costs a network round trip and several seconds either way."""
    dispatched = []

    monkeypatch.setitem(
        TOOLS,
        "web_search",
        lambda content: dispatched.append(content) or _RESULTS,
    )
    _scripted(
        monkeypatch,
        [
            {"tool": "web_search", "content": "query one", "done": False},
            {"tool": "web_search", "content": "query two", "done": False},
        ],
    )

    Orchestrator(max_steps=3).run("Quelles sont les nouveautés de Qwen3 ?")

    assert dispatched == ["query one"]


def test_switching_to_web_fetch_after_a_search_still_works(monkeypatch):
    """The guard is keyed on web_search alone. Following a search with
    a fetch is the behaviour the steering hint asks for and must stay
    open -- otherwise the guard would forbid the very move it wants."""
    monkeypatch.setitem(TOOLS, "web_fetch", lambda content: "page text")
    _scripted(
        monkeypatch,
        [
            {"tool": "web_search", "content": "nouveautés Qwen3", "done": False},
            {"tool": "web_fetch", "content": "https://example.org/q", "done": True},
        ],
    )

    result = Orchestrator(max_steps=3).run("Quelles sont les nouveautés de Qwen3 ?")

    assert result.ok
    assert result.tool == "web_fetch"
    assert result.output == "page text"


def test_a_single_search_is_untouched(monkeypatch):
    _scripted(
        monkeypatch,
        [{"tool": "web_search", "content": "articles Zig", "done": True}],
    )

    result = Orchestrator(max_steps=3).run("Trouve-moi des articles sur Zig")

    assert result.ok
    assert result.tool == "web_search"
    assert result.output == _RESULTS
