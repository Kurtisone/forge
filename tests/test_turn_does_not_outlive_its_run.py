"""
The turn channel, and the promise it was not keeping.

forge/turn.py says of itself that it "follows subtrace.py: a module the
orchestrator sets at the top of a run and clears when it finishes", and
turn.get_input() documents its return as "the raw message, or '' outside
a run (direct calls, tests)". Nothing called turn.clear(), so neither
sentence was true.

WHY IT WAS REACHABLE RATHER THAN THEORETICAL

api.py serves every request from ONE ThreadPoolExecutor with two
workers. `POST /chat` runs the orchestrator on a worker and leaves the
message in that thread; `POST /run` runs any registered graph on the
SAME pool without going through the orchestrator at all (api.py's
run_graph builds the graph and calls g.run directly). So a /run lands on
a worker still holding the last /chat message -- and WHICH of the two
workers picks it up decides the answer.

The three readers, and what each does with a stale message:

  graphs/research.py   `turn.get_input() or query` -- prefers the stale
                       message to the real query, so a web answer about
                       beef stew gets a footer about a container named
                       in an unrelated earlier turn.
  graphs/delegate.py   `turn.get_input().strip() or state.user_input`
                       -- writes the wrong request into a job's
                       objective, which is the exact failure turn.py was
                       created to fix.
  tools/memory.py      `asked = turn.get_input()`, then refuses to
                       remember if it looks like a question -- a stale
                       message ending in "?" turns a `remember` into a
                       recall of someone else's question. A write
                       silently becoming an unrelated read.

graphs/sysadmin.py reads it too and is not affected: it cross-checks the
router's restatement, precisely because this channel can go stale.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import forge.orchestrator as orch_mod
from forge import turn
from forge.orchestrator import Orchestrator
from forge.tools import registry as tool_registry

MESSAGE = "Pourquoi searxng a redémarré ?"


@pytest.fixture(autouse=True)
def _clean():
    turn.clear()
    yield
    turn.clear()


def _router_says_chat(monkeypatch):
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt, **kw: json.dumps({"tool": "chat", "content": "x"}),
    )


def test_the_channel_still_carries_the_message_during_the_run(monkeypatch):
    """
    The fix must not silence the feature. A tool running inside the run
    sees the raw message -- which is the whole reason turn.py exists,
    since tools otherwise receive only the router's restatement.
    """
    seen = {}

    def a_tool(content):
        seen["during"] = turn.get_input()
        return "ok"

    monkeypatch.setitem(tool_registry.TOOLS, "chat", a_tool)
    _router_says_chat(monkeypatch)

    Orchestrator().run(MESSAGE)

    assert seen["during"] == MESSAGE


def test_it_is_gone_once_the_run_returns(monkeypatch):
    monkeypatch.setitem(tool_registry.TOOLS, "chat", lambda c: "ok")
    _router_says_chat(monkeypatch)

    Orchestrator().run(MESSAGE)

    assert turn.get_input() == ""


def test_it_is_gone_even_when_the_run_raises(monkeypatch):
    """
    A `finally`, not a line before `return`. A run that dies partway
    leaves the worker thread in the pool either way, and the next
    request to land on it is the one that would read the corpse.
    """

    def boom(prompt, **kw):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(orch_mod, "call_llm", boom)
    monkeypatch.setattr(orch_mod.Orchestrator, "_finish", _raising_finish)

    with pytest.raises(RuntimeError):
        Orchestrator().run(MESSAGE)

    assert turn.get_input() == ""


def _raising_finish(self, state, remember):  # pragma: no cover - helper
    raise RuntimeError("router exploded")


def test_a_later_graph_on_the_same_worker_reads_nothing(monkeypatch):
    """
    The reproduction, in the shape api.py actually has: ONE worker, a
    /chat turn, then a /run of a graph directly on the same thread.

    Before the fix this returned the previous turn's message. It is not
    asserted through research.names_something_local because the point is
    narrower and belongs here: the channel, not what any one reader does
    with it.
    """
    monkeypatch.setitem(tool_registry.TOOLS, "chat", lambda c: "ok")
    _router_says_chat(monkeypatch)

    pool = ThreadPoolExecutor(max_workers=1)
    try:
        pool.submit(Orchestrator().run, MESSAGE).result()
        # exactly what api.py's run_graph does: no orchestrator, no
        # turn.set_input, same pool.
        leaked = pool.submit(turn.get_input).result()
    finally:
        pool.shutdown(wait=True)

    assert leaked == "", f"a later request on this worker read {leaked!r}"
