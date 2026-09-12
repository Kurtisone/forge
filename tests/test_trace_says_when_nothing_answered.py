"""
A run can finish cleanly and answer nothing, and the trace must say so.

Reported from a real run: the router invented a hostname, `web_fetch`
returned "[error] could not resolve host: ..." as its output -- a
string, no exception -- and the log shows both verdicts side by side:

    tool.result tool='web_fetch' length=68
    memory.not_indexed reason='non-answer reply' chars=68

The indexing path recognised it. The trace drew a green tick. Two
verdicts on one string, in one function, disagreeing.

`ok` was never the right thing to read. It is a rendering directive,
and every site that sets it says so -- graphs/recall.py's error node
writes `ok = True  # surface as message, not crash`. Setting it False
instead is not the fix either: `remember` is derived from it, and
3.20.1 is the bug report from the last time those two were confused
(the user's own question vanished with the answer that never came).

So the verdict is computed once, on the single exit path, and both
consumers read the same field. Once is not a convention here:
outcome.taken() reads and clears, so a second caller gets None and
would disagree with the first by construction.
"""

import json

import pytest

import forge.orchestrator as orch_mod
from forge import non_answer, outcome
from forge.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def _clean():
    outcome.clear()
    yield
    outcome.clear()


@pytest.fixture
def traces(tmp_path, monkeypatch):
    import forge.trace as trace_mod

    path = tmp_path / "traces.jsonl"
    monkeypatch.setattr(trace_mod, "TRACE_FILE", str(path))
    return path


def _last(path):
    return json.loads(path.read_text().strip().split("\n")[-1])


def _routes_to(monkeypatch, tool, content, output):
    from forge.tools.registry import TOOLS

    monkeypatch.setitem(TOOLS, tool, lambda c: output)
    monkeypatch.setattr(
        orch_mod,
        "call_llm",
        lambda prompt: json.dumps({"tool": tool, "content": content}),
    )


class TestTheReportedRun:
    def test_a_tool_that_returned_an_error_string_is_not_an_answer(
        self, traces, monkeypatch
    ):
        _routes_to(
            monkeypatch,
            "web_fetch",
            "https://www.xn--buf-hoa.example/recipe",
            f"{non_answer.ERROR_PREFIX}could not resolve host: nope",
        )

        result = Orchestrator(max_steps=1).run("la recette du bœuf bourguignon")

        # ok stays true: the failure reached the user as a message,
        # which is the behaviour, not the bug.
        assert result.ok is True
        assert _last(traces)["not_answered"] == "non-answer reply"

    def test_a_real_answer_leaves_the_field_empty(self, traces, monkeypatch):
        _routes_to(
            monkeypatch, "web_fetch", "https://example.org", "Le bœuf mijote 3 heures."
        )

        Orchestrator(max_steps=1).run("la recette du bœuf bourguignon")

        assert _last(traces)["not_answered"] is None
        assert _last(traces)["ok"] is True


class TestItAgreesWithTheIndexingDecision:
    """
    The two cannot drift, because there is one computation. This is
    the property the bug was the absence of.
    """

    def test_structural_refusal_reaches_the_trace_too(self, traces, monkeypatch):
        """
        A run that reported itself through forge/outcome.py, with a
        reply whose WORDING says nothing is wrong.
        """
        from forge.tools.registry import TOOLS

        def tool(_content):
            outcome.do_not_index("recall: no results above the distance cutoff")
            return "Je vais regarder ça pour toi."

        monkeypatch.setitem(TOOLS, "recall", tool)
        monkeypatch.setattr(
            orch_mod,
            "call_llm",
            lambda prompt: json.dumps({"tool": "recall", "content": "ma voiture"}),
        )

        Orchestrator(max_steps=1).run("quelle voiture ?")

        assert (
            _last(traces)["not_answered"]
            == "recall: no results above the distance cutoff"
        )

    def test_the_verdict_is_read_once_so_it_cannot_leak(self):
        """
        outcome.taken() clears on read. Asking twice is how the second
        asker gets a different answer from the first -- which is the
        shape of the bug, not a detail of it.
        """
        agent = Orchestrator()
        outcome.do_not_index("recall: no results")
        assert agent._not_an_answer("peu importe") == "recall: no results"
        assert agent._not_an_answer("Le port est 8080.") is None


class TestTheUiHasThreeStates:
    """
    Source assertions, same precedent and same caveat as
    test_ui_security.py: no JS runtime here, so this is a tripwire.
    """

    def _source(self):
        from pathlib import Path

        import forge.api as api_mod

        return (Path(api_mod.__file__).parent / "static" / "index.html").read_text(
            encoding="utf-8"
        )

    def test_the_trace_card_reads_not_answered(self):
        assert "t.not_answered" in self._source()

    def test_a_blank_run_is_neither_green_nor_red(self):
        src = self._source()
        assert ".trace-blank" in src
        assert "trace-blank-note" in src
