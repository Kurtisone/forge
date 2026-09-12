"""
The run-level "this turn did not answer" verdict, end to end.

Two properties matter more than the happy path: the verdict must not
survive into the next turn, and it must mark BOTH messages of the
exchange rather than only the reply.
"""

import json

import pytest

from forge import memory, non_answer, outcome
from forge.orchestrator import Orchestrator


@pytest.fixture(autouse=True)
def _clean_channel():
    outcome.clear()
    yield
    outcome.clear()


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    monkeypatch.setattr(memory, "MEMORY_FILE", str(path))
    return path


def _history(path):
    return json.loads(path.read_text(encoding="utf-8"))["history"]


class TestChannel:
    def test_empty_by_default(self):
        assert outcome.pending() is None

    def test_taken_clears(self):
        outcome.do_not_index("recall: no results")
        assert outcome.pending() == "recall: no results"
        assert outcome.taken() == "recall: no results"
        assert outcome.pending() is None
        assert outcome.taken() is None

    def test_a_reason_is_always_recorded(self):
        # An empty reason must not read back as "no verdict" -- the
        # caller said something failed.
        outcome.do_not_index("")
        assert outcome.pending() is not None


class TestAnsweredDecision:
    def test_a_plain_answer_is_indexable(self):
        assert Orchestrator()._indexable("Tu as un Steam Deck.") is True

    def test_reported_failure_wins_over_a_fluent_reply(self):
        # The point of the structural half: the text says nothing is
        # wrong, the run says otherwise.
        outcome.do_not_index("recall: no results above the distance cutoff")
        assert Orchestrator()._indexable("Je vais regarder ça pour toi.") is False

    def test_non_answer_text_alone_is_enough(self):
        assert Orchestrator()._indexable(non_answer.NOTHING_CLOSE_ENOUGH) is False

    def test_verdict_is_consumed_so_the_next_turn_is_clean(self):
        outcome.do_not_index("recall: no results")
        agent = Orchestrator()
        assert agent._indexable("peu importe") is False
        assert agent._indexable("Le port est 8080.") is True


class TestPersistence:
    def test_a_normal_exchange_carries_no_mark(self, store):
        memory.add_exchange("quel port ?", "8080")
        assert all("index" not in m for m in _history(store))

    def test_both_messages_of_a_dead_exchange_are_marked(self, store):
        memory.add_exchange(
            "Quel est le modèle de ma voiture ?", non_answer.NOTHING_CLOSE_ENOUGH
        )
        memory.add_exchange(
            "Quel est le modèle de ma voiture ?",
            non_answer.NOTHING_CLOSE_ENOUGH,
            index=False,
        )

        first, second = _history(store)[:2], _history(store)[2:]
        assert all("index" not in m for m in first)
        assert [m["index"] for m in second] == [False, False]

    def test_the_exchange_is_still_stored_and_visible(self, store):
        memory.add_exchange("et ma voiture ?", "[error] recall failed", index=False)

        history = _history(store)
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "et ma voiture ?"


class TestRecallReportsItself:
    def test_every_recall_reports_itself_even_a_good_one(self, monkeypatch):
        # The rule taken on 2026-08-23: a recall answer was rebuilt
        # from the store, so writing it back gives the store a second,
        # worse copy of what it already holds -- worse because the
        # copy carries the question. #138 was exactly that, an archived
        # recall whose reply was already partial when written, and it
        # outranked everything on "Tu peux me lister mon matériel ?".
        from forge.graphs import recall

        monkeypatch.setattr(
            recall.memory_tool,
            "search",
            lambda q, **kw: [
                {
                    "id": 308,
                    "kind": "fact",
                    "distance": 0.8366,
                    "project": None,
                    "content": "NiPoGi AM06PRO, Ryzen 5500U, 32 Go",
                    "created_at": "2026-08-23",
                }
            ],
        )
        monkeypatch.setattr(recall, "call_llm", lambda p: "Tu as un Ryzen 5500U.")

        answer = recall.run("Quel processeur a mon NiPoGi ?")

        # The answer is good and the user sees it...
        assert "5500U" in answer
        # ...and it still must not be written back.
        assert outcome.pending() is not None

    def test_cutoff_refusal_reaches_the_channel(self, monkeypatch):
        from forge.graphs import recall

        monkeypatch.setattr(
            recall.memory_tool, "search", lambda q, **kw: [{"id": 1, "distance": 9.0}]
        )
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.95)

        recall.run("Quel est le modèle de ma voiture ?")

        assert outcome.pending() is not None

    def test_a_turn_that_never_touched_recall_reports_nothing(self):
        # The rule is about recall, not about every turn. Anything
        # else keeps being indexed exactly as before.
        assert outcome.pending() is None
        assert Orchestrator()._indexable("Le port est 8080.") is True


class TestAGraphThatEndsWithoutAnswering:
    """
    The structural half, generalised past recall.

    Every graph's refusal node sets `ok = True` on purpose, so the user
    reads a message rather than a crash -- and that is exactly what
    erases the fact the store needs. recall.py was told to report it in
    August; research, review, sysadmin and the default fallback were
    not, and the real store shows what that cost: on 2026-09-12, 22 of
    its 195 archived entries are the assistant refusing.

    Declared on the node rather than derived from its output, because
    deriving it from the output is the wording test that already lives
    in forge/non_answer.py. The two halves are worth having precisely
    because they fail differently.
    """

    def _graph(self, answers):
        from forge.graph import Graph

        def node(state):
            state.final_output = "rien"
            return state

        g = Graph("probe")
        g.add_node("only", node, answers=answers)
        return g

    def test_ending_on_a_node_that_does_not_answer_reports_it(self):
        self._graph(answers=False).run("pourquoi ?")

        assert outcome.pending() is not None

    def test_ending_anywhere_else_reports_nothing(self):
        self._graph(answers=True).run("pourquoi ?")

        assert outcome.pending() is None

    def test_the_reason_names_the_graph_and_the_node(self):
        self._graph(answers=False).run("pourquoi ?")

        reason = outcome.pending()
        assert "probe" in reason
        assert "only" in reason

    def test_research_with_nothing_found_reports_it(self, monkeypatch):
        from forge.graphs import research

        monkeypatch.setattr(research.web_search, "search", lambda q: [])

        state = research.build().run("pourquoi searxng a redémarré")

        assert non_answer.is_non_answer(state.final_output)
        assert outcome.pending() is not None

    def test_a_research_answer_is_still_indexed(self, monkeypatch):
        from forge.graphs import research

        monkeypatch.setattr(
            research.web_search,
            "search",
            lambda q: [{"title": "t", "url": "http://x", "snippet": "s"}],
        )
        monkeypatch.setattr(research.web_fetch, "run", lambda url: "le contenu")
        monkeypatch.setattr(
            research, "call_llm", lambda p, grammar=None: "SearXNG a redémarré."
        )

        research.build().run("pourquoi searxng a redémarré")

        assert outcome.pending() is None

    def test_an_unresolvable_sysadmin_target_reports_it(self, monkeypatch):
        """
        `#19` and `#26` of the real store, both of them the deterministic
        node that exists because a model asked to diagnose the wrong
        subsystem will do it fluently. Both were indexed.
        """
        from forge.graphs import sysadmin

        # _run_fixed is sysadmin's one external boundary -- same level
        # tests/test_sysadmin.py mocks at.
        monkeypatch.setattr(sysadmin, "_run_fixed", lambda cmd, timeout: "")

        state = sysadmin.build().run(
            "pourquoi forge-inexistant plante ?",
            initial_context={"target_hint": "forge-inexistant", "question": None},
        )

        assert non_answer.is_non_answer(state.final_output)
        assert outcome.pending() is not None
