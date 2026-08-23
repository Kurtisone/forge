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
        outcome.no_answer("recall: no results")
        assert outcome.pending() == "recall: no results"
        assert outcome.taken() == "recall: no results"
        assert outcome.pending() is None
        assert outcome.taken() is None

    def test_a_reason_is_always_recorded(self):
        # An empty reason must not read back as "no verdict" -- the
        # caller said something failed.
        outcome.no_answer("")
        assert outcome.pending() is not None


class TestAnsweredDecision:
    def test_plain_answer_is_answered(self):
        assert Orchestrator()._answered("Tu as un Steam Deck.") is True

    def test_reported_failure_wins_over_a_fluent_reply(self):
        # The point of the structural half: the text says nothing is
        # wrong, the run says otherwise.
        outcome.no_answer("recall: no results above the distance cutoff")
        assert Orchestrator()._answered("Je vais regarder ça pour toi.") is False

    def test_non_answer_text_alone_is_enough(self):
        assert Orchestrator()._answered(non_answer.NOTHING_CLOSE_ENOUGH) is False

    def test_verdict_is_consumed_so_the_next_turn_is_clean(self):
        outcome.no_answer("recall: no results")
        agent = Orchestrator()
        assert agent._answered("peu importe") is False
        assert agent._answered("Le port est 8080.") is True


class TestPersistence:
    def test_a_normal_exchange_carries_no_mark(self, store):
        memory.add_exchange("quel port ?", "8080")
        assert all("answered" not in m for m in _history(store))

    def test_both_messages_of_a_dead_exchange_are_marked(self, store):
        memory.add_exchange(
            "Quel est le modèle de ma voiture ?", non_answer.NOTHING_CLOSE_ENOUGH
        )
        memory.add_exchange(
            "Quel est le modèle de ma voiture ?",
            non_answer.NOTHING_CLOSE_ENOUGH,
            answered=False,
        )

        first, second = _history(store)[:2], _history(store)[2:]
        assert all("answered" not in m for m in first)
        assert [m["answered"] for m in second] == [False, False]

    def test_the_exchange_is_still_stored_and_visible(self, store):
        memory.add_exchange("et ma voiture ?", "[error] recall failed", answered=False)

        history = _history(store)
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[0]["content"] == "et ma voiture ?"


class TestRecallReportsItself:
    def test_cutoff_refusal_reaches_the_channel(self, monkeypatch):
        from forge.graphs import recall

        monkeypatch.setattr(
            recall.memory_tool, "search", lambda q: [{"id": 1, "distance": 9.0}]
        )
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.95)

        recall.run("Quel est le modèle de ma voiture ?")

        assert outcome.pending() is not None

    def test_a_successful_recall_reports_nothing(self, monkeypatch):
        from forge.graphs import recall

        monkeypatch.setattr(
            recall.memory_tool,
            "search",
            lambda q: [
                {
                    "id": 1,
                    "kind": "fact",
                    "distance": 0.4,
                    "project": None,
                    "content": "Possède un Steam Deck",
                    "created_at": "2026-08-23",
                }
            ],
        )
        monkeypatch.setattr(recall, "call_llm", lambda p: "Tu as un Steam Deck.")

        recall.run("Tu peux me lister mon matériel ?")

        assert outcome.pending() is None
