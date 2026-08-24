"""
Tests for the rescue pass: asking again, in other words, once the
cutoff has dropped everything.

The placement is what most of these pin. A recall whose first pass
keeps something must never reach the expansion -- not the variants,
not the search, and above all not the model call. The mechanism is
free on every question that already works or it is not worth having.
"""

from forge.graphs import recall
from forge.types import AgentState


def _hits(*distances):
    return [
        {"id": i, "kind": "fact", "content": f"entry {i}", "distance": d}
        for i, d in enumerate(distances)
    ]


def _state(query="Tu peux me lister mon matériel ?"):
    return AgentState(user_input=query, context={"query": query}, max_steps=4)


def _never(*a, **kw):  # pragma: no cover - the assertion is that it doesn't run
    raise AssertionError("the rescue pass ran on a question that already worked")


class TestItOnlyRunsWhenEverythingWasDropped:
    def test_a_first_pass_that_keeps_something_never_expands(self, monkeypatch):
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "llm")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(0.72))
        monkeypatch.setattr(recall.expansion, "variants", _never)
        monkeypatch.setattr(recall.memory_tool, "search_many", _never)

        state = recall._recall_node(_state())

        assert [r["distance"] for r in state.context["results"]] == [0.72]

    def test_an_empty_store_is_not_a_rescue_case(self, monkeypatch):
        """
        No rows at all means the store is empty, not that the question
        was badly worded. Rephrasing an empty store is a model call
        spent on nothing.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "llm")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: [])
        monkeypatch.setattr(recall.expansion, "variants", _never)

        state = recall._recall_node(_state())

        assert not state.ok

    def test_with_no_cutoff_it_can_never_fire(self, monkeypatch):
        """
        Nothing is ever dropped without a cutoff, so there is never a
        failure to rescue. graphs/recall.py says so at startup; this
        pins the behaviour behind the warning.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", None)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "llm")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.4))
        monkeypatch.setattr(recall.expansion, "variants", _never)

        state = recall._recall_node(_state())

        assert [r["distance"] for r in state.context["results"]] == [1.4]

    def test_off_means_off(self, monkeypatch):
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "off")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.4))
        monkeypatch.setattr(recall.expansion, "variants", _never)

        state = recall._recall_node(_state())

        assert not state.ok
        assert "mémoire" in state.final_output


class TestWhenItRuns:
    def test_a_rescued_entry_reaches_synthesis(self, monkeypatch):
        """
        The measured case: the first phrasing brings back nothing
        within the cutoff, another phrasing brings back the entry that
        answers.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.05))
        monkeypatch.setattr(
            recall.expansion, "variants", lambda q, mode: ["processeur mémoire"]
        )
        monkeypatch.setattr(
            recall.memory_tool,
            "search_many",
            lambda queries, **kw: [
                dict(_hits(0.72)[0], matched_query="processeur mémoire")
            ],
        )

        state = recall._recall_node(_state())

        assert state.ok
        assert [r["distance"] for r in state.context["results"]] == [0.72]

    def test_the_rescue_is_marked_for_the_sub_trace(self, monkeypatch):
        """
        A rescued answer looks exactly like an ordinary one in the UI
        unless it says so, and the two are worth telling apart while
        the mechanism is young.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.05))
        monkeypatch.setattr(recall.expansion, "variants", lambda q, mode: ["autre"])
        monkeypatch.setattr(
            recall.memory_tool, "search_many", lambda queries, **kw: _hits(0.72)
        )

        state = recall._recall_node(_state())

        assert state.context["expanded"] is True

    def test_the_cutoff_still_governs_what_comes_back(self, monkeypatch):
        """
        The rescue widens the search, it does not lower the bar. A
        rephrasing that finds nothing nearer changes nothing.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.05))
        monkeypatch.setattr(recall.expansion, "variants", lambda q, mode: ["autre"])
        monkeypatch.setattr(
            recall.memory_tool, "search_many", lambda queries, **kw: _hits(0.95, 1.2)
        )

        state = recall._recall_node(_state())

        assert not state.ok
        assert state.final_output == recall.non_answer.NOTHING_CLOSE_ENOUGH

    def test_the_original_query_is_not_searched_again(self, monkeypatch):
        """
        Its rows were just dropped. Re-embedding it would buy a second
        copy of a result already in hand.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.05))
        monkeypatch.setattr(recall.expansion, "variants", lambda q, mode: ["autre"])
        seen = {}
        monkeypatch.setattr(
            recall.memory_tool,
            "search_many",
            lambda queries, **kw: (seen.update(queries=queries), _hits(0.72))[1],
        )

        recall._recall_node(_state("Tu peux me lister mon matériel ?"))

        assert seen["queries"] == ["autre"]

    def test_no_variants_means_no_search(self, monkeypatch):
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.05))
        monkeypatch.setattr(recall.expansion, "variants", lambda q, mode: [])
        monkeypatch.setattr(recall.memory_tool, "search_many", _never)

        assert not recall._recall_node(_state()).ok

    def test_an_embedding_failure_mid_rescue_is_not_an_error_the_user_reads(
        self, monkeypatch
    ):
        """
        The first search reached the server, so this is a failure
        between the two calls. The answer without the rescue is the
        answer they were getting anyway.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.05))
        monkeypatch.setattr(recall.expansion, "variants", lambda q, mode: ["autre"])

        def dead(queries, **kw):
            raise recall.rag.EmbeddingError("connection refused")

        monkeypatch.setattr(recall.memory_tool, "search_many", dead)

        state = recall._recall_node(_state())

        assert not state.ok
        assert state.final_output == recall.non_answer.NOTHING_CLOSE_ENOUGH
        assert "[error]" not in state.final_output


class TestWhatTheRescueIsAllowedToSee:
    def test_it_never_searches_archived_conversation(self, monkeypatch):
        """
        Measured 2026-08-24. Rephrased as "processeur mémoire disque",
        the hardware question put #307 -- the fact that answers it --
        at 0.9435, and #167 at 0.8777. #167 is an archived exchange
        where someone asked to display a Containerfile and got the
        file back: long, dense with technical nouns, and it beats a
        one-line fact on almost any technical question. With it in
        scope no threshold admits 0.9435 and refuses 0.8777.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: _hits(1.05))
        monkeypatch.setattr(recall.expansion, "variants", lambda q, mode: ["autre"])
        seen = {}
        monkeypatch.setattr(
            recall.memory_tool,
            "search_many",
            lambda queries, **kw: (seen.update(kw), _hits(0.72))[1],
        )

        recall._recall_node(_state())

        assert seen["exclude_kind"] == recall.rag.ARCHIVED_KIND

    def test_the_first_pass_still_searches_everything(self, monkeypatch):
        """
        Archived transcript is not excluded because it is worthless.
        It is excluded from the RESCUE because that pass has already
        loosened the query, and loosening the query while keeping the
        noisiest half of the store in scope is what manufactures the
        intruder.
        """
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.88)
        monkeypatch.setattr(recall, "RECALL_EXPANSION", "terms")
        seen = {}
        monkeypatch.setattr(
            recall.memory_tool,
            "search",
            lambda q, **kw: (seen.update(kw), _hits(0.72))[1],
        )

        recall._recall_node(_state())

        assert "exclude_kind" not in seen
