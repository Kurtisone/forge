"""
The detector and the things it has to detect, pinned together.

These tests deliberately call the PRODUCERS rather than asserting on
the constants. A constant compared to itself proves nothing: the
failure this guards against is a producer that stops using the
constant (or rewords the sentence around it), and only running the
producer can see that.
"""

import pytest

from forge import non_answer


class TestDetector:
    def test_empty_is_a_non_answer(self):
        assert non_answer.is_non_answer("")
        assert non_answer.is_non_answer("   \n ")
        assert non_answer.is_non_answer(None)  # type: ignore[arg-type]

    def test_real_answers_are_not(self):
        for text in (
            "Tu as un Steam Deck et un NiPoGi AM06PRO.",
            "Le port est 8080.",
            "Voici les erreurs trouvées dans le fichier : ...",
        ):
            assert not non_answer.is_non_answer(text)

    def test_leading_whitespace_does_not_hide_a_marker(self):
        assert non_answer.is_non_answer("\n  [error] recall failed: boom")

    def test_marker_in_the_middle_is_left_alone(self):
        # A synthesis quoting a log line is an answer. Only the opening
        # of the reply decides -- see the module docstring.
        assert not non_answer.is_non_answer(
            "Le service a redémarré trois fois ; le journal montre "
            "[error] cannot bind port 8080 à chaque tentative."
        )


class TestRecallProducers:
    def test_cutoff_refusal_is_recognised(self, monkeypatch):
        from forge.graphs import recall

        monkeypatch.setattr(
            recall.memory_tool, "search", lambda q, **kw: [{"id": 1, "distance": 9.0}]
        )
        monkeypatch.setattr(recall, "RECALL_MAX_DISTANCE", 0.95)

        state = recall.build().run("Quel est le modèle de ma voiture ?")

        assert non_answer.is_non_answer(state.final_output)

    def test_empty_store_is_recognised(self, monkeypatch):
        from forge.graphs import recall

        monkeypatch.setattr(recall.memory_tool, "search", lambda q, **kw: [])

        state = recall.build().run("Quel est le modèle de ma voiture ?")

        assert non_answer.is_non_answer(state.final_output)

    @pytest.mark.parametrize(
        "raw",
        [
            "Respond in plain text",  # prompt leak
            "",  # nothing generated
        ],
    )
    def test_synthesis_failures_are_recognised(self, raw):
        from forge.graphs import recall

        assert non_answer.is_non_answer(recall._clean_synthesis_response(raw))

    def test_a_real_synthesis_is_not_flagged(self):
        from forge.graphs import recall

        cleaned = recall._clean_synthesis_response(
            "Tu as un Steam Deck sous SteamOS et un NiPoGi AM06PRO."
        )
        assert not non_answer.is_non_answer(cleaned)


class TestDefaultGraphProducers:
    def test_provider_failure_is_recognised(self, monkeypatch):
        from forge.errors import ProviderError
        from forge.graphs import default

        def boom(*a, **k):
            raise ProviderError("backend down")

        monkeypatch.setattr(default, "call_llm", boom)

        state = default.build().run("bonjour")

        assert non_answer.is_non_answer(state.final_output)


class TestProducersFoundInTheStore:
    """
    Four producers that were writing their own refusal and were not in
    the closed set. None of them was found by reading the code: they
    were found by reading the real store on 2026-09-12, where 22 of
    195 archived entries are the assistant refusing and this module
    recognised none of them.

    Same discipline as every other class here -- the producer runs.
    Asserting on the constants would pass while the producer wrote
    something else.
    """

    def test_research_with_nothing_found_is_recognised(self, monkeypatch):
        from forge.graphs import research

        monkeypatch.setattr(research.web_search, "search", lambda q: [])

        state = research.build().run("pourquoi searxng a redémarré")

        assert non_answer.is_non_answer(state.final_output)

    def test_web_search_with_nothing_found_is_recognised(self, monkeypatch):
        from forge.tools import web_search

        monkeypatch.setattr(web_search, "last_unresponsive", list)

        assert non_answer.is_non_answer(web_search._format_results("tarte tatin", []))

    def test_a_search_backend_failure_is_not_the_same_claim(self, monkeypatch):
        """
        Both are recognised, and they must not be recognised as the
        same thing: an empty web and a search that did not run are one
        HTTP response and opposite claims. The first lets the model
        answer from its weights, and it sounds just as confident.
        """
        from forge.tools import web_search

        monkeypatch.setattr(web_search, "last_unresponsive", lambda: ["duckduckgo"])

        output = web_search._format_results("tarte tatin", [])

        assert non_answer.is_non_answer(output)
        assert output.startswith(non_answer.ERROR_PREFIX)

    def test_an_unresolvable_sysadmin_target_is_recognised(self):
        from forge.graphs import sysadmin
        from forge.types import AgentState

        state = AgentState(
            user_input="pourquoi searxng a redémarré ?",
            max_steps=1,
            context={"target_missed": "searxng", "units": [], "containers": []},
        )

        assert non_answer.is_non_answer(
            sysadmin._target_missed_node(state).final_output
        )

    def test_logs_that_could_not_be_collected_are_recognised(self):
        from forge.graphs import sysadmin
        from forge.types import AgentState

        state = AgentState(
            user_input="pourquoi searxng a redémarré ?",
            max_steps=1,
            context={
                "log_source": "podman logs searxng",
                "collect_failed": "command timed out after 0s",
                "target_hint": "searxng",
            },
        )

        assert non_answer.is_non_answer(
            sysadmin._collect_failed_node(state).final_output
        )


class TestDelegationProducers:
    """
    The job dialogue re-asking mid-job. Five of the real store's
    archived entries are copies of these two sentences, each one a
    user turn followed by Forge asking again -- a question with no
    answer in it, which is the shape that outranks the real answer to
    the same question.
    """

    @pytest.fixture(autouse=True)
    def _runner(self):
        from forge import runner
        from forge.executors import EchoExecutor

        r = runner.JobRunner(EchoExecutor(), timeout=5)
        runner.set_runner(r)
        yield r
        r.stop()
        runner.set_runner(None)

    def test_a_question_instead_of_an_answer_is_recognised(self):
        from forge import delegation, jobs

        job = jobs.create({})
        jobs.transition(job.id, jobs.AWAITING_USER, pending_field="objective")

        assert non_answer.is_non_answer(delegation.intercept("C'est à dire ?"))

    def test_an_unreadable_confirmation_is_recognised(self):
        from forge import delegation, jobs

        job = jobs.create({"objective": "a", "workspace": "b"})
        jobs.transition(job.id, jobs.AWAITING_USER, pending_field=delegation.CONFIRM)

        assert non_answer.is_non_answer(delegation.intercept("les tests passent"))
