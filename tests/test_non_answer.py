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
