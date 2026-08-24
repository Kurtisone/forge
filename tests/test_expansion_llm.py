"""
Tests for the model-backed half of query expansion.

The model is substituted everywhere here. What is pinned is the
contract around it: the grammar llama.cpp has to accept, what is read
out of an answer, and -- the one that matters most -- that no failure
of the call ever reaches the caller as an exception. A rephrasing that
does not arrive costs a search that was already going to fail.
"""

import pytest

from forge import expansion, gbnf
from forge.errors import ProviderError


class TestGrammar:
    def test_llama_cpp_would_accept_it(self):
        """
        A rule name llama.cpp's lexer rejects is answered with 400 on
        every completion, not with a bad variant -- v3.10 spent a
        debugging cycle on `payload_call` before the underscore was
        found.
        """
        gbnf.validate(expansion._GRAMMAR)

    def test_it_admits_exactly_three_strings(self):
        assert expansion._GRAMMAR.count("string ws") == 3


class TestParse:
    def test_a_plain_array(self):
        assert expansion._parse('["un deux", "trois quatre"]') == [
            "un deux",
            "trois quatre",
        ]

    def test_a_fenced_array(self):
        """
        The grammar only exists on llama.cpp. On ollama or OpenRouter
        the same call runs unconstrained and comes back fenced.
        """
        raw = '```json\n["un deux", "trois quatre"]\n```'

        assert expansion._parse(raw) == ["un deux", "trois quatre"]

    def test_an_array_with_a_sentence_around_it(self):
        raw = 'Voici trois requêtes :\n["un deux", "trois quatre"]\nVoilà.'

        assert expansion._parse(raw) == ["un deux", "trois quatre"]

    def test_non_strings_are_dropped_not_raised_on(self):
        """Two strings and a number is still two usable queries."""
        assert expansion._parse('["un deux", 7, null, "trois quatre"]') == [
            "un deux",
            "trois quatre",
        ]

    @pytest.mark.parametrize(
        "raw",
        [
            '{"tool": "chat", "content": "..."}',
            "je ne peux pas répondre",
            "",
        ],
    )
    def test_anything_that_is_not_an_array_raises(self, raw):
        with pytest.raises(ValueError):
            expansion._parse(raw)


class TestFromLlm:
    def test_the_variants_come_back(self, monkeypatch):
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: (
                '["processeur mémoire disque", "NiPoGi AM06PRO", "32 Go de RAM"]'
            ),
        )

        assert expansion._from_llm("Tu peux me lister mon matériel ?") == [
            "processeur mémoire disque",
            "NiPoGi AM06PRO",
            "32 Go de RAM",
        ]

    def test_the_call_is_grammar_constrained(self, monkeypatch):
        seen = {}

        def fake(prompt, grammar=None):
            seen["grammar"] = grammar
            return '["a b", "c d", "e f"]'

        monkeypatch.setattr(expansion, "call_llm", fake)
        expansion._from_llm("Quel processeur ?")

        assert seen["grammar"] == expansion._GRAMMAR

    def test_a_dead_provider_yields_no_variants_and_no_exception(self, monkeypatch):
        """
        The contract. Everything this returns is a suggestion about
        where else to look, filtered afterwards by the same cutoff --
        so a failed call makes recall unhelped, never wrong.
        """

        def dead(prompt, grammar=None):
            raise ProviderError("connection refused")

        monkeypatch.setattr(expansion, "call_llm", dead)

        assert expansion._from_llm("Quel processeur ?") == []

    def test_an_unreadable_answer_yields_no_variants_and_no_exception(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            expansion, "call_llm", lambda prompt, grammar=None: "désolé, je ne sais pas"
        )

        assert expansion._from_llm("Quel processeur ?") == []

    def test_the_worked_example_is_not_taken_for_an_answer(self, monkeypatch):
        """
        This model copies a worked example verbatim when the input is
        unfamiliar -- measured on sysadmin (2026-08-11) and on recall
        (2026-08-16). The example's subject is the repository's own
        standing unanswerable question, so a variant about it is
        recognisable and harmless to drop.
        """
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: (
                '["recette tarte tatin pommes", "processeur NiPoGi", "cuisson tarte tatin moule"]'
            ),
        )

        assert expansion._from_llm("Quel processeur a mon NiPoGi ?") == [
            "processeur NiPoGi"
        ]

    def test_someone_whose_store_is_about_baking_may_still_ask(self, monkeypatch):
        """
        The leak check is asked of the QUERY first. A real question
        about the example's subject must not be the one thing this
        refuses to expand.
        """
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: (
                '["recette tarte tatin pommes", "cuisson tarte tatin moule", "tarte tatin caramel"]'
            ),
        )

        assert len(expansion._from_llm("Ma recette de tarte tatin ?")) == 3


class TestLlmMode:
    def test_it_carries_the_deterministic_variants_too(self, monkeypatch):
        """
        Built on top of `terms`, not instead of it: the free rewrites
        cost an embedding call each and do not depend on the model
        having had a good day.
        """
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: '["processeur mémoire disque"]',
        )

        produced = expansion.variants("Tu peux me lister mon matériel ?", "llm")

        assert "processeur mémoire disque" in produced
        assert "lister mon matériel" in produced

    def test_a_dead_provider_leaves_the_free_ones(self, monkeypatch):
        def dead(prompt, grammar=None):
            raise ProviderError("connection refused")

        monkeypatch.setattr(expansion, "call_llm", dead)

        assert expansion.variants("Tu peux me lister mon matériel ?", "llm") == [
            "lister mon matériel",
            "lister matériel",
        ]

    def test_the_total_is_still_capped(self, monkeypatch):
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: '["un deux", "trois quatre", "cinq six"]',
        )

        produced = expansion.variants("Tu peux me lister mon matériel ?", "llm")

        assert len(produced) == expansion.MAX_VARIANTS

    def test_off_never_calls_the_model(self, monkeypatch):
        def must_not_run(prompt, grammar=None):  # pragma: no cover
            raise AssertionError("the model was called with expansion off")

        monkeypatch.setattr(expansion, "call_llm", must_not_run)

        assert expansion.variants("Tu peux me lister mon matériel ?", "off") == []

    def test_terms_never_calls_the_model(self, monkeypatch):
        def must_not_run(prompt, grammar=None):  # pragma: no cover
            raise AssertionError("terms mode spent a model call")

        monkeypatch.setattr(expansion, "call_llm", must_not_run)

        expansion.variants("Tu peux me lister mon matériel ?", "terms")
