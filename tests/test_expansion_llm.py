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

    def test_a_one_word_rewrite_is_unrepresentable(self):
        """
        Measured 2026-08-24: two of three real questions came back as
        single nouns -- ['config', 'développement', 'matériel'] and
        ['conteneurs', 'conteneurs', 'conteneurs']. Every one was
        dropped by keep(), every rescue silently cancelled. Filtering
        after the fact left the mechanism doing nothing; the floor
        belongs where the model cannot produce it.
        """
        assert 'word (" " word)+' in expansion._GRAMMAR


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
    def test_it_does_not_carry_the_deterministic_variants(self, monkeypatch):
        """
        `terms` used to be appended here on the theory that a free
        variant costs nothing. Measured 2026-08-24, it costs two
        things: it pushes distances up on this embedding model (four
        questions out of four), and its rows compete for the merge's
        top_k slots, so a variant that finds nothing useful can push
        the rescued entry out of the list.
        """
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: '["processeur mémoire disque"]',
        )

        produced = expansion.variants("Tu peux me lister mon matériel ?", "llm")

        assert produced == ["processeur mémoire disque"]

    def test_a_dead_provider_leaves_nothing_rather_than_the_losers(self, monkeypatch):
        """
        No rescue beats a rescue built out of the rewrites that lost
        the measurement.
        """

        def dead(prompt, grammar=None):
            raise ProviderError("connection refused")

        monkeypatch.setattr(expansion, "call_llm", dead)

        assert expansion.variants("Tu peux me lister mon matériel ?", "llm") == []

    def test_the_total_is_still_capped(self, monkeypatch):
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: (
                '["un deux", "trois quatre", "cinq six", "sept huit", "neuf dix"]'
            ),
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


class TestWhatTheLogSays:
    def test_the_count_is_taken_after_the_filtering(self, monkeypatch, caplog):
        """
        Measured 2026-08-24: asked about containers, the model answered
        ["conteneurs", "conteneurs", "conteneurs"]. All three were
        dropped as one-word rewrites, the search that followed had
        nothing to search with, and the log said kept=3.
        """
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: '["conteneurs", "conteneurs", "conteneurs"]',
        )
        events = []
        monkeypatch.setattr(
            expansion.log, "event", lambda name, **f: events.append((name, f))
        )

        assert (
            expansion.variants("Qu'est-ce que j'utilise comme conteneurs ?", "llm")
            == []
        )

        name, fields = events[-1]
        assert name == "recall.expansion_llm"
        assert fields["proposed"] == 3
        assert fields["kept"] == 0

    def test_a_collapse_to_nothing_is_said_out_loud(self, monkeypatch, caplog):
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: '["conteneurs", "conteneurs", "conteneurs"]',
        )

        with caplog.at_level("WARNING"):
            expansion.variants("Qu'est-ce que j'utilise comme conteneurs ?", "llm")

        assert "nothing to search with" in caplog.text

    def test_a_normal_call_says_nothing_alarming(self, monkeypatch, caplog):
        monkeypatch.setattr(
            expansion,
            "call_llm",
            lambda prompt, grammar=None: '["processeur mémoire disque"]',
        )

        with caplog.at_level("WARNING"):
            expansion.variants("Tu peux me lister mon matériel ?", "llm")

        assert "nothing to search with" not in caplog.text


def test_the_prompt_forbids_guessing_a_product_name():
    """
    Measured 2026-08-24 and then measured again. Asked "Qu'est-ce que
    j'utilise comme conteneurs ?", the model rewrote it with two
    brands the question never mentioned, on a store whose answer says
    podman -- so the expansion searched for the wrong ecosystem and
    pushed the fact that answers from 1.0400 to 1.0760.

    The first attempt at a rule NAMED those two brands as the thing
    not to write. They came back anyway, in every draw, and the prompt
    grew 75 tokens for it. Naming the wrong answer put the wrong
    answer in front of the model; the rule is positive now, and says
    where the nouns may come from instead of which ones are banned.
    """
    assert "Docker" not in expansion._PROMPT
    assert "comes from the question" in expansion._PROMPT
