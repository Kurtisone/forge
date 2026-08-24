"""
Tests for the deterministic half of query expansion.

No model, no network: everything here is a pure function on a string.
The measured case -- "Tu peux me lister mon matériel ?" -- is the one
that opens the file, because it is the question that produced the lot.
"""

import pytest

from forge import expansion


class TestTerms:
    def test_the_conversational_frame_comes_off(self):
        assert "lister mon matériel" in expansion.terms(
            "Tu peux me lister mon matériel ?"
        )

    def test_the_content_words_survive_alone(self):
        assert "lister matériel" in expansion.terms("Tu peux me lister mon matériel ?")

    def test_identifiers_keep_their_shape(self):
        """
        `sqlite-vec`, `NiPoGi`, `Q4_K_M` are the only vocabulary this
        store has that nothing else does. A tokenizer that splits or
        lowercases them throws away the words that make a hit.
        """
        produced = " ".join(expansion.terms("Quel processeur a mon NiPoGi AM06PRO ?"))

        assert "NiPoGi" in produced
        assert "AM06PRO" in produced

    def test_underscores_survive_inside_a_token(self):
        produced = " ".join(expansion.terms("On tourne toujours en Q4_K_M ?"))

        assert "Q4_K_M" in produced

    def test_a_query_already_written_as_terms_produces_nothing(self):
        """
        Both rewrites fold back to the query, so there is nothing to
        add and no embedding call to pay for.
        """
        assert expansion.terms("processeur NiPoGi") == []

    def test_the_two_rewrites_are_not_repeated_when_they_agree(self):
        produced = expansion.terms("Tu peux me donner le port du serveur ?")

        assert len(produced) == len(set(produced))

    def test_a_variant_never_comes_back_as_a_single_word(self):
        """
        Same argument rag._MIN_ENTRY_WORDS makes about a one-word
        entry, from the search side: it matches everything and answers
        nothing.
        """
        for variant in expansion.terms("Quel est mon processeur ?"):
            assert len(variant.split()) >= 2

    def test_trailing_politeness_is_dropped(self):
        produced = expansion.terms("Tu peux me lister mes projets s'il te plaît ?")

        assert all("plaît" not in v for v in produced)


class TestKeep:
    def test_a_variant_that_is_the_query_again_is_dropped(self):
        """Different clothes, same question: an embedding call for nothing."""
        assert (
            expansion.keep(["Le PROCESSEUR du NiPoGi !"], "le processeur du nipogi")
            == []
        )

    def test_duplicates_are_dropped_by_folded_form(self):
        kept = expansion.keep(["processeur NiPoGi", "Processeur nipogi"], "matériel")

        assert kept == ["processeur NiPoGi"]

    def test_an_answer_sized_candidate_is_dropped(self):
        long_one = (
            "Ton NiPoGi AM06PRO embarque un Ryzen 5500U " + "et beaucoup de RAM " * 6
        )

        assert expansion.keep([long_one], "matériel") == []

    def test_the_number_of_variants_is_capped(self):
        candidates = [f"variante numéro {i}" for i in range(20)]

        assert len(expansion.keep(candidates, "matériel")) == expansion.MAX_VARIANTS

    def test_order_is_preserved(self):
        kept = expansion.keep(["premier candidat", "second candidat"], "matériel")

        assert kept == ["premier candidat", "second candidat"]


class TestVariants:
    def test_off_produces_nothing(self):
        assert expansion.variants("Tu peux me lister mon matériel ?", "off") == []

    def test_an_empty_query_produces_nothing(self):
        assert expansion.variants("   ", "terms") == []

    def test_an_unknown_mode_is_off_and_says_so(self, caplog):
        """
        Falling back to the full behaviour on a typo would turn
        RECALL_EXPANSION=trems into work nobody asked for. Falling back
        to none of it costs exactly what the feature cost before it
        existed.
        """
        with caplog.at_level("WARNING"):
            assert expansion.variants("Quel processeur ?", "trems") == []

        assert "trems" in caplog.text

    @pytest.mark.parametrize("mode", expansion.MODES)
    def test_every_declared_mode_is_handled_without_warning(self, mode, caplog):
        with caplog.at_level("WARNING"):
            expansion.variants("Tu peux me lister mon matériel ?", mode)

        assert "unknown RECALL_EXPANSION" not in caplog.text


def test_an_apostrophe_does_not_smuggle_a_one_word_variant_through():
    """
    `m'appelle` is one word to a reader and two to anything that
    splits on punctuation. The word count is taken on the variant, not
    on its folded form, or "Comment je m'appelle ?" expands to a
    dangling verb -- the exact shape rag.remember refuses to store.
    """
    assert expansion.terms("Comment je m'appelle ?") == []
