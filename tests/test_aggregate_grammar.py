"""
Tests for the grammar that writes the aggregate.

The closure gate says a word the sources never used must not reach the
store. `invented` can only report that after the model has written it;
the grammar makes it unwritable. Same move this repository has made
eleven times now, for the same reason -- a rule the model is asked to
follow is one it follows most of the time.

The check that matters most is the rule names: llama.cpp's lexer
rejects underscores and answers 400 to every completion when it finds
one, which is a dead call rather than a degraded one.
"""

from collections import Counter

import pytest

from forge import aggregate, gbnf

SOURCES = [
    "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM",
    "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go",
]
COMMON = Counter({"de": 40, "le": 40, "et": 40, "avec": 40, "un": 40, "a": 40})
LIMIT = 10


@pytest.fixture
def built():
    return aggregate.grammar(SOURCES, COMMON, LIMIT)


def test_llama_cpp_would_accept_it(built):
    gbnf.validate(built)


def test_rule_names_use_only_characters_llama_cpp_accepts(built):
    assert gbnf.check_rule_names(built) == []


def test_every_source_word_is_reachable(built):
    for word in ("NiPoGi", "AM06PRO", "Ryzen", "5500U", "SSD", "Matériel"):
        assert f'"{word}"' in built


def test_a_word_that_is_neither_a_source_word_nor_common_is_unreachable(built):
    """
    The whole point: `Nvidia` is a plausible completion and nobody
    wrote it. Under this grammar it cannot be sampled at all.
    """
    for invented in ("Nvidia", "Intel", "512"):
        assert f'"{invented}"' not in built


def test_source_words_keep_the_case_they_were_written_in(built):
    """
    An aggregate spelling it `nipogi` would be a worse entry than the
    telegraphic ones it replaces.
    """
    assert '"NiPoGi"' in built
    assert '"AM06PRO"' in built


def test_a_common_word_can_start_the_sentence(built):
    assert '"Avec"' in built or '"Le"' in built


def test_the_sentence_cannot_be_two_words(built):
    """
    Three explicit repetitions, not {3,}: bounded repetition arrived
    in llama.cpp later than the rest of GBNF, and a grammar the server
    refuses is a 400 rather than a degraded call.
    """
    assert "{" not in built
    assert built.count("aggregate-sep aggregate-word") == 3


def test_the_lexicon_and_the_gate_agree(built):
    """
    One definition used twice. If the grammar could emit a word
    `invented` would then refuse, a llama.cpp run and an ollama run
    would disagree about what this tier is allowed to write.
    """
    allowed = aggregate.lexicon(SOURCES, COMMON, LIMIT)
    emittable = {
        literal.strip('"')
        for literal in built.split("aggregate-word ::= ")[1]
        .splitlines()[0]
        .split(" | ")
    }
    # Through the gate's own tokenizer, which is the whole relationship
    # between the two: the grammar emits `AM06PRO` whole, the gate sees
    # `am`, `06`, `pro`, and every piece is in the lexicon.
    pieces = {piece for literal in emittable for piece in aggregate.tokens(literal)}
    assert pieces <= allowed
