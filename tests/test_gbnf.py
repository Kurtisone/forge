"""
Tests for forge.gbnf: the checks every grammar Forge generates has to
pass before it is sent to llama-server.

These used to live in tests/test_grammar.py, where they only ever ran
against the router's grammar. v3.13 adds a second grammar (the
delegation spec), and a check that only covers one grammar is not a
check, it's a coincidence.
"""

import pytest

from forge import gbnf
from forge.router.grammar import build_router_grammar


def test_underscore_is_not_a_word_char_for_llama_cpp():
    """
    The single fact this module exists to pin down. llama.cpp's lexer
    accepts [a-zA-Z0-9-]; an underscore ends the name early, so
    `payload_call` reads as the rule `payload` followed by garbage.
    """
    assert gbnf.is_word_char("a")
    assert gbnf.is_word_char("7")
    assert gbnf.is_word_char("-")
    assert not gbnf.is_word_char("_")


def test_validate_accepts_the_real_router_grammar():
    """
    The grammar that has been in production since v3.10 must pass, or
    every call would silently run unconstrained -- a much quieter and
    much worse failure than the one this module prevents.
    """
    gbnf.validate(build_router_grammar(available_tools=["chat", "code", "files"]))


def test_validate_names_the_rule_llama_cpp_would_reject():
    """
    llama-server's own 400 body says only "failed to parse grammar" --
    the rule name appears nowhere in it. Recovering that name is the
    reason to validate on this side at all, so it has to be IN the
    message, not merely detected.
    """
    grammar = 'root ::= spec_call\nspec_call ::= "{" "}"\n'
    with pytest.raises(gbnf.GrammarError) as excinfo:
        gbnf.validate(grammar)
    assert "spec_call" in str(excinfo.value)


def test_validate_rejects_an_undefined_reference():
    grammar = 'root ::= "{" field "}"\n'
    with pytest.raises(gbnf.GrammarError) as excinfo:
        gbnf.validate(grammar)
    assert "field" in str(excinfo.value)


def test_validate_rejects_a_grammar_with_no_root():
    grammar = 'entry ::= "{" "}"\n'
    with pytest.raises(gbnf.GrammarError):
        gbnf.validate(grammar)


def test_validate_rejects_an_empty_grammar():
    with pytest.raises(gbnf.GrammarError):
        gbnf.validate("   \n")


def test_string_literals_are_not_read_as_rule_references():
    """
    A JSON key spelled "\\"tool\\"" inside a rule body is a literal,
    not a reference to a rule named `tool`. Getting this wrong is what
    made the old hand-maintained keyword list necessary.
    """
    grammar = 'root ::= "{" "\\"tool\\"" ":" value "}"\nvalue ::= "1"\n'
    assert gbnf.check_references(grammar) == []


def test_char_classes_are_not_read_as_rule_references():
    grammar = "root ::= [a-zA-Z]+ ws\nws ::= [ \\t\\n]*\n"
    assert gbnf.check_references(grammar) == []
