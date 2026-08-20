"""
The router grammar applied to every LLM call in Forge, including the
graph syntheses that spend four paragraphs asking for plain text.
These pin the fix: what the prose grammars admit, what they make
unreachable, and the fact that recall passes one.

The character classes are checked by translating them to Python
regexes, which happens to be exact -- GBNF's negated class syntax and
Python's are the same here, hex escapes included. Same move as
tests/test_grammar.py reimplementing llama.cpp's is_word_char(): there
is no GBNF engine in this environment, so the alternative is asserting
on the grammar TEXT, which passes whatever the grammar means.
"""

import re

import pytest

from forge import gbnf, prose_grammar
from forge.graphs import recall as recall_mod
from forge.types import AgentState


def _as_regex(grammar: str) -> re.Pattern:
    head = re.search(r"head ::= (\[.*\])", grammar).group(1)
    tail = re.search(r"tail ::= (\[.*\])", grammar).group(1)
    return re.compile(f"{head}{tail}*", re.DOTALL)


SENTENCE_RE = _as_regex(prose_grammar.SENTENCE)
PROSE_RE = _as_regex(prose_grammar.PROSE)


@pytest.mark.parametrize("grammar", [prose_grammar.SENTENCE, prose_grammar.PROSE])
def test_the_grammars_are_ones_llama_cpp_can_parse(grammar):
    """An unparseable grammar is a 400 on every completion, and
    _grammar_for() drops it and runs UNCONSTRAINED -- which is the
    state this module exists to leave. `payload_call` already cost a
    total router outage this way."""
    gbnf.validate(grammar)
    for name in gbnf.rule_names(grammar):
        assert all(gbnf.is_word_char(c) for c in name), name


@pytest.mark.parametrize("pattern", [SENTENCE_RE, PROSE_RE])
def test_a_routing_decision_is_unreachable(pattern):
    assert not pattern.fullmatch('{"tool":"chat","content":"Tu as un Steam Deck."}')


@pytest.mark.parametrize("pattern", [SENTENCE_RE, PROSE_RE])
def test_padding_the_object_with_whitespace_does_not_get_round_it(pattern):
    """head excludes whitespace as well as braces, so the cheapest
    way out -- emit a space, then the object -- is closed too."""
    for prefix in (" ", "\n", "\t", "  "):
        assert not pattern.fullmatch(f'{prefix}{{"tool":"chat","content":"x"}}')


@pytest.mark.parametrize("pattern", [SENTENCE_RE, PROSE_RE])
def test_a_real_answer_is_admitted(pattern):
    assert pattern.fullmatch("Tu as un Steam Deck et un NiPoGi.")


def test_a_sentence_is_one_line():
    assert not SENTENCE_RE.fullmatch("Première ligne.\nSeconde ligne.")


def test_prose_may_span_lines():
    assert PROSE_RE.fullmatch("Premier paragraphe.\n\nSecond paragraphe.")


def test_prose_keeps_braces_legal_inside_the_text():
    """Deliberate asymmetry with SENTENCE. research quotes web pages
    and sysadmin quotes logs; forbidding braces there would trade the
    wrong output for a different wrong output. Constraining the first
    character is enough to make a top-level object unreachable."""
    assert PROSE_RE.fullmatch('Le service a émis {"code": 500} avant de tomber.')


def test_recall_asks_for_a_sentence_grammar(monkeypatch):
    """The regression this locks down: recall calling call_llm(prompt)
    with no grammar, which providers/llama_cpp._grammar_for() reads as
    'use the router's'. Nothing about that is visible at the call
    site, which is how it survived four rounds of prompt work.

    The synthesize node is exercised directly rather than through a
    graph run: the recall node ahead of it reaches the embedding
    server, which is not what is under test here (same reasoning as
    calling review's _run_tests_node directly, where a full run dies
    on WORKSPACE_DIR confinement long before the interesting step).
    """
    seen = []

    def fake_llm(prompt, grammar=None):
        seen.append(grammar)
        return "Tu as un Steam Deck."

    monkeypatch.setattr(recall_mod, "call_llm", fake_llm)

    state = AgentState(user_input="Quel matériel ?", max_steps=4)
    state.context = {
        "query": "Quel matériel ?",
        "results": [
            {"kind": "fact", "content": "Possède un Steam Deck", "project": None}
        ],
    }

    recall_mod._synthesize_node(state)

    assert state.final_output == "Tu as un Steam Deck."
    assert seen == [prose_grammar.SENTENCE]


# The two places a routing decision is genuinely what is wanted, so
# inheriting the router grammar is correct. Everything else that calls
# the model wants prose and must say so.
_ROUTER_CALLERS = {"forge/orchestrator.py", "forge/graphs/default.py"}


def _call_llm_sites():
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src"
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == "call_llm":
                yield str(path.relative_to(root)), node


def test_every_model_call_says_which_shape_it_wants():
    """
    The invariant, rather than the four fixes.

    A new graph that calls call_llm(prompt) inherits the router's
    grammar silently -- nothing at the call site says so, which is how
    recall, review, research and sysadmin all ended up sampling a
    routing decision while their prompts begged for plain text. This
    fails on the next one instead of shipping the fifth instance.
    """
    unconstrained = [
        module
        for module, node in _call_llm_sites()
        if module not in _ROUTER_CALLERS
        and not any(kw.arg == "grammar" for kw in node.keywords)
        and len(node.args) < 2
    ]
    assert not unconstrained, (
        "these call the model without saying what shape they expect, so "
        f"they get the router's grammar: {unconstrained}"
    )
