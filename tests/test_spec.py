"""
Tests for forge.spec: the delegation spec.

The theme running through these is the split the module is built on --
the grammar decides the SHAPE, the code decides COMPLETENESS -- plus
the single-source-of-truth property that the router's prompt/grammar
pair had to learn the hard way.
"""

import pytest

from forge import gbnf, spec
from forge.errors import SpecParseError


def test_grammar_is_one_llama_cpp_would_accept():
    gbnf.validate(spec.build_spec_grammar())


def test_grammar_and_prompt_list_the_same_fields():
    """
    The router shipped a prompt and a grammar maintained separately,
    so a tool present in one and absent from the other produced a
    model naming something the grammar forbade. Both are generated
    from _FIELDS here; this is what notices if that ever stops being
    true.
    """
    grammar = spec.build_spec_grammar()
    prompt = spec.prompt_fields()
    for name in spec.FIELD_NAMES:
        assert name in grammar
        assert name in prompt


def test_an_underscored_field_name_is_fine_here(monkeypatch):
    """
    Pinning down a difference that is easy to get backwards. In
    router/grammar.py a tool name becomes a RULE name, where
    llama.cpp's lexer rejects underscores. Here a field name becomes a
    JSON key inside a string literal, where it is just text -- so this
    must NOT raise, and the guard below must not overreach into
    forbidding it.
    """
    ok = spec.Field(
        name="acceptance_criteria", kind="text", required=True, label="X", question="?"
    )
    monkeypatch.setattr(spec, "_FIELDS", (ok,))
    gbnf.validate(spec.build_spec_grammar())


def test_a_field_name_that_would_break_the_literal_fails_at_build(monkeypatch):
    """
    _FIELDS is a tuple a future field gets appended to. A quote in a
    name closes the GBNF string literal early, and llama-server then
    answers 400 to EVERY completion -- the symptom is a dead runtime,
    not a bad spec. It has to fail here, with the name in the message.
    """
    bad = spec.Field(
        name='objec"tive', kind="text", required=True, label="X", question="?"
    )
    monkeypatch.setattr(spec, "_FIELDS", (bad,))
    with pytest.raises(gbnf.GrammarError) as excinfo:
        spec.build_spec_grammar()
    assert "objec" in str(excinfo.value)


def test_parse_reads_a_plain_object():
    s = spec.parse(
        '{"objective": "a", "workspace": "b", "acceptance": ["c"], '
        '"constraints": [], "context": ""}'
    )
    assert s.objective == "a"
    assert s.acceptance == ["c"]


def test_parse_survives_a_fence_and_trailing_prose():
    """
    Under the grammar this would be a bare json.loads. The grammar
    only exists on llama.cpp -- ollama and OpenRouter run the same
    call unconstrained, and that is where the fences come from.
    """
    raw = '```json\n{"objective": "a", "acceptance": ["c"]}\n```\nVoilà.'
    assert spec.parse(raw).objective == "a"


def test_parse_drops_unknown_keys_rather_than_raising():
    s = spec.parse('{"objective": "a", "deadline": "demain"}')
    assert s.objective == "a"
    assert not hasattr(s, "deadline")


def test_parse_treats_null_as_absent():
    assert spec.parse('{"objective": "a", "context": null}').context == ""


def test_parse_splits_a_string_given_where_a_list_belongs():
    """
    An unconstrained model answers a list field with lines of text
    often enough to handle here rather than lose the whole spec over.
    """
    s = spec.parse('{"objective": "a", "acceptance": "tests verts\\nruff propre"}')
    assert s.acceptance == ["tests verts", "ruff propre"]


def test_parse_rejects_an_answer_with_no_object():
    """
    Typed, like every other runtime failure in Forge: the caller has
    to be able to tell "the model didn't produce a spec" from "the
    provider fell over" without parsing a message string.
    """
    with pytest.raises(SpecParseError):
        spec.parse("je ne sais pas quoi mettre")


def test_parse_rejects_malformed_json():
    with pytest.raises(SpecParseError):
        spec.parse('{"objective": "a", }{')


def test_missing_reports_only_required_holes():
    s = spec.Spec(objective="a", workspace="b", acceptance=["c"])
    assert spec.missing(s) == []
    assert not s.constraints and not s.context


def test_missing_ignores_blank_values():
    s = spec.Spec(objective="   ", workspace="b", acceptance=["c"])
    s.set("objective", "   ")
    assert "objective" in spec.missing(s)


def test_missing_follows_field_order():
    assert spec.missing(spec.Spec()) == ["objective", "workspace", "acceptance"]


def test_next_question_asks_one_at_a_time():
    """
    A 9B given three questions in one turn answers one and a half of
    them, and nothing deterministic can then say which field the
    answer belongs to -- the exact ambiguity awaiting_user exists to
    avoid.
    """
    name, question = spec.next_question(spec.Spec())
    assert name == "objective"
    assert question.strip()


def test_next_question_is_none_once_required_fields_are_filled():
    s = spec.Spec(objective="a", workspace="b", acceptance=["c"])
    assert spec.next_question(s) is None


def test_render_shows_empty_optional_fields_rather_than_hiding_them():
    """
    The user approves the rendered spec before anything is queued. A
    field that silently disappears cannot be noticed as wrong.
    """
    out = spec.render(spec.Spec(objective="a", workspace="b", acceptance=["c"]))
    assert "—" in out
    assert "Contraintes" in out


def test_set_normalises_a_list_field_from_free_text():
    s = spec.Spec()
    s.set("acceptance", "  un  \n\n deux \n")
    assert s.acceptance == ["un", "deux"]
