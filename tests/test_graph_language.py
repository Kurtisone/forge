"""
recall answered French questions in English until v3.12's dettes batch
named the language in last position. review, research and sysadmin
have the identical problem for the identical reason -- an English
prompt body pulling the answer toward English -- and were left out at
the time with a note that forge/lang.py was reusable as-is.

These pin that they now use it, that the instruction really is last in
what gets sent, and that an unrecognised question adds nothing.
"""

import pytest

from forge import lang, prose_grammar
from forge.graphs import research as research_mod
from forge.graphs import review as review_mod
from forge.graphs import sysadmin as sysadmin_mod
from forge.types import AgentState


def _state(**context):
    s = AgentState(user_input=context.get("question", ""), max_steps=6)
    s.context = dict(context)
    return s


def _capture(monkeypatch, module, answer="Réponse."):
    sent = []

    def fake_llm(prompt, grammar=None):
        sent.append((prompt, grammar))
        return answer

    monkeypatch.setattr(module, "call_llm", fake_llm)
    return sent


def _run_research(monkeypatch, question, answer="Réponse."):
    sent = _capture(monkeypatch, research_mod, answer)
    research_mod._synthesize_node(
        _state(query=question, results=[], fetched=[], question=question)
    )
    return sent


def _run_review(monkeypatch, question, answer="Réponse."):
    sent = _capture(monkeypatch, review_mod, answer)
    review_mod._llm_review_node(
        _state(
            file_name="a.py",
            file_content="x = 1",
            question=question,
            test_path=None,
            test_output=None,
        )
    )
    return sent


def _run_sysadmin(monkeypatch, question, answer="Réponse."):
    sent = _capture(monkeypatch, sysadmin_mod, answer)
    sysadmin_mod._synthesize_node(
        _state(question=question, log_source="journalctl -k", collected_logs="rien")
    )
    return sent


RUNNERS = [_run_research, _run_review, _run_sysadmin]


@pytest.mark.parametrize("run", RUNNERS)
def test_the_detected_language_is_named_last(monkeypatch, run):
    sent = run(monkeypatch, "Pourquoi le service a-t-il redémarré ce matin ?")

    prompt = sent[0][0]
    assert "French" in prompt
    # Last position is the point. Everything above competes with an
    # English prompt body.
    assert prompt.rstrip().endswith("Every word of it.")


@pytest.mark.parametrize("run", RUNNERS)
def test_an_undetectable_question_says_nothing_about_language(monkeypatch, run):
    """Empty rather than a default. Forcing an answer into the wrong
    language is worse than the bug, because the model complies."""
    sent = run(monkeypatch, "nginx 8080")

    prompt = sent[0][0]
    assert "Every word of it." not in prompt


@pytest.mark.parametrize("run", RUNNERS)
def test_an_answer_in_the_wrong_language_is_asked_again(monkeypatch, run):
    sent = run(
        monkeypatch,
        "Pourquoi le service a-t-il redémarré ce matin ?",
        answer="The service restarted because the port was already in use.",
    )

    assert len(sent) == 2
    assert "wrong language" in sent[1][0]
    # The retry keeps the prose grammar: a second call under the router
    # grammar would come back as JSON and be "fixed" into an error.
    assert sent[1][1] == prose_grammar.PROSE


@pytest.mark.parametrize("run", RUNNERS)
def test_a_matching_answer_costs_exactly_one_call(monkeypatch, run):
    sent = run(monkeypatch, "Pourquoi le service a-t-il redémarré ce matin ?")
    assert len(sent) == 1


@pytest.mark.parametrize("run", RUNNERS)
def test_the_retry_can_be_turned_off(monkeypatch, run):
    for module in (research_mod, review_mod, sysadmin_mod):
        monkeypatch.setattr(module, "ENFORCE_ANSWER_LANGUAGE", False)

    sent = run(
        monkeypatch,
        "Pourquoi le service a-t-il redémarré ce matin ?",
        answer="The service restarted because the port was already in use.",
    )

    assert len(sent) == 1


def test_the_wording_has_exactly_one_definition():
    """Four graphs need the identical instruction. Four copies of a
    string that must stay identical is how a fix drifts -- the reason
    this moved out of recall in the first place."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "src" / "forge"
    holders = [
        p.relative_to(root).as_posix()
        for p in root.rglob("*.py")
        if "Every word of it." in p.read_text()
    ]
    assert holders == ["lang.py"], holders


def test_an_error_answer_is_never_retried_for_language():
    """An "[error] ..." string is Forge's own message, not the model's
    answer. Detecting its language and asking again would spend a call
    to translate an error."""
    calls = []
    out = lang.enforce(
        "Pourquoi le service a-t-il redémarré ?",
        "[error] LLM unavailable: down",
        retry=lambda line: calls.append(line) or "ignored",
    )
    assert out.startswith("[error]")
    assert calls == []
