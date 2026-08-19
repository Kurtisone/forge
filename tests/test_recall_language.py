"""
recall must answer in the language it was asked in.

The prompt has carried "Write in the same language as the question"
since the graph was written, and recall still answered French
questions in English -- the prompt body is English, the entries are
French, and the model followed the prompt. That sentence asked it to
infer the target language for itself.

Two halves, tested separately:
  1. the detected language is NAMED in the prompt, in last position;
  2. the answer is CHECKED afterwards, because half of this is still
     a wording fix, and wording fixes have lost six times here.

The retry is priced: it fires only on a run that was already wrong,
never when either language is uncertain, and never twice.
"""

import forge.graphs.recall as recall_mod
from forge.graphs.recall import build as build_recall

FR_QUESTION = "Tu peux me lister mon matériel ?"
FR_ANSWER = "Tu as un Steam Deck et un NiPoGi."
EN_ANSWER = "You have a Steam Deck and a NiPoGi."

_ENTRIES = [
    {"kind": "fact", "content": "Possède un Steam Deck", "project": None},
    {"kind": "fact", "content": "Possède un NiPoGi", "project": None},
]


def _run(monkeypatch, replies, question=FR_QUESTION):
    """Run the graph over a scripted list of model replies."""
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(recall_mod.memory_tool, "search", lambda q: _ENTRIES)
    monkeypatch.setattr(recall_mod, "call_llm", fake_llm)
    state = build_recall().run(question, initial_context={"query": question})
    return state, calls


def test_the_detected_language_is_named_last_in_the_prompt(monkeypatch):
    _, calls = _run(monkeypatch, [FR_ANSWER])

    assert "Write your answer in French." in calls[0]
    # Last position is the point: everything above it is an English
    # prompt body pulling the answer the other way.
    assert calls[0].rstrip().endswith("Write your answer in French. Every word of it.")


def test_an_undetectable_question_says_nothing_about_language(monkeypatch):
    """
    Guessing here would be worse than the bug: the model obeys.
    """
    _, calls = _run(monkeypatch, ["8080"], question="8080")

    assert "Write your answer in" not in calls[0]


def test_an_answer_in_the_wrong_language_is_asked_again(monkeypatch):
    state, calls = _run(monkeypatch, [EN_ANSWER, FR_ANSWER])

    assert len(calls) == 2
    assert "previous answer was in the wrong language" in calls[1]
    assert state.final_output == FR_ANSWER


def test_a_matching_answer_costs_exactly_one_call(monkeypatch):
    state, calls = _run(monkeypatch, [FR_ANSWER])

    assert len(calls) == 1
    assert state.final_output == FR_ANSWER


def test_a_retry_that_fails_the_same_way_keeps_the_first_answer(monkeypatch):
    """
    Twice wrong is the model's limit, not a reason to hand back the
    worse of two answers -- and an answer in the wrong language still
    has the right content, which is more than an error message has.
    """
    state, calls = _run(monkeypatch, [EN_ANSWER, "You still have a Steam Deck."])

    assert len(calls) == 2  # never a third
    assert state.final_output == EN_ANSWER


def test_the_retry_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(recall_mod, "RECALL_ENFORCE_LANGUAGE", False)
    state, calls = _run(monkeypatch, [EN_ANSWER, FR_ANSWER])

    assert len(calls) == 1
    assert state.final_output == EN_ANSWER


def test_an_unusable_first_answer_is_not_retried_for_language(monkeypatch):
    """
    The example-leak and prompt-leak refusals are already final. Their
    error strings are French, so a language check would fire on them
    and spend a call re-asking a question that was never about
    language.
    """
    leaked = "Tu as un serveur exemple-hôte et un onduleur modèle-fictif."
    state, calls = _run(monkeypatch, [leaked, FR_ANSWER], question="list my hardware")

    assert len(calls) == 1
    assert state.final_output.startswith("[error]")


def test_a_provider_failure_during_the_retry_is_still_a_provider_failure(monkeypatch):
    from forge.errors import ProviderError

    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return EN_ANSWER
        raise ProviderError("llama-server is down")

    monkeypatch.setattr(recall_mod.memory_tool, "search", lambda q: _ENTRIES)
    monkeypatch.setattr(recall_mod, "call_llm", fake_llm)

    state = build_recall().run(FR_QUESTION, initial_context={"query": FR_QUESTION})

    assert not state.ok
    assert "LLM unavailable" in state.final_output
