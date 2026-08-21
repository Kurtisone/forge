"""
Tests for what `review` says when it only read part of the file.

Real run, 2026-08-21: a review of notes.md opened with "defect number
one: the document stops abruptly mid-sentence, which suggests a badly
executed copy-paste". The document did not stop abruptly. Forge cut it
at 8 000 characters and told the model, in the passive voice and at the
very end, "... (truncated at 8000 chars)". The model reviewed our own
scissors and the user was never told the review covered a fifth of the
file.

Two different failures wearing one symptom:
  - the model treats the cut as content   (wording, fixed in the prompt)
  - the user cannot tell it happened      (structural, fixed in code)

Only the second is testable without a model, and it is the one that
matters more, so it is the one pinned hardest.
"""

import pytest

from forge.graphs import review
from forge.types import AgentState


def _state(path, question="Que peut-on améliorer ?"):
    return AgentState(
        user_input=str(path),
        context={"file_path": str(path), "question": question},
        max_steps=5,
    )


@pytest.fixture
def big_file(tmp_path):
    """A file comfortably past the cut, with a recognisable tail."""
    path = tmp_path / "notes.md"
    path.write_text("a" * (review._MAX_FILE_CHARS + 5_000) + "TAIL")
    return path


@pytest.fixture
def small_file(tmp_path):
    path = tmp_path / "small.md"
    path.write_text("just a few characters")
    return path


def test_a_short_file_is_not_marked_as_truncated(small_file):
    state = review._read_file_node(_state(small_file))

    assert "truncated_from" not in state.context
    assert state.context["file_content"] == "just a few characters"


def test_the_cut_marker_names_forge_and_both_sizes(big_file):
    """
    The old marker said "truncated at 8000 chars" with no subject. At
    the end of a file that stops mid-sentence, that reads as something
    the file says about itself.
    """
    state = review._read_file_node(_state(big_file))
    content = state.context["file_content"]

    assert "TAIL" not in content, "the tail should not have been sent"
    assert "Forge" in content, "the marker must name who did the cutting"
    assert str(review._MAX_FILE_CHARS) in content
    assert str(len(big_file.read_text())) in content, (
        "the marker must state the real size, or the model cannot tell "
        "whether it is missing a line or thirty pages"
    )


def test_the_prompt_tells_the_model_not_to_review_the_cut(big_file, monkeypatch):
    captured = {}

    def fake_call_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Une revue tout à fait ordinaire du fichier."

    monkeypatch.setattr(review, "call_llm", fake_call_llm)

    state = review._read_file_node(_state(big_file))
    review._llm_review_node(state)

    prompt = captured["prompt"]
    assert "NOTE:" in prompt
    assert "not the author's" in prompt
    assert str(len(big_file.read_text())) in prompt


def test_an_untruncated_review_carries_no_note(small_file, monkeypatch):
    """
    The note costs prompt tokens on every review, and most reviews are
    of files well under the cut. It must not be paid when it does not
    apply.
    """
    captured = {}

    def fake_call_llm(prompt, **kwargs):
        captured["prompt"] = prompt
        return "Une revue ordinaire."

    monkeypatch.setattr(review, "call_llm", fake_call_llm)

    state = review._read_file_node(_state(small_file))
    state = review._llm_review_node(state)

    assert "NOTE:" not in captured["prompt"]
    assert "Revue partielle" not in state.final_output


def test_the_user_is_told_the_review_is_partial(big_file, monkeypatch):
    """
    The deterministic half, and the point of the whole change: this
    holds whatever the model answers, including when it says nothing
    about coverage at all.
    """
    monkeypatch.setattr(
        review, "call_llm", lambda prompt, **kw: "Le fichier est bien structuré."
    )

    state = review._read_file_node(_state(big_file))
    state = review._llm_review_node(state)

    assert "Revue partielle" in state.final_output
    assert str(review._MAX_FILE_CHARS) in state.final_output
    assert state.final_output.startswith("Le fichier est bien structuré.")


def test_the_footer_states_a_coverage_percentage(big_file, monkeypatch):
    """
    "8000 of 13004 characters" is arithmetic the reader has to do while
    deciding whether to trust the review. 62 % is the decision.
    """
    monkeypatch.setattr(review, "call_llm", lambda prompt, **kw: "Rien à signaler.")

    state = review._read_file_node(_state(big_file))
    state = review._llm_review_node(state)

    expected = round(100 * review._MAX_FILE_CHARS / len(big_file.read_text()))
    assert f"{expected} %" in state.final_output


def test_no_partial_review_footer_on_an_error(big_file, monkeypatch):
    """
    Run #9942466c: the copy guard fired, and the truncation footer went
    on anyway -- "[error] the model copied the file" followed by
    "partial review: 40 % of the file", which reads as though a review
    happened and covered 40 % of it.
    """
    monkeypatch.setattr(
        review, "call_llm", lambda prompt, **kw: big_file.read_text()[:2000]
    )

    state = review._read_file_node(_state(big_file))
    state = review._llm_review_node(state)

    assert state.final_output.startswith("[error]")
    assert "Revue partielle" not in state.final_output
