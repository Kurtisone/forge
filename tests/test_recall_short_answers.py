"""
recall's prompt asks for ONE SHORT SENTENCE. The shared unwrap
minimums say a short answer is degenerate. Those two cannot both be
right, and until 2026-08-16 the second one won.

Observed on the Deck: asked "quel port utilise le serveur ?" over
entries naming a port, recall produced the correct answer -- "Le
serveur utilise le port 8080.", 6 words, 32 chars -- and the user was
shown the raw {"tool":...,"content":...} envelope instead, because 6 <
8 words and 32 < 40 chars. The defaults were calibrated on review,
whose prompt asks for a multi-sentence synthesis; recall inherited them
and asks for the opposite.

The floor still has to stop something. The degenerate case actually
seen here is the model emitting the NEVER DO THIS example's own
content: the three characters "...".
"""

from forge.graphs import recall
from forge.text_cleaning import MIN_UNWRAP_CHARS, MIN_UNWRAP_WORDS

WRAPPED = '{{"tool":"chat","content":"{}"}}'


def test_the_case_that_was_failing():
    """Below both shared minimums, and correct."""
    answer = "Le serveur utilise le port 8080."
    assert len(answer.split()) < MIN_UNWRAP_WORDS
    assert len(answer) < MIN_UNWRAP_CHARS
    assert recall._clean_synthesis_response(WRAPPED.format(answer)) == answer


def test_a_very_short_answer_still_unwraps():
    answer = "Le port est 8080."
    assert recall._clean_synthesis_response(WRAPPED.format(answer)) == answer


def test_the_degenerate_echo_is_still_refused():
    """The NEVER DO THIS example's own content, seen live."""
    out = recall._clean_synthesis_response(WRAPPED.format("..."))
    assert out == WRAPPED.format("...")


def test_the_other_graphs_keep_the_shared_minimums():
    """recall lowering its own floor must not lower anyone else's --
    review is where the defaults were measured."""
    from forge.graphs import review

    out = review._clean_review_response(WRAPPED.format("..."))
    assert "..." in out
    assert not out.startswith("...")
