"""
A recall answer must never be the prompt's own GOOD ANSWER example.

graphs/sysadmin.py hit this in production on 2026-08-11 and grew
_EXAMPLE_LEAK_FRAGMENTS for it. recall hit the same thing on
2026-08-16, surfaced by the /no_think experiment: asked "quel port
utilise le serveur ?" over entries about a port, it returned the
example sentence about this box's hardware instead -- with no leak
marker in it, so nothing downstream could tell it from a real answer.
The prefix was kept in the end, but the hazard it was masking is real:
graphs/sysadmin.py met this failure in production WITH the prefix in
place, so masking is not protection.

The example was rewritten to fictional placeholders precisely so a copy
IS detectable. The old one named real hardware, so a legitimate recall
over entries about that hardware would have reproduced it word for
word; that is why it is not kept as a permanent net the way sysadmin
keeps its previous example.
"""

from forge.graphs import recall


def test_the_example_is_detectable_because_it_is_fictional():
    for fragment in recall._EXAMPLE_LEAK_FRAGMENTS:
        assert fragment in recall._SYNTHESIS_PROMPT


def test_a_copied_example_is_refused():
    out = recall._clean_synthesis_response(
        "Tu as un serveur exemple-hôte et un onduleur modèle-fictif."
    )
    assert out.startswith("[error]")
    assert "recopié" in out


def test_a_copied_example_wrapped_in_router_json_is_refused():
    """The grammar wraps every synthesis in the router's JSON shape, so
    the check has to survive unwrapping, not run before it."""
    out = recall._clean_synthesis_response(
        '{"tool":"chat","content":"Tu as un serveur exemple-hôte '
        'et un onduleur modèle-fictif."}'
    )
    assert out.startswith("[error]")


def test_a_real_answer_about_real_hardware_passes():
    """The old example named this box's hardware. Keeping it as a
    detection fragment would have failed exactly this answer."""
    answer = "Tu as un Steam Deck et un Dell R710."
    assert recall._clean_synthesis_response(answer) == answer


def test_an_ordinary_answer_passes():
    answer = "Le serveur utilise le port 8080."
    assert recall._clean_synthesis_response(answer) == answer
