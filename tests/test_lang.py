"""
forge.lang: French, English, or an honest "don't know".

The third answer is what the tests are mostly about. Forcing an
answer into the wrong language is worse than the bug this module
exists for, because the model obeys -- so every thin or split case
below must come back UNKNOWN rather than guess.
"""

import pytest

from forge import lang


@pytest.mark.parametrize(
    "text",
    [
        "Tu peux me lister mon matériel ?",
        "quel port utilise le serveur ?",
        "Qu'est-ce que j'avais dit sur le NiPoGi ?",
        "rappelle-moi mes priorités",
        "Le serveur utilise le port 8080.",
    ],
)
def test_french_is_detected(text):
    assert lang.detect(text) == "fr"


@pytest.mark.parametrize(
    "text",
    [
        "what did I say about the NiPoGi?",
        "The server uses port 8080.",
        "You have a NiPoGi with a Ryzen 5500U.",
        "list my hardware",
        "as soon as we can do that",
    ],
)
def test_english_is_detected(text):
    assert lang.detect(text) == "en"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Forge",
        "8080",
        "ok",
        "podman compose build forge",
        "NiPoGi Ryzen 5500U",
    ],
)
def test_thin_evidence_returns_unknown(text):
    assert lang.detect(text) is None


def test_no_marker_belongs_to_both_languages():
    """
    A word in both tables adds noise to both counts and evidence to
    neither, which is the one way this table can quietly stop working.
    """
    assert not (lang._MARKERS["fr"] & lang._MARKERS["en"])


def test_accents_tip_a_tie_but_do_not_overrule_the_count():
    # One accented proper noun against nothing: French.
    assert lang.detect("déployé") == "fr"
    # The same accent against a sentence's worth of English function
    # words: still English.
    assert lang.detect("I have already deployed that to the café server") == "en"


def test_mismatch_names_the_language_the_answer_should_have_been_in():
    assert (
        lang.mismatch("quel port utilise le serveur ?", "The server uses port 8080.")
        == "French"
    )


def test_mismatch_is_silent_when_the_languages_agree():
    assert (
        lang.mismatch("quel port utilise le serveur ?", "Le serveur utilise le port.")
        is None
    )


@pytest.mark.parametrize(
    "question,answer",
    [
        ("8080", "The server uses port 8080."),  # question undecided
        ("quel port utilise le serveur ?", "8080"),  # answer undecided
        ("ok", "ok"),  # both undecided
    ],
)
def test_mismatch_needs_both_sides_to_be_certain(question, answer):
    assert lang.mismatch(question, answer) is None
