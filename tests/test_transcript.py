"""
Unit tests for forge.transcript.

The test that matters here is the round trip: `split(render(msgs))`
must equal `blocks(msgs)`. Compaction takes the first path (messages
-> units) and the migration in deploy/rag_resplit.py takes the second
(stored text -> units), and if the two ever disagree the store ends up
holding two differently-sliced halves -- which shows up as a retrieval
distance with no explanation, not as a failure.
"""

from forge import transcript


def _m(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def test_render_matches_the_stored_format():
    assert transcript.render([_m("user", "salut"), _m("assistant", "bonjour")]) == (
        "user: salut\nassistant: bonjour"
    )


def test_a_unit_is_a_user_turn_plus_what_answered_it():
    messages = [
        _m("user", "quel port ?"),
        _m("assistant", "8080"),
        _m("user", "et l'hôte ?"),
        _m("assistant", "localhost"),
    ]

    assert transcript.blocks(messages) == [
        "user: quel port ?\nassistant: 8080",
        "user: et l'hôte ?\nassistant: localhost",
    ]


def test_messages_before_the_first_user_turn_are_their_own_unit():
    messages = [
        _m("system", "[12 messages compactés]"),
        _m("assistant", "suite"),
        _m("user", "question"),
        _m("assistant", "réponse"),
    ]

    assert transcript.blocks(messages) == [
        "system: [12 messages compactés]\nassistant: suite",
        "user: question\nassistant: réponse",
    ]


def test_split_round_trips_through_render():
    messages = [
        _m("system", "ouverture"),
        _m("user", "quel port ?"),
        _m("assistant", "8080"),
        _m("user", "multi\nligne"),
        _m("assistant", "réponse\nsur deux lignes"),
    ]

    assert transcript.split(transcript.render(messages)) == transcript.blocks(messages)


def test_split_keeps_text_that_has_no_role_prefix():
    # A migration that silently returned [] here would delete the entry
    # it was asked to re-slice.
    assert transcript.split("juste du texte") == ["juste du texte"]


def test_split_keeps_text_appearing_before_the_first_prefix():
    units = transcript.split("préambule\nuser: question\nassistant: réponse")

    assert units == ["préambule", "user: question\nassistant: réponse"]


def test_split_of_empty_text_is_empty():
    assert transcript.split("   ") == []


def test_a_single_exchange_stays_one_unit():
    text = "user: une question\nassistant: une réponse"

    assert transcript.split(text) == [text]
