"""
Unit tests for forge.transcript.

The test that matters here is that the two entry points are the same
pipeline: `split(render(msgs))` must equal `units(msgs)`. Compaction
takes the first path (messages -> units) and the migration in
deploy/rag_resplit.py takes the second (stored text -> units).

That equality is not hypothetical maintenance. The first version of
this module shared only the CUTTING, and the migration of 2026-08-22
wrote a dozen entries whose entire content is a pointer to another
entry, plus some raw router JSON -- because compaction filtered those
out before cutting and the migration did not.
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


def test_the_two_entry_points_are_the_same_pipeline():
    messages = [
        _m("system", "ouverture"),
        _m("user", "quel port ?"),
        _m("assistant", "8080"),
        _m("system", transcript.pointer(59, [12])),
        _m("user", "multi\nligne"),
        _m(
            "assistant",
            '{"tool": "chat", "content": "une réponse assez longue '
            'pour être considérée comme substantielle par le désenveloppage"}',
        ),
        _m("assistant", "réponse\nsur deux lignes"),
    ]

    assert transcript.split(transcript.render(messages)) == transcript.units(messages)


def test_parse_is_the_inverse_of_render():
    messages = [
        _m("user", "quel port ?"),
        _m("assistant", "8080\net rien d'autre"),
    ]

    assert transcript.parse(transcript.render(messages)) == messages


def test_an_earlier_pointer_never_reaches_the_store():
    messages = [
        _m("system", transcript.pointer(59, [12])),
        _m("user", "une question"),
        _m("assistant", "une réponse"),
    ]

    assert transcript.units(messages) == ["user: une question\nassistant: une réponse"]


def test_a_pointer_is_recognised_whatever_role_carries_it():
    """
    The check looks at the shape, not the speaker. A block re-parsed by
    the migration can hand a pointer back under whatever role happened
    to precede it in the stored text.
    """
    assert transcript.units([_m("assistant", transcript.pointer(9, [3]))]) == []
    assert (
        transcript.units([{"role": None, "content": transcript.pointer(9, [3])}]) == []
    )


def test_the_pointer_builder_matches_its_own_detector():
    for ids in ([], [7], [7, 8, 9]):
        assert transcript.POINTER_RE.match(transcript.pointer(12, ids))


def test_an_entry_that_holds_only_a_pointer_yields_nothing():
    assert transcript.split(f"system: {transcript.pointer(59, [12])}") == []


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
