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


# --- units that answered nothing -------------------------------------
#
# See forge/non_answer.py for the measurement these exist for: an
# exchange whose reply says nothing is a near-copy of its own question,
# which makes it the closest possible neighbour of anyone asking it
# again.


def test_a_refusal_and_its_question_go_together():
    from forge import non_answer

    messages = [
        _m("user", "Tu peux me lister mon matériel ?"),
        _m("assistant", non_answer.NOTHING_CLOSE_ENOUGH),
    ]

    assert transcript.units(messages) == []


def test_the_question_is_not_kept_on_its_own():
    # The failure this guards against is a per-message filter: drop the
    # reply, keep the question, and the entry left behind is worse than
    # the one removed.
    from forge import non_answer

    messages = [
        _m("user", "Quel est le modèle de ma voiture ?"),
        _m("assistant", non_answer.NOTHING_CLOSE_ENOUGH),
        _m("user", "Et mon matériel ?"),
        _m("assistant", "Un Steam Deck et un NiPoGi AM06PRO."),
    ]

    units = transcript.units(messages)

    assert units == [
        "user: Et mon matériel ?\nassistant: Un Steam Deck et un NiPoGi AM06PRO."
    ]


def test_a_marked_exchange_is_dropped_whatever_it_says():
    # The structural half: the reply reads like an answer, the run said
    # otherwise when it was persisted.
    messages = [
        {"role": "user", "content": "et ma voiture ?", "index": False},
        {"role": "assistant", "content": "Je vais regarder ça.", "index": False},
    ]

    assert transcript.units(messages) == []


def test_a_mark_on_either_message_is_enough():
    messages = [
        _m("user", "et ma voiture ?"),
        {"role": "assistant", "content": "Je vais regarder ça.", "index": False},
    ]

    assert transcript.units(messages) == []


def test_a_unit_with_no_reply_is_left_alone():
    # Same shape as the problem, different cause: the eviction window
    # ended on a user turn. Nothing failed.
    assert transcript.units([_m("user", "et mon matériel ?")]) == [
        "user: et mon matériel ?"
    ]


def test_one_real_reply_is_enough_to_keep_the_unit():
    from forge import non_answer

    messages = [
        _m("user", "Tu peux me lister mon matériel ?"),
        _m("assistant", f"{non_answer.ERROR_PREFIX}first try failed"),
        _m("assistant", "Un Steam Deck et un NiPoGi AM06PRO."),
    ]

    assert len(transcript.units(messages)) == 1


def test_the_mark_is_the_one_thing_the_two_paths_cannot_share():
    # Deliberate, and pinned so it stays a decision. render() does not
    # write the mark down and parse() cannot recover it, so a block
    # already in the store can only ever be judged on its text.
    messages = [
        {"role": "user", "content": "et ma voiture ?", "index": False},
        {"role": "assistant", "content": "Je vais regarder ça.", "index": False},
    ]

    assert transcript.units(messages) == []
    assert transcript.split(transcript.render(messages)) == [
        "user: et ma voiture ?\nassistant: Je vais regarder ça."
    ]


def test_dropped_reports_exactly_what_units_left_behind():
    from forge import non_answer

    messages = [
        _m("user", "Tu peux me lister mon matériel ?"),
        _m("assistant", non_answer.NOTHING_CLOSE_ENOUGH),
        _m("user", "et le port ?"),
        _m("assistant", "8080."),
    ]

    kept = transcript.units(messages)
    gone = transcript.dropped(messages)

    assert kept == ["user: et le port ?\nassistant: 8080."]
    assert gone == [
        (
            "user: Tu peux me lister mon matériel ?\nassistant: "
            f"{non_answer.NOTHING_CLOSE_ENOUGH}"
        )
    ]


def test_nothing_is_both_kept_and_dropped():
    # The migration prints one list and writes the other. A unit
    # appearing in both, or in neither, is a report that lies about
    # what was written.
    from forge import non_answer

    messages = [
        _m("user", "une question"),
        _m("assistant", "une réponse"),
        _m("user", "une autre"),
        _m("assistant", f"{non_answer.ERROR_PREFIX}boom"),
        _m("user", "une troisième"),
    ]

    kept = transcript.units(messages)
    gone = transcript.dropped(messages)

    assert set(kept) & set(gone) == set()
    assert len(kept) + len(gone) == 3


def test_split_dropped_reads_stored_text():
    from forge import non_answer

    text = transcript.render(
        [
            _m("user", "Quel est le modèle de ma voiture ?"),
            _m("assistant", non_answer.NOTHING_CLOSE_ENOUGH),
        ]
    )

    kept, gone = transcript.split_partition(text)

    assert kept == []
    assert len(gone) == 1
