"""
Tests for the half of this tier that used to be a model.

The branch shipped with the aggregate SENTENCE written by a 9B under a
GBNF grammar, and the real store measured what that costs: seven calls
across two runs on 2026-09-11, zero aggregates written. Both subjects
failed the same way -- the answer ran to the grammar's maximum item
count every single time, because nothing in a lexicon of WORDS makes
reusing a word expensive and "do not leave anything out" concatenates.

What replaced it is arithmetic on DETAILS, and these are its rules.
The one that matters most is in `distinct`: a detail is dropped only
in favour of a detail that contains every one of its words. That is
what keeps a negation from being folded into its own opposite, and it
is the reason this module can write into the store without a model and
without a truth check.
"""

from forge import aggregate
from forge.aggregate import Detail


def entry(entry_id: int, content: str) -> dict:
    return {"id": entry_id, "kind": "fact", "content": content, "project": None}


NIPOGI = (
    entry(17, "Le NiPoGi a 32 Go de RAM"),
    entry(
        307,
        "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go",
    ),
    entry(
        315,
        "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible, services Podman",
    ),
)


# --- What a note is made of ------------------------------------------------


def test_a_label_and_its_details():
    label, details = aggregate.labelled(
        "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM"
    )
    assert label == "Matériel"
    assert details == ["NiPoGi AM06PRO", "processeur Ryzen 5500U", "32 Go de RAM"]


def test_a_note_without_a_separator_is_one_detail():
    """
    Guessing where the commas would have gone is how an aggregate
    starts asserting things nobody wrote.
    """
    assert aggregate.labelled("Le proxy podman écoute sur un socket unix") == (
        None,
        ["Le proxy podman écoute sur un socket unix"],
    )


def test_a_colon_in_the_middle_of_a_sentence_is_not_a_label():
    label, details = aggregate.labelled(
        "Le proxy podman écoute sur un socket unix : il est en lecture seule"
    )
    assert label is None
    assert len(details) == 1


def test_a_semicolon_separates_details_too():
    _, details = aggregate.labelled("Le chat s'appelle Mimi; il a 3 ans")
    assert details == ["Le chat s'appelle Mimi", "il a 3 ans"]


# --- Deduplication, and the invariant under it -----------------------------


def test_a_detail_is_dropped_by_one_that_has_all_of_its_words():
    kept = aggregate.distinct(
        [
            Detail("processeur Ryzen 5500U", 307),
            Detail("5500U", 315),
        ]
    )
    assert [d.text for d in kept] == ["processeur Ryzen 5500U"]


def test_the_spelled_out_surface_wins_a_tie():
    """
    `SSD 256 Go` and `SSD 256Go` tokenize identically, so containment
    cannot order them. The telegram loses: docs/memory.md spent six
    campaigns establishing that an instruction-tuned embedding model
    retrieves one worst.
    """
    kept = aggregate.distinct([Detail("SSD 256Go", 315), Detail("SSD 256 Go", 307)])
    assert [d.text for d in kept] == ["SSD 256 Go"]


def test_a_negation_is_never_folded_into_its_own_opposite():
    """
    THE reason containment is over every word and not over the
    informative ones. `pas` and `ne` are connectives by frequency --
    exactly the words an informative-word rule ignores -- so under that
    rule the store would quietly answer the opposite of what it holds.
    """
    kept = aggregate.distinct(
        [
            Detail("Le NiPoGi a 32 Go de RAM", 17),
            Detail("Le NiPoGi n'a pas 32 Go de RAM", 400),
        ]
    )
    assert "Le NiPoGi n'a pas 32 Go de RAM" in [d.text for d in kept]


def test_two_quantities_sharing_a_unit_both_survive():
    kept = aggregate.distinct(
        [Detail("32 Go de RAM", 307), Detail("256 Go de SSD", 307)]
    )
    assert len(kept) == 2


def test_the_order_is_the_order_the_notes_were_written_in():
    kept = aggregate.distinct(
        [
            Detail("NiPoGi AM06PRO", 307),
            Detail("Arch", 315),
            Detail("Ansible", 315),
        ]
    )
    assert [d.text for d in kept] == ["NiPoGi AM06PRO", "Arch", "Ansible"]


def test_a_tie_is_settled_without_moving_the_line():
    """
    The two surfaces of one word set sit at different places in the
    notes. Which one wins is a question about spelling; WHERE the line
    ends up must not depend on the answer, or two stores that hold the
    same facts in a different order read differently.
    """
    kept = aggregate.distinct(
        [
            Detail("SSD 256Go", 315),
            Detail("Arch", 315),
            Detail("SSD 256 Go", 307),
        ]
    )
    assert [d.text for d in kept] == ["SSD 256 Go", "Arch"]


# --- The line a subject folds into -----------------------------------------


def test_the_head_is_the_first_label_the_sources_carry():
    merged = aggregate.merge(NIPOGI, "nipogi")
    assert merged.head == "Matériel"


def test_a_subject_with_no_label_is_named_as_somebody_wrote_it():
    """
    `NiPoGi`, not `nipogi` and not `Nipogi`. An aggregate spelling it
    wrong is a worse entry than the telegrams it replaces.
    """
    merged = aggregate.merge(
        (entry(1, "Le NiPoGi a 32 Go de RAM"), entry(2, "NiPoGi AM06PRO, Arch")),
        "nipogi",
    )
    assert merged.head == "NiPoGi"


def test_a_second_label_becomes_a_detail_instead_of_disappearing():
    """
    It is a word of a source, so the coverage gate is entitled to look
    for it. Dropping it silently is what that gate exists to refuse.
    """
    merged = aggregate.merge(
        (
            entry(1, "Serveur : Dell R710, 64 Go de RAM"),
            entry(2, "Machines : le Dell R710 est au garage"),
        ),
        "r710",
    )
    assert merged.head == "Serveur"
    assert "Machines" in [d.text for d in merged.details]


def test_the_label_and_its_own_list_are_not_split():
    """
    MEASURED on the block the real store produces, three passes with
    the arms rotated: the line that opened `Matériel : Le NiPoGi a 32
    Go de RAM, NiPoGi AM06PRO, ...` lost its own first item in every
    answer, and the same details with the label's own list restored
    behind it kept them 3/3. A label and the list under it were typed
    in one line by one person; another entry's sentence between them
    reads as part of the label.
    """
    merged = aggregate.merge(NIPOGI, "nipogi")

    first = merged.details[0]
    assert first.source == 307
    assert first.text == "NiPoGi AM06PRO"


def test_the_real_store_group_folds_into_one_line():
    merged = aggregate.merge(NIPOGI, "nipogi")
    assert merged.text == (
        "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, SSD 256 Go, "
        "Le NiPoGi a 32 Go de RAM, Arch, Ansible, services Podman."
    )


def test_every_word_of_the_result_was_typed_by_somebody():
    """
    Closure, no longer as a gate on an answer but as a property of the
    writer. The grammar used to constrain which WORDS could be
    emitted; this constrains which sentences can.
    """
    merged = aggregate.merge(NIPOGI, "nipogi")
    sources = [e["content"] for e in NIPOGI]
    freq = aggregate.frequencies([{"content": c} for c in sources])
    allowed = aggregate.lexicon(sources, freq, limit=1)
    assert aggregate.invented(merged.text, allowed) == []


def test_it_answers_the_same_way_twice():
    assert aggregate.merge(NIPOGI, "nipogi").text == (
        aggregate.merge(NIPOGI, "nipogi").text
    )


def test_nothing_a_source_says_is_lost():
    merged = aggregate.merge(NIPOGI, "nipogi")
    freq = aggregate.frequencies(list(NIPOGI) + [{"content": "de le a et un"}] * 40)
    for source in NIPOGI:
        assert aggregate.uncovered(merged.text, source["content"], freq, 10) == []


def test_the_result_is_not_a_concatenation_of_the_notes():
    """
    What the model did on every one of the seven calls measured on
    2026-09-11. The length is the tell: the merge is shorter than the
    notes it stands for, and that is what the budget gate is about to
    check.
    """
    merged = aggregate.merge(NIPOGI, "nipogi")
    assert len(merged.text) < sum(len(e["content"]) for e in NIPOGI)


def test_a_repeated_pair_is_not_produced_by_the_merge():
    merged = aggregate.merge(NIPOGI, "nipogi")
    freq = aggregate.frequencies(list(NIPOGI) + [{"content": "de le a et un"}] * 40)
    assert aggregate.repeated(merged.text, freq, 10) == []


def test_an_aggregate_can_be_read_back_as_a_note_and_merged_again():
    """
    An aggregate is an entry like any other, so the next pass will
    read it as a source. Its shape has to survive that round trip or
    the second fold starts from a line this module cannot parse.
    """
    first = aggregate.merge(NIPOGI, "nipogi")
    label, details = aggregate.labelled(first.text)
    assert label == "Matériel"
    assert details[0] == "NiPoGi AM06PRO"


# --- When there is nothing to write ----------------------------------------


def test_a_note_contained_in_another_names_the_one_that_speaks_for_it():
    """
    `Possède un Steam Deck` against the same sentence continued. There
    is no line to compose here: one of the two already says the whole
    subject, and composing one anyway would replace a sentence the
    user typed with a rearrangement of it.
    """
    merged = aggregate.merge(
        (
            entry(1, "Possède un Steam Deck"),
            entry(
                317,
                "Possède un Steam Deck sous SteamOS, fait tourner des "
                "conteneurs Podman dessus",
            ),
        ),
        "steam",
    )
    assert merged.speaker == 317


def test_two_notes_that_each_say_something_new_have_no_speaker():
    merged = aggregate.merge(NIPOGI, "nipogi")
    assert merged.speaker is None


def test_a_head_taken_from_another_entry_means_there_is_a_line_to_write():
    """
    The label is a word of its own source. If it came from an entry
    that contributed nothing else, that entry still said something the
    speaker does not, and absorbing it would lose it.
    """
    merged = aggregate.merge(
        (
            entry(1, "Machines : le Dell R710"),
            entry(2, "Le Dell R710 est au garage, 64 Go de RAM"),
        ),
        "r710",
    )
    assert merged.speaker is None
    assert merged.head == "Machines"
