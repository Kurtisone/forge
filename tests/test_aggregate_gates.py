"""
Tests for the half of the aggregation tier that is arithmetic.

Which entries belong to one subject is an enumerable choice, and this
repository's standing rule sends those to code or to a grammar, never
to a prompt. So grouping is a frequency count, and the two gates that
police what a model is then allowed to write are set comparisons.

The deliberate entries are the real store's, verbatim from
docs/memory.md, because the three NiPoGi lines are the case this tier
exists for: they overlap, no pair is identical, and the only merging
this codebase had (remember_many's exact-duplicate check) finds
nothing in them.

THE CORPUS IS BIGGER THAN THE POOL ON PURPOSE. What makes a word a
connective is how much of the store uses it, and eleven deliberate
entries are not enough for `avec` to look like one. The real store has
~190 rows, most of them archived transcript; ARCHIVE below stands in
for that, and leaving it out is how the first version of these tests
had the closure gate refusing the word `un`.
"""

from forge import aggregate

NIPOGI = [
    {
        "id": 1,
        "content": "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, "
        "32 Go de RAM, SSD 256 Go",
    },
    {
        "id": 2,
        "content": "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, "
        "Ansible, services Podman",
    },
    {"id": 3, "content": "Le NiPoGi a 32 Go de RAM"},
]

OTHERS = [
    {"id": 4, "content": "Possède un Steam Deck sous SteamOS"},
    {"id": 5, "content": "Ne pas épingler les messages avec des emojis"},
    {"id": 6, "content": "Le chat de la voisine s'appelle Pistache"},
]

#: Stands in for the archived half of the real store: ordinary French,
#: no subject in common, and enough of it that connectives recur. It is
#: deliberately repetitive -- what is under test is that `et`, `une`,
#: `a` and `avec` show up everywhere and therefore identify nothing,
#: not that the filler reads well.
ARCHIVE = [
    {
        "id": 100 + i,
        "content": (
            f"user: {question} assistant: Oui, et c'est une réponse "
            f"avec un détail : le reste est de ce côté, il y a tout"
        ),
    }
    for i, question in enumerate(
        [
            "Tu peux me dire ce que tu sais ?",
            "Et le reste ?",
            "Merci",
            "Bonjour",
            "Ça marche ?",
            "On continue ?",
            "Tu as le fichier ?",
            "Un souci ?",
            "Et après ?",
            "C'est tout ?",
            "Redis-moi",
            "OK",
            "Tu confirmes ?",
            "Et la suite ?",
            "Tu peux relire ?",
            "C'est bon pour toi ?",
            "Tu as vu le message ?",
            "On garde ça ?",
            "Tu notes ?",
            "Et ensuite ?",
        ]
    )
]

STORE = NIPOGI + OTHERS + ARCHIVE
MAX_DF = 0.2


def _corpus():
    freq = aggregate.frequencies(STORE)
    return freq, aggregate.ceiling(len(STORE), MAX_DF)


def test_a_number_glued_to_its_unit_is_two_tokens():
    """
    unicode61 reads `32Go` as one token. This tokenizer must not: the
    entries worth aggregating are precisely the telegraphic ones that
    glue them, and a gate blind to that would refuse the case it
    exists for.
    """
    assert aggregate.tokens("32Go RAM") == ["32", "go", "ram"]
    assert aggregate.tokens("32 Go de RAM") == ["32", "go", "de", "ram"]


def test_the_three_overlapping_entries_form_one_subject():
    freq, limit = _corpus()
    found = aggregate.subjects(NIPOGI + OTHERS, freq, limit)
    assert [s.ids for s in found] == [[1, 2, 3]]


def test_the_subject_is_named_by_a_word_a_human_can_read():
    """
    `nipogi`, `ram`, `go` and `32` all name the same three entries and
    all produce the same group. The name only ever reaches a log line,
    and "folded 3 entries about 32" is not an answer to "what was
    folded".
    """
    freq, limit = _corpus()
    assert aggregate.subjects(NIPOGI + OTHERS, freq, limit)[0].term == "nipogi"


def test_rarest_first_would_have_split_this_group():
    """
    Pins the rule against the reading it replaced. The rarest token
    shared by any two NiPoGi entries is `06`, out of AM06PRO: it names
    two of the three and would leave the third ungrouped forever.
    """
    freq, limit = _corpus()
    assert len(aggregate.subjects(NIPOGI + OTHERS, freq, limit)[0].ids) == 3


def test_a_lone_entry_is_not_a_subject():
    freq, limit = _corpus()
    assert aggregate.subjects(OTHERS, freq, limit) == []


def test_a_common_word_never_names_a_subject():
    """
    `le` and `de` are in a third of this store and name nothing.
    Grouping on one would produce an aggregate of everything, which is
    the ranking the hot tier refuses under another name.
    """
    freq, limit = _corpus()
    pair = [
        {"id": 200, "content": "Le serveur Dell R710 est au garage"},
        {"id": 201, "content": "Le chat de la voisine s'appelle Pistache"},
    ]
    assert aggregate.subjects(pair, freq, limit) == []


def test_an_entry_belongs_to_one_group_only():
    freq, limit = _corpus()
    seen: set[int] = set()
    for subject in aggregate.subjects(STORE, freq, limit):
        assert not seen & set(subject.ids)
        seen.update(subject.ids)


def test_coverage_passes_when_the_aggregate_carries_every_source_word():
    freq, limit = _corpus()
    written = (
        "Matériel : le NiPoGi AM06PRO tourne sous Arch, processeur Ryzen "
        "5500U, 32 Go de RAM, SSD 256 Go, Ansible et des services Podman"
    )
    for entry in NIPOGI:
        assert aggregate.uncovered(written, entry["content"], freq, limit) == []


def test_coverage_names_the_detail_that_went_missing():
    freq, limit = _corpus()
    written = "Le NiPoGi AM06PRO a un processeur Ryzen 5500U et 32 Go de RAM"
    missing = aggregate.uncovered(written, NIPOGI[0]["content"], freq, limit)
    assert "ssd" in missing
    assert "256" in missing


def test_coverage_catches_the_category_word_going_missing():
    """
    docs/memory.md records "a fact should name its category" being
    killed for the embedding channel and resurrected for the word
    channel: #307 came back at rank 1 on 2026-08-25 BECAUSE it
    contains `matériel`. An aggregate that drops it costs exactly that.
    """
    freq, limit = _corpus()
    written = "Le NiPoGi AM06PRO, Arch, Ryzen 5500U, 32 Go de RAM, SSD 256 Go"
    assert "matériel" in aggregate.uncovered(written, NIPOGI[0]["content"], freq, limit)


def test_closure_refuses_a_word_that_came_from_nowhere():
    freq, limit = _corpus()
    allowed = aggregate.lexicon([e["content"] for e in NIPOGI], freq, limit)
    written = "Le NiPoGi AM06PRO a 32 Go de RAM et une carte graphique Nvidia"
    assert aggregate.invented(written, allowed) == ["carte", "graphique", "nvidia"]


def test_closure_lets_connectives_through():
    """
    The aggregate has to be a sentence, and a sentence needs words
    that identify nothing. Refusing those would make the gate
    unsatisfiable rather than strict.
    """
    freq, limit = _corpus()
    allowed = aggregate.lexicon([e["content"] for e in NIPOGI], freq, limit)
    written = "Le NiPoGi AM06PRO a 32 Go de RAM, avec un SSD de 256 Go"
    assert aggregate.invented(written, allowed) == []


def test_closure_catches_a_number_nobody_wrote():
    """
    The dangerous case is not a new noun, it is a plausible number. A
    512 among 256s reads like a fact and is one nobody stated.
    """
    freq, limit = _corpus()
    allowed = aggregate.lexicon([e["content"] for e in NIPOGI], freq, limit)
    assert aggregate.invented("Le NiPoGi a un SSD de 512 Go", allowed) == ["512"]


def test_grouping_is_stable_across_runs():
    freq, limit = _corpus()
    first = aggregate.subjects(NIPOGI + OTHERS, freq, limit)
    second = aggregate.subjects(list(reversed(NIPOGI + OTHERS)), freq, limit)
    assert [(s.term, s.ids) for s in first] == [(s.term, s.ids) for s in second]


# --- The real store, 2026-09-11 -------------------------------------------
#
# The eleven deliberate entries as they stood when the first version of
# the grouping rule was measured against a copy. Two of the three
# subjects it found were not subjects, and this is the fixture that
# keeps them out.

REAL = [
    {"id": 1, "content": "Possède un Steam Deck"},
    {"id": 2, "content": "Possède un Dell R710, configuration à préciser plus tard"},
    {"id": 17, "content": "Le NiPoGi a 32 Go de RAM"},
    {
        "id": 307,
        "content": "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, "
        "32 Go de RAM, SSD 256 Go",
    },
    {"id": 308, "content": "NiPoGi AM06PRO processeur Ryzen 5500U"},
    {"id": 309, "content": "Le proxy podman écoute sur un socket unix"},
    {"id": 310, "content": "J'utilise aardvark-dns pour la résolution"},
    {
        "id": 314,
        "content": "Services tournant sous podman : forge, forge-llm, "
        "forge-embedding, searxng",
    },
    {
        "id": 315,
        "content": "NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, "
        "Ansible, services Podman",
    },
    {
        "id": 317,
        "content": "Possède un Steam Deck sous SteamOS, fait tourner des "
        "conteneurs Podman dessus",
    },
    {"id": 999, "content": "Ne pas épingler les messages avec des emojis"},
]

#: 195 archived rows, which is what the real store carries alongside.
BIG_ARCHIVE = [
    {
        "id": 2000 + i,
        "content": f"user: question {i} ? assistant: Oui, et c'est une réponse "
        "avec un détail : le reste est de ce côté, il y a tout dans la liste",
    }
    for i in range(195)
]


def _real():
    corpus = REAL + BIG_ARCHIVE
    return aggregate.frequencies(corpus), aggregate.ceiling(len(corpus), MAX_DF)


def test_podman_is_a_topic_and_never_a_subject():
    """
    Measured: the naming word alone grouped a unix socket, a service
    list, a hardware spec and a Steam Deck, because all four say
    `podman`. They share that word and nothing else.
    """
    freq, limit = _real()
    for subject in aggregate.subjects(REAL, freq, limit):
        assert set(subject.ids) != {309, 314, 315, 317}
        assert 309 not in subject.ids


def test_possede_is_a_verb_and_never_a_subject():
    """
    Measured: #1 and #2 were folded into one entry that merged a Steam
    Deck and a Dell R710, saving three estimated tokens. They share
    `possède` and nothing else.
    """
    freq, limit = _real()
    for subject in aggregate.subjects(REAL, freq, limit):
        assert set(subject.ids) != {1, 2}


def test_the_groups_the_real_store_should_produce():
    freq, limit = _real()
    found = {tuple(s.ids) for s in aggregate.subjects(REAL, freq, limit)}
    assert (307, 308, 315) in found
    assert (1, 317) in found


def test_a_group_shares_more_than_its_name():
    freq, limit = _real()
    for subject in aggregate.subjects(REAL, freq, limit):
        assert len(aggregate.shared(subject.entries, freq, limit)) >= 2


# --- The two sentences the real store produced, 2026-09-11 ----------------


def test_the_false_copula_is_what_no_gate_can_see():
    """
    The floor of this design, pinned so nobody mistakes the gates for
    a truth check. This passed closure, coverage, quorum AND budget,
    and folded three entries a human had typed:

        Le NiPoGi AM06PRO, un matériel de la NiPoGi AM06PRO, est un
        processeur Ryzen 5500U, [...]

    A mini PC is not a processor. Arithmetic on words will never see
    that, which is why `grammar` stopped asking for a sentence -- the
    shape that made a copula reachable at all.
    """
    freq, limit = _real()
    sources = [e["content"] for e in REAL if e["id"] in (17, 307, 315)]
    written = (
        "Le NiPoGi AM06PRO, un matériel de la NiPoGi AM06PRO, est un "
        "processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go, Arch, Ansible, "
        "services Podman"
    )
    assert aggregate.invented(written, aggregate.lexicon(sources, freq, limit)) == []
    for source in sources:
        assert aggregate.uncovered(written, source, freq, limit) == []


def test_the_repetition_gate_catches_both_of_them():
    freq, limit = _real()
    nipogi = (
        "Le NiPoGi AM06PRO, un matériel de la NiPoGi AM06PRO, est un "
        "processeur Ryzen 5500U, 32 Go de RAM"
    )
    steam = (
        "Possède un Steam Deck et un Steam Deck sous SteamOS, fait tourner "
        "des conteneurs Podman dessus"
    )
    assert "nipogi am" in aggregate.repeated(nipogi, freq, limit)
    assert "steam deck" in aggregate.repeated(steam, freq, limit)


def test_a_correct_list_reusing_a_unit_word_is_not_a_repetition():
    """
    `32 Go de RAM, SSD 256 Go` uses `go` twice and is right to. The
    pairs are `32 go` and `256 go`, which are different -- a rule at
    the word level would have refused the one aggregate this tier
    exists to produce.
    """
    freq, limit = _real()
    written = (
        "Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go"
    )
    assert aggregate.repeated(written, freq, limit) == []


def test_a_pair_with_a_connective_in_it_is_not_a_repetition():
    """`32 Go de RAM, 256 Go de SSD` repeats `go de`, which names nothing."""
    freq, limit = _real()
    written = "Matériel : 32 Go de RAM, 256 Go de SSD"
    assert aggregate.repeated(written, freq, limit) == []
