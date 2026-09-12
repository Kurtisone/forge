"""
One fact per subject, out of several that overlap.

WHAT THIS IS FOR. The hot tier (v3.18) put the whole deliberate store
in the synthesis prompt and five real runs on 2026-08-26 said two
things at once. It works -- "Quels sont tous les ordinateurs que je
possède ?" came back three for three out of eleven lines, with
retrieval returning nothing at all. And it does not solve the question
it was opened for: on "Tu peux me lister mon matériel ?" the same
block produced one entry, then two, then one. That is not a failure to
enumerate, it is a judgement of SCOPE, and it is a defensible one
given three NiPoGi entries that overlap without any pair being
identical:

    - [fact] Matériel : NiPoGi AM06PRO, processeur Ryzen 5500U, 32 Go de RAM
    - [fact] NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible
    - [fact] Le NiPoGi a 32 Go de RAM

No deterministic test merges those and there are zero exact duplicates
in the whole store, so `remember_many`'s duplicate check -- the only
merging this codebase had -- finds nothing. GBNF was the candidate
structural fix for the scope judgement and the computers run closed
it: the model enumerates correctly when the lines do not overlap.

BOTH HALVES ARE ARITHMETIC, AND THE SECOND ONE STOPPED BEING A MODEL
ON 2026-09-11.

Choosing WHICH entries belong to one subject is enumerable, and this
repository's standing rule is that an enumerable choice goes in a
grammar or in code, never in a prompt. So the grouping here is
arithmetic: the rarest word that still names several entries is the
name of a subject. No model call, no threshold nobody measured, and
the same answer every time it runs.

Writing the line was a model under a GBNF grammar, and the real store
measured what that costs. Seven calls across two runs, zero aggregates
written, and the same failure every time: the answer ran to the
grammar's maximum item count. `nipogi` came back as its three notes
concatenated; `steam` as a cycle of three items repeated four times.
Nothing in a lexicon of WORDS makes reusing a word expensive, so "say
each thing ONCE" was a rule of the prompt, enforced by nothing, set
against "do not leave anything out", which was enforced by a gate.

Moving the grammar's unit from the word to the DETAIL -- each
candidate detail emittable at most once, in order, the list ending
when they run out -- fixed the repetition and the runaway structurally
and left one failure standing: the model omits. Measured the same day,
it dropped `32 Go de RAM` and `Arch` from the NiPoGi group and
relabelled it `NiPoGi`, losing the word `matériel` that docs/memory.md
records as the reason #307 is reachable at all on the word channel.

Which is the finding: once the choice is enumerated, the only freedom
left to the model is to OMIT. So the line is now composed here, out of
the details the sources already contain -- see `merge`.

THE GATES, AND WHAT EACH ONE STILL ANSWERS.

An aggregate stands in for text somebody wrote. docs/memory.md states
the danger from the other side, about expanding a query versus
expanding a fact: a wrong expansion of a fact writes something nobody
said into memory, where it is indistinguishable from something they
did, and it stays. So:

  coverage   every informative word of a source must appear in the
             aggregate. A source that fails is LEFT ACTIVE -- the
             aggregate is still written, the block simply does not
             shrink for that one. The tier under-performs visibly
             instead of dropping a detail silently.

  closure    every informative word of the aggregate must come from
             its sources. Under a deterministic writer this can no
             longer fail, and it stays as the assertion that says so:
             the day anything in this module composes a word rather
             than copying one, the write is refused rather than
             discovered later in the store.

  repetition no pair of informative words twice. It used to catch a
             model concatenating its notes; what it catches now is two
             details saying one thing in different words, which set
             arithmetic cannot merge and must not pretend to have.

WHY THE WORD COUNTS ARE COMPUTED HERE AND NOT ASKED OF FTS5.

rag.informative_terms asks the index, and its docstring says why: the
lexical channel's count has to be folded exactly like the matching
that will use it, or the two drift. Nothing here ever matches the
index. These gates compare one text to another, and the frequency is
used only to decide which words are too common to carry meaning. A
local count with this module's own tokenizer is therefore the
self-consistent choice, not a shortcut.

It also lets the tokenizer differ where it has to. unicode61 reads
`32Go` as one token; this one splits at the digit/letter boundary,
because the entries worth aggregating are exactly the telegraphic ones
that glue numbers to units, and a gate that refused to fold `32Go RAM`
would refuse the case it exists for.

A word that appears in NO entry is INFORMATIVE here, which is the
other deliberate difference from rag.informative_terms -- that one
drops a zero-frequency term as unmatchable. A word nobody has ever
written is the most suspicious thing an aggregate can contain, and
dropping it would open a hole straight through the closure gate.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

from forge import rag
from forge.config import (
    COMPACTION_AGGREGATE,
    COMPACTION_AGGREGATE_MAX_DF,
    COMPACTION_AGGREGATE_MIN_SOURCES,
)
from forge.logger import log
from forge.tokens import estimate_tokens

#: Word-ish tokens, with a digit/letter boundary treated as a word
#: boundary. `32Go` -> `32`, `go`; `AM06PRO` -> `am`, `06`, `pro`.
#: Length 1 is kept: French connectives include `a`, `à`, `y`, and a
#: detail that loses them stops being the sentence somebody wrote.
_TOKEN_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)

#: The same text as somebody wrote it. NOT split at the digit/letter
#: boundary, because `AM06PRO`, `5500U` and `32Go` are single words on
#: the page, and this is the tokenizer that reads a name back off the
#: page -- `surface` uses it to spell a subject `NiPoGi` rather than
#: `Nipogi`.
#:
#: The two stand in a deliberate order: everything this one produces,
#: _TOKEN_RE splits into pieces the lexicon already holds, so a text
#: built out of source words passes the closure gate by construction.
_SURFACE_RE = re.compile(r"[^\W_]+", re.UNICODE)


def tokens(text: str) -> list[str]:
    """The lowercased tokens of a text, in order, duplicates kept."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def frequencies(entries: list[dict]) -> Counter:
    """How many entries each token appears in (not how many times)."""
    counter: Counter = Counter()
    for entry in entries:
        counter.update(set(tokens(entry["content"])))
    return counter


def ceiling(total: int, max_df: float) -> int:
    """
    The document-frequency above which a word names nothing.

    At least one, always -- the same floor rag.informative_terms uses
    and for the same reason: on a store of ten rows a strict fraction
    refuses every word of the only entry that answers a question.
    """
    return max(1, int(max_df * total))


def informative(text: str, freq: Counter, limit: int) -> set[str]:
    """
    The tokens of *text* that identify something.

    A token nobody else wrote (frequency 0 or 1) is as informative as
    it gets. A token in more entries than *limit* names half the store
    and names nothing.
    """
    return {t for t in tokens(text) if freq.get(t, 0) <= limit}


def common(freq: Counter, limit: int) -> set[str]:
    """
    The words of the store that identify nothing -- its connectives,
    read off the store instead of off a maintained list.

    Same choice rag.informative_terms makes and for the same reason: a
    stopword list is an artefact that is wrong for whatever gets
    written next, while a count over the actual entries is right by
    construction and needs no French, which matters for a corpus
    holding NiPoGi, busctl and aardvark-dns.
    """
    return {term for term, df in freq.items() if df > limit}


def lexicon(sources: list[str], freq: Counter, limit: int) -> set[str]:
    """
    Every word an aggregate of *sources* is allowed to use: the words
    of its own sources, plus the words that identify nothing.

    ONE definition, used twice -- as the alternation of the GBNF
    grammar that constrains sampling, and as the check that catches a
    provider with no grammar. Two definitions of the same rule is the
    drift this codebase has a module (forge/non_answer.py) and a test
    dedicated to preventing.
    """
    allowed = common(freq, limit)
    for source in sources:
        allowed.update(tokens(source))
    return allowed


@dataclass(frozen=True)
class Subject:
    """A set of entries that talk about the same thing, and its name."""

    term: str
    entries: tuple[dict, ...]

    @property
    def ids(self) -> list[int]:
        return [e["id"] for e in self.entries]

    def sources(self) -> list[str]:
        return [e["content"] for e in self.entries]


def shared(entries: tuple[dict, ...], freq: Counter, limit: int) -> set[str]:
    """The informative words every entry of a group has."""
    sets = [informative(e["content"], freq, limit) for e in entries]
    return set.intersection(*sets) if sets else set()


def subjects(
    entries: list[dict],
    freq: Counter,
    limit: int,
    min_sources: int = 2,
    min_shared: int = 2,
) -> list[Subject]:
    """
    Group entries by subject, deterministically, with no model call.

    THE RULE, in two halves. A candidate group is the set of entries
    sharing one informative word. It is a SUBJECT only if its entries
    share at least *min_shared* informative words -- the naming word,
    and something else. Take the best one, remove its entries from the
    pool, repeat.

    The second half is not a refinement, it is what makes the first
    one usable, and the real store said so. Measured 2026-09-11
    against a copy of the 11 deliberate entries, the naming word alone
    produced two groups that are not subjects:

        podman   #309 the podman proxy listens on a unix socket
                 #314 services running under podman
                 #315 NiPoGi AM06PRO, Arch, ... services Podman
                 #317 Steam Deck under SteamOS, runs Podman containers

        possède  #1 Possède un Steam Deck
                 #2 Possède un Dell R710

    `podman` is a topic and `possède` is a verb. Neither names a
    thing, and the aggregate written for the second one merged two
    different machines into one entry before the budget gate could
    notice it had saved three tokens. What separates them from a real
    subject is exactly this: #1 and #2 share ONE informative word,
    while #17 and #307 share `nipogi`, `32`, `go` and `ram`.

    NO SECOND CEILING. The first draft of this fix also capped how
    much of the POOL a naming word may cover, on the grounds that
    `podman` names four deliberate entries out of eleven. It would
    have worked, and it is not here: the only value that separates
    `podman` (4) from `nipogi` (3) on this store is a third, which is
    a number fitted to one measurement. The shared-vocabulary rule
    refuses both groups on its own and needs nothing calibrated.

    Largest group first, then the tightest shared vocabulary, then the
    longer name. Rarest-first was the reading before this one and it
    splits the group it aims at: the rarest token shared by two NiPoGi
    entries is `06`, out of AM06PRO.

    *freq* and *limit* are computed over the WHOLE store, not over the
    pool. What makes a word common is how much of the store uses it;
    eleven deliberate entries are not enough corpus for `avec` to look
    like a connective, and the caller has the rest of the store.

    An entry belongs to ONE group. Overlapping groups would produce
    two aggregates each speaking for the same source, which
    supersession refuses to represent (no chains).

    Entries already superseded never reach this function: it is fed
    from rag.hot_entries, which skips them.
    """
    pool = {e["id"]: e for e in entries}
    out: list[Subject] = []

    while True:
        postings: dict[str, list[int]] = {}
        for entry in pool.values():
            for token in informative(entry["content"], freq, limit):
                postings.setdefault(token, []).append(entry["id"])

        candidates = []
        for term, ids in postings.items():
            if len(ids) < min_sources:
                continue
            group = tuple(pool[i] for i in sorted(ids))
            vocabulary = shared(group, freq, limit)
            if len(vocabulary) < min_shared:
                continue
            candidates.append((-len(ids), -len(vocabulary), -len(term), term, group))

        if not candidates:
            return out

        *_, term, group = min(candidates)
        out.append(Subject(term=term, entries=group))
        for entry in group:
            pool.pop(entry["id"])


def uncovered(aggregate: str, source: str, freq: Counter, limit: int) -> list[str]:
    """
    The informative words of *source* the aggregate does not carry.

    Empty means this source can be folded. Non-empty is reported and
    the source stays active -- see the module docstring on why this
    gate under-performs rather than deletes.
    """
    carried = set(tokens(aggregate))
    return sorted(informative(source, freq, limit) - carried)


def repeated(text: str, freq: Counter, limit: int) -> list[str]:
    """
    Pairs of informative words this text uses twice.

    THE FAILURE THIS CATCHES IS CONCATENATION, measured 2026-09-11 on
    the real store, in both sentences the model produced:

        Le NiPoGi AM06PRO, un matériel de la NiPoGi AM06PRO, est un
        processeur Ryzen 5500U, 32 Go de RAM, SSD 256 Go, [...]

        Possède un Steam Deck et un Steam Deck sous SteamOS, fait
        tourner des conteneurs Podman dessus.

    Both passed closure, coverage, quorum -- and one passed budget and
    folded three entries. Nothing in the lexicon makes reusing a word
    cost anything, and "do not leave anything out" pushes straight
    here.

    BOTH words of the pair have to be informative, which is what keeps
    this from refusing a correct list. `32 Go de RAM, SSD 256 Go` uses
    `go` twice and is right to; the pairs are `32 go` and `256 go`,
    which are different. A rule at the word level would have refused
    it, and a rule that lets one common word into the pair would refuse
    `32 Go de RAM, 256 Go de SSD` over `go de`.

    NOT A TRUTH CHECK, and neither is anything else in this module.
    The first sentence above says a mini PC IS a processor, and no
    arithmetic on words will ever see that -- which is why the grammar
    stopped asking for a sentence. See `grammar`.
    """
    words = tokens(text)
    strong = informative(text, freq, limit)
    seen: set[tuple[str, str]] = set()
    twice: set[str] = set()
    for pair in pairwise(words):
        if pair[0] not in strong or pair[1] not in strong:
            continue
        if pair in seen:
            twice.add(" ".join(pair))
        seen.add(pair)
    return sorted(twice)


def invented(aggregate: str, allowed: set[str]) -> list[str]:
    """
    The words of *aggregate* that are not in its lexicon.

    Non-empty means the aggregate is not written at all: it asserts
    something about the user that no entry of theirs says, and once
    stored it is indistinguishable from something they wrote.
    """
    return sorted(set(tokens(aggregate)) - allowed)


# --- The details, which are what the notes are made of ---------------------

#: What separates one detail from the next inside a note. Comma and
#: semicolon, because that is what the store actually holds:
#: `NiPoGi AM06PRO, Arch, 5500U, 32Go RAM, SSD 256Go, Ansible`.
#:
#: A note with no separator at all is ONE detail, and that is the
#: honest reading of it rather than a degenerate case -- a sentence
#: nobody punctuated is a sentence whose pieces nobody separated, and
#: guessing where they would have is how an aggregate starts asserting
#: things.
_DETAIL_RE = re.compile(r"\s*[;,]\s*")

#: How many words may precede the colon before it stops being a label.
#:
#: `Matériel : NiPoGi AM06PRO, ...` is a label and a list. `Le proxy
#: podman écoute sur un socket unix : il est en lecture seule` is a
#: sentence that happens to contain a colon, and taking its first nine
#: words as the head of a list would produce a line nobody wrote in a
#: shape nobody uses.
_MAX_LABEL_WORDS = 3


def labelled(content: str) -> tuple[str | None, list[str]]:
    """
    A note as it is actually written: an optional label, then details.

    This is the tokenizer of this tier, one level up from words. The
    branch's first version worked on words -- the grammar's unit was
    the word, the repetition gate compared pairs of words -- and the
    real store said what that costs: nothing in a lexicon of words
    makes reusing a word expensive, so "do not leave anything out"
    concatenated the notes and every gate downstream was left arguing
    about the debris.

    The detail is the unit the notes are already written in. Two notes
    that overlap overlap by detail, and a merge that keeps details
    whole is a recombination of what is stored rather than a rewrite
    of it.
    """
    label, body = None, content
    head, colon, rest = content.partition(":")
    if colon and rest.strip() and len(head.split()) <= _MAX_LABEL_WORDS:
        label, body = head.strip(), rest
    return label, [d.strip() for d in _DETAIL_RE.split(body) if d.strip()]


@dataclass(frozen=True)
class Detail:
    """One detail, and the entry it was written in."""

    text: str
    source: int

    @property
    def words(self) -> frozenset[str]:
        return frozenset(tokens(self.text))


#: Among details that say the same words, which surface is kept: the
#: one written out in full.
#:
#: `SSD 256 Go` and `SSD 256Go` tokenize identically -- the tokenizer
#: of this module splits at the digit/letter boundary on purpose -- so
#: something has to choose, and it has to choose the same way every
#: time or two runs of this pass produce two different stores. The
#: spelled-out form wins on the one axis docs/memory.md has measured:
#: an instruction-tuned embedding model retrieves a telegram worst,
#: and `#313` spent six campaigns unreachable for being one.
def _spelled_out(detail: Detail) -> tuple[int, int]:
    return (-len(detail.text.split()), -len(detail.text))


def distinct(details: list[Detail]) -> list[Detail]:
    """
    The details that survive deduplication, in the order written.

    ONE RULE, AND IT IS THE SAFETY PROPERTY OF THIS WHOLE TIER: a
    detail is dropped only in favour of a detail that contains EVERY
    ONE of its words -- connectives included, not just the informative
    ones.

    Informative-word containment was the obvious reading and it is
    wrong in a way no arithmetic recovers from. `pas`, `ne`, `jamais`,
    `sans` are connectives by frequency, so they are exactly the words
    an informative-word rule ignores, and under it `Le NiPoGi n'a pas
    32 Go de RAM` is contained in `Le NiPoGi a 32 Go de RAM` and gets
    folded into its own opposite. Requiring every word inverts that:
    a negation carries a word its positive does not, so the negation
    can never be the one dropped.

    Stated as the invariant it is: **nothing that says more is ever
    deleted by something that says less.** No stopword list, no
    negation list, no language -- which matters for the same corpus of
    NiPoGi, busctl and aardvark-dns that the frequency count exists
    for.

    Equal word sets are the one case containment cannot order, and
    `_spelled_out` breaks the tie. The position is the first one the
    word set appeared at, so which surface wins never moves the line.
    """
    best: dict[frozenset[str], tuple[int, Detail]] = {}
    for position, detail in enumerate(details):
        words = detail.words
        if not words:
            continue
        seen = best.get(words)
        if seen is None:
            best[words] = (position, detail)
        elif _spelled_out(detail) < _spelled_out(seen[1]):
            best[words] = (seen[0], detail)

    survivors = [
        (position, detail)
        for words, (position, detail) in best.items()
        if not any(words < other for other in best if other != words)
    ]
    return [detail for _, detail in sorted(survivors, key=lambda pair: pair[0])]


def surface(term: str, entries: tuple[dict, ...]) -> str:
    """
    The subject's name as somebody wrote it -- `NiPoGi`, not `nipogi`
    and not `Nipogi`.

    The first occurrence in the entries, in their own order, so this
    answers the same way on every run. `capitalize()` is the fallback
    for a term that is a piece of a longer word (`am`, out of
    AM06PRO) and therefore has no surface of its own.
    """
    for entry in entries:
        for written in _SURFACE_RE.findall(entry["content"]):
            if written.lower() == term:
                return written
    return term.capitalize()


@dataclass(frozen=True)
class Merged:
    """The line a subject folds into, and what it is made of."""

    head: str
    details: tuple[Detail, ...]

    #: The entry that already carries every surviving detail, when one
    #: does. Then there is nothing to write: that entry can speak for
    #: the others as it stands. See `_absorb`.
    speaker: int | None = None

    @property
    def text(self) -> str:
        return f"{self.head} : " + ", ".join(d.text for d in self.details) + "."


def merge(entries: tuple[dict, ...], term: str) -> Merged:
    """
    One labelled list out of several overlapping notes, deterministically.

    THE HEAD IS THE FIRST LABEL THE SOURCES CARRY, and the subject's
    own name when none of them carries one. A label is the user's word
    for what the list is about -- `Matériel` is the word docs/memory.md
    records as the reason `#307` comes back at rank 1 on the word
    channel -- so inventing a better one is both unnecessary and the
    move this module exists to refuse. A second, different label is
    not thrown away either: it becomes a detail, where the coverage
    gate can see it.

    THE ITEMS ARE THE SOURCES' OWN DETAILS, VERBATIM, deduplicated by
    `distinct` and left in the order they were written. Every word of
    the result was typed by the person it describes, which is the
    strongest form of the closure gate this tier ever had -- stronger
    than the grammar that used to enforce it, because a grammar
    constrains which words may be emitted and this constrains which
    SENTENCES may be.

    AND SOMETIMES THERE IS NOTHING TO WRITE. When every surviving
    detail turns out to belong to one entry, that entry already says
    the whole subject and the others are older, shorter versions of
    it -- `Possède un Steam Deck` against `Possède un Steam Deck sous
    SteamOS, fait tourner des conteneurs Podman dessus`. `speaker`
    names it, and `_absorb` folds the rest into it without composing
    anything at all. The head is allowed to come from the speaker or
    to be the subject's own name; a head taken from ANOTHER entry
    means that entry said something this one does not, so the line
    gets written after all.
    """
    head: str | None = None
    head_source: int | None = None
    collected: list[Detail] = []
    for entry in entries:
        label, details = labelled(entry["content"])
        if label is not None:
            if head is None:
                head, head_source = label, entry["id"]
            else:
                collected.append(Detail(label, entry["id"]))
        collected.extend(Detail(text, entry["id"]) for text in details)

    survivors = tuple(distinct(collected))
    written_by = {d.source for d in survivors}
    speaker = None
    if len(written_by) == 1 and head_source in (None, *written_by):
        speaker = written_by.pop()

    return Merged(head or surface(term, entries), survivors, speaker)


#: How much shorter an aggregate has to be before folding is worth it.
#:
#: NOT a ``>=``, which is what shipped first and what the real store
#: caught on 2026-09-11: an aggregate of two entries came back at 24
#: estimated tokens against 27, and the same run logged
#: ``tokens.estimate_drift ... error_pct=21.4``. The gate was comparing
#: two estimates whose error was seven times the gap it was measuring.
#: A margin wider than the estimator's observed drift is the smallest
#: honest version of "shorter".
_BUDGET_MARGIN = 0.25

# --- The pass, which runs in compaction and nowhere else -------------------

#: How much of the store the frequency count reads. The corpus is only
#: used to decide which words identify nothing, so a cap degrades
#: gracefully -- list_entries is newest-first, and the words that
#: matter are recent by construction. Same bound and same reasoning as
#: tools/memory._VOCABULARY_ENTRIES.
_CORPUS_ENTRIES = 2000


def _project(entries: tuple[dict, ...]) -> str | None:
    """
    The project an aggregate belongs to: the one its sources share, or
    none at all.

    A project is a namespace -- rag._already_stored says so from the
    other side, where the same sentence filed under two projects is
    two statements about two things. An aggregate of sources from two
    namespaces belongs to neither.
    """
    projects = {e.get("project") for e in entries}
    return projects.pop() if len(projects) == 1 else None


def run_pass(
    conn, *, max_df: float, min_sources: int, write: bool = True
) -> list[dict]:
    """
    Aggregate what can be aggregated, and report what happened.

    *write* is what bench/rag_aggregate.py turns off to show what this
    would do without doing it. It is a parameter of the pass rather
    than a second implementation in the harness, because the thing
    worth reading before a fold is what the fold WILL be, and a
    harness that computes it separately is a harness that can be
    right about a pass that is wrong.

    NOTHING IS WRITTEN UNLESS IT IS GOING TO REPLACE SOMETHING. Every
    gate is a comparison between texts, so all of them run BEFORE the
    entry is stored:

      closure   a word from nowhere -- the subject is abandoned. It
                cannot fire under this writer; it is the assertion
                that says so.
      repetition two details saying one thing in different words --
                the subject is abandoned.
      coverage  a source whose detail went missing stays active.
      quorum    fewer than *min_sources* foldable sources left, and
                the aggregate would be one more overlapping line in
                the block rather than one fewer.
      budget    an aggregate no shorter than what it folds makes the
                block bigger, which is the opposite of the job.

    Returns one dict per subject, whether it was written or not. The
    caller logs it; bench/rag_aggregate.py prints it.

    NEVER RAISES. The pass runs after a compaction that has already
    happened and already committed, so nothing here may turn that into
    an error the user reads -- which mattered more when a provider
    could be down, and still holds for a store that cannot be written.
    """
    entries = rag.hot_entries(conn)
    if len(entries) < min_sources:
        return []

    corpus = rag.list_entries(conn, limit=_CORPUS_ENTRIES)
    freq = frequencies(corpus)
    limit = ceiling(len(corpus), max_df)

    report: list[dict] = []
    for subject in subjects(entries, freq, limit, min_sources):
        report.append(_one(conn, subject, freq, limit, min_sources, write))
    return report


def _coverage(
    subject: Subject, written: str, freq: Counter, limit: int
) -> tuple[list[dict], dict]:
    """Which sources the sentence can speak for, and what the rest lost."""
    foldable, held_back = [], {}
    for entry in subject.entries:
        missing = uncovered(written, entry["content"], freq, limit)
        if missing:
            held_back[entry["id"]] = missing
        else:
            foldable.append(entry)
    return foldable, held_back


def _absorb(
    conn,
    subject: Subject,
    speaker: int,
    freq: Counter,
    limit: int,
    write: bool,
    outcome: dict,
) -> dict:
    """
    Fold a subject into the entry that already says all of it.

    THE CHEAPEST FOLD THERE IS, and the safest. Nothing is composed,
    nothing is stored, and the text that survives is one the user
    typed -- so the two questions every other path has to answer here
    have no content: there is no word from nowhere and no detail that
    could go missing.

    NO QUORUM AND NO BUDGET. Both exist to judge NEW TEXT. Quorum
    refuses an entry that stands in for a single other entry, because
    that is a rewrite of somebody's note; absorbing one note into
    another rewrites nothing, and the block is one line shorter for
    it. Budget compares what a line costs against what it saves, and
    this one costs nothing.

    THE NAMESPACE IS THE ONE THING IT STILL HAS TO CHECK. A project is
    a namespace -- rag._already_stored says so from the other side --
    and hiding an entry of one project behind an entry of another
    makes it unreachable from the block under a name nobody filed it
    with. Those sources stay active.

    What this can do is fold a note into a longer note that contradicts
    it, if the contradiction is spelled with words the shorter one also
    uses. `distinct` is what stops that, one level down: a detail is
    only ever dropped by a detail that contains EVERY one of its words,
    so `n'a pas` can never be absorbed by `a`.
    """
    keeper = next(e for e in subject.entries if e["id"] == speaker)
    foldable, held_back = [], {}
    for entry in subject.entries:
        if entry["id"] == speaker:
            continue
        if entry.get("project") != keeper.get("project"):
            held_back[entry["id"]] = ["(another project)"]
            continue
        missing = uncovered(keeper["content"], entry["content"], freq, limit)
        if missing:
            held_back[entry["id"]] = missing
        else:
            foldable.append(entry["id"])

    outcome = {**outcome, "written": keeper["content"], "into": speaker}
    if not foldable:
        return {**outcome, "refused": "quorum", "held_back": held_back}

    if not write:
        return {**outcome, "folds": foldable, "held_back": held_back}

    return {
        **outcome,
        "folded": rag.supersede(conn, foldable, speaker),
        "held_back": held_back,
    }


def _one(
    conn,
    subject: Subject,
    freq: Counter,
    limit: int,
    min_sources: int,
    write: bool = True,
) -> dict:
    """
    One subject, from its entries to the line that speaks for them.

    THE ORDER IS THE POINT. `merge` composes the line out of details
    the sources already contain, then every gate compares that text to
    those texts, and only then does anything reach the store. An
    aggregate that would be one more overlapping line in the block
    instead of one fewer is never written at all.

    Closure comes first and can no longer fail -- see the module
    docstring. It is kept where a gate on a model used to be, so that
    a writer which one day composes a word rather than copying one is
    stopped here rather than found later in the store.
    """
    sources = subject.sources()
    outcome: dict = {"subject": subject.term, "sources": subject.ids}
    merged = merge(subject.entries, subject.term)

    if merged.speaker is not None:
        return _absorb(conn, subject, merged.speaker, freq, limit, write, outcome)

    written = merged.text

    strangers = invented(written, lexicon(sources, freq, limit))
    if strangers:
        log.warning(
            "aggregate: %r not written -- it says %s, and no entry does. An "
            "aggregate is a recombination of what is already stored.",
            subject.term,
            ", ".join(repr(s) for s in strangers),
        )
        return {
            **outcome,
            "written": written,
            "refused": "closure",
            "invented": strangers,
        }

    # Two details saying one thing in different words. Set arithmetic
    # cannot merge `32 Go de RAM` with `mémoire de 32 Go` -- they share
    # no word set and neither contains the other -- so the line would
    # carry both. Refusing is the honest outcome: the block keeps two
    # overlapping entries instead of gaining a third that repeats
    # itself.
    twice = repeated(written, freq, limit)
    if twice:
        log.warning(
            "aggregate: %r would say %s twice -- two details say one thing in "
            "different words, which is not something arithmetic can merge",
            subject.term,
            ", ".join(repr(t) for t in twice),
        )
        return {
            **outcome,
            "written": written,
            "refused": "repetition",
            "repeated": twice,
        }

    foldable, held_back = _coverage(subject, written, freq, limit)

    if len(foldable) < min_sources:
        return {
            **outcome,
            "written": written,
            "refused": "quorum",
            "held_back": held_back,
        }

    cost = estimate_tokens(render_lines([written]))
    saved = estimate_tokens(render_lines([e["content"] for e in foldable]))
    if cost > saved * (1 - _BUDGET_MARGIN):
        return {
            **outcome,
            "written": written,
            "refused": "budget",
            "tokens": (cost, saved),
        }

    if not write:
        return {
            **outcome,
            "written": written,
            "folds": [e["id"] for e in foldable],
            "held_back": held_back,
            "tokens": (cost, saved),
        }

    try:
        entry_id = rag.remember(
            conn, kind="fact", content=written, project=_project(subject.entries)
        )
    except (rag.DegenerateEntry, rag.EmbeddingError) as e:
        log.warning("aggregate: %r could not be stored (%s)", subject.term, e)
        return {**outcome, "written": written, "refused": "store"}

    folded = rag.supersede(conn, [e["id"] for e in foldable], entry_id)
    return {
        **outcome,
        "written": written,
        "id": entry_id,
        "folded": folded,
        "held_back": held_back,
        "tokens": (cost, saved),
    }


def render_lines(contents: list[str]) -> str:
    """
    The contents as the hot block will carry them, which is the only
    shape in which their cost is the cost that matters. A bare
    len(content) would count the text and not the line.
    """
    return "\n".join(f"- [fact] {c}" for c in contents)


def maybe_aggregate() -> list[dict]:
    """
    The entry point compaction calls. Returns the report, empty when
    the knob is off or nothing could be folded.

    IN COMPACTION AND NOWHERE ELSE, which is a placement argument and
    not a convenience. The router normalises what it writes -- measured
    2026-08-25, byte-identical output with the category word gone -- so
    extraction cannot live on the write path. And recall is the latency
    path: the hot block is a stable prefix whose prefill is paid once,
    and a pass that rewrote it mid-conversation would cost that cache
    on every turn. Compaction is rare, already off the answer's
    critical path, and already the place this store is fed.

    The pass no longer costs a model call, which removes the argument
    about latency and leaves the one about the KV cache standing. A
    block that changes between two recalls is re-prefilled whatever
    wrote it.

    SWALLOWS EVERYTHING. A compaction that has already committed must
    not be turned into an error the user reads because the pass that
    runs after it did not work.
    """
    if not COMPACTION_AGGREGATE:
        return []

    try:
        conn = rag.get_connection()
        try:
            report = run_pass(
                conn,
                max_df=COMPACTION_AGGREGATE_MAX_DF,
                min_sources=COMPACTION_AGGREGATE_MIN_SOURCES,
            )
        finally:
            conn.close()
    except Exception as e:  # noqa: BLE001 - see the docstring
        log.warning("aggregate: the pass did not run (%s)", e)
        return []

    for item in report:
        log.event("aggregate.subject", **item)

    folded = sum(len(i.get("folded") or []) for i in report)
    if folded:
        log.event(
            "aggregate.run",
            subjects=len(report),
            written=sum(1 for i in report if i.get("id")),
            folded=folded,
        )
    return report
