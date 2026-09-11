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

TWO HALVES, AND ONLY ONE OF THEM IS ALLOWED TO BE A MODEL.

Choosing WHICH entries belong to one subject is enumerable, and this
repository's standing rule is that an enumerable choice goes in a
grammar or in code, never in a prompt. So the grouping here is
arithmetic: the rarest word that still names several entries is the
name of a subject. No model call, no threshold nobody measured, and
the same answer every time it runs.

Writing the aggregate SENTENCE is the other half, and that one needs a
model -- see the grammar in this module, which is what keeps the model
inside the vocabulary it was given.

THE TWO GATES, AND WHY THEY POINT IN OPPOSITE DIRECTIONS.

An aggregate is text nobody wrote, standing in for text somebody did.
docs/memory.md already states the danger from the other side, about
expanding a query versus expanding a fact: a wrong expansion of a fact
writes something nobody said into memory, where it is
indistinguishable from something they did, and it stays. So:

  coverage   every informative word of a source must appear in the
             aggregate. A source that fails is LEFT ACTIVE -- the
             aggregate is still written, the block simply does not
             shrink for that one. The tier under-performs visibly
             instead of dropping a detail silently.

  closure    every informative word of the aggregate must come from
             its sources. An aggregate that fails is NOT WRITTEN AT
             ALL. This is the gate that separates "a fact aggregated"
             from "a fact invented", and it is a precondition to the
             write rather than to the fold, because a hallucinated
             entry is harmful whether or not anything is folded into
             it.

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

from forge import rag
from forge.config import (
    COMPACTION_AGGREGATE,
    COMPACTION_AGGREGATE_MAX_DF,
    COMPACTION_AGGREGATE_MIN_SOURCES,
)
from forge.errors import ProviderError
from forge.llm import call_llm
from forge.logger import log
from forge.text_cleaning import strip_think_blocks, try_unwrap_router_json
from forge.tokens import estimate_tokens

#: Word-ish tokens, with a digit/letter boundary treated as a word
#: boundary. `32Go` -> `32`, `go`; `AM06PRO` -> `am`, `06`, `pro`.
#: Length 1 is kept: French connectives include `a`, `à`, `y`, and the
#: grammar built from this lexicon has to be able to form a sentence.
_TOKEN_RE = re.compile(r"[0-9]+|[^\W\d_]+", re.UNICODE)

#: The same text as the model will be allowed to write it. NOT split at
#: the digit/letter boundary, because `AM06PRO`, `5500U` and `32Go` are
#: single words on the page and a grammar that could only emit their
#: pieces would have to glue them back with an empty separator -- which
#: also glues `32` to `RAM`.
#:
#: The two tokenizers stand in a deliberate order: everything this one
#: produces, _TOKEN_RE splits into pieces the lexicon already holds. So
#: anything the grammar can emit passes the closure gate by
#: construction, and a test pins that rather than leaving it to be
#: noticed when llama.cpp and ollama start disagreeing.
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


def subjects(
    entries: list[dict], freq: Counter, limit: int, min_sources: int = 2
) -> list[Subject]:
    """
    Group entries by subject, deterministically, with no model call.

    THE RULE: the largest set of entries sharing one informative word
    is a subject. Take it, remove those entries from the pool, repeat
    until no word names *min_sources* of what is left.

    Largest first, and not rarest first. Rarest-first was the obvious
    reading of "the most specific word names the subject" and it
    splits the group it is aimed at: on the real NiPoGi entries the
    rarest shared token is `06`, out of `AM06PRO`, which names two of
    the three and leaves the third alone forever. The word that names
    the WHOLE group is the one worth grouping on, and `informative`
    is already what stops a word that names the whole store from
    qualifying.

    Ties are broken by the longer word, then alphabetically. Both
    halves matter: three of the real entries are named equally well by
    `nipogi`, `ram`, `go` and `32`, they all produce the SAME group,
    and `Subject.term` only ever reaches a log line -- where `32` is
    unreadable and `nipogi` is the answer to "what was folded?".

    *freq* and *limit* are computed over the WHOLE store, not over the
    pool. What makes a word common is how much of the store uses it;
    eleven deliberate entries are not enough corpus for `avec` to look
    like a connective, and the caller has the rest of the store.

    An entry belongs to ONE group. Overlapping groups would produce
    two aggregates each speaking for the same source, which
    supersession refuses to represent (no chains) -- better to refuse
    it here, where the reason is legible.

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

        candidates = sorted(
            (
                (-len(ids), -len(term), term, sorted(ids))
                for term, ids in postings.items()
                if len(ids) >= min_sources
            )
        )
        if not candidates:
            return out

        _, _, term, ids = candidates[0]
        out.append(Subject(term=term, entries=tuple(pool[i] for i in ids)))
        for entry_id in ids:
            pool.pop(entry_id)


def uncovered(aggregate: str, source: str, freq: Counter, limit: int) -> list[str]:
    """
    The informative words of *source* the aggregate does not carry.

    Empty means this source can be folded. Non-empty is reported and
    the source stays active -- see the module docstring on why this
    gate under-performs rather than deletes.
    """
    carried = set(tokens(aggregate))
    return sorted(informative(source, freq, limit) - carried)


def invented(aggregate: str, allowed: set[str]) -> list[str]:
    """
    The words of *aggregate* that are not in its lexicon.

    Non-empty means the aggregate is not written at all: it asserts
    something about the user that no entry of theirs says, and once
    stored it is indistinguishable from something they wrote.
    """
    return sorted(set(tokens(aggregate)) - allowed)


# --- The sentence, and the only half a model is allowed to write -----------


def surfaces(sources: list[str]) -> set[str]:
    """Every word of the sources, as written on the page."""
    return {w for source in sources for w in _SURFACE_RE.findall(source)}


def _alternatives(sources: list[str], freq: Counter, limit: int) -> list[str]:
    """
    Every literal the grammar will let the model emit.

    Source words keep the case they were WRITTEN in -- `NiPoGi`,
    `AM06PRO`, `SSD` -- because an aggregate spelling them `nipogi`
    would be a worse entry than the ones it replaces, and the entry
    that made this tier necessary is the one written as a telegram in
    the first place.

    Common words get a lowercase and a capitalised form, which is the
    whole of what a sentence needs: one of them starts it.
    """
    alternatives: set[str] = set()
    for written in surfaces(sources):
        alternatives.add(written)
        # Case variants only for words that are only letters. Folding
        # AM06PRO would offer `Am06pro`, which is not a spelling of
        # anything and is one more way for the sentence to be worse
        # than the notes it replaces.
        if written.isalpha():
            alternatives.add(written.lower())
            alternatives.add(written.capitalize())
    for word in common(freq, limit):
        alternatives.add(word)
        alternatives.add(word.capitalize())
    return sorted(alternatives)


def _escape(literal: str) -> str:
    return literal.replace("\\", "\\\\").replace('"', '\\"')


def grammar(sources: list[str], freq: Counter, limit: int) -> str:
    """
    A GBNF grammar whose entire vocabulary is this subject's lexicon.

    THE CLOSURE GATE, MOVED INTO THE SAMPLER. `invented` can only
    report that a word came from nowhere once the model has written
    it; an alternation the word is not in means it cannot be written.
    This repository has now reached for that move eleven times, and
    the reason has not changed: a rule the model is ASKED to follow is
    one it follows most of the time, and the times it does not are the
    ones nobody sees. `invented` stays, because the grammar only
    exists on llama.cpp -- on ollama or OpenRouter the same call runs
    unconstrained and the check is all there is.

    It also fixes the LANGUAGE for free. Every literal here came out
    of entries the user wrote, so the sentence is in their language
    without a single word of the prompt saying so.

    At least four words, expressed as three explicit repetitions
    rather than `{3,}`: llama.cpp's support for bounded repetition
    arrived later than the rest of GBNF, and a grammar the server
    refuses is a 400 on the call rather than a degraded one. There is
    no upper bound here -- a budget check is arithmetic and belongs
    where the block is measured, not in a sampler.

    Rule names are hyphenated. llama.cpp's lexer builds names out of
    is_word_char(), which accepts [a-zA-Z0-9-] and NOT underscore; see
    forge/gbnf.py for the debugging cycle that cost.
    """
    words = " | ".join(f'"{_escape(w)}"' for w in _alternatives(sources, freq, limit))
    return (
        "root ::= aggregate-word (aggregate-sep aggregate-word) "
        "(aggregate-sep aggregate-word) (aggregate-sep aggregate-word)* "
        '"."?\n'
        f"aggregate-word ::= {words}\n"
        'aggregate-sep ::= " " | ", " | " : "\n'
    )


#: The instruction. Short on purpose: everything the model could get
#: wrong about WHICH words to use is already impossible, so the prompt
#: only has to say what the sentence is for. /no_think matches every
#: other non-router call in this codebase.
PROMPT = """/no_think
These notes were written at different times and all say something about
the same thing. Write ONE sentence that says everything they say,
keeping every detail: every model number, every quantity, every name.

Do not add anything. Do not leave anything out. Do not comment on the
notes -- the sentence replaces them and will be read on its own, by
someone who will never see this list.

Notes about {subject}:
{sources}

The sentence:"""


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


def _clean(raw: str) -> str:
    """
    Same treatment the four graphs give their syntheses, and the same
    one compaction's llm_summary strategy has a paragraph about. No
    grammar means the router's, so the model can answer with a routing
    decision -- and here that decision would not be shown to anyone
    who could see it was wrong, it would be WRITTEN INTO THE STORE as
    a fact about the user.
    """
    text = strip_think_blocks(raw)
    unwrapped = try_unwrap_router_json(text, "aggregate")
    return (unwrapped if unwrapped is not None else text).strip()


def run_pass(conn, *, max_df: float, min_sources: int) -> list[dict]:
    """
    Aggregate what can be aggregated, and report what happened.

    NOTHING IS WRITTEN UNLESS IT IS GOING TO REPLACE SOMETHING. Every
    gate is a comparison between texts, so all of them run BEFORE the
    entry is stored:

      closure   a word from nowhere -- the subject is abandoned.
      coverage  a source whose detail went missing stays active.
      quorum    fewer than *min_sources* foldable sources left, and
                the aggregate would be one more overlapping line in
                the block rather than one fewer.
      budget    an aggregate no shorter than what it folds makes the
                block bigger, which is the opposite of the job.

    Returns one dict per subject, whether it was written or not. The
    caller logs it; bench/rag_aggregate.py prints it.

    NEVER RAISES on a model failure. The pass runs after a compaction
    that has already happened and already committed; a provider that
    is down must not turn that into an error the user reads.
    """
    entries = rag.hot_entries(conn)
    if len(entries) < min_sources:
        return []

    corpus = rag.list_entries(conn, limit=_CORPUS_ENTRIES)
    freq = frequencies(corpus)
    limit = ceiling(len(corpus), max_df)

    report: list[dict] = []
    for subject in subjects(entries, freq, limit, min_sources):
        report.append(_one(conn, subject, freq, limit, min_sources))
    return report


def _one(conn, subject: Subject, freq: Counter, limit: int, min_sources: int) -> dict:
    sources = subject.sources()
    outcome: dict = {"subject": subject.term, "sources": subject.ids}

    try:
        raw = call_llm(
            PROMPT.format(subject=subject.term, sources="\n".join(sources)),
            grammar=grammar(sources, freq, limit),
        )
    except ProviderError as e:
        log.warning(
            "aggregate: %r not written, the provider failed (%s)", subject.term, e
        )
        return {**outcome, "written": None, "refused": "provider"}

    written = _clean(raw)
    if not written:
        return {**outcome, "written": None, "refused": "empty"}

    allowed = lexicon(sources, freq, limit)
    strangers = invented(written, allowed)
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

    foldable, held_back = [], {}
    for entry in subject.entries:
        missing = uncovered(written, entry["content"], freq, limit)
        if missing:
            held_back[entry["id"]] = missing
        else:
            foldable.append(entry)

    if len(foldable) < min_sources:
        return {
            **outcome,
            "written": written,
            "refused": "quorum",
            "held_back": held_back,
        }

    cost = estimate_tokens(render_lines([written]))
    saved = estimate_tokens(render_lines([e["content"] for e in foldable]))
    if cost >= saved:
        return {
            **outcome,
            "written": written,
            "refused": "budget",
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

    SWALLOWS EVERYTHING. A compaction that has already committed must
    not be turned into an error the user reads because a model call
    failed afterwards.
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
