"""
The replies Forge produces when it has nothing to say, in one place.

WHY THIS EXISTS

Measured on the Deck on 2026-08-22, against the real store, one day
after compaction started indexing one entry per exchange. Asked "Tu
peux me lister mon matériel ?", the store handed back at rank 1, at
distance 0.4519, an entry whose whole content is that same question
followed by Forge declining to answer it. The entry that actually
holds the hardware came second, at 0.7891.

That is not a bad row that happened to win. It is the shape of the
unit doing exactly what it was built to do. A retrieval unit is a user
turn plus whatever answered it, so the question is IN the text; an
exchange whose reply says nothing is therefore a near-copy of the
question and nothing else, which makes it the closest possible
neighbour of anyone asking it again. The emptier the entry, the better
it matches. Those refusals were already in the store on 2026-08-19,
buried inside whole compacted blocks where they were diluted along
with everything else; the granularity that fixed the dilution is what
made them competitive.

WHAT A CLOSED SET IS WORTH, MEASURED

Read against the real store on 2026-09-12, 195 archived entries: 22 of
them are the assistant refusing, and this module recognised **none**.
Eight were Forge's own fixed strings, from four producers that had
never been registered here -- the research graph's empty search, the
two sysadmin nodes written in code, and the delegation flow re-asking
mid-job. That is what a closed set with no way to discover its own
members costs: it stays correct about what it knows and silently
misses what nobody told it.

The other fourteen are the model's own prose and none of them is
catchable here. Four were recalls, which stopped being indexed at all
on 2026-08-23; the rest are a chat turn declining, or a sysadmin or
research synthesis reporting that the logs and the search results do
not answer the question. That last family is the one worth naming as
open: the run has the material to know -- it collected the logs -- and
nothing in it reports the verdict, so the only trace is a sentence a
model chose.

WHAT THIS CATCHES, AND WHAT IT CANNOT

Only the fixed strings Forge writes itself. `[error] ` is already the
codebase-wide prefix for "this is a failure being surfaced as a
message rather than raised" -- five modules emit it -- so the closed
set below is small and mostly already a convention.

It does NOT catch a refusal the MODEL wrote in its own words ("je n'ai
pas cette information"), which is prose like any other. Nothing
text-based can, and guessing at it with a phrase list would start
dropping real answers. The structural half of the problem is handled
elsewhere, by forge/outcome.py: a run that knows it failed says so,
and the exchange is marked when it is persisted. This module is the
half that works on text -- which is the only half available to
deploy/rag_resplit.py, since every block already in the store was
written down long before any run could mark it.

WHY startswith, AND WHY THAT IS THE SAFE DIRECTION

A real answer could conceivably open with "[error] " -- a sysadmin
synthesis quoting the first line of a log. The two mistakes are not
the same size. A false positive costs one exchange not indexed, out of
a store that holds hundreds. A false negative costs an entry that
outranks the real answer to its own question, which is the bug this
exists for. So the test is deliberately permissive.

DRIFT

Producers import the constants from here rather than repeating the
literals, and a test asserts that what each producer emits is
recognised by is_non_answer(). Two places stating the same string, one
of them edited, is how POINTER_RE nearly went silent -- and the
failure here is silent in the same way: a refusal that stops matching
gets indexed as if it were an answer, and shows up months later as a
retrieval nobody can explain.
"""

# Surfaced failures. "[error] " is the existing convention across
# graphs/recall.py, graphs/research.py, graphs/review.py and
# graphs/sysadmin.py; "[no memory] " and "Tool error: " are the two
# others already in use.
ERROR_PREFIX = "[error] "
NO_MEMORY_PREFIX = "[no memory] "
TOOL_ERROR_PREFIX = "Tool error: "

# graphs/default.py's fallback node, which is not a prefix anyone
# would have guessed: it was found by a test running the producer
# rather than reading the constants, which is the whole argument for
# testing it that way.
SOMETHING_WENT_WRONG_PREFIX = "Something went wrong: "

# Four producers that were writing their own refusal and were not in
# this set, found by reading the real store rather than the code --
# 22 of its 195 archived entries are refusals and this module caught
# NONE of them (2026-09-12). Eight were these.
#
# `[no results] ` is the web's answer being empty, from the research
# graph and the web_search tool. It is deliberately NOT what
# web_search emits when no engine answered: that one is `[error] `,
# because "the web has nothing" and "Forge did not get to look" are
# the same HTTP response and opposite claims.
NO_RESULTS_PREFIX = "[no results] "

# The two sysadmin nodes written entirely in code, for the case where
# a model asked to diagnose the wrong subsystem would have done it
# fluently.
TARGET_MISSED_PREFIX = "[cible introuvable] "
COLLECT_FAILED_PREFIX = "[collecte impossible] "

# The one case where "these logs do not answer the question" needs no
# judgement: there are no logs. Measured 2026-09-12 with
# bench/sysadmin_verdict.py -- asked whether an EMPTY log block
# contained what was needed to answer "pourquoi searxng a redémarré ?",
# the model said yes. See the sysadmin graph's docstring for the other
# seven fixtures and the three arms that failed the same way.
NOTHING_COLLECTED_PREFIX = "[rien à lire] "

# The delegation flow re-asking mid-job. These two are French
# sentences rather than bracketed markers, which is a weaker test and
# is accepted for the reason stated under `startswith` below: a chat
# turn opening with "Je n'ai pas compris." is not an answer the store
# loses anything by dropping, while five copies of the job dialogue
# are five near-copies of a question with no answer in them.
DID_NOT_UNDERSTAND_PREFIX = "Je n'ai pas compris. Réponds"
NOT_YOUR_DECISION_PREFIX = "Je ne peux pas répondre à la place"

# Whole replies, emitted verbatim.
BACKEND_UNAVAILABLE = "The model backend is unavailable."
NOTHING_CLOSE_ENOUGH = "Je n'ai rien d'assez proche en mémoire pour répondre à ça."

_PREFIXES = (
    ERROR_PREFIX,
    NO_MEMORY_PREFIX,
    TOOL_ERROR_PREFIX,
    SOMETHING_WENT_WRONG_PREFIX,
    NO_RESULTS_PREFIX,
    TARGET_MISSED_PREFIX,
    COLLECT_FAILED_PREFIX,
    NOTHING_COLLECTED_PREFIX,
    DID_NOT_UNDERSTAND_PREFIX,
    NOT_YOUR_DECISION_PREFIX,
)
_EXACT = (BACKEND_UNAVAILABLE, NOTHING_CLOSE_ENOUGH)


def is_non_answer(text: str) -> bool:
    """
    Whether this reply is one of Forge's own ways of saying nothing.

    Takes the text of a single assistant turn. Whether a whole exchange
    should be indexed is a question about the group, not this message
    -- see forge.transcript.worth_indexing.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    return stripped.startswith(_PREFIXES) or stripped in _EXACT
