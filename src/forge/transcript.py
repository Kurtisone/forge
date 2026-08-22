"""
One definition of how a conversation is written down, and where one
exchange ends.

Two callers need the same answer to that question and, until this
module existed, were free to disagree. compaction.py renders evicted
messages and hands the text to the vector store; the migration in
deploy/rag_resplit.py has to take a block that was rendered weeks ago
and cut it the same way. A second copy of the boundary rule would put
the live path and the migration out of step, and the symptom would be
a store whose old and new entries are sliced differently -- invisible
from the outside, surfacing only as a retrieval distance nobody can
account for.

WHY AN EXCHANGE, AND NOT THE WHOLE BLOCK

Measured on the Deck against the real store on 2026-08-22. The
question "Quels outils as-tu accès ?" sits at distance 0.9386 from the
entry that answers it nearly word for word -- because that entry is a
whole compacted block. A short fact answering "Le serveur de test
tourne sur quel port ?" sits at 0.671. Burying a sentence in a long
block costs roughly 0.27 of distance, which is larger than the entire
gap that separated a hit from a miss that day (0.0422).

The cause is in rag._embed: text past EMBEDDING_MAX_CHARS is split,
each chunk embedded, and the chunk vectors AVERAGED into one. The mean
of a dozen unrelated chunks is close to no question in particular. The
fix is not a better average -- it is to stop asking one vector to
stand for a dozen subjects.

WHY AN EXCHANGE, AND NOT A SINGLE MESSAGE

The other direction fails too. A user turn alone retrieves a question
rather than an answer, and an assistant reply on its own has lost its
subject: "oui, 8080" answers nothing when it is all you get back. A
user turn plus whatever answered it is the smallest unit that still
stands on its own.

KNOWN LIMIT, accepted. `split` finds turn boundaries by looking for a
role prefix at the start of a line, so a message whose own content
contains a line beginning with "user: " produces a boundary in the
wrong place. Nothing is lost when that happens -- the text is still
stored, in two pieces instead of one -- and the alternative (a
structured archive format) would not be readable by the blocks already
in the store, which is the case this has to serve.
"""

import re

# Every role forge.memory writes. Kept as one tuple because the regex
# below and any future caller must agree on the closed set: a role
# missing here is not a boundary, so its message silently joins the
# previous turn.
ROLES = ("user", "assistant", "system")

_SPEAKER = re.compile(rf"^({'|'.join(ROLES)}): ", re.MULTILINE)


def render(messages: list[dict]) -> str:
    """Write messages down the way the store has always held them."""
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def blocks(messages: list[dict]) -> list[str]:
    """
    Group messages into retrieval units and render each one.

    A new unit starts at every `user` message; anything that follows
    belongs to it. Messages appearing before the first user turn form a
    unit of their own rather than being dropped or glued to the turn
    after them -- an evicted window does not necessarily begin on a
    user message.
    """
    groups: list[list[dict]] = []
    for m in messages:
        if m.get("role") == "user" or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    return [render(g) for g in groups]


def split(text: str) -> list[str]:
    """
    Cut already-rendered text into the same units `blocks` would have
    produced from the messages it came from.

    This is the inverse used by the migration: entries written before
    the split existed are one long string, and re-slicing them is the
    only way the store's old half ends up shaped like its new half.

    Text with no role prefix at all comes back as a single unit --
    something is in there, and refusing to return it would silently
    delete an entry during a migration. Text before the first prefix
    is returned as its own unit for the same reason.
    """
    marks = list(_SPEAKER.finditer(text))
    if not marks:
        return [text] if text.strip() else []

    pieces: list[tuple[str, str]] = []
    lead = text[: marks[0].start()]
    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        pieces.append((mark.group(1), text[mark.start() : end]))

    units: list[str] = []
    if lead.strip():
        units.append(lead)
    for role, piece in pieces:
        if role == "user" or not units:
            units.append(piece)
        else:
            units[-1] += piece

    # render() joins with exactly one "\n", so a unit that was followed
    # by another ends with that separator. Removing exactly one newline
    # is the faithful inverse; stripping all of them would eat a blank
    # line a message actually ended with.
    return [u.removesuffix("\n") for u in units]
