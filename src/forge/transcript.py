"""
One definition of how a conversation is written down, where one
exchange ends, and what is worth indexing.

Two callers need the same answers and, until this module existed, were
free to disagree. compaction.py renders evicted messages and hands the
text to the vector store; the migration in deploy/rag_resplit.py has to
take a block that was rendered weeks ago and treat it the same way.

The first version of this module shared only the CUTTING, and the
disagreement moved rather than disappeared: compaction filtered out
compaction pointers and unwrapped router JSON before cutting, and the
migration did neither. The real migration on 2026-08-22 duly wrote a
dozen entries whose entire content is "[59 messages précédents
compactés -- voir mémoire vectorielle #12]", plus a handful of raw
router JSON out of the old entry #9. So the pipeline is shared here
whole, and the two entry points are now the same sequence:

    units(messages)  =  blocks(indexable(messages))
    split(text)      =  units(parse(text))

A test asserts that equality on the same input. Sharing one step out
of three is how the second step drifts.

WHY AN EXCHANGE, AND NOT THE WHOLE BLOCK

Measured on the Deck against the real store on 2026-08-22, then again
after this change. The question "Quels outils as-tu accès ?" sat at
0.9386 from the entry that answers it nearly word for word -- because
that entry was a whole compacted block. Re-sliced, the same question
against the same content sits at 0.7695. "Tu peux me lister mon
matériel" went from 0.8934 to 0.4498. bench/rag_dilution.py isolates
the effect on a planted sentence and puts it at 0.4450.

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

KNOWN LIMIT, accepted. `parse` finds turn boundaries by looking for a
role prefix at the start of a line, so a message whose own content
contains a line beginning with "user: " produces a boundary in the
wrong place. Nothing is lost when that happens -- the text is still
stored, in two pieces instead of one -- and the alternative (a
structured archive format) could not read the blocks already in the
store, which is the case this has to serve.
"""

import re

from forge.text_cleaning import try_unwrap_router_json

# Every role forge.memory writes. Kept as one tuple because the regex
# below and any future caller must agree on the closed set: a role
# missing here is not a boundary, so its message silently joins the
# previous turn.
ROLES = ("user", "assistant", "system")

_SPEAKER = re.compile(rf"^({'|'.join(ROLES)}): ", re.MULTILINE)

# What a pointer left behind by compaction looks like, in the one form
# that has to be recognised again later -- see `indexable`.
#
# Builder and detector live side by side and a test pins them
# together. Two places stating the same string, one of them edited, is
# the drift that already bit router/grammar.py against
# router/prompt.py; here the failure is silent, because a pointer that
# stops matching gets indexed as if it were conversation.
POINTER_RE = re.compile(r"^\[\d+ messages précédents compactés")


def pointer(message_count: int, ids: list[int]) -> str:
    """The history message left in place of an evicted block."""
    if not ids:
        return (
            f"[{message_count} messages précédents compactés -- "
            f"rien d'indexable, aucune entrée mémoire créée]"
        )
    where = (
        f"#{ids[0]}" if len(ids) == 1 else f"#{ids[0]}-#{ids[-1]} ({len(ids)} entrées)"
    )
    return (
        f"[{message_count} messages précédents compactés -- "
        f"voir mémoire vectorielle {where}, cherchable via !recall]"
    )


def render(messages: list[dict]) -> str:
    """
    Write messages down the way the store has always held them.

    A message with no role is written without a prefix, so text that
    arrived without one does not come back with a speaker invented for
    it.
    """
    return "\n".join(
        f"{m['role']}: {m['content']}" if m.get("role") else m["content"]
        for m in messages
    )


def parse(text: str) -> list[dict]:
    """
    Inverse of `render`: recover messages from text already written
    down. Text before the first role prefix, or text with no prefix at
    all, comes back as one message with role None.
    """
    marks = list(_SPEAKER.finditer(text))
    if not marks:
        return [{"role": None, "content": text}] if text.strip() else []

    messages: list[dict] = []
    lead = text[: marks[0].start()]
    if lead.strip():
        messages.append({"role": None, "content": lead.removesuffix("\n")})

    for i, mark in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        body = text[mark.end() : end]
        # render() joins with exactly one "\n", so a message followed by
        # another ends with that separator. Removing exactly one is the
        # faithful inverse; stripping all of them would eat a blank line
        # a message actually ended with.
        messages.append({"role": mark.group(1), "content": body.removesuffix("\n")})

    return messages


def indexable(messages: list[dict]) -> list[dict]:
    """
    Drop what is not conversation, and unwrap what is wearing an
    envelope, before any of it reaches the vector store.

    Both cases were found by reading the real store on 2026-08-22, the
    first day anything could enumerate it without asking it a question.

    A previous compaction pointer is a reference to another entry.
    Indexed, it becomes a memory whose entire content is "N messages
    were compacted, see #12" -- it answers no question and sits at
    middling distance from all of them. The block it points at stays
    reachable through search; only the textual chain is not rebuilt,
    which is the honest trade for not indexing a signpost as if it were
    the road.

    Entry #9 of that store held raw router JSON -- {"tool": "code",
    ...} -- swallowed from an assistant turn by an older version that
    did not unwrap tool output. The envelope is the noise; the content
    inside it is a real answer, so it is unwrapped rather than dropped.

    The pointer check does not look at the role. It did at first, and
    that was a guess about who wrote it: a pointer reaching the
    migration has been through render and parse, and a block whose
    text starts mid-turn can hand it back under whatever role happened
    to precede it. The shape is the evidence, not the speaker.
    """
    kept = []
    for m in messages:
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if POINTER_RE.match(content):
            continue
        unwrapped = try_unwrap_router_json(content, "compaction")
        kept.append({**m, "content": unwrapped if unwrapped is not None else content})
    return kept


def blocks(messages: list[dict]) -> list[str]:
    """
    Group messages into retrieval units and render each one.

    A new unit starts at every `user` message; anything that follows
    belongs to it. Messages appearing before the first user turn form a
    unit of their own rather than being dropped or glued to the turn
    after them -- an evicted window does not necessarily begin on a
    user message.

    Grouping only, no filtering. `units` is what callers want.
    """
    groups: list[list[dict]] = []
    for m in messages:
        if m.get("role") == "user" or not groups:
            groups.append([m])
        else:
            groups[-1].append(m)
    return [render(g) for g in groups]


def units(messages: list[dict]) -> list[str]:
    """What the vector store should hold for these messages."""
    return blocks(indexable(messages))


def split(text: str) -> list[str]:
    """
    The same thing, for text written down before any of this existed.
    The migration's only entry point.
    """
    return units(parse(text))
