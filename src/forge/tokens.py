"""
Local token estimation.

The delegated counter (types.Usage, filled by each provider and summed
in metrics.py) is exact, but it only exists AFTER a call. Two things
need a number before any call happens:

  - compaction, which decides whether to evict history at persist time,
    with only the history list in hand and the next turn's prompt not
    yet built;
  - the context gauge, which has to say something on page load.

Neither can wait for a backend to answer. That is the real reason this
module exists -- covering a backend that reports no counts at all is a
secondary benefit.

It follows that estimation must be pure Python with no network: putting
llama-server's exact /tokenize on that path would make persisting a
turn depend on the inference server being reachable. A rough number
that always works beats an exact one that can hang.

CALIBRATION (measured on the Deck, 2026-08-16, by pairing len(prompt)
with the prompt_tokens llama-server reported for that same call):

    router prompt, 4 log samples ......... 4.00 - 4.01 chars/token
    sysadmin synthesis (logs + French) ... 4.00
    review synthesis (code + English) .... 4.01
    research synthesis (English) ......... 4.01
    recall synthesis (short French) ...... 4.16
    recall in production (French memory
      entries, 2977 chars / 810 tokens) .. 3.67

Eleven samples, range 3.67-4.16. The bulk sits at 4.00 with surprising
regularity; both outliers are dense French, where accented characters
cost more under BPE.

CHARS_PER_TOKEN is deliberately set BELOW the lowest observed ratio
rather than at the mean. The estimate decides when to shed context, so
the two errors are not symmetric: overestimating means compacting a
little early, underestimating means overflowing the window. A ratio of
4.0 would have understated the production recall prompt by 9%, in
exactly the wrong direction.

3.6 rather than the 3.67 that sample actually measured, because sitting
exactly on the observed minimum leaves no room at all: the next French
prompt slightly denser than that one understates again. The first draft
of this module used 3.7 -- a tidy rounding of 3.67 that is on the WRONG
side of it, and understated that very sample by 5 tokens. The
calibration test in tests/test_tokens.py caught it; the round number
looked right and was not.

The number is checked against reality on every call -- see
llm.call_llm, which knows both len(prompt) and the reported
prompt_tokens and logs when they disagree. That turns a heuristic into
a watched heuristic: if the ratio drifts (different model, different
dominant language), the log says so instead of a context overflow
saying it later.
"""

CHARS_PER_TOKEN = 3.6

# Every message costs a little more than its text: a role prefix, a
# separator, whatever the template puts around it. Measured against
# _format_history the overhead is a handful of tokens per message;
# 4 is a round number on the safe side.
PER_MESSAGE_OVERHEAD = 4


def estimate_tokens(text: str) -> int:
    """
    Rough token count for *text*. Never raises, never touches the
    network, and is intentionally biased high (see CHARS_PER_TOKEN).
    """
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN) + 1


def estimate_messages(messages: list[dict]) -> int:
    """
    Rough token count for a history block, counting the per-message
    framing that estimate_tokens alone would miss.

    Accepts anything dict-shaped with a "content" -- a malformed entry
    contributes its overhead and nothing else rather than raising,
    because this runs on the persistence path where a crash would cost
    the user their turn.
    """
    total = 0
    for m in messages:
        content = m.get("content") if isinstance(m, dict) else None
        total += PER_MESSAGE_OVERHEAD
        if isinstance(content, str):
            total += estimate_tokens(content)
    return total
