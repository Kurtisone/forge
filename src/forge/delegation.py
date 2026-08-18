"""
Where a user message meets a waiting job.

This runs BEFORE the router, and that is the whole design. When a job
is waiting on an answer, the next message belongs to it because there
is exactly one waiting job -- a fact, checked in code, not a judgement
the 9B is asked to make. Routing it instead would mean asking the
model "is this an answer to the question you asked, or a new
request?", which is the same class of question that lost to a
deterministic check six times on this repo.

It also costs nothing. An intercepted turn makes no LLM call at all:
no routing, no recall, no compaction. Answering "src/forge" to "which
folder?" should not cost a full prefill on an APU.

The matching is deliberately narrow. Cancellation words are matched
against the WHOLE normalised message, never as substrings -- "n'annule
pas les tests" contains "annule" and must not stop a job. Missing a
cancellation costs the user a second attempt; catching a false one
throws away work they asked for.
"""

import unicodedata

from forge import jobs, runner, spec
from forge.logger import log

#: pending_field value used while a completed spec waits for approval.
#: Not a spec field: it is a question about the spec, not part of it.
CONFIRM = "__confirm__"

_CANCEL_WORDS = frozenset(
    {
        "annule",
        "annuler",
        "annulle",
        "stop",
        "arrete",
        "arreter",
        "laisse tomber",
        "abandonne",
        "oublie",
    }
)

_CONFIRM_WORDS = frozenset(
    {"oui", "ok", "okay", "go", "vas y", "valide", "lance", "c est bon", "parfait"}
)

# A cancellation is a short message. Past this, the word is being used
# inside a sentence about something else.
_MAX_KEYWORD_CHARS = 20


def _normalise(text: str) -> str:
    """Lowercase, unaccented, punctuation-free, single-spaced."""
    stripped = unicodedata.normalize("NFD", text.strip().lower())
    stripped = "".join(c for c in stripped if unicodedata.category(c) != "Mn")
    kept = [c if c.isalnum() or c.isspace() else " " for c in stripped]
    return " ".join("".join(kept).split())


def _is_keyword(text: str, words: frozenset[str]) -> bool:
    if len(text) > _MAX_KEYWORD_CHARS:
        return False
    return _normalise(text) in words


def intercept(user_input: str) -> str | None:
    """
    Handle *user_input* as delegation traffic, or return None to let
    the router do its job.

    None is the common case and has to stay cheap: one read of the
    jobs file when nothing is pending.
    """
    job = jobs.awaiting_user()
    if job is None:
        return _maybe_cancel_running(user_input)

    if _is_keyword(user_input, _CANCEL_WORDS):
        jobs.transition(job.id, jobs.CANCELLED)
        return f"Job {job.id} annulé."

    if job.pending_field == CONFIRM:
        return _handle_confirmation(job, user_input)

    return _fill_field(job, user_input)


def _maybe_cancel_running(user_input: str) -> str | None:
    """
    A bare "annule" with a job in flight stops it.

    Only when something is actually running: outside that, the word
    belongs to the conversation and hijacking it would make Forge
    unable to discuss cancelling anything.
    """
    if not _is_keyword(user_input, _CANCEL_WORDS):
        return None

    live = [
        j
        for j in jobs.all_jobs()
        if j.status in (jobs.READY, jobs.RUNNING) and not j.is_terminal
    ]
    if not live:
        return None
    if len(live) > 1:
        # Same rule as awaiting_user: with two candidates there is no
        # deterministic answer, so the user picks rather than Forge
        # guessing which work to throw away.
        listing = ", ".join(str(j.id) for j in live)
        return f"Plusieurs jobs en cours ({listing}). Lequel annuler ?"

    job = live[0]
    runner.get_runner().cancel(job.id)
    log.info("job %d cancelled from the thread", job.id)
    return f"Job {job.id} annulé."


def _fill_field(job: jobs.Job, user_input: str) -> str:
    """Record the answer and ask the next question, or show the spec."""
    current = spec.Spec(**{k: v for k, v in job.spec.items() if k in spec.FIELD_NAMES})
    current.set(job.pending_field, user_input)

    job.spec = current.to_dict()
    jobs.save(job)

    question = spec.next_question(current)
    if question is not None:
        name, text = question
        jobs.transition(job.id, jobs.AWAITING_USER, pending_field=name)
        return text

    jobs.transition(job.id, jobs.AWAITING_USER, pending_field=CONFIRM)
    return (
        f"{spec.render(current)}\n\n"
        "Je lance ? Réponds « oui » pour valider, « annule » pour abandonner."
    )


def _handle_confirmation(job: jobs.Job, user_input: str) -> str:
    """
    Approval is a yes-list, not an interpretation.

    Anything else re-asks rather than guessing. Reading "pas encore, il
    manque les tests" as approval would queue work on a spec the user
    was in the middle of correcting, and that is the one mistake this
    step exists to prevent.
    """
    if not _is_keyword(user_input, _CONFIRM_WORDS):
        return (
            "Je n'ai pas compris. Réponds « oui » pour lancer le job, "
            "« annule » pour l'abandonner."
        )

    jobs.transition(job.id, jobs.READY)
    runner.get_runner().submit(job.id)
    return f"Job {job.id} lancé."
