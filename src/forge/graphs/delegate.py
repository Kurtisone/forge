"""
Forge delegate graph: draft a spec -> open a job -> ask or confirm.

A deterministic sequence, not a router-driven chain, for the reason
graphs/research.py and graphs/recall.py already document: this model
class cannot reliably decide "I have enough now, do the next thing".
The router makes exactly ONE decision here (call "delegate"), and
everything after it is fixed.

The single LLM call is an ACCELERATOR, not a requirement. It reads
whatever the user already said into the spec fields, so that someone
who wrote a complete request in one sentence is not then interviewed
about it. If it fails -- bad JSON, provider down, a model that fills
nothing -- the job is still created and delegation.py interviews from
the first field. Nothing here is a dead end, because the interview
alone can produce a complete spec.

That call is the first in Forge to run under a grammar that is not
the router's (v3.13 lot 1). Which is why it does not need
try_unwrap_router_json() the way every other graph does: those exist
precisely because the router grammar used to apply to every call in
the process.

The prompt's one hard rule is that an unknown field stays EMPTY. An
invented workspace path is the worst possible output here -- it reads
as a complete spec, skips the question that would have caught it, and
sends an implementer at the wrong directory. An empty field costs one
question.

Nodes:
  draft_node  -- one grammar-constrained LLM call filling what it can
  open_node   -- creates the job, asks the first missing question or
                 shows the spec for approval

Usage (Python):
  from forge.graphs.delegate import run
  print(run("répare le cache KV dans src/forge"))
"""

from forge import delegation, jobs, spec
from forge.errors import ForgeError, ProviderError
from forge.graph import Graph
from forge.llm import call_llm
from forge.logger import log
from forge.types import AgentState

_PROMPT = """Tu prépares une fiche de tâche à confier à un développeur.

Demande de l'utilisateur :
{request}

Remplis ces champs à partir de la demande, en JSON :
{fields}

RÈGLE ABSOLUE : si la demande ne dit rien sur un champ, laisse-le vide
("" ou []). N'invente jamais un chemin, un critère ou une contrainte.
Un champ vide sera demandé à l'utilisateur ; un champ inventé ne sera
jamais vérifié."""


def draft_node(state: AgentState) -> AgentState:
    """One constrained call; an empty spec is a valid outcome."""
    prompt = _PROMPT.format(
        request=state.user_input.strip(),
        fields=spec.prompt_fields(),
    )

    try:
        raw = call_llm(prompt, grammar=spec.build_spec_grammar())
        drafted = spec.parse(raw)
    except (ProviderError, ForgeError) as e:
        # Degraded, not failed: the interview can fill every field on
        # its own, so a model that cannot draft costs questions, not
        # the feature.
        log.warning("delegate: could not draft a spec (%s), interviewing instead", e)
        drafted = spec.Spec()

    state.context["spec"] = drafted
    return state


def open_node(state: AgentState) -> AgentState:
    """
    Create the job, then hand the conversation to delegation.py.

    Deliberately reusing the same question/confirmation code the
    interception uses rather than writing a first-turn variant of it:
    two paths asking the questions would be two places for the
    wording, the ordering and the CONFIRM sentinel to drift apart.
    """
    drafted: spec.Spec = state.context["spec"]

    existing = jobs.awaiting_user()
    if existing is not None:
        # jobs.py allows only one job to wait at a time, and it is
        # right to: the next message has to belong to exactly one of
        # them. Refusing here turns that invariant into a sentence
        # instead of a stack trace.
        state.final_output = (
            f"Le job {existing.id} attend déjà une réponse. "
            "Réponds-lui ou annule-le avant d'en ouvrir un autre."
        )
        state.final_tool = "delegate"
        return state

    job = jobs.create(drafted.to_dict())
    question = spec.next_question(drafted)

    if question is not None:
        name, text = question
        jobs.transition(job.id, jobs.AWAITING_USER, pending_field=name)
        state.final_output = f"Job {job.id} ouvert.\n\n{text}"
    else:
        jobs.transition(job.id, jobs.AWAITING_USER, pending_field=delegation.CONFIRM)
        state.final_output = (
            f"Job {job.id} ouvert.\n\n{spec.render(drafted)}\n\n"
            "Je lance ? Réponds « oui » pour valider, « annule » pour abandonner."
        )

    state.final_tool = "delegate"
    return state


def build() -> Graph:
    graph = Graph(name="delegate", max_steps=4)
    graph.add_node("draft", draft_node)
    graph.add_node("open", open_node)
    graph.add_edge("draft", "open")
    graph.set_entry("draft")
    return graph


def run(request: str) -> str:
    """Draft a spec for *request* and open a job for it."""
    state = build().run(request, initial_context={"request": request})
    return state.final_output or ""
