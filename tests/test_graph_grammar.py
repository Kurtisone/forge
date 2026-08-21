"""
Which calls constrain the model, and which deliberately do not.

The reasoning here was wrong once, in the direction that looks
obviously right, so it is written down rather than left to the next
reader's judgement.

recall, review, research and sysadmin all call call_llm(prompt) with
no grammar, so providers/llama_cpp._grammar_for() gives them the
ROUTER's -- while their prompts spend paragraphs asking for plain
text. That is exactly what a bug looks like, and it is why the
warning "model wrapped a substantive answer in router-style JSON" has
been in the logs since v3.11.

Measured on the Deck, 2026-08-21, by giving each of them a prose
grammar instead (bench/prose_grammar_ab.py reproduces it):

    graph      router arm            prose arm
    recall     23 tok, correct       '<answer>', 8 chars, nothing else
    review     72 tok, found the bug 'NO_THINK:', 5 tokens
    research   86 tok, correct       1536 tok, hit n_predict, output
                                     was the prompt read back
    sysadmin  164 tok, correct       '<analysis>' block, 314 tok

The router grammar was never only stopping JSON. It was suppressing
the scaffolding this model reaches for, and it was the only hard
TERMINATOR in the loop -- an object cannot run past its closing brace,
free prose runs to n_predict. The graph prompts have never been
written for unconstrained decoding.

So the wrapping is not a defect to remove. It is a shape the graphs
know how to undo, and try_unwrap_router_json() is the seam that makes
the fallback safe rather than a workaround for it.
"""

import ast
import pathlib

_SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Callers that want a routing decision, where the fallback is not just
# safe but exactly right.
_ROUTER_CALLERS = {"forge/orchestrator.py", "forge/graphs/default.py"}

# The one caller that names its own grammar and is right to. delegate
# asks for a spec under spec.build_spec_grammar() -- a CLOSED shape
# with required fields and a terminator, which is what the router
# grammar also is. That is the difference from the prose callers
# below: constraining the model to a different structure works,
# unconstraining it does not.
_OWN_GRAMMAR_CALLERS = {"forge/graphs/delegate.py"}

# Callers whose prompt asks for prose and which take the fallback
# ANYWAY, on the measurement above. Each one must undo the shape it
# gets -- which is what the second test checks.
_PROSE_CALLERS = {
    "forge/graphs/recall.py",
    "forge/graphs/review.py",
    "forge/graphs/research.py",
    "forge/graphs/sysadmin.py",
    "forge/compaction.py",
}


def _call_llm_sites():
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if name == "call_llm":
                yield path.relative_to(_SRC).as_posix(), node


def test_no_call_site_has_appeared_that_this_file_has_not_considered():
    """
    A new module calling the model silently inherits the router's
    grammar. Whether that is right depends on what its prompt asks
    for, and the answer is not obvious in either direction -- so the
    failure mode this closes is nobody deciding.
    """
    known = _ROUTER_CALLERS | _PROSE_CALLERS | _OWN_GRAMMAR_CALLERS
    unknown = sorted({module for module, _ in _call_llm_sites()} - known)
    assert not unknown, (
        "these call the model and are listed in neither set, so they take "
        f"the router grammar by accident rather than on purpose: {unknown}. "
        "Read this file's header before adding a grammar to them."
    )


def test_the_prose_callers_still_take_the_router_grammar():
    """
    The regression this locks down is not the original bug. It is the
    FIX for the original bug, which was written, tested, deployed, and
    measured worse than what it replaced. Adding grammar= back here
    without a bench run that says otherwise reintroduces empty recall
    answers.
    """
    constrained = sorted(
        module
        for module, node in _call_llm_sites()
        if module in _PROSE_CALLERS
        and (any(kw.arg == "grammar" for kw in node.keywords) or len(node.args) >= 2)
    )
    assert not constrained, (
        f"{constrained} now pass a grammar. On 2026-08-21 that produced "
        "'<answer>' from recall and a 1536-token prompt echo from research. "
        "Run bench/prose_grammar_ab.py before changing this."
    )


def test_every_prose_caller_can_undo_the_shape_it_gets():
    """
    Taking the router grammar is only survivable because each of these
    unwraps it afterwards. A prose caller without that call is the
    version of this bug that reaches the user.
    """
    missing = [
        module
        for module in sorted(_PROSE_CALLERS)
        if "try_unwrap_router_json" not in (_SRC / module).read_text()
    ]
    assert not missing, (
        f"{missing} take the router grammar but never undo it, so a routing "
        "decision reaches the user verbatim."
    )
