"""
Tests that every graph-based tool publishes its internal steps.

The debt: `sysadmin` showed the user a step-by-step timeline and every
other graph tool showed one opaque box -- while `review` ran three
nodes, `research` three, `recall` two and `delegate` several, each
already recorded as a TraceStep by graph.py's Node.execute and each
dropped at the wrapper boundary.

Nothing was missing except the four calls to publish.
"""

import pytest

from forge import subtrace


class _Step:
    def __init__(self, name, ok=True, ms=7):
        self.decision_tool = name
        self.tool_ok = ok
        self.duration_ms = ms


class _State:
    def __init__(self, *names):
        self.trace = [_Step(n) for n in names]
        self.context = {}
        self.final_output = "sortie"


def test_node_names_alone_produce_a_usable_timeline():
    """
    `details` is optional on purpose: node names are already meaningful
    ("read_file", "search", "synthesize"), so a graph that supplies
    nothing still gets a real timeline rather than nothing at all.
    """
    steps = subtrace.from_state(_State("read_file", "llm_review"))

    assert [s["label"] for s in steps] == ["read_file", "llm_review"]
    assert all(s["detail"] == "" for s in steps)
    assert all(s["ok"] for s in steps)
    assert steps[0]["duration_ms"] == 7


def test_a_detail_is_only_computed_for_nodes_that_ran():
    """
    Callables rather than strings, because a detail is computed from
    the finished state and an eager dict would have to build one for
    nodes the run never reached.
    """
    exploded = []

    def boom():  # pragma: no cover - must not run
        exploded.append(True)
        raise AssertionError("detail computed for a node that never ran")

    steps = subtrace.from_state(
        _State("search"), {"search": lambda: "ok", "never_reached": boom}
    )

    assert not exploded
    assert steps[0]["detail"] == "ok"


@pytest.mark.parametrize(
    "module_name",
    ["review", "research", "recall", "sysadmin", "delegate"],
)
def test_every_graph_tool_publishes(module_name):
    """
    The regression. A graph tool that stops publishing goes back to
    being one opaque box, and nothing else would notice.
    """
    import importlib

    source = importlib.import_module(f"forge.graphs.{module_name}").__file__

    with open(source, encoding="utf-8") as handle:
        text = handle.read()

    assert "subtrace.publish(" in text, (
        f"graphs/{module_name}.py runs a graph and throws its own trace "
        f"away at the wrapper boundary"
    )
