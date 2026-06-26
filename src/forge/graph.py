"""
Forge Graph Execution Engine (v3).

A Graph is a set of Nodes connected by conditional Edges. Execution
passes AgentState from node to node until a terminal node is reached
or MAX_STEPS is exhausted.

Design goals (same as the rest of Forge):
- No external dependencies.
- Composable with the existing orchestrator: the Orchestrator stays
  the entry point; the graph is an optional execution model for
  multi-step flows.
- Typed at every boundary: state in, state out.
- Observable: every node execution records a TraceStep.

Vocabulary (borrowed from LangGraph, simplified):
  Node   — a unit of work: receives AgentState, returns AgentState.
  Edge   — a directed connection from one node to another.
  Graph  — a named set of nodes + edges with one entry point.

Conditional edges: a node can declare multiple outgoing edges with
a condition function (state) -> bool. The first edge whose condition
is True is taken. An unconditional edge (condition=None) always fires.
A node with no outgoing edge is terminal.

Example — a two-node router/fallback graph:

    graph = Graph("simple")
    graph.add_node("router", router_node)
    graph.add_node("fallback", fallback_node)
    graph.add_edge("router", "fallback",
                   condition=lambda s: not s.ok)
    result = graph.run("hello")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from forge.errors import LoopGuardError
from forge.logger import log
from forge.types import AgentState, TraceStep


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

NodeFn = Callable[[AgentState], AgentState]


@dataclass
class Edge:
    """A directed connection between two nodes, with an optional condition."""
    to_node: str
    condition: Optional[Callable[[AgentState], bool]] = None

    def matches(self, state: AgentState) -> bool:
        return self.condition is None or self.condition(state)


@dataclass
class Node:
    """
    A named unit of work. fn receives AgentState and returns AgentState.

    The function must:
    - never raise (catch internally and mark state.ok=False)
    - always return the state object (mutated or replaced)
    """
    name: str
    fn: NodeFn
    edges: list[Edge] = field(default_factory=list)

    def execute(self, state: AgentState) -> AgentState:
        started = time.monotonic()
        log.event("graph.node.enter", node=self.name)

        ts = TraceStep(step=state.steps_taken + 1)
        ts.decision_tool = self.name
        state.steps_taken += 1

        try:
            state = self.fn(state)
            ts.tool_ok = state.ok
        except Exception as e:  # noqa: BLE001
            log.error("node %r raised: %s", self.name, e)
            state.ok = False
            state.error = str(e)
            ts.tool_ok = False
            ts.tool_error = str(e)

        ts.duration_ms = int((time.monotonic() - started) * 1000)
        state.trace.append(ts)
        log.event("graph.node.exit", node=self.name, ok=state.ok,
                  duration_ms=ts.duration_ms)
        return state

    def next_node(self, state: AgentState) -> Optional[str]:
        """Return the name of the next node to execute, or None if terminal."""
        for edge in self.edges:
            if edge.matches(state):
                return edge.to_node
        return None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

class Graph:
    """
    A named, runnable directed graph of Nodes.

    Usage:
        graph = Graph("my_graph")
        graph.add_node("step_a", fn_a)
        graph.add_node("step_b", fn_b)
        graph.add_edge("step_a", "step_b")           # unconditional
        graph.add_edge("step_b", "step_a",            # conditional loop
                       condition=lambda s: not s.ok)
        result = graph.run("user input")
    """

    def __init__(self, name: str, max_steps: int = 10):
        self.name = name
        self.max_steps = max_steps
        self._nodes: dict[str, Node] = {}
        self._entry: Optional[str] = None

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def add_node(self, name: str, fn: NodeFn) -> "Graph":
        """Register a node. The first node added becomes the entry point."""
        self._nodes[name] = Node(name=name, fn=fn)
        if self._entry is None:
            self._entry = name
        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        condition: Optional[Callable[[AgentState], bool]] = None,
    ) -> "Graph":
        """Add an edge from from_node to to_node, optionally conditional."""
        if from_node not in self._nodes:
            raise ValueError(f"node {from_node!r} not registered")
        if to_node not in self._nodes:
            raise ValueError(f"node {to_node!r} not registered")
        self._nodes[from_node].edges.append(Edge(to_node=to_node,
                                                  condition=condition))
        return self

    def set_entry(self, name: str) -> "Graph":
        """Override the default entry point."""
        if name not in self._nodes:
            raise ValueError(f"node {name!r} not registered")
        self._entry = name
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, user_input: str,
            history: Optional[list[dict]] = None) -> AgentState:
        """
        Execute the graph starting from the entry node.
        Returns the final AgentState (use .to_result() for AgentResult).
        """
        if not self._entry:
            raise ValueError("graph has no nodes")

        state = AgentState(
            user_input=user_input,
            max_steps=self.max_steps,
            history=history or [],
        )

        current = self._entry
        visited_sequence: list[str] = []

        for _ in range(self.max_steps):
            node = self._nodes.get(current)
            if node is None:
                log.error("graph: unknown node %r", current)
                state.ok = False
                state.error = f"unknown node: {current!r}"
                break

            visited_sequence.append(current)
            state = node.execute(state)

            # Cycle detection: same node twice in a row with same
            # ok/error state means we're spinning.
            if len(visited_sequence) >= 2:
                if (visited_sequence[-1] == visited_sequence[-2]
                        and state.ok == (visited_sequence[-1] == current)):
                    err = LoopGuardError(
                        f"graph cycle detected at node {current!r}"
                    )
                    log.error(str(err))
                    state.ok = False
                    state.error = str(err)
                    break

            nxt = node.next_node(state)
            if nxt is None:
                # Terminal node reached
                log.event("graph.terminal", node=current)
                break
            current = nxt
        else:
            err = LoopGuardError(
                f"graph {self.name!r}: max_steps={self.max_steps} exhausted"
            )
            log.error(str(err))
            state.ok = False
            state.error = str(err)

        if state.final_output is None:
            state.final_output = ""
        return state
