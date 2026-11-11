"""Session runner for TaskGraphs: executes a graph against the single-session
World in dependency order, with a deterministic, armed-once interruption.

The interrupt is the graph-scope analogue of the single-session armed one-shot
failure injection: `after_commit` kills the session after node k's effect
committed but before the session acknowledged it (the resume-seam
ambiguity); `before_commit` kills it before anything at node k committed.
"""
from __future__ import annotations

import hashlib
import os
import sys
import uuid
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from world import World  # noqa: E402

INTERRUPT_TIMINGS = ("before_commit", "after_commit")


class SessionInterrupted(Exception):
    """Session died mid-graph. The orchestrator sees run-level failure."""


@dataclass
class Interrupt:
    node_index: int
    timing: str  # INTERRUPT_TIMINGS

    def __post_init__(self):
        assert self.timing in INTERRUPT_TIMINGS


# ---------------------------------------------------------------------------
# Key postures (single-session client postures, per-node)
# ---------------------------------------------------------------------------

def naive_key(graph, node) -> str:
    """Fresh UUID per call — the unprompted-client posture."""
    return uuid.uuid4().hex


def stable_key(graph, node) -> str:
    """Key derived from graph + node identity: survives sessions/retries."""
    raw = graph.graph_id + node.node_id + node.tool + str(sorted(node.args.items()))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


KEY_POSTURES = {"naive": naive_key, "stable": stable_key}


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """One continuous run over (a scope of) a graph's nodes, in topo order.

    `acknowledged` is what THIS SESSION recorded as done before dying —
    distinct from the ledger's ground truth. An `after_commit` interrupt at
    node k leaves k committed in the ledger but absent from `acknowledged`.
    """

    def __init__(self, world: World, graph, key_posture: str = "naive"):
        self.world = world
        self.graph = graph
        self.key_for = KEY_POSTURES[key_posture]
        self.acknowledged: set = set()

    def run(self, scope=None, interrupt: Interrupt = None) -> None:
        """Execute node indices in `scope` (default: all) in order.
        Raises SessionInterrupted if the armed interrupt fires."""
        scope = list(scope) if scope is not None else range(len(self.graph.nodes))
        for idx in scope:
            node = self.graph.nodes[idx]
            key = self.key_for(self.graph, node)
            if interrupt is not None and idx == interrupt.node_index:
                if interrupt.timing == "before_commit":
                    raise SessionInterrupted(f"interrupted before node {idx}")
                self.world.call(node.tool, node.args, key, self.graph.graph_id)
                raise SessionInterrupted(f"interrupted after node {idx} commit")
            self.world.call(node.tool, node.args, key, self.graph.graph_id)
            self.acknowledged.add(idx)
