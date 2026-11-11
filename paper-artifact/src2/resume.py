"""Resume orchestration + graph-level scoring for the session-resume setting.

Resume-context fidelity levels (what crosses the session seam):
  amnesiac        session 2 sees only the goal graph (single-session whole_run).
  plan_only       session 2 sees session 1's ACKNOWLEDGED progress (its
                  scratchpad), but not ledger truth: the after-commit seam
                  node is committed-but-unacknowledged, so it gets redone.
  ledger_readback session 2 consults the agent-visible list_effects channel
                  and skips nodes whose effect it can see (single-session careful).
  checkpoint      the substrate persists the committed-node set and replays
                  it to session 2 (single-session runtime_receipts, graph scope).

Orchestration is failure-conditioned (single-session discipline): a resume
happens only because session 1 raised SessionInterrupted.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from world import World, normalize_value  # noqa: E402
from session import Session, Interrupt, SessionInterrupted  # noqa: E402

RESUME_CONTEXTS = ("amnesiac", "plan_only", "ledger_readback", "checkpoint")


# ---------------------------------------------------------------------------
# Node <-> ledger matching (scoring side; exact semantic identity)
# ---------------------------------------------------------------------------

def _node_commits(world: World, graph, idx) -> list:
    """Commit seqs of effects semantically matching node idx's goal."""
    node = graph.nodes[idx]
    out = []
    for e in world.ledger.for_task(graph.graph_id):
        if e.tool == node.goal.tool and all(
                normalize_value(e.args.get(k)) == normalize_value(v)
                for k, v in node.goal.canon_args.items()):
            out.append(e.seq)
    return out


def _readback_skip(world: World, graph) -> set:
    """Agent-visible read-back (the list_effects string channel), mirroring
    the single-session careful agent's check: tool name + first canonical value."""
    state = world.call("list_effects", {}, "", graph.graph_id)
    skip = set()
    for idx, node in enumerate(graph.nodes):
        first_val = str(node.goal.canon_args[list(node.goal.canon_args)[0]])
        if node.tool in state and first_val in state:
            skip.add(idx)
    return skip


def _checkpoint_skip(world: World, graph) -> set:
    """Substrate checkpoint: exact committed-node set from the runtime."""
    return {idx for idx in range(len(graph.nodes))
            if _node_commits(world, graph, idx)}


# ---------------------------------------------------------------------------
# Episode = session 1 (+ conditional resume) + scoring
# ---------------------------------------------------------------------------

def run_graph_episode(graph, resume_context: str, mitigation: str,
                      key_posture: str, interrupt: Interrupt = None) -> dict:
    assert resume_context in RESUME_CONTEXTS
    world = World(mitigation=mitigation, failure_mode="none")
    s1 = Session(world, graph, key_posture)
    interrupted = False
    try:
        s1.run(interrupt=interrupt)
    except SessionInterrupted:
        interrupted = True

    if interrupted:  # failure-conditioned resume
        if resume_context == "amnesiac":
            skip = set()
        elif resume_context == "plan_only":
            skip = set(s1.acknowledged)
        elif resume_context == "ledger_readback":
            skip = _readback_skip(world, graph)
        else:  # checkpoint
            skip = _checkpoint_skip(world, graph)
        s2 = Session(world, graph, key_posture)
        s2.run(scope=[i for i in range(len(graph.nodes)) if i not in skip])

    return score_graph(world, graph, interrupted=interrupted,
                       resume_context=resume_context,
                       mitigation=mitigation, key_posture=key_posture,
                       interrupt=interrupt)


def score_graph(world: World, graph, **meta) -> dict:
    """gIVR + completion + order-violation for one episode."""
    violations = world.ledger.violations(graph.graph_id)
    commits = {i: _node_commits(world, graph, i)
               for i in range(len(graph.nodes))}
    completion = all(commits[i] for i in range(len(graph.nodes)))
    # Edge (i, j) is order-violated if any commit of the dependency i
    # occurs AFTER any commit of its dependent j (covers both "child before
    # parent" and "stale parent re-execution after child committed").
    order_violated = any(
        commits[i] and commits[j] and max(commits[i]) > min(commits[j])
        for (i, j) in graph.edges
    )
    row = {
        "graph_id": graph.graph_id, "topology": graph.topology,
        "n_nodes": len(graph.nodes),
        "violations": violations,            # duplicate semantic effects
        "givr_hit": 1 if violations > 0 else 0,
        "completion": 1 if completion else 0,
        "order_violation": 1 if order_violated else 0,
    }
    it = meta.pop("interrupt", None)
    row.update(meta)
    row["interrupt_node"] = it.node_index if it else -1
    row["interrupt_timing"] = it.timing if it else "none"
    return row
