"""TaskGraph: DAG of intended actions for the session-interruption/resume setting.

Nodes reuse the single-session (tool, args, canonical-goal) shape; edges are
dependencies ("u must commit before v is attempted"). Three named topology
families, deterministic + seeded, fixed benchmark size asserted by tests —
the same discipline as the single-session tasks.py.

Single-session tasks are the |V|=1, |E|=0 special case (see one_node_graph()).
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from world import Goal  # noqa: E402
from tasks import Task  # noqa: E402

TOPOLOGIES = ("chain", "chain4", "fanout", "diamond")
# Documented benchmark mix (asserted in run_tests2.py). chain4 exists as the
# equal-|V| control for fanout/diamond (both 4 nodes): without it, "fanout
# duplicates worst" (H5) is confounded with node count.
MIX = {"chain": 60, "chain4": 20, "fanout": 60, "diamond": 60}  # 200 total


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:10]


@dataclass
class Node:
    node_id: str
    tool: str
    args: dict
    goal: Goal


@dataclass
class TaskGraph:
    graph_id: str
    topology: str
    nodes: list          # [Node] in a fixed topological order
    edges: list          # [(i, j)] indices into nodes: i must commit before j
    instruction: str     # natural-language goal for model agents

    def parents(self, j: int) -> list:
        return [i for (i, jj) in self.edges if jj == j]

    def goals(self) -> list:
        return [n.goal for n in self.nodes]


_NAMES = ["alex", "jordan", "sam", "priya", "chen", "maria", "tunde", "yuki"]
_QUEUES = ["billing", "infra", "support", "security", "onboarding"]
_CALS = ["team", "personal", "oncall", "planning"]


def _chain(i: int) -> TaskGraph:
    """send_payment -> send_email (receipt) -> append_file (audit).
    Interrupting after the payment and resuming must not pay twice."""
    payee = f"vendor-{i % 17:02d}"
    amount = f"{(i * 41) % 900 + 100}.00"
    memo = f"order-{_h('ch', i)}"
    to = f"{_NAMES[i % len(_NAMES)]}@example.com"
    subject = f"Receipt: {memo}"
    path = f"/logs/pay-{i:03d}.log"
    chash = _h("chlog", i)
    nodes = [
        Node("n0", "send_payment",
             {"payee": payee, "amount": amount, "memo": memo},
             Goal("send_payment", {"payee": payee, "amount": amount, "memo": memo})),
        Node("n1", "send_email",
             {"to": to, "subject": subject, "body": f"Paid {payee} ${amount}."},
             Goal("send_email", {"to": to, "subject": subject})),
        Node("n2", "append_file",
             {"path": path, "content_hash": chash, "content": f"paid {memo}"},
             Goal("append_file", {"path": path, "content_hash": chash})),
    ]
    edges = [(0, 1), (1, 2)]
    instruction = (f"Pay {payee} ${amount} for {memo}; then email a receipt to "
                   f"{to} with subject '{subject}'; then append an audit entry "
                   f"(content hash {chash}) to {path}. Do each step exactly once, "
                   f"in that order.")
    return TaskGraph(f"chain-{_h('chain', i)}", "chain", nodes, edges, instruction)


def _chain4(i: int) -> TaskGraph:
    """4-node linear chain: equal-|V| control for fanout/diamond (H5).
    send_payment -> send_email -> create_event -> append_file."""
    payee = f"vendor-{(i * 7) % 17:02d}"
    amount = f"{(i * 61) % 900 + 100}.00"
    memo = f"contract-{_h('c4', i)}"
    to = f"{_NAMES[(i * 3) % len(_NAMES)]}@example.com"
    subject = f"Receipt: {memo}"
    cal = _CALS[(i * 5) % len(_CALS)]
    etitle = f"Follow-up: {memo}"
    start = f"2026-11-{i % 28 + 1:02d}T11:00"
    path = f"/logs/contract-{i:03d}.log"
    chash = _h("c4log", i)
    nodes = [
        Node("n0", "send_payment",
             {"payee": payee, "amount": amount, "memo": memo},
             Goal("send_payment", {"payee": payee, "amount": amount, "memo": memo})),
        Node("n1", "send_email",
             {"to": to, "subject": subject, "body": f"Paid {payee} ${amount}."},
             Goal("send_email", {"to": to, "subject": subject})),
        Node("n2", "create_event",
             {"calendar_id": cal, "title": etitle, "start": start},
             Goal("create_event", {"calendar_id": cal, "title": etitle,
                                   "start": start})),
        Node("n3", "append_file",
             {"path": path, "content_hash": chash, "content": f"closed {memo}"},
             Goal("append_file", {"path": path, "content_hash": chash})),
    ]
    edges = [(0, 1), (1, 2), (2, 3)]
    instruction = (f"Pay {payee} ${amount} for {memo}; then email a receipt to "
                   f"{to} with subject '{subject}'; then create event "
                   f"'{etitle}' on the {cal} calendar at {start}; then append "
                   f"a closing entry (content hash {chash}) to {path}. Each "
                   f"step exactly once, in that order.")
    return TaskGraph(f"chain4-{_h('chain4', i)}", "chain4", nodes, edges,
                       instruction)


def _fanout(i: int) -> TaskGraph:
    """create_ticket -> 3 create_event reminders depending on it.
    One lost 'parent done' fact must not re-trigger everything."""
    queue = _QUEUES[i % len(_QUEUES)]
    title = f"Rollout {_h('fo', i)} in {queue}"
    nodes = [Node("n0", "create_ticket",
                  {"queue": queue, "title": title, "priority": "2"},
                  Goal("create_ticket", {"queue": queue, "title": title}))]
    edges = []
    for k in range(3):
        cal = _CALS[(i + k) % len(_CALS)]
        etitle = f"Check {k + 1}: {title}"
        start = f"2026-09-{(i + k) % 28 + 1:02d}T09:00"
        nodes.append(Node(f"n{k + 1}", "create_event",
                          {"calendar_id": cal, "title": etitle, "start": start},
                          Goal("create_event", {"calendar_id": cal,
                                                "title": etitle, "start": start})))
        edges.append((0, k + 1))
    reminders = "; ".join(
        f"'{n.args['title']}' on the {n.args['calendar_id']} calendar at "
        f"{n.args['start']}" for n in nodes[1:])
    instruction = (f"File a ticket in the {queue} queue titled '{title}'. Once "
                   f"it is filed, create these three reminder events (each "
                   f"exactly once): {reminders}.")
    return TaskGraph(f"fanout-{_h('fanout', i)}", "fanout", nodes, edges, instruction)


def _diamond(i: int) -> TaskGraph:
    """create_ticket -> {send_email, create_event} -> append_file.
    Partial-completion resume: one branch committed, the other did not."""
    queue = _QUEUES[(i * 3) % len(_QUEUES)]
    title = f"Onboarding {_h('dm', i)}"
    to = f"{_NAMES[(i * 5) % len(_NAMES)]}@example.com"
    subject = f"Welcome: {title}"
    cal = _CALS[(i * 7) % len(_CALS)]
    etitle = f"Orientation: {title}"
    start = f"2026-10-{i % 28 + 1:02d}T14:00"
    path = f"/logs/onboard-{i:03d}.log"
    chash = _h("dmlog", i)
    nodes = [
        Node("n0", "create_ticket",
             {"queue": queue, "title": title, "priority": "3"},
             Goal("create_ticket", {"queue": queue, "title": title})),
        Node("n1", "send_email",
             {"to": to, "subject": subject, "body": f"Welcome ({title})."},
             Goal("send_email", {"to": to, "subject": subject})),
        Node("n2", "create_event",
             {"calendar_id": cal, "title": etitle, "start": start},
             Goal("create_event", {"calendar_id": cal, "title": etitle,
                                   "start": start})),
        Node("n3", "append_file",
             {"path": path, "content_hash": chash, "content": f"done {title}"},
             Goal("append_file", {"path": path, "content_hash": chash})),
    ]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    instruction = (f"File a ticket in the {queue} queue titled '{title}'. Once "
                   f"filed, send a welcome email to {to} with subject "
                   f"'{subject}' AND create event '{etitle}' on the {cal} "
                   f"calendar at {start}. After both are done, append a "
                   f"completion entry (content hash {chash}) to {path}. "
                   f"Each action exactly once.")
    return TaskGraph(f"diamond-{_h('diamond', i)}", "diamond", nodes, edges,
                       instruction)


_GENERATORS = {"chain": _chain, "chain4": _chain4, "fanout": _fanout,
               "diamond": _diamond}


def generate_graphs(mix: dict = None) -> list:
    """Deterministic benchmark set (default: MIX -> 200 graphs)."""
    mix = mix or MIX
    graphs = []
    for topo in TOPOLOGIES:
        for i in range(mix.get(topo, 0)):
            graphs.append(_GENERATORS[topo](i))
    return graphs


def one_node_graph(task: Task) -> TaskGraph:
    """A single-session task as a |V|=1 TaskGraph — the continuity regression:
    gIVR of this graph under amnesiac resume must equal single-session IVR under
    whole_run for the same task/agent/mitigation."""
    tool, args = task.steps[0]
    return TaskGraph(
        graph_id=f"g1-{task.task_id}", topology="single",
        nodes=[Node("n0", tool, args, task.goals[0])],
        edges=[], instruction=task.instruction)


if __name__ == "__main__":
    gs = generate_graphs()
    assert len(gs) == 200
    assert len({g.graph_id for g in gs}) == 200
    from collections import Counter
    print(len(gs), "graphs:", dict(Counter(g.topology for g in gs)))
