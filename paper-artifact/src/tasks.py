"""Deterministic task generator: 320 tasks from 8 templates across 4 families.

Honest counting: each family below is a *template*; the
per_domain parameterizations are clones that vary surface content only.
The benchmark's structural diversity is the template count (8), not the
task count (320) — the paper must report both.

Families:
  atomic         5 templates (email/payments/tickets/files/calendar):
                 one goal side effect per task.
  multi_step     1 template: two dependent effects (file ticket, then send
                 confirmation email referencing it) — a whole-run retry
                 must not duplicate the already-committed first step.
  read_then_act  1 template: instruction explicitly says to check existing
                 effects before acting — tests whether agents follow a
                 stated check-first protocol.
  ambiguous      1 template: goal stated loosely in prose (no exact args
                 given) — tests arg fidelity of model agents and matcher
                 robustness to paraphrase.

Deterministic (seeded) so the benchmark is reproducible and diffable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from world import Goal


@dataclass
class Task:
    task_id: str
    domain: str
    tool: str          # tool of the first effectful step (back-compat)
    args: dict         # args of the first effectful step (back-compat)
    goal: Goal         # first goal (back-compat)
    instruction: str   # natural-language goal, used by model agents
    family: str = "atomic"
    steps: list = field(default_factory=list)   # [(tool, args), ...]
    goals: list = field(default_factory=list)   # [Goal, ...] — ALL must hold
    # k-th valid effectful call that receives the injected failure (World
    # fail_on_call). 1 for every released task; varied only by long chains.
    fail_on_call: int = 1

    @property
    def ordered(self) -> bool:
        """Goals must commit in listed order (step 2 depends on step 1)."""
        return self.family in ("multi_step", "long_chain")

    def __post_init__(self):
        if not self.steps:
            self.steps = [(self.tool, self.args)]
        if not self.goals:
            self.goals = [self.goal]


_NAMES = ["alex", "jordan", "sam", "priya", "chen", "maria", "tunde", "yuki"]
_QUEUES = ["billing", "infra", "support", "security", "onboarding"]
_CALS = ["team", "personal", "oncall", "planning"]


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Atomic family: 5 templates, one effect each
# ---------------------------------------------------------------------------

def _email_tasks(n):
    for i in range(n):
        to = f"{_NAMES[i % len(_NAMES)]}@example.com"
        subject = f"Q3 status update #{i:03d}"
        args = {"to": to, "subject": subject, "body": f"Status body {i}"}
        yield ("email", "send_email", args, {"to": to, "subject": subject},
               f"Send an email to {to} with subject '{subject}'.")


def _payment_tasks(n):
    for i in range(n):
        payee = f"vendor-{i % 17:02d}"
        amount = f"{(i * 37) % 900 + 100}.00"
        memo = f"invoice-{_h('inv', i)}"
        args = {"payee": payee, "amount": amount, "memo": memo}
        yield ("payments", "send_payment", args, dict(args),
               f"Pay {payee} ${amount} for {memo}.")


def _ticket_tasks(n):
    for i in range(n):
        queue = _QUEUES[i % len(_QUEUES)]
        title = f"Incident {_h('t', i)}: retry storm in {queue}"
        args = {"queue": queue, "title": title, "priority": str(i % 4 + 1)}
        yield ("tickets", "create_ticket", args, {"queue": queue, "title": title},
               f"File a ticket in the {queue} queue titled '{title}'.")


def _file_tasks(n):
    for i in range(n):
        path = f"/logs/audit-{i:03d}.log"
        chash = _h("content", i)
        args = {"path": path, "content_hash": chash, "content": f"entry {i}"}
        # instruction must state every canonical field: a model agent only
        # sees the instruction, never task.args (pilot audit 2026-07-11)
        yield ("files", "append_file", args, {"path": path, "content_hash": chash},
               f"Append audit entry {i} (content hash {chash}) to {path}.")


def _calendar_tasks(n):
    for i in range(n):
        cal = _CALS[i % len(_CALS)]
        title = f"Sync {_h('c', i)}"
        start = f"2026-08-{i % 28 + 1:02d}T10:00"
        args = {"calendar_id": cal, "title": title, "start": start}
        yield ("calendar", "create_event", args, dict(args),
               f"Create event '{title}' on the {cal} calendar at {start}.")


_ATOMIC_GENERATORS = (_email_tasks, _payment_tasks, _ticket_tasks,
                      _file_tasks, _calendar_tasks)


# ---------------------------------------------------------------------------
# Structured families: multi_step, read_then_act, ambiguous
# ---------------------------------------------------------------------------

def _multi_step_tasks(n):
    """Two dependent effects: file a ticket, then email a confirmation that
    references it. A whole-run retry after step 2's failure must not
    re-commit step 1."""
    for i in range(n):
        queue = _QUEUES[i % len(_QUEUES)]
        title = f"Escalation {_h('ms', i)} in {queue}"
        to = f"{_NAMES[i % len(_NAMES)]}@example.com"
        subject = f"Filed: {title}"
        ticket_args = {"queue": queue, "title": title, "priority": "2"}
        email_args = {"to": to, "subject": subject,
                      "body": f"Ticket '{title}' has been filed in {queue}."}
        yield dict(
            domain="multi", family="multi_step",
            steps=[("create_ticket", ticket_args), ("send_email", email_args)],
            goals=[Goal("create_ticket", {"queue": queue, "title": title}),
                   Goal("send_email", {"to": to, "subject": subject})],
            instruction=(f"File a ticket in the {queue} queue titled '{title}', "
                         f"then send an email to {to} with subject '{subject}' "
                         f"confirming the ticket was filed."),
        )


def _read_then_act_tasks(n):
    """The instruction states the check-first protocol explicitly. Measures
    whether agents follow a stated protocol, separating 'did not know to
    check' from 'knew and still duplicated'."""
    for i in range(n):
        payee = f"vendor-{(i * 3) % 17:02d}"
        amount = f"{(i * 53) % 900 + 100}.00"
        memo = f"renewal-{_h('rta', i)}"
        args = {"payee": payee, "amount": amount, "memo": memo}
        yield dict(
            domain="payments", family="read_then_act",
            steps=[("send_payment", args)],
            goals=[Goal("send_payment", dict(args))],
            instruction=(f"First use list_effects to check whether the payment "
                         f"of ${amount} to {payee} for {memo} has already been "
                         f"made. Only if it has NOT been made, pay {payee} "
                         f"${amount} for {memo}. Never pay twice."),
        )


def _ambiguous_tasks(n):
    """The goal is stated loosely (no quoted exact args). Canonical args are
    still defined for scoring; the normalized matcher (world.normalize_value)
    absorbs formatting variation, and per-episode transcripts allow manual
    audit of residual paraphrase mismatches."""
    for i in range(n):
        name = _NAMES[i % len(_NAMES)]
        to = f"{name}@example.com"
        subject = f"Weekly metrics digest {_h('amb', i)}"
        args = {"to": to, "subject": subject, "body": f"Digest body {i}"}
        yield dict(
            domain="email", family="ambiguous",
            steps=[("send_email", args)],
            goals=[Goal("send_email", {"to": to, "subject": subject})],
            instruction=(f"Make sure {name} (reachable at {to}) receives this "
                         f"week's metrics digest — the one titled "
                         f"{subject}. One copy is enough."),
        )


_STRUCTURED_GENERATORS = (_multi_step_tasks, _read_then_act_tasks,
                          _ambiguous_tasks)

TEMPLATE_COUNT = len(_ATOMIC_GENERATORS) + len(_STRUCTURED_GENERATORS)  # 8
FAMILIES = ("atomic", "multi_step", "read_then_act", "ambiguous")


def generate_tasks(per_domain: int = 40) -> list[Task]:
    """All 8 templates x per_domain parameterizations (default 320 tasks)."""
    tasks = []
    for gen in _ATOMIC_GENERATORS:
        for domain, tool, args, canon, instruction in gen(per_domain):
            tid = f"{domain}-{_h(domain, tool, str(sorted(args.items())))}"
            tasks.append(Task(
                task_id=tid, domain=domain, tool=tool, args=args,
                goal=Goal(tool=tool, canon_args=canon),
                instruction=instruction, family="atomic",
            ))
    for gen in _STRUCTURED_GENERATORS:
        for spec in gen(per_domain):
            tool, args = spec["steps"][0]
            tid = (f"{spec['family']}-"
                   f"{_h(spec['family'], tool, str(sorted(args.items())))}")
            tasks.append(Task(
                task_id=tid, domain=spec["domain"], tool=tool, args=args,
                goal=spec["goals"][0], instruction=spec["instruction"],
                family=spec["family"], steps=spec["steps"], goals=spec["goals"],
            ))
    return tasks


# ---------------------------------------------------------------------------
# Tier-1 E1-5: longer chains with a mid-plan failure (NOT part of
# generate_tasks, so the 320-task / 8-template benchmark is unchanged)
# ---------------------------------------------------------------------------

LONG_TEMPLATES = ("long3", "long4")


def _long_chain_spec(i: int, length: int) -> dict:
    """long3: send_payment -> send_email (receipt) -> append_file (audit),
    the shape of src2/graph._chain. long4 adds create_event before the file
    step, the shape of _chain4. The instruction states every canonical value
    and the order, in the plain "then" style of the multi_step family."""
    tag = f"l{length}"
    payee = f"vendor-{(i * 5 + length) % 17:02d}"
    amount = f"{(i * 43 + 7 * length) % 900 + 100}.00"
    memo = f"order-{_h(tag, i)}"
    to = f"{_NAMES[(i + length) % len(_NAMES)]}@example.com"
    subject = f"Receipt: {memo}"
    path = f"/logs/order-{tag}-{i:03d}.log"
    chash = _h(tag + "log", i)
    pay = {"payee": payee, "amount": amount, "memo": memo}
    email = {"to": to, "subject": subject, "body": f"Paid {payee} ${amount}."}
    steps = [("send_payment", pay), ("send_email", email)]
    goals = [Goal("send_payment", dict(pay)),
             Goal("send_email", {"to": to, "subject": subject})]
    text = (f"Pay {payee} ${amount} for {memo}; then email a receipt to {to} "
            f"with subject '{subject}'; ")
    if length == 4:
        cal = _CALS[(i * 3) % len(_CALS)]
        etitle = f"Review: {memo}"
        start = f"2026-12-{i % 28 + 1:02d}T15:00"
        ev = {"calendar_id": cal, "title": etitle, "start": start}
        steps.append(("create_event", ev))
        goals.append(Goal("create_event", dict(ev)))
        text += f"then create event '{etitle}' on the {cal} calendar at {start}; "
    steps.append(("append_file", {"path": path, "content_hash": chash,
                                  "content": f"paid {memo}"}))
    goals.append(Goal("append_file", {"path": path, "content_hash": chash}))
    text += f"then append an audit entry (content hash {chash}) to {path}."
    return dict(domain=tag, steps=steps, goals=goals, instruction=text,
                fail_on_call=i % length + 1)   # k balanced over 1..L


def generate_long_tasks(per_template: int = 16) -> list[Task]:
    """E1-5 task set: long3 + long4, per_template each (default 32 tasks).
    The injected failure lands on effectful call k = i mod L + 1."""
    tasks = []
    for length in (3, 4):
        for i in range(per_template):
            spec = _long_chain_spec(i, length)
            tool, args = spec["steps"][0]
            tid = (f"long_chain-"
                   f"{_h('long_chain', spec['domain'], tool, str(sorted(args.items())))}")
            tasks.append(Task(
                task_id=tid, domain=spec["domain"], tool=tool, args=args,
                goal=spec["goals"][0], instruction=spec["instruction"],
                family="long_chain", steps=spec["steps"], goals=spec["goals"],
                fail_on_call=spec["fail_on_call"]))
    return tasks


if __name__ == "__main__":
    ts = generate_tasks()
    assert len(ts) == TEMPLATE_COUNT * 40 == 320
    assert len({t.task_id for t in ts}) == 320, "task ids must be unique"
    from collections import Counter
    print(len(ts), "tasks:", dict(Counter(t.family for t in ts)),
          f"({TEMPLATE_COUNT} templates)")
