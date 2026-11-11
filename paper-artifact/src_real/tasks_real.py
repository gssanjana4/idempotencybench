"""Tasks of the real-API slice: 16 single payments + 16 two-payment plans.

Same template and wording as the simulator's payment tasks
(src/tasks._payment_tasks: "Pay {payee} ${amount} for {memo}.") and the
simulator's chain plans (src2/graph._chain: "...; then ... Do each step
exactly once, in that order."). The reference (memo) is unique per episode
(derived from the episode id), so no two episodes ever share a payment.
"""
from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.join(HERE, "src2"))

from world import Goal  # noqa: E402
from tasks import Task  # noqa: E402
from graph import TaskGraph, Node  # noqa: E402

N_TASKS = 16


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:10]


def task_id(i: int) -> str:
    return f"rpay-{i:02d}"


def plan_id(i: int) -> str:
    return f"rplan-{i:02d}"


def payment_task(i: int, episode_id: str) -> Task:
    payee = f"vendor-{i % 17:02d}"
    amount = f"{(i * 37) % 900 + 100}.00"
    memo = f"invoice-{_h('real', episode_id)}"
    args = {"payee": payee, "amount": amount, "memo": memo}
    return Task(task_id=task_id(i), domain="payments", tool="send_payment",
                args=args, goal=Goal("send_payment", dict(args)),
                instruction=f"Pay {payee} ${amount} for {memo}.", family="atomic")


def payment_plan(i: int, episode_id: str) -> TaskGraph:
    """Pay A, then pay B (edge 0 -> 1). Interrupted after A commits."""
    pa, pb = f"vendor-{(i * 7) % 17:02d}", f"vendor-{(i * 7 + 5) % 17:02d}"
    aa, ab = f"{(i * 61) % 900 + 100}.00", f"{(i * 43 + 17) % 900 + 100}.00"
    h = _h("realplan", episode_id)
    ma, mb = f"order-{h}-a", f"order-{h}-b"
    nodes = [Node("n0", "send_payment", {"payee": pa, "amount": aa, "memo": ma},
                  Goal("send_payment", {"payee": pa, "amount": aa, "memo": ma})),
             Node("n1", "send_payment", {"payee": pb, "amount": ab, "memo": mb},
                  Goal("send_payment", {"payee": pb, "amount": ab, "memo": mb}))]
    instruction = (f"Pay {pa} ${aa} for {ma}; then pay {pb} ${ab} for {mb}. "
                   f"Do each step exactly once, in that order.")
    return TaskGraph(plan_id(i), "pay2", nodes, [(0, 1)], instruction)
