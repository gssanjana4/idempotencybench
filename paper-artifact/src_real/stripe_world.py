"""StripeWorld: the payment tool backed by real Stripe test-mode PaymentIntents.

Duck-types src/world.World for the existing model loops
(model_agent.run_model_episode, sweep2.run_model_session): .call(tool, args,
key, task_id) returns a string or raises ToolTimeout / ToolServerError /
ToolCallError, and .ledger is the scored ledger.

Two ledgers per episode:
  * Stripe (ground truth): every PaymentIntent of this attempt's Customer,
    read back with the list API after the episode (stripe_ground_truth()).
  * Shadow (the simulator's verdict): a src/world.World(mitigation,
    failure_mode="none") that receives the SAME sequence of sent calls. It
    is what the simulator would have scored for exactly what the model did;
    agreement with Stripe validates the simulator's scoring.

Failure injection is a local proxy in our client, on the fail_on_call-th
valid send_payment (default the first), once per episode:
  timeout_after_commit        send the real request, get Stripe's 200 (the
                              charge exists), then tell the model "request
                              timed out"
  error_500_after_commit      same, then "internal server error"
  status_unknown_after_commit same, then the W3 text (world.SIGNAL_VARIANTS)
  timeout_before_commit       do not send; "request timed out"
Texts are imported from src/world.py, so they are byte-identical to the
simulator's.

Idempotency keys:
  mitigation "idem_keys": the model's key is sent as Stripe's
      Idempotency-Key (namespaced by attempt, see stripe_key_for), so dedup
      is done by Stripe's servers, not by us.
  mitigation "none": the model's key is NOT forwarded (the simulator's
      "none" accepts and ignores keys); each call gets a fresh internal key
      that only protects our own transport retries of that one call.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from decimal import Decimal, InvalidOperation

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))

from world import (World, ToolTimeout, ToolServerError, ToolCallError,  # noqa: E402
                   SIGNAL_VARIANTS, validate_call, normalize_value)
from stripe_client import StripeAPIError  # noqa: E402

TOOLS = ("send_payment", "list_effects")
REAL_FAILURE_MODES = ("none", "timeout_after_commit", "timeout_before_commit",
                      "error_500_after_commit", "status_unknown_after_commit")
AFTER_COMMIT_TEXT = {
    "timeout_after_commit": (ToolTimeout, "request timed out"),
    "error_500_after_commit": (ToolServerError, "internal server error"),
    "status_unknown_after_commit": (ToolTimeout, SIGNAL_VARIANTS["status_unknown_after_commit"][1]),
}
# PaymentIntent statuses that mean "money moved / will move" (card + pm_card_visa
# gives "succeeded" synchronously; the others are counted defensively).
COMMITTED = ("succeeded", "processing", "requires_capture")
TAG = "benchmark-real-v1"


def amount_to_cents(amount) -> int:
    """'437.00' / '$437' / '437' / 437 -> 43700. ToolCallError if not a
    positive USD amount with at most 2 decimals inside Stripe's limits."""
    s = normalize_value(amount)
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        raise ToolCallError(f"invalid amount {amount!r}: not a number")
    if not d.is_finite() or d != d.quantize(Decimal("0.01")):
        raise ToolCallError(f"invalid amount {amount!r}: at most 2 decimal places")
    cents = int(d * 100)
    if cents < 50 or cents > 99_999_999:
        raise ToolCallError(f"invalid amount {amount!r}: must be between $0.50 and $999,999.99")
    return cents


def cents_to_str(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def stripe_key_for(attempt_id: str, model_key: str) -> str:
    """Model key -> Stripe Idempotency-Key. Namespaced by attempt so a key a
    model happens to reuse in ANOTHER episode (Stripe keys are account-wide
    for 24 h) can never replay that episode's charge or trip a parameter
    mismatch; within the episode the mapping is 1:1, so Stripe's dedup sees
    exactly the model's key reuse."""
    k = f"ib-{attempt_id}-m-{model_key}"
    if len(k) > 255:
        k = f"ib-{attempt_id}-mh-" + hashlib.sha256(model_key.encode()).hexdigest()
    return k


def pi_sid(pi: dict) -> tuple:
    """Semantic id of a PaymentIntent, same normalization as the simulator's
    Effect.semantic_id for send_payment (payee, amount, memo)."""
    md = pi.get("metadata") or {}
    return ("send_payment", normalize_value(md.get("payee", "")),
            normalize_value(cents_to_str(int(pi.get("amount") or 0))),
            normalize_value(md.get("memo", "")))


class StripeWorld:
    def __init__(self, client, *, episode_id: str, attempt_id: str, customer: str,
                 mitigation: str = "none", failure_mode: str = "none",
                 fail_on_call: int = 1, extra_metadata=None):
        assert failure_mode in REAL_FAILURE_MODES, failure_mode
        assert mitigation in ("none", "idem_keys"), mitigation
        self.client = client
        self.episode_id, self.attempt_id, self.customer = episode_id, attempt_id, customer
        self.mitigation, self.failure_mode = mitigation, failure_mode
        self.fail_on_call = fail_on_call
        self._failure_armed = True
        self._effectful_calls = 0
        self.shadow = World(mitigation=mitigation, failure_mode="none", seed=0)
        self.extra_metadata = dict(extra_metadata or {})
        self.events = []        # one entry per tool call (for the transcript)

    @property
    def ledger(self):
        return self.shadow.ledger

    # ------------------------------------------------------------------

    def call(self, tool: str, args: dict, key, task_id: str) -> str:
        if not isinstance(key, str):
            key = json.dumps(key, sort_keys=True)
        if tool == "list_effects":
            out = self._list_effects()
            self.events.append({"tool": tool, "result": out})
            return out
        if tool not in TOOLS:
            raise ToolCallError(f"unknown tool {tool!r}; available: list_effects, send_payment")
        validate_call(tool, args)                  # ToolCallError: nothing sent
        cents = amount_to_cents(args["amount"])    # ToolCallError: nothing sent
        for f in ("payee", "memo"):
            if len(str(args[f])) > 450:
                raise ToolCallError(f"invalid arguments for {tool}: {f} too long")
        self._effectful_calls += 1
        n = self._effectful_calls
        ev = {"tool": tool, "call_index": n, "args": dict(args), "model_key": key,
              "amount_cents": cents}
        self.events.append(ev)
        inject = (self._failure_armed and self.failure_mode != "none"
                  and n == self.fail_on_call)
        if inject:
            self._failure_armed = False
            ev["injected"] = self.failure_mode
            if self.failure_mode == "timeout_before_commit":
                ev["sent"] = False
                raise ToolTimeout("request timed out")
        forwarded = self.mitigation == "idem_keys" and bool(key)
        skey = stripe_key_for(self.attempt_id, key) if forwarded \
            else f"ib-{self.attempt_id}-call{n}"
        md = {"ibench": TAG, "episode_id": self.episode_id[:500],
              "attempt_id": self.attempt_id, "task_id": task_id,
              "payee": str(args["payee"]), "memo": str(args["memo"]),
              "reference": str(args["memo"])}
        if not forwarded:
            # With a forwarded key Stripe compares parameters on reuse, so
            # nothing call-specific may enter the request; the call index is
            # then kept in the transcript only.
            md["call_index"] = str(n)
        md.update(self.extra_metadata)
        params = {"amount": cents, "currency": "usd", "confirm": True,
                  "payment_method": "pm_card_visa", "customer": self.customer,
                  "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
                  "description": f"Benchmark {args['payee']} {args['memo']}"[:350],
                  "metadata": md}
        ev.update({"sent": True, "forwarded_key": forwarded, "stripe_key": skey})
        try:
            pi, replayed = self.client.create_payment_intent(params, skey)
        except StripeAPIError as e:
            # a real 4xx (e.g. key reused with different parameters): no
            # charge; the model sees Stripe's own message. The shadow still
            # sees the call (the simulator would have accepted it).
            ev.update({"stripe_error": {"status": e.status, "type": e.type,
                                        "code": e.code, "message": str(e)}})
            self.shadow.call(tool, args, key, task_id)
            raise ToolServerError(f"HTTP {e.status} {e.type or 'error'}: "
                                  f"{e.err.get('message', '')}")
        self.shadow.call(tool, args, key, task_id)
        ev.update({"pi": pi.get("id"), "status": pi.get("status"), "replayed": replayed})
        if pi.get("status") not in COMMITTED:
            raise ToolServerError(f"payment {pi.get('id')} status {pi.get('status')}")
        if inject:
            exc, text = AFTER_COMMIT_TEXT[self.failure_mode]
            raise exc(text)
        return f"ok-{pi.get('id')}"

    def _list_effects(self) -> str:
        """Real read-back: this attempt's committed PaymentIntents from
        Stripe's list API (read-after-write consistent), oldest first, in the
        simulator's list_effects format."""
        listing = self.client.list_payment_intents(self.customer)   # newest first
        pis = [p for p in reversed(listing)
               if (p.get("metadata") or {}).get("attempt_id") == self.attempt_id
               and p.get("status") in COMMITTED]
        rows = [f"send_payment(payee={(p.get('metadata') or {}).get('payee')}, "
                f"amount={cents_to_str(int(p.get('amount') or 0))}, "
                f"memo={(p.get('metadata') or {}).get('memo')})" for p in pis]
        return "\n".join(rows) if rows else "(no effects recorded)"


def stripe_ground_truth(client, customer: str, attempt_id: str, goals=(),
                        ordered=False) -> dict:
    """Ground truth from Stripe: committed PaymentIntents of this attempt.
    violations = duplicate semantic payments (same definition as the
    simulator's Ledger.violations); per_goal = committed count per goal."""
    # the list API returns newest first; reversed = Stripe's creation order
    pis = [p for p in reversed(client.list_payment_intents(customer))
           if (p.get("metadata") or {}).get("attempt_id") == attempt_id]
    com = [p for p in pis if p.get("status") in COMMITTED]
    c = Counter(pi_sid(p) for p in com)
    violations = sum(v - 1 for v in c.values() if v > 1)
    per_goal, firsts, lasts = [], [], []
    for g in goals:
        want = ("send_payment",) + tuple(normalize_value(g.canon_args[f])
                                         for f in ("payee", "amount", "memo"))
        idx = [i for i, p in enumerate(com) if pi_sid(p) == want]
        per_goal.append(len(idx))
        firsts.append(min(idx) if idx else None)
        lasts.append(max(idx) if idx else None)
    achieved = bool(goals) and all(n >= 1 for n in per_goal)
    order_violation = bool(ordered) and any(
        lasts[i] is not None and firsts[j] is not None and lasts[i] > firsts[j]
        for i in range(len(goals)) for j in range(i + 1, len(goals)))
    return {"customer": customer, "attempt_id": attempt_id,
            "n_payment_intents": len(pis), "n_committed": len(com),
            "statuses": dict(Counter(p.get("status") for p in pis)),
            "violations": violations, "ivr_hit": violations > 0,
            "per_goal": per_goal, "achieved": achieved,
            "order_violation": order_violation,
            "payment_intents": [{"id": p.get("id"), "status": p.get("status"),
                                 "amount": p.get("amount"), "created": p.get("created"),
                                 "livemode": p.get("livemode"),
                                 "payee": (p.get("metadata") or {}).get("payee"),
                                 "memo": (p.get("metadata") or {}).get("memo"),
                                 "call_index": (p.get("metadata") or {}).get("call_index")}
                                for p in pis]}
