"""Cumulative spend cap for the v32 experiments (hard stop at $50 by default).

One append-only JSONL ledger is shared by every process of every v32 run
(any number of launches, sequential or parallel). Lines:

  {"type": "reserve", "id": ..., "usd": worst_case, ...}  before money can be spent
  {"type": "settle",  "id": ..., "usd": actual, ...}      when the actual cost is known
  {"type": "charge",  "usd": actual, ...}                 one-off actual cost (key check)

Committed spend = sum(settle) + sum(charge) + sum(reserve whose id has no
settle). An unsettled reservation (a kill between request and response)
keeps counting at its worst case (at most ~$0.06 for one Fable 5 call), so
the guard can only over-count, never under-count.

reserve() is atomic across processes (fcntl lock on the ledger): it re-reads
the ledger, and refuses (raises SpendCapReached) if committed + worst_case
would exceed the cap. Every request that can cost money is reserved first,
at a worst case that bounds its bill (max_tokens output; input tokens bounded
from the request size), so the actual cumulative spend cannot pass the cap.

SpendCapReached's message contains "HTTP 402" and "credit balance": the two
strings on which src/tier1.py already stops OpenRouter and Anthropic
episodes, so the existing runner stops the provider instead of retrying.
"""
from __future__ import annotations

import fcntl
import json
import os
import threading
import time
import uuid

DEFAULT_CAP_USD = 50.0


class SpendCapReached(RuntimeError):
    def __init__(self, committed, need, cap):
        super().__init__(
            f"spend guard: committed ${committed:.4f} + worst case ${need:.4f} > cap ${cap:.2f} "
            f"(stops this provider like HTTP 402 / credit balance too low; completed episodes are kept)")
        self.committed, self.need, self.cap = committed, need, cap


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SpendGuard:
    def __init__(self, ledger_path: str, cap_usd: float = DEFAULT_CAP_USD):
        self.path = ledger_path
        self.cap = float(cap_usd)
        self._lock = threading.Lock()

    # -- reading -------------------------------------------------------------
    def entries(self) -> list:
        out = []
        if not os.path.exists(self.path):
            return out
        with open(self.path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue          # a line cut off by a kill mid-write
        return out

    @staticmethod
    def summarize(entries) -> dict:
        res, settled = {}, {}
        charged = 0.0
        for e in entries:
            if e.get("type") == "reserve":
                res[e["id"]] = e
            elif e.get("type") == "settle":
                settled[e["id"]] = float(e["usd"])
            elif e.get("type") == "charge":
                charged += float(e["usd"])
        open_res = {k: v for k, v in res.items() if k not in settled}
        actual = sum(settled.values()) + charged
        held = sum(float(v["usd"]) for v in open_res.values())
        by_prov = {}
        for e in entries:
            if e.get("type") in ("settle", "charge"):
                p = e.get("provider", "?")
                by_prov[p] = by_prov.get(p, 0.0) + float(e["usd"])
        return {"actual": actual, "held": held, "committed": actual + held,
                "open": open_res, "by_provider": by_prov}

    def committed(self) -> float:
        return self.summarize(self.entries())["committed"]

    def remaining(self) -> float:
        return self.cap - self.committed()

    # -- writing -------------------------------------------------------------
    def _locked(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        return open(self.path, "a")

    def _append(self, rec):
        line = json.dumps(rec) + "\n"
        # a line cut off by a kill has no newline: start a fresh line so the
        # new record is not glued onto it (and lost with it)
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    line = "\n" + line
        with open(self.path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def reserve(self, usd: float, **meta) -> str:
        rid = meta.pop("rid", None) or uuid.uuid4().hex[:16]
        with self._lock, self._locked() as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                c = self.committed()
                if c + usd > self.cap + 1e-12:
                    raise SpendCapReached(c, usd, self.cap)
                self._append({"type": "reserve", "id": rid, "usd": round(float(usd), 8),
                              "at": _now(), **meta})
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)
        return rid

    def settle(self, rid: str, usd: float, **meta):
        with self._lock, self._locked() as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                self._append({"type": "settle", "id": rid, "usd": round(float(usd), 8),
                              "at": _now(), **meta})
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def charge(self, usd: float, **meta):
        with self._lock, self._locked() as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            try:
                self._append({"type": "charge", "usd": round(float(usd), 8), "at": _now(), **meta})
            finally:
                fcntl.flock(lk, fcntl.LOCK_UN)

    def check(self, need: float = 0.0):
        """Raise SpendCapReached unless committed + need <= cap."""
        c = self.committed()
        if c + need > self.cap + 1e-12:
            raise SpendCapReached(c, need, self.cap)


# ---------------------------------------------------------------------------
# Worst-case bounds
# ---------------------------------------------------------------------------

TOOL_OVERHEAD_TOKENS = 600   # tool-use system prompt + tool definitions framing


def input_tokens_bound(payload) -> int:
    """Upper bound on input tokens from the request size: 1 token per 2
    characters of the JSON body (English/JSON runs at ~4 characters per
    token), plus a fixed tool-use overhead."""
    n = len(json.dumps(payload, default=str))
    return n // 2 + TOOL_OVERHEAD_TOKENS


def worst_usd(payload, max_tokens: int, price_in: float, price_out: float, discount: float = 1.0) -> float:
    """price_* in USD per million tokens."""
    return discount * (input_tokens_bound(payload) * price_in + int(max_tokens) * price_out) / 1e6


CACHE_WRITE_MULT, CACHE_READ_MULT = 1.25, 0.1   # = src/sweep.py


def anthropic_call_cost(model, comp, prices) -> float:
    """List price x logged tokens of one Anthropic completion, cache tokens at
    1.25x / 0.1x the input price: the rule of src/sweep.episode_cost."""
    p_in, p_out = prices[model]
    u = comp.usage or {}
    return ((u.get("input") or 0) * p_in + (u.get("output") or 0) * p_out
            + (u.get("cache_write") or 0) * p_in * CACHE_WRITE_MULT
            + (u.get("cache_read") or 0) * p_in * CACHE_READ_MULT) / 1e6


class GuardedProvider:
    """Wraps a synchronous provider (OpenRouterProvider or AnthropicProvider):
    reserve the worst case before each completion, settle at the logged cost
    after it (OpenRouter usage.cost; Anthropic list price x tokens, cache
    tokens included). worst_in_mult = 1.25 for Anthropic with prompt caching
    (every input token could be a cache write). A failed call settles at 0
    (OpenRouter bills only completions it returns; the Anthropic API does not
    bill a request it rejects)."""

    def __init__(self, inner, guard: SpendGuard, prices: dict, cost_fn=None, worst_in_mult=1.0):
        self.inner, self.guard, self.prices = inner, guard, prices
        self.cost_fn = cost_fn
        self.worst_in_mult = float(worst_in_mult)
        self.name = inner.name

    def complete(self, model, system, tools, messages, max_tokens, temperature):
        p_in, p_out = self.prices[model]
        payload = {"system": system, "tools": tools, "messages": _jsonable(messages)}
        w = worst_usd(payload, max_tokens, p_in * self.worst_in_mult, p_out)
        rid = self.guard.reserve(w, provider=self.name, model=model, kind="call")
        try:
            comp = self.inner.complete(model, system, tools, messages, max_tokens, temperature)
        except Exception:
            self.guard.settle(rid, 0.0, provider=self.name, model=model, kind="call_failed")
            raise
        if self.cost_fn is not None:
            c = self.cost_fn(model, comp, self.prices)
        else:
            c = float((comp.provider_meta or {}).get("cost") or 0.0)
            if c == 0.0 and comp.usage:   # no usage.cost returned: price the tokens
                c = ((comp.usage.get("input") or 0) * p_in + (comp.usage.get("output") or 0) * p_out) / 1e6
        self.guard.settle(rid, c, provider=self.name, model=model, kind="call")
        return comp

    def history_assistant(self, completion):
        return self.inner.history_assistant(completion)

    def history_tool_results(self, results):
        return self.inner.history_tool_results(results)


def _jsonable(x):
    """Message history may hold SDK content objects; size them by their dict form."""
    if isinstance(x, list):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if hasattr(x, "model_dump"):
        return x.model_dump()
    if hasattr(x, "__dict__") and not isinstance(x, (str, int, float, bool)):
        return {k: _jsonable(v) for k, v in vars(x).items()}
    return x
