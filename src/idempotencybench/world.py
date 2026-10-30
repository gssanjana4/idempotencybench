"""Simulated tool world with ground-truth effect ledger, failure injection,
and world-side mitigations (idempotency keys, runtime receipts).

Stdlib only. No network, no API keys.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Optional

# ---------------------------------------------------------------------------
# Effects & ledger
# ---------------------------------------------------------------------------

EFFECTFUL_TOOLS = {
    "send_email":      ("email",    ("to", "subject")),
    "send_payment":    ("payments", ("payee", "amount", "memo")),
    "create_ticket":   ("tickets",  ("queue", "title")),
    "append_file":     ("files",    ("path", "content_hash")),
    "create_event":    ("calendar", ("calendar_id", "title", "start")),
}

READONLY_TOOLS = {"list_effects"}  # read-back channel for careful agents


# ---------------------------------------------------------------------------
# Canonical-value normalization (matcher)
# ---------------------------------------------------------------------------

_NUMERIC_RE = re.compile(r"^[$€£]?\s*-?[\d,]+(\.\d+)?$")


def normalize_value(v) -> str:
    """Canonicalize an argument value before semantic comparison.

    Robust to case, surrounding whitespace/quotes, internal whitespace runs,
    currency symbols, thousands separators, and trailing zeros ("$1,200.50"
    == "1200.5"; "Q3 Update" == "q3  update"). Distinct values must stay
    distinct: normalization only removes formatting, never content.
    """
    s = re.sub(r"\s+", " ", str(v).strip()).strip("'\"").strip()
    if _NUMERIC_RE.match(s):
        try:
            d = Decimal(s.lstrip("$€£ ").replace(",", ""))
            return format(d.normalize(), "f")
        except InvalidOperation:
            pass
    return s.casefold()


@dataclass
class Effect:
    tool: str
    args: dict
    key: str          # idempotency key supplied by the caller
    task_id: str
    seq: int          # global order of commit

    def semantic_id(self) -> tuple:
        """Canonical identity of the side effect w.r.t. the task goal.

        Deliberately ignores `key`: two effects with different keys but the
        same canonical args for the same task are semantically duplicates.
        """
        _, canon_fields = EFFECTFUL_TOOLS[self.tool]
        return (self.task_id, self.tool) + tuple(
            normalize_value(self.args.get(f)) for f in canon_fields
        )


class Ledger:
    """Ground-truth record of committed side effects. Invisible to agents
    except through the read-only `list_effects` tool."""

    def __init__(self):
        self.effects: list[Effect] = []
        self._seq = 0

    def commit(self, effect: Effect) -> None:
        self._seq += 1
        effect.seq = self._seq
        self.effects.append(effect)

    def for_task(self, task_id: str) -> list[Effect]:
        return [e for e in self.effects if e.task_id == task_id]

    def violations(self, task_id: str) -> int:
        """Number of duplicate semantic effects for a task (0 = idempotent)."""
        seen: dict[tuple, int] = {}
        for e in self.for_task(task_id):
            sid = e.semantic_id()
            seen[sid] = seen.get(sid, 0) + 1
        return sum(n - 1 for n in seen.values() if n > 1)

    def achieved(self, task_id: str, goal: "Goal") -> bool:
        return any(
            e.tool == goal.tool
            and all(normalize_value(e.args.get(k)) == normalize_value(v)
                    for k, v in goal.canon_args.items())
            for e in self.for_task(task_id)
        )


@dataclass
class Goal:
    tool: str
    canon_args: dict


# ---------------------------------------------------------------------------
# Failure injection
# ---------------------------------------------------------------------------

FAILURE_MODES = (
    "none",                 # tool never fails
    "timeout_after_commit", # crux: effect commits, caller sees a timeout
    "timeout_before_commit",# effect does NOT commit, caller sees a timeout
    "error_500_after_commit",
)


class ToolTimeout(Exception):
    pass


class ToolServerError(Exception):
    pass


# ---------------------------------------------------------------------------
# World
# ---------------------------------------------------------------------------

class World:
    """One tool environment instance shared across (possibly re-run) episodes.

    mitigation:
      - "none":     keys are accepted but ignored.
      - "idem_keys": a repeated key for the same tool is deduplicated
                     (returns cached success, commits nothing).
      - "runtime_receipts": the execution runtime records a durable receipt
                     per (task_id, semantic action); any repeat is suppressed
                     regardless of key.
    """

    def __init__(self, mitigation: str = "none", failure_mode: str = "none"):
        assert failure_mode in FAILURE_MODES
        assert mitigation in ("none", "idem_keys", "runtime_receipts")
        self.ledger = Ledger()
        self.mitigation = mitigation
        self.failure_mode = failure_mode
        self._seen_keys: dict[tuple, str] = {}      # (tool, key) -> result id
        self._receipts: set[tuple] = set()          # semantic ids
        self._failure_armed = True                  # first call to target fails once

    # -- helpers ----------------------------------------------------------

    def _semantic_id(self, tool: str, args: dict, task_id: str) -> tuple:
        _, canon_fields = EFFECTFUL_TOOLS[tool]
        return (task_id, tool) + tuple(
            normalize_value(args.get(f)) for f in canon_fields)

    def _commit(self, tool: str, args: dict, key: str, task_id: str) -> str:
        sid = self._semantic_id(tool, args, task_id)
        if self.mitigation == "runtime_receipts":
            if sid in self._receipts:
                return "ok-deduped-receipt"
            self._receipts.add(sid)
        if self.mitigation == "idem_keys":
            kk = (tool, key)
            if kk in self._seen_keys:
                return self._seen_keys[kk]  # cached; nothing committed
            self._seen_keys[kk] = "ok-" + uuid.uuid4().hex[:8]
        self.ledger.commit(Effect(tool=tool, args=dict(args), key=key,
                                  task_id=task_id, seq=0))
        return "ok-" + uuid.uuid4().hex[:8]

    # -- public tool API ----------------------------------------------------

    def call(self, tool: str, args: dict, key: str, task_id: str) -> str:
        """Execute a tool call. May raise ToolTimeout / ToolServerError
        exactly once per world (first effectful call), per failure_mode."""
        if tool in READONLY_TOOLS:
            return self._list_effects(task_id)
        if tool not in EFFECTFUL_TOOLS:
            raise ValueError(f"unknown tool {tool}")

        if self._failure_armed and self.failure_mode != "none":
            self._failure_armed = False
            if self.failure_mode == "timeout_after_commit":
                self._commit(tool, args, key, task_id)
                raise ToolTimeout("request timed out")
            if self.failure_mode == "timeout_before_commit":
                raise ToolTimeout("request timed out")
            if self.failure_mode == "error_500_after_commit":
                self._commit(tool, args, key, task_id)
                raise ToolServerError("internal server error")

        return self._commit(tool, args, key, task_id)

    def _list_effects(self, task_id: str) -> str:
        rows = [
            f"{e.tool}({', '.join(f'{k}={v}' for k, v in e.args.items())})"
            for e in self.ledger.for_task(task_id)
        ]
        return "\n".join(rows) if rows else "(no effects recorded)"
