"""Simulated tool world with ground-truth effect ledger, failure injection,
and world-side mitigations (idempotency keys, runtime receipts).

Stdlib only. No network, no API keys.
"""
from __future__ import annotations

import hashlib
import random
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
# ISO date-time with zero seconds: "2026-08-01T10:00:00" == "2026-08-01T10:00"
_ISO_ZERO_SEC_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}):00$", re.I)


def normalize_value(v) -> str:
    """Canonicalize an argument value before semantic comparison.

    Robust to case, surrounding whitespace/quotes, internal whitespace runs,
    currency symbols, thousands separators, trailing zeros ("$1,200.50"
    == "1200.5"; "Q3 Update" == "q3  update"), and zero seconds in an ISO
    date-time ("2026-08-01T10:00:00" == "2026-08-01T10:00"). Distinct values
    must stay distinct: normalization only removes formatting, never content.
    """
    s = re.sub(r"\s+", " ", str(v).strip()).strip("'\"").strip()
    s = _ISO_ZERO_SEC_RE.sub(r"\1", s)
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

    def first_seq(self, task_id: str, goal: "Goal"):
        """Commit order of the first effect satisfying `goal`, or None."""
        for e in self.for_task(task_id):
            if e.tool == goal.tool and all(
                    normalize_value(e.args.get(k)) == normalize_value(v)
                    for k, v in goal.canon_args.items()):
                return e.seq
        return None

    def outcome(self, task_id: str, goals: list, ordered: bool = False) -> dict:
        """Single source of truth for episode outcome.

        omission: some goal effect was never committed.
        misorder: every goal committed, but (for ordered tasks) a dependent
                  effect committed before the effect it depends on.
        success:  every goal committed, in dependency order where required.
        Exactly one of {success, omission, misorder} is True.
        """
        seqs = [self.first_seq(task_id, g) for g in goals]
        omission = any(s is None for s in seqs)
        in_order = (not omission) and (not ordered or seqs == sorted(seqs))
        return {"success": (not omission) and in_order,
                "omission": omission,
                "misorder": (not omission) and not in_order}

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


# Tier-1 E1-6 failure-signal variants (minimal pairs). In every variant the
# effect COMMITS; only the error text the caller sees changes. Kept OUT of
# FAILURE_MODES so the scripted grid and its released CSVs are unchanged.
# name -> (exception class, text); "{request_id}" is filled per call.
SIGNAL_VARIANTS = {
    "error_500_body_after_commit": (
        "server", 'HTTP 500 Internal Server Error\nRetry-After: 2\n'
                  '{{"error":"internal_error","request_id":"{request_id}"}}'),
    "error_503_retry_after_after_commit": (
        "server", "HTTP 503 Service Unavailable\nRetry-After: 2"),
    "status_unknown_after_commit": (
        "timeout", "The request may have been processed; its outcome is unknown."),
    "timeout_504_after_commit": ("timeout", "504 Gateway Timeout"),
}

# R2 (wording robustness) paraphrases of W3 `status_unknown_after_commit`.
# Raised exactly like W3: the effect COMMITS, then a timeout-class error
# (ToolTimeout, same is_error flag) whose text is the only difference. The
# model sees "ERROR: " + text (model_agent / sweep2 prefix), so the texts
# below omit the "ERROR: " that the pre-registration quotes. Kept OUT of
# SIGNAL_VARIANTS and FAILURE_MODES so E1-6 and the scripted grid are unchanged.
WORDING_VARIANTS = {
    "status_unknown_p1_after_commit": (
        "timeout", "Connection lost after the request was sent. "
                   "It is not known whether it was completed."),
    "status_unknown_p2_after_commit": (
        "timeout", "The server did not confirm the result. "
                   "The action may or may not have taken effect."),
    "status_unknown_p3_after_commit": (
        "timeout", "Request status unknown; it might have succeeded."),
}


# R4 (outcome unknown BEFORE commit) failure modes. The injected failure fires
# on the same call as every other mode (fail_on_call-th valid effectful call)
# and NOTHING commits, exactly like `timeout_before_commit`; only the error
# text / class the caller sees differs. The texts are the after-commit ones,
# taken from the same objects, so they are byte-identical:
#   error_500_before_commit      -> ToolServerError("internal server error")
#                                   (= error_500_after_commit's text and class)
#   status_unknown_before_commit -> ToolTimeout(W3 text)
#                                   (= SIGNAL_VARIANTS["status_unknown_after_commit"])
# Kept OUT of FAILURE_MODES, SIGNAL_VARIANTS and WORDING_VARIANTS so the
# scripted grid, E1-6, R2 and R3 are unchanged.
PRECOMMIT_VARIANTS = {
    "error_500_before_commit": ("server", "internal server error"),
    "status_unknown_before_commit": SIGNAL_VARIANTS["status_unknown_after_commit"],
}


class ToolCallError(ValueError):
    """Invalid call (unknown tool, or missing/malformed required arguments).
    Raised BEFORE anything commits and before the injected failure is armed,
    so a malformed call is never a ledger effect and never consumes the
    failure. Model loops return it to the model as a tool error."""


def validate_call(tool: str, args: dict) -> None:
    """Check an effectful call against its schema (required = canonical
    fields, string-valued). Raises ToolCallError if invalid."""
    if tool not in EFFECTFUL_TOOLS:
        raise ToolCallError(
            f"unknown tool {tool!r}; available: "
            f"{', '.join(sorted(EFFECTFUL_TOOLS) + sorted(READONLY_TOOLS))}")
    if "_unparsed_arguments" in args:
        raise ToolCallError(f"arguments for {tool} were not valid JSON")
    _, canon = EFFECTFUL_TOOLS[tool]
    bad = [f for f in canon
           if isinstance(args.get(f), bool)
           or not isinstance(args.get(f), (str, int, float))
           or not str(args.get(f)).strip()]
    if bad:
        raise ToolCallError(
            f"invalid arguments for {tool}: missing or empty required "
            f"field(s) {', '.join(bad)}")


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

    def __init__(self, mitigation: str = "none", failure_mode: str = "none",
                 fail_on_call: int = 1, seed=None):
        """fail_on_call: the injected failure fires on the k-th VALID
        effectful call (default 1 = first call, the released behaviour).
        seed: if not None, ack ids come from random.Random(seed) instead of
        uuid4, so a run is reproducible (default None = released behaviour)."""
        assert failure_mode in FAILURE_MODES or failure_mode in SIGNAL_VARIANTS \
            or failure_mode in WORDING_VARIANTS or failure_mode in PRECOMMIT_VARIANTS
        assert mitigation in ("none", "idem_keys", "runtime_receipts")
        assert fail_on_call >= 1
        self.fail_on_call = fail_on_call
        self._effectful_calls = 0                   # valid effectful calls so far
        self._rng = random.Random(seed) if seed is not None else None
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
        if self.mitigation == "idem_keys" and key:
            # Only a NON-EMPTY repeated key deduplicates. A keyless call must
            # always commit: an empty key is the absence of a key, not a key.
            # (Regression: before this guard, (tool, "") was cached, so a
            # keyless retry was silently swallowed and never scored as a
            # violation -- understating IVR for any model that omits keys.)
            kk = (tool, key)
            if kk in self._seen_keys:
                return self._seen_keys[kk]  # cached; nothing committed
            self._seen_keys[kk] = "ok-" + self._ack_hex()
        self.ledger.commit(Effect(tool=tool, args=dict(args), key=key,
                                  task_id=task_id, seq=0))
        return "ok-" + self._ack_hex()

    def _ack_hex(self) -> str:
        if self._rng is None:
            return uuid.uuid4().hex[:8]
        return f"{self._rng.getrandbits(32):08x}"

    # -- public tool API ----------------------------------------------------

    def call(self, tool: str, args: dict, key: str, task_id: str) -> str:
        """Execute a tool call. May raise ToolTimeout / ToolServerError
        exactly once per world (on the fail_on_call-th valid effectful call,
        default the first), per failure_mode."""
        if tool in READONLY_TOOLS:
            return self._list_effects(task_id)
        validate_call(tool, args)  # ToolCallError: nothing commits
        self._effectful_calls += 1

        if self._failure_armed and self.failure_mode != "none" \
                and self._effectful_calls == self.fail_on_call:
            self._failure_armed = False
            if self.failure_mode == "timeout_after_commit":
                self._commit(tool, args, key, task_id)
                raise ToolTimeout("request timed out")
            if self.failure_mode == "timeout_before_commit":
                raise ToolTimeout("request timed out")
            if self.failure_mode == "error_500_after_commit":
                self._commit(tool, args, key, task_id)
                raise ToolServerError("internal server error")
            if self.failure_mode in SIGNAL_VARIANTS or self.failure_mode in WORDING_VARIANTS:
                kind, text = (SIGNAL_VARIANTS.get(self.failure_mode)
                              or WORDING_VARIANTS[self.failure_mode])
                self._commit(tool, args, key, task_id)
                rid = "req_" + hashlib.sha256(
                    f"{task_id}|{self._effectful_calls}".encode()).hexdigest()[:12]
                exc = ToolServerError if kind == "server" else ToolTimeout
                raise exc(text.format(request_id=rid))
            if self.failure_mode in PRECOMMIT_VARIANTS:   # R4: nothing commits
                kind, text = PRECOMMIT_VARIANTS[self.failure_mode]
                raise (ToolServerError if kind == "server" else ToolTimeout)(text)

        return self._commit(tool, args, key, task_id)

    def _list_effects(self, task_id: str) -> str:
        rows = [
            f"{e.tool}({', '.join(f'{k}={v}' for k, v in e.args.items())})"
            for e in self.ledger.for_task(task_id)
        ]
        return "\n".join(rows) if rows else "(no effects recorded)"
