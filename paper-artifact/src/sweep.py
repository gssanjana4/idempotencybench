"""Full multi-model sweep for the benchmark (the paper's Results data).

Design: K models x n=32 tasks/cell (4 per template,
so every family is covered) x r=2 samples, Wilson 95% CIs, resumable.

Cells per model (budget-conscious):
  core:        {base, careful} x {none, runtime_receipts} @ timeout_after_commit
  keys:        base x idem_keys @ timeout_after_commit
  by-failure:  base x none @ {none, timeout_before_commit, error_500_after_commit}
  => 8 cells x 32 tasks x 2 samples = 512 episodes per model.

Usage:
  python3 src/sweep.py --dry-run          # counts only; no key, no spend
  python3 src/sweep.py --launch           # runs (requires ANTHROPIC_API_KEY);
                                          # resumable: completed episodes skipped
  python3 src/sweep.py --aggregate        # raw JSONL -> results/sweep_summary.csv

Models: env BENCH_SWEEP_MODELS (comma-separated), default below.
NEVER launched automatically — requires an explicit go-ahead (budget).
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from world import World
from tasks import generate_tasks

DEFAULT_MODELS = "claude-fable-5,claude-sonnet-5,claude-haiku-4-5-20251001"
MODELS = [m for m in os.environ.get(
    "BENCH_SWEEP_MODELS", DEFAULT_MODELS).split(",") if m]

PER_TEMPLATE = 4     # 4 x 8 templates = 32 tasks per cell (n >= 30)
SAMPLES = 2          # r

# (prompt, mitigation, failure_mode)
CELLS = [
    ("base",    "none",             "timeout_after_commit"),
    ("base",    "runtime_receipts", "timeout_after_commit"),
    ("careful", "none",             "timeout_after_commit"),
    ("careful", "runtime_receipts", "timeout_after_commit"),
    ("base",    "idem_keys",        "timeout_after_commit"),
    ("base",    "none",             "none"),
    ("base",    "none",             "timeout_before_commit"),
    ("base",    "none",             "error_500_after_commit"),
    # Added 2026-07-13: closes the headline gap — does prompt-level care
    # fix the majority-rate 500-error violations?
    ("careful", "none",             "error_500_after_commit"),
]

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# BENCH_OUT_DIR: write a new run somewhere other than results/ so the
# released data is never appended to or overwritten (default unchanged).
OUT_DIR = os.environ.get("BENCH_OUT_DIR", os.path.join(HERE, "results"))
RAW_PATH = os.path.join(OUT_DIR, "sweep_raw.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "sweep_summary.csv")
TRANSCRIPT_DIR = os.path.join(OUT_DIR, "sweep_transcripts")
# BENCH_LIMIT: stop after this many new episodes (smoke runs); 0 = no limit.
LIMIT = int(os.environ.get("BENCH_LIMIT", "0"))
# BENCH_BUDGET_USD: hard spend cap for this OUT_DIR, from logged usage (all
# transcripts already in TRANSCRIPT_DIR count); 0 = no cap (default unchanged).
# Same rule as src2/sweep2.py: stop BEFORE an episode could exceed the cap.
BUDGET_USD = float(os.environ.get("BENCH_BUDGET_USD", "0"))
# USD per million tokens (input, output). Anthropic list prices 2026-10-25.
PRICES = {"claude-fable-5": (10.0, 50.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}


# Anthropic prompt-caching multipliers on the input price (5-minute cache).
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1


def price_for(model):
    for prefix, pr in PRICES.items():
        if model.startswith(prefix):
            return pr
    return None


def _all_turns(t):
    for d in t.get("deliveries") or [t]:
        yield from d.get("turns", [])
    for sess in t.get("sessions", []):
        yield from sess.get("turns", [])


def episode_cost(t) -> float:
    """USD of one transcript. Anthropic models: list price x logged tokens.
    OpenRouter: the usage.cost OpenRouter returned per turn (recorded as
    provider_meta.cost); 0 if absent -- the OpenRouter key limit is the cap."""
    pr = price_for(t.get("model", ""))
    if pr is not None:
        u = t.get("usage") or {}
        if not u and t.get("sessions"):
            u = {"input": 0, "output": 0}
            for sess in t["sessions"]:
                su = (sess.get("meta") or {}).get("usage") or {}
                for kk, v in su.items():
                    u[kk] = u.get(kk, 0) + (v or 0)
        # Prompt caching (R2 runs): uncached input at list price, cache writes
        # at 1.25x and cache reads at 0.1x the input price. Older transcripts
        # have no cache_* fields, so their cost is unchanged.
        return ((u.get("input") or 0) * pr[0] + (u.get("output") or 0) * pr[1]
                + (u.get("cache_write") or 0) * pr[0] * CACHE_WRITE_MULT
                + (u.get("cache_read") or 0) * pr[0] * CACHE_READ_MULT) / 1e6
    return sum(float((turn.get("provider_meta") or {}).get("cost") or 0)
               for turn in _all_turns(t))


def spent_usd(transcript_dir=None) -> float:
    d = transcript_dir or TRANSCRIPT_DIR
    total = 0.0
    if not os.path.isdir(d):
        return total
    for fn in os.listdir(d):
        try:
            with open(os.path.join(d, fn)) as f:
                total += episode_cost(json.load(f))
        except Exception:
            continue
    return total


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def episode_id(model, prompt, mitigation, failure_mode, task_id, sample) -> str:
    return f"{model}__{prompt}__{mitigation}__{failure_mode}__{task_id}__s{sample}"


def plan():
    tasks = generate_tasks(per_domain=PER_TEMPLATE)
    episodes = [
        (model, prompt, mit, fm, task, s)
        for model in MODELS
        for (prompt, mit, fm) in CELLS
        for task in tasks
        for s in range(SAMPLES)
    ]
    return tasks, episodes


def _done_ids() -> set:
    done = set()
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["episode_id"])
                except Exception:
                    pass
    return done


def dry_run():
    tasks, episodes = plan()
    done = _done_ids()
    remaining = [e for e in episodes if episode_id(
        e[0], e[1], e[2], e[3], e[4].task_id, e[5]) not in done]
    print(f"models ({len(MODELS)}): {', '.join(MODELS)}")
    print(f"cells/model: {len(CELLS)}  tasks/cell: {len(tasks)}  samples: {SAMPLES}")
    print(f"episodes total: {len(episodes)}  done: {len(done)}  remaining: {len(remaining)}")
    print(f"est. API calls (~3/episode): ~{3 * len(remaining)}")
    print("No spend performed. Launch with --launch after the budget go.")


def launch():
    from model_agent import (run_model_episode, _score_and_log,
                             BASE_SYSTEM, CAREFUL_SYSTEM)
    from runlock import acquire
    if not acquire(OUT_DIR, "sweep", MODELS):
        return 0
    import anthropic
    client = anthropic.Anthropic()

    _, episodes = plan()
    done = _done_ids()
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    ran = failed = 0
    spent = spent_usd() if BUDGET_USD else 0.0
    if BUDGET_USD:
        import providers
        providers.RECORD_COST = True   # count OpenRouter usage.cost toward the cap
        print(f"spent so far in {OUT_DIR}: ${spent:.2f} of ${BUDGET_USD:.2f}")
    max_ep = 0.0   # worst-case cost of one episode, to stop BEFORE the cap
    for model, prompt, mit, fm, task, s in episodes:
        eid = episode_id(model, prompt, mit, fm, task.task_id, s)
        if eid in done:
            continue
        if BUDGET_USD and spent + max(max_ep, 0.5) > BUDGET_USD:
            print(f"!! budget ${BUDGET_USD:.2f} would be exceeded "
                  f"(spent ${spent:.2f}); stopping")
            break
        system = CAREFUL_SYSTEM if prompt == "careful" else BASE_SYSTEM
        world = World(mitigation=mit, failure_mode=fm)
        try:
            transcript = run_model_episode(client, world, task, system,
                                           model=model)
        except Exception as e:  # isolate one bad episode; no row => retried next run
            if "HTTP 402" in str(e):
                raise  # out of credits: stop the whole run
            failed += 1
            print(f"!! episode {eid} failed, will retry on next run: {e!s:.300}")
            continue
        transcript["sample"] = s
        cell = f"{model}_{prompt}_{mit}_{fm}"
        v, ok = _score_and_log(world, task, transcript, TRANSCRIPT_DIR,
                               f"{cell}__s{s}")
        sc = transcript["score"]
        row = {"episode_id": eid, "model": model, "prompt": prompt,
               "mitigation": mit, "failure_mode": fm, "task_id": task.task_id,
               "family": task.family, "sample": s,
               "violations": v, "achieved": ok,
               "omission": sc["omission"], "misorder": sc["misorder"],
               "truncated": transcript.get("truncated", False),
               "invalid_calls": len(transcript.get("invalid_calls", []))}
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        if BUDGET_USD:
            c = episode_cost(transcript)
            spent += c
            max_ep = max(max_ep, c)
        ran += 1
        if LIMIT and ran >= LIMIT:
            print(f"BENCH_LIMIT={LIMIT} reached")
            break
        if ran % 25 == 0:
            print(f"...{ran} episodes this run")
    print(f"launch complete: {ran} new episodes -> {RAW_PATH}"
          + (f"; {failed} failed (rerun to retry)" if failed else ""))
    aggregate()
    return failed


def aggregate():
    if not os.path.exists(RAW_PATH):
        print(f"no raw data at {RAW_PATH}"); return
    # One row per episode_id. Two concurrent launches (before src/runlock.py)
    # could run the same episode twice; keep the LAST row, whose transcript
    # is the one on disk (same filename, later write wins).
    by_id: dict[str, dict] = {}
    with open(RAW_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["episode_id"]] = r
    cells: dict[tuple, list] = {}
    for r in by_id.values():
        key = (r["model"], r["prompt"], r["mitigation"], r["failure_mode"])
        cells.setdefault(key, []).append(r)
    rows = []
    for (model, prompt, mit, fm), eps in sorted(cells.items()):
        n = len(eps)
        kv = sum(1 for e in eps if e["violations"] > 0)
        ks = sum(1 for e in eps if e["achieved"])
        ko = sum(1 for e in eps if e.get("omission"))
        km = sum(1 for e in eps if e.get("misorder"))
        vlo, vhi = wilson_ci(kv, n)
        # Task-level stats: the r samples of one
        # task are correlated, so episode-level Wilson CIs understate
        # uncertainty. A task "violates" if ANY of its samples violates.
        by_task: dict[str, bool] = {}
        for e in eps:
            by_task[e["task_id"]] = by_task.get(e["task_id"], False) or \
                (e["violations"] > 0)
        nt = len(by_task)
        kt = sum(1 for v in by_task.values() if v)
        tlo, thi = wilson_ci(kt, nt)
        rows.append({
            "model": model, "prompt": prompt, "mitigation": mit,
            "failure_mode": fm, "n_episodes": n,
            "ivr": round(kv / n, 4),
            "ivr_ci_lo": round(vlo, 4), "ivr_ci_hi": round(vhi, 4),
            "n_tasks": nt,
            "ivr_task": round(kt / nt, 4),
            "ivr_task_ci_lo": round(tlo, 4), "ivr_task_ci_hi": round(thi, 4),
            "success": round(ks / n, 4),
            "omission": round(ko / n, 4),
            "misorder": round(km / n, 4),
            "violation_depth": round(
                sum(e["violations"] for e in eps) / n, 4),
        })
    released = os.path.join(HERE, "results", "sweep_summary.csv")
    if (os.path.abspath(SUMMARY_PATH) == released
            and os.environ.get("BENCH_WRITE_RELEASED") != "1"):
        print(f"refusing to overwrite released {SUMMARY_PATH}; set "
              f"BENCH_OUT_DIR, or BENCH_WRITE_RELEASED=1 on purpose")
        return
    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} cells -> {SUMMARY_PATH}")


if __name__ == "__main__":
    if "--launch" in sys.argv:
        sys.exit(1 if launch() else 0)
    elif "--aggregate" in sys.argv:
        aggregate()
    else:
        dry_run()
