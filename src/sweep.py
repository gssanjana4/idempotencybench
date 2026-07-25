"""Full multi-model sweep for IdempotencyBench (the paper's Results data).

Design (mock-review W5 bar): K models x n=32 tasks/cell (4 per template,
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

Models: env IBENCH_SWEEP_MODELS (comma-separated), default below.
NEVER launched automatically — requires the author's explicit go (budget).
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
    "IBENCH_SWEEP_MODELS", DEFAULT_MODELS).split(",") if m]

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
    # fix the majority-rate 500-error violations? (mock-review round 2)
    ("careful", "none",             "error_500_after_commit"),
]

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(HERE, "results", "sweep_raw.jsonl")
SUMMARY_PATH = os.path.join(HERE, "results", "sweep_summary.csv")
TRANSCRIPT_DIR = os.path.join(HERE, "results", "sweep_transcripts")


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
    import anthropic
    client = anthropic.Anthropic()

    _, episodes = plan()
    done = _done_ids()
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    ran = 0
    for model, prompt, mit, fm, task, s in episodes:
        eid = episode_id(model, prompt, mit, fm, task.task_id, s)
        if eid in done:
            continue
        system = CAREFUL_SYSTEM if prompt == "careful" else BASE_SYSTEM
        world = World(mitigation=mit, failure_mode=fm)
        transcript = run_model_episode(client, world, task, system, model=model)
        transcript["sample"] = s
        cell = f"{model}_{prompt}_{mit}_{fm}"
        v, ok = _score_and_log(world, task, transcript, TRANSCRIPT_DIR,
                               f"{cell}__s{s}")
        row = {"episode_id": eid, "model": model, "prompt": prompt,
               "mitigation": mit, "failure_mode": fm, "task_id": task.task_id,
               "family": task.family, "sample": s,
               "violations": v, "achieved": ok}
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        ran += 1
        if ran % 25 == 0:
            print(f"...{ran} episodes this run")
    print(f"launch complete: {ran} new episodes -> {RAW_PATH}")
    aggregate()


def aggregate():
    if not os.path.exists(RAW_PATH):
        print(f"no raw data at {RAW_PATH}"); return
    cells: dict[tuple, list] = {}
    with open(RAW_PATH) as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["prompt"], r["mitigation"], r["failure_mode"])
            cells.setdefault(key, []).append(r)
    rows = []
    for (model, prompt, mit, fm), eps in sorted(cells.items()):
        n = len(eps)
        kv = sum(1 for e in eps if e["violations"] > 0)
        ks = sum(1 for e in eps if e["achieved"])
        vlo, vhi = wilson_ci(kv, n)
        # Task-level stats (mock-review round 2, R2): the r samples of one
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
            "violation_depth": round(
                sum(e["violations"] for e in eps) / n, 4),
        })
    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} cells -> {SUMMARY_PATH}")


if __name__ == "__main__":
    if "--launch" in sys.argv:
        launch()
    elif "--aggregate" in sys.argv:
        aggregate()
    else:
        dry_run()
