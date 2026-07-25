"""IdempotencyBench harness: runs the full grid and writes results/summary.csv.

Zero-cost by default (scripted agents). `--fable5` runs the real-model pilot
via model_agent.py (requires ANTHROPIC_API_KEY).
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from world import World, FAILURE_MODES
from tasks import generate_tasks
from agents import AGENTS, RETRY_MODES, run_episode

MITIGATIONS = ("none", "idem_keys", "runtime_receipts")


def run_grid(tasks, agents=("naive", "careful", "stable_key"),
             retry_modes=RETRY_MODES, mitigations=MITIGATIONS,
             failure_mode="timeout_after_commit"):
    """Returns rows of aggregate metrics per (agent, retry_mode, mitigation)."""
    rows = []
    for agent_name in agents:
        for retry_mode in retry_modes:
            for mitigation in mitigations:
                violations = 0
                successes = 0
                omissions = 0
                depth = 0
                for task in tasks:
                    world = World(mitigation=mitigation,
                                  failure_mode=failure_mode)
                    run_episode(AGENTS[agent_name], world, task, retry_mode)
                    v = world.ledger.violations(task.task_id)
                    ok = all(world.ledger.achieved(task.task_id, g)
                             for g in task.goals)
                    violations += 1 if v > 0 else 0
                    depth += v
                    successes += 1 if ok else 0
                    omissions += 0 if ok else 1
                n = len(tasks)
                rows.append({
                    "agent": agent_name,
                    "retry_mode": retry_mode,
                    "mitigation": mitigation,
                    "failure_mode": failure_mode,
                    "n_tasks": n,
                    "ivr": round(violations / n, 4),
                    "violation_depth": round(depth / n, 4),
                    "success": round(successes / n, 4),
                    "omission": round(omissions / n, 4),
                })
    return rows


def run_failure_sweep(tasks, failure_modes=FAILURE_MODES, **kw):
    """Runs the full (agent x retry x mitigation) grid once per failure mode
    and concatenates the rows. Each row carries its `failure_mode` label so
    the resulting CSV is a single tidy table across all injection modes.

    Covers the crux mode `timeout_after_commit` plus `timeout_before_commit`
    (nothing commits, so a retry is *correct* and should NOT be penalized) and
    `error_500_after_commit` (a different post-commit signal). `none` is the
    clean baseline.
    """
    rows = []
    for fm in failure_modes:
        rows.extend(run_grid(tasks, failure_mode=fm, **kw))
    return rows


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def print_matrix(rows):
    print(f"{'agent':<12}{'retry_mode':<12}{'mitigation':<18}"
          f"{'IVR':>7}{'success':>9}{'omission':>10}")
    for r in rows:
        print(f"{r['agent']:<12}{r['retry_mode']:<12}{r['mitigation']:<18}"
              f"{r['ivr']:>7.2f}{r['success']:>9.2f}{r['omission']:>10.2f}")


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if "--fable5" in sys.argv:
        from model_agent import run_fable5_pilot
        run_fable5_pilot(os.path.join(here, "results", "fable5_pilot.csv"))
        return
    tasks = generate_tasks(per_domain=40)  # 8 templates x 40 = 320 tasks

    # Default single-mode grid (crux failure mode) -> summary.csv
    rows = run_grid(tasks)
    write_csv(rows, os.path.join(here, "results", "summary.csv"))
    print_matrix(rows)
    print(f"\n{len(tasks)} tasks -> results/summary.csv")

    # Full sweep across all failure modes -> summary_by_failure.csv
    sweep_rows = run_failure_sweep(tasks)
    write_csv(sweep_rows, os.path.join(here, "results", "summary_by_failure.csv"))
    print(f"{len(sweep_rows)} rows across {len(FAILURE_MODES)} failure modes "
          f"-> results/summary_by_failure.csv")


if __name__ == "__main__":
    main()
