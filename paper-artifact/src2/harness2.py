"""Session-resume harness: scripted-substrate sweep over the TaskGraph benchmark.

Sweep: 200 graphs x every interrupt point x {before,after}_commit x
4 resume contexts x 3 mitigations x 2 key postures, plus a no-interrupt
baseline per configuration. Zero-cost, deterministic, stdlib only.

Outputs (tidy CSVs, one row per aggregate cell):
  results/graph_summary.csv      per (context, mitigation, posture, timing)
  results/graph_by_topology.csv  adds (topology, interrupt_depth) for H1/H5
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from graph import generate_graphs  # noqa: E402
from session import Interrupt, INTERRUPT_TIMINGS  # noqa: E402
from resume import run_graph_episode, RESUME_CONTEXTS  # noqa: E402

MITIGATIONS = ("none", "idem_keys", "runtime_receipts")
POSTURES = ("naive", "stable")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_sweep(graphs=None):
    graphs = graphs or generate_graphs()
    rows = []
    for ctx in RESUME_CONTEXTS:
        for mit in MITIGATIONS:
            for posture in POSTURES:
                for g in graphs:
                    # no-interrupt baseline
                    rows.append(run_graph_episode(g, ctx, mit, posture))
                    for k in range(len(g.nodes)):
                        for timing in INTERRUPT_TIMINGS:
                            rows.append(run_graph_episode(
                                g, ctx, mit, posture,
                                interrupt=Interrupt(k, timing)))
    return rows


def _agg(rows_subset):
    n = len(rows_subset)
    return {
        "n_episodes": n,
        "givr": round(sum(r["givr_hit"] for r in rows_subset) / n, 4),
        "mean_violations": round(
            sum(r["violations"] for r in rows_subset) / n, 4),
        "completion": round(sum(r["completion"] for r in rows_subset) / n, 4),
        "order_violation_rate": round(
            sum(r["order_violation"] for r in rows_subset) / n, 4),
    }


def aggregate(rows):
    summary, by_topo = [], []
    keys = sorted({(r["resume_context"], r["mitigation"], r["key_posture"],
                    r["interrupt_timing"]) for r in rows})
    for ctx, mit, posture, timing in keys:
        sub = [r for r in rows if
               (r["resume_context"], r["mitigation"], r["key_posture"],
                r["interrupt_timing"]) == (ctx, mit, posture, timing)]
        summary.append({"resume_context": ctx, "mitigation": mit,
                        "key_posture": posture, "interrupt_timing": timing,
                        **_agg(sub)})
        topo_keys = sorted({(r["topology"], r["interrupt_node"]) for r in sub})
        for topo, k in topo_keys:
            tsub = [r for r in sub if (r["topology"], r["interrupt_node"])
                    == (topo, k)]
            by_topo.append({"resume_context": ctx, "mitigation": mit,
                            "key_posture": posture, "interrupt_timing": timing,
                            "topology": topo, "interrupt_depth": k,
                            **_agg(tsub)})
    return summary, by_topo


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    rows = run_sweep()
    summary, by_topo = aggregate(rows)
    write_csv(summary, os.path.join(HERE, "results", "graph_summary.csv"))
    write_csv(by_topo, os.path.join(HERE, "results", "graph_by_topology.csv"))
    print(f"{len(rows)} episodes -> {len(summary)} summary cells, "
          f"{len(by_topo)} by-topology cells")
    # Console teaser: the headline axis (naive posture, after-commit)
    print(f"\n{'context':<16}{'mitigation':<18}{'gIVR':>7}{'compl':>7}{'OVR':>7}")
    for r in summary:
        if r["key_posture"] == "naive" and r["interrupt_timing"] == "after_commit":
            print(f"{r['resume_context']:<16}{r['mitigation']:<18}"
                  f"{r['givr']:>7.2f}{r['completion']:>7.2f}"
                  f"{r['order_violation_rate']:>7.2f}")


if __name__ == "__main__":
    main()
