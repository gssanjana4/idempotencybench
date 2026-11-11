"""Session-resume figures from results/sweep2_summary.csv (model sweep).

fig1: gIVR by resume context, grouped by model, with Wilson CIs -- the
      headline (amnesiac vs everything else).
fig2: stale re-execution rate by resume context -- the hazard single-call
      IVR cannot express.

Usage: python3 src2/plot_sweep2.py
Writes figures/sweep2_givr.pdf and figures/sweep2_ovr.pdf (paper-sized).
"""
from __future__ import annotations

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(HERE, "results", "sweep2_summary.csv")
FIGDIR = os.path.join(HERE, "figures")

CELL_ORDER = [
    ("amnesiac", "none", "amnesiac"),
    ("plan_only", "none", "plan only"),
    ("ledger_readback", "none", "ledger\nreadback"),
    ("amnesiac", "runtime_receipts", "checkpoint\n(receipts)"),
]
SHORT = {"claude-fable-5": "Fable 5", "claude-sonnet-5": "Sonnet 5",
         "claude-haiku-4-5-20251001": "Haiku 4.5"}


def load():
    with open(SUMMARY) as f:
        return list(csv.DictReader(f))


def grouped_bar(rows, metric, lo_key, hi_key, ylabel, title, out):
    models = sorted({r["model"] for r in rows},
                    key=lambda m: list(SHORT).index(m) if m in SHORT else 99)
    idx = {(r["model"], r["resume_context"], r["mitigation"]): r for r in rows}
    n_cells, n_models = len(CELL_ORDER), len(models)
    width = 0.8 / max(n_models, 1)
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for mi, model in enumerate(models):
        xs, ys, los, his = [], [], [], []
        for ci, (ctx, mit, _) in enumerate(CELL_ORDER):
            r = idx.get((model, ctx, mit))
            if not r:
                continue
            v = float(r[metric])
            xs.append(ci + mi * width - 0.4 + width / 2)
            ys.append(v)
            los.append(max(0.0, v - float(r[lo_key])))
            his.append(max(0.0, float(r[hi_key]) - v))
        ax.bar(xs, ys, width * 0.92, label=SHORT.get(model, model),
               yerr=[los, his], capsize=2, error_kw={"linewidth": 0.8})
    ax.set_xticks(range(n_cells))
    ax.set_xticklabels([lbl for _, _, lbl in CELL_ORDER], fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, out), bbox_inches="tight")
    print("wrote", os.path.join(FIGDIR, out))


def main():
    rows = load()
    grouped_bar(rows, "givr", "givr_lo", "givr_hi",
                "graph-IVR", "Duplicate actions across the session seam",
                "sweep2_givr.pdf")
    grouped_bar(rows, "ovr", "ovr_lo", "ovr_hi",
                "stale re-execution rate",
                "Stale re-execution across the seam",
                "sweep2_ovr.pdf")


if __name__ == "__main__":
    main()
