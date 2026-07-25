"""IdempotencyBench results plotting.

Reads results/summary.csv (produced by harness.py) and renders an IVR matrix
as small-multiple heatmaps -- one panel per agent, rows = retry mode,
columns = world-side mitigation, cell color/label = Idempotency Violation Rate.

Usage:
    python3 src/plot_results.py                # summary.csv -> figures/ivr_matrix.png
    python3 src/plot_results.py --input results/summary_by_failure.csv \
            --output figures/ivr_matrix_by_failure.png

Zero-cost: reads only local CSVs, makes no API calls. Requires matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# Stable display order (matches harness.py); anything else is appended as seen.
RETRY_ORDER = ("no_retry", "tool_retry", "agent_retry", "whole_run")
MITIGATION_ORDER = ("none", "idem_keys", "runtime_receipts")
AGENT_ORDER = ("naive", "careful", "stable_key")


def _ordered(values, preferred):
    seen = list(dict.fromkeys(values))
    head = [v for v in preferred if v in seen]
    tail = [v for v in seen if v not in preferred]
    return head + tail


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No data rows in {path}")
    return rows


def plot(rows, out_path, title=None, paper=False):
    """paper=True sizes for an ACL two-column-span figure* (~6.3in wide),
    PDF output recommended; screen default otherwise."""
    import matplotlib
    matplotlib.use("Agg")  # headless / no display
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # If the CSV carries multiple failure modes, plot the crux mode only and
    # note it; the by-failure CSV is better viewed one mode at a time.
    failure_modes = _ordered([r.get("failure_mode", "") for r in rows], ())
    if len(failure_modes) > 1:
        crux = "timeout_after_commit" if "timeout_after_commit" in failure_modes else failure_modes[0]
        rows = [r for r in rows if r.get("failure_mode", "") == crux]
        mode_label = f"{crux} (crux; {len(failure_modes)} modes in source)"
    else:
        mode_label = failure_modes[0] if failure_modes and failure_modes[0] else "all"

    agents = _ordered([r["agent"] for r in rows], AGENT_ORDER)
    retries = _ordered([r["retry_mode"] for r in rows], RETRY_ORDER)
    mitigations = _ordered([r["mitigation"] for r in rows], MITIGATION_ORDER)

    # index (agent, retry, mitigation) -> ivr
    ivr = {}
    for r in rows:
        ivr[(r["agent"], r["retry_mode"], r["mitigation"])] = float(r["ivr"])

    # green (safe, IVR=0) -> red (violating, IVR=1)
    cmap = LinearSegmentedColormap.from_list(
        "ivr", ["#1a9850", "#fee08b", "#d73027"])

    n = len(agents)
    if paper:
        figsize = (6.3, 2.6)
        fs_title, fs_tick, fs_cell, fs_label = 8, 6, 6, 7
    else:
        figsize = (3.6 * n + 0.8, 3.8)
        fs_title, fs_tick, fs_cell, fs_label = 11, 8, 9, 9
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes[0]

    for col, (ax, agent) in enumerate(zip(axes, agents)):
        grid = [[ivr.get((agent, rm, mit), float("nan")) for mit in mitigations]
                for rm in retries]
        im = ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(agent, fontsize=fs_title, fontweight="bold")
        ax.set_xticks(range(len(mitigations)))
        ax.set_xticklabels(mitigations, rotation=30, ha="right", fontsize=fs_tick)
        ax.set_yticks(range(len(retries)))
        # Only the leftmost panel carries the retry-mode labels; inner panels
        # would otherwise crowd the gap between subplots.
        ax.set_yticklabels(retries if col == 0 else [""] * len(retries),
                           fontsize=fs_tick)
        for i in range(len(retries)):
            for j in range(len(mitigations)):
                v = grid[i][j]
                if v != v:  # NaN
                    txt, color = "-", "#666666"
                else:
                    txt = f"{v:.2f}"
                    color = "white" if (v < 0.15 or v > 0.85) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        color=color, fontsize=fs_cell)
        ax.set_xticks([x - 0.5 for x in range(1, len(mitigations))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(retries))], minor=True)
        ax.grid(which="minor", color="white", linewidth=1.5)
        ax.tick_params(which="minor", length=0)

    axes[0].set_ylabel("retry mode", fontsize=fs_label)
    for ax in axes:
        ax.set_xlabel("mitigation", fontsize=fs_label)

    cbar = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
    cbar.set_label("IVR" if paper else "Idempotency Violation Rate (IVR)",
                   fontsize=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)

    if not paper:  # the paper figure gets its caption from LaTeX
        n_tasks = rows[0].get("n_tasks", "?")
        suptitle = title or (
            f"IdempotencyBench: IVR matrix  |  failure mode: {mode_label}  "
            f"|  n={n_tasks} tasks/cell")
        fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.02)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description="Plot IdempotencyBench IVR matrix.")
    ap.add_argument("--input", default=os.path.join(here, "results", "summary.csv"))
    ap.add_argument("--output", default=os.path.join(here, "figures", "ivr_matrix.png"))
    ap.add_argument("--title", default=None)
    ap.add_argument("--paper", action="store_true",
                    help="size for ACL figure* (6.3in), no suptitle")
    args = ap.parse_args()

    rows = load_rows(args.input)
    out = plot(rows, args.output, title=args.title, paper=args.paper)
    print(f"Wrote {out} from {os.path.relpath(args.input, here)} "
          f"({len(rows)} rows read)")


if __name__ == "__main__":
    main()
