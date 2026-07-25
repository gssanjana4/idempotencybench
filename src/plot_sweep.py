"""Model-results figure: IVR by failure mode per model, Wilson 95% CIs.
Source: results/sweep_summary.csv (base prompt, mitigation=none cells).
Usage: python3 src/plot_sweep.py [--paper]
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_LABELS = {
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "claude-fable-5": "Fable 5",
    "claude-sonnet-5": "Sonnet 5",
}
MODE_ORDER = ["none", "timeout_before_commit", "timeout_after_commit",
              "error_500_after_commit"]
MODE_LABELS = ["none", "timeout\nbefore commit", "timeout\nafter commit",
               "error 500\nafter commit"]


def main():
    paper = "--paper" in sys.argv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(HERE, "results",
                                                 "sweep_summary.csv"))))
    rows = [r for r in rows if r["prompt"] == "base"
            and r["mitigation"] == "none"]

    models = list(MODEL_LABELS)
    fig, ax = plt.subplots(figsize=(6.3, 2.4) if paper else (8, 4))
    width = 0.26
    for mi, model in enumerate(models):
        xs, ys, lo, hi = [], [], [], []
        for xi, mode in enumerate(MODE_ORDER):
            r = next(r for r in rows if r["model"] == model
                     and r["failure_mode"] == mode)
            xs.append(xi + (mi - 1) * width)
            ys.append(float(r["ivr"]))
            lo.append(float(r["ivr"]) - float(r["ivr_ci_lo"]))
            hi.append(float(r["ivr_ci_hi"]) - float(r["ivr"]))
        ax.bar(xs, ys, width * 0.92, label=MODEL_LABELS[model],
               yerr=[lo, hi], capsize=2,
               error_kw={"linewidth": 0.8})
    fs = 7 if paper else 10
    ax.set_xticks(range(len(MODE_ORDER)))
    ax.set_xticklabels(MODE_LABELS, fontsize=fs)
    ax.set_ylabel("IVR", fontsize=fs)
    ax.set_ylim(0, 1.0)
    ax.tick_params(labelsize=fs)
    ax.legend(fontsize=fs, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    if not paper:
        ax.set_title("IVR by failure mode (base prompt, no mitigation; "
                     "n=64/cell, Wilson 95% CI)")
    out = os.path.join(HERE, "figures",
                       "sweep_ivr_paper.pdf" if paper else "sweep_ivr.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
