# v34b copy of make_figures_v34.py: same data and markers; panels 0.33 in shorter;
# output fig_results_v34b.pdf.
#!/usr/bin/env python3
"""Results figure for the merged paper (fig:results), generated from the
analysis outputs; nothing in it is typed by hand.

Reads (read-only):
  results/analysis_v3/numbers.json     main-grid cells (bh.*, v2.*, xp.*, e1_6.*)
  results/analysis_v3/r2_wording.json  R2 same-batch cells (r2.A.*, r2.B.*)
  results/analysis_v3/r3_wording.json  R3 same-batch cells (r3.A.*, r3.B.*)
Writes:
  paper-merged/fig_results.pdf  (and fig_results.png for a quick look)

Left panel: per model, the share of resumed sessions that repeated a committed
step (graph IVR) with no resume note (amnesiac), with the two-sentence note
(bare_hint) and, where R2 ran, with its three paraphrases. Right panel: per
model, the share of single-session episodes that duplicated after a timeout, a
bare "internal server error", the "outcome unknown" status and, where R2 ran,
its three paraphrases. Every point carries its registered interval (graph- or
task-level 95% Wilson). Same-batch cells are used for the models R2 (Claude)
or R3 (OpenRouter models; Fable 5's error cells) covers, when a model's two
baseline cells are complete under the registered 90% rule; the other models
show their main-grid cells. A cell below the 90% rule is never drawn.
Black-and-white safe: conditions differ by marker shape and fill, not by
colour; a model whose cells all come from one same-batch launch sits on a
shaded slot (R2: plain grey; R3: hatched), and markers at exactly 0 are
staggered in height so that each stays visible. Deterministic output: the PDF
carries no creation date, and fonts are embedded as TrueType (no Type 3).
Called from build.sh before pdflatex.
"""
import json, math, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
# results/analysis_v3 of this archive (read-only); A3_DIR overrides the location
A3 = os.environ.get("A3_DIR") or os.path.join(os.path.dirname(HERE), "results", "analysis_v3")

MODELS = [("haiku", "Haiku 4.5"), ("sonnet", "Sonnet 5"), ("fable", "Fable 5"),
          ("gpt", "GPT-\n5.5"), ("gemini", "Gemini\n"), ("qwen", "Qwen3.6-\n27B")]
# x position per model; the last slot is wider so the two longest labels do not touch
XPOS = [0, 1, 2, 3, 4, 5.3]
PARAS = ("p1", "p2", "p3")

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.4,
    # STIX ships with matplotlib, so the output does not depend on system fonts,
    # and it matches the paper's Times-like text
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8, "axes.titlesize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.markersize": 3.6,
})


def load():
    nj = json.load(open(os.path.join(A3, "numbers.json")))["numbers"]
    out = [nj]
    for name in ("r2_wording.json", "r3_wording.json"):
        p = os.path.join(A3, name)
        d = json.load(open(p))["numbers"] if os.path.exists(p) else {}
        # the registered 90% rule: an incomplete R2 / R3 cell is never drawn
        out.append({k: e for k, e in d.items()
                    if not (isinstance(e, dict) and str(e.get("status", "")).startswith("incomplete"))})
    return tuple(out)


def cell(d, key):
    """(rate, lo, hi, n_units) or None; n_units = graphs or tasks of the interval."""
    e = d.get(key)
    if not isinstance(e, dict) or e.get("value") is None or "ci" not in e:
        return None
    units = e.get("T") or e.get("G")
    if not units:
        # bh.* restart cells record only n episodes: two samples per graph, and
        # one Sonnet 5 episode refused before the seam (31 episodes, 16 graphs)
        units = math.ceil(e["n"] / 2) if "n" in e else None
    return (float(e["value"]), float(e["ci"][0]), float(e["ci"][1]), units)


ZERO_STAGGER = 0.035   # height offset between successive markers at exactly 0 (one model slot)
SHADE = "0.93"          # background of a slot whose cells all come from the R2 same-batch launch
SHADE_R3 = "0.84"       # hatch colour of a slot whose cells all come from the R3 same-batch launch

# condition -> (label, marker, fill, grey, x offset)
LEFT = [
    # markers 0.2 apart (about 6 pt at the printed size, wider than a marker), and
    # exact zeros are staggered in height (draw), so four zeros (Haiku 4.5) stay apart
    ("amnesiac", "no note", "o", "full", "0.0", -0.40),
    ("bare_hint", "two-sentence note", "s", "none", "0.0", -0.20),
    ("hint_p1", "para. 1", "^", "none", "0.35", 0.0),
    ("hint_p2", "para. 2", "v", "none", "0.35", 0.20),
    ("hint_p3", "para. 3", "D", "none", "0.35", 0.40),
]
RIGHT = [
    ("toA", "“timed out”", "o", "none", "0.0", -0.42),
    ("e500", "“server error”", "o", "full", "0.0", -0.25),
    ("w3", "“unknown”", "s", "none", "0.0", -0.08),
    ("unknown_p1", "para. 1", "^", "none", "0.35", 0.09),
    ("unknown_p2", "para. 2", "v", "none", "0.35", 0.26),
    ("unknown_p3", "para. 3", "D", "none", "0.35", 0.43),
]


def same_batch(r2, r3, part, m, base, ctrl, metric, spec):
    """The same-batch cells of model m from R2 (Claude) or R3 (OpenRouter models;
    Fable 5's error cells), when both baseline cells are complete; else None."""
    for tag, d in (("R2", r2), ("R3", r3)):
        if cell(d, f"{tag[:2].lower()}.{part}.{m}.{base}:{metric}") and \
                cell(d, f"{tag[:2].lower()}.{part}.{m}.{ctrl}:{metric}"):
            out = {}
            for c, *_ in spec:
                v = cell(d, f"{tag[:2].lower()}.{part}.{m}.{c}:{metric}")
                if v:
                    out[c] = v
            return out, tag
    return None


def left_data(nj, r2, r3, m):
    """Restart cells. R2 same batch where it ran (Haiku 4.5, Fable 5), R3 where
    it ran and its baseline cells are complete (OpenRouter models), else the main
    grid: October for the Claude models, the cross-provider batch for the others."""
    out, src = {}, None
    sb = same_batch(r2, r3, "A", m, "amnesiac", "bare_hint", "givr", LEFT)
    if sb:
        out, src = sb
    else:
        b = "oct" if m in ("haiku", "sonnet", "fable") else "xp"
        src = "grid"
        for c in ("amnesiac", "bare_hint"):
            v = cell(nj, f"bh.{m}.{b}.{c}:givr")
            if v:
                out[c] = v
    return out, src


def right_data(nj, r2, r3, m):
    """Single-session cells. R2 same batch where it ran (Haiku 4.5, Sonnet 5),
    R3 where it ran (Fable 5 and the OpenRouter models), else the main grid:
    July for the Claude models, October for the others."""
    out, src = {}, None
    sb = same_batch(r2, r3, "B", m, "e500", "toA", "ivr", RIGHT)
    if sb:
        out, src = sb
    else:
        b = "v2" if m in ("haiku", "sonnet", "fable") else "xp"
        src = "grid"
        for c in ("toA", "e500"):
            v = cell(nj, f"{b}.{m}.base.none.{c}:ivr")
            if v:
                out[c] = v
        v = cell(nj, f"e1_6.{m}.base.none.w3:ivr")
        if v:
            out["w3"] = v
    return out, src


def draw(ax, spec, data_fn, nj, r2, r3, ylabel, unit):
    units = set()
    labels = []
    for i, (m, disp) in enumerate(MODELS):
        data, src = data_fn(nj, r2, r3, m)
        labels.append(disp if "\n" in disp else "%s\n%s" % (disp.rsplit(" ", 1)[0], disp.rsplit(" ", 1)[1]) if " " in disp else disp)
        if src == "R2":   # the data source, visible without reading the label
            ax.axvspan(XPOS[i] - 0.5, XPOS[i] + 0.5, color=SHADE, lw=0, zorder=0)
        elif src == "R3":   # R3 slots are hatched, so the two launches stay distinct in print
            ax.axvspan(XPOS[i] - 0.5, XPOS[i] + 0.5, facecolor="white", edgecolor=SHADE_R3,
                       hatch="///", lw=0, zorder=0)
        zeros = 0
        for c, lab, mk, fill, grey, dx in spec:
            v = data.get(c)
            if not v:
                continue
            rate, lo, hi, n = v
            if n:
                units.add(n)
            x = XPOS[i] + dx
            y = rate
            if rate == 0.0:   # stagger exact zeros so that each marker stays visible
                y = ZERO_STAGGER * (zeros % 2)
                zeros += 1
            ax.plot([x, x], [lo, hi], color=grey, lw=0.8, solid_capstyle="butt", zorder=2)
            ax.plot([x], [y], marker=mk, color=grey, markerfacecolor=grey if fill == "full" else "white",
                    markeredgewidth=0.9, linestyle="none", zorder=3)
        if i < len(MODELS) - 1:
            ax.axvline((XPOS[i] + XPOS[i + 1]) / 2, color="0.85", lw=0.5, zorder=1)
    ax.set_xticks(XPOS)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.04, 1.06)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlim(-0.6, XPOS[-1] + 0.6)
    ax.yaxis.grid(True, color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    nstr = ", ".join(str(u) for u in sorted(units, reverse=True))
    ax.set_ylabel("%s (n = %s %s)" % (ylabel, nstr, unit), labelpad=3)
    handles = [Line2D([], [], marker=mk, color=grey, markerfacecolor=grey if fill == "full" else "white",
                      markeredgewidth=0.9, linestyle="none", label=lab) for _, lab, mk, fill, grey, _ in spec]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.30), frameon=False, ncol=3,
              columnspacing=0.9, borderaxespad=0.0, handlelength=0.8, handletextpad=0.3, fontsize=8)


def main():
    nj, r2, r3 = load()
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.3, 2.42))
    draw(axl, LEFT, left_data, nj, r2, r3, "gIVR", "graphs")
    draw(axr, RIGHT, right_data, nj, r2, r3, "IVR", "tasks")
    axl.set_title("(a) Restart: resume note", loc="left")
    axr.set_title("(b) Retry: error text after a commit", loc="left")
    fig.subplots_adjust(left=0.085, right=0.997, top=0.92, bottom=0.39, wspace=0.22)
    out = os.path.join(os.environ.get("FIG_OUT_DIR") or HERE, "fig_results_v34b")
    meta = {"CreationDate": None, "ModDate": None, "Creator": None, "Producer": None,
            "Title": "Results figure"}
    fig.savefig(out + ".pdf", metadata=meta, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote %s.pdf" % out)


if __name__ == "__main__":
    main()
