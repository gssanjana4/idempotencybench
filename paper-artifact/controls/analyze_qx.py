#!/usr/bin/env python3
"""Analyse a quick-experiment run (or any tier-1 raw.jsonl) the way the paper does:
per cell k/n with the task-level (or graph-level) 95% Wilson interval, lookups,
completion and omission; each cell against a same-run baseline with the paired
sign-flip permutation test over tasks/graphs (10,000 flips, two-sided), plus the
exact template-level sign-flip test for single-session cells.

  python3 -I -B scripts/analyze_qx.py RAW.jsonl --base error_500_after_commit
  python3 -I -B scripts/analyze_qx.py RAW.jsonl --base amnesiac
Self-check on released data (must print Qwen 42/64 -> 4/64, as in the paper):
  python3 -I -B scripts/analyze_qx.py ../v27_fix/results/r3_wording/raw.jsonl --base error_500_after_commit --model qwen
"""
import argparse
import itertools
import json
import math
import random
from collections import defaultdict


def wilson_cluster(vals_by_unit, z=1.96):
    """Wilson interval on the unit means (task- or graph-level), as in the paper:
    n = number of units, p = mean of unit means."""
    units = list(vals_by_unit.values())
    n = len(units)
    if n == 0:
        return (0.0, 1.0)
    p = sum(sum(u) / len(u) for u in units) / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def signflip(diffs, n_flips=10000, seed=0):
    diffs = [d for d in diffs]
    obs = abs(sum(diffs))
    if all(d == 0 for d in diffs):
        return 1.0
    if len(diffs) <= 16:   # exact
        hits = tot = 0
        for signs in itertools.product((1, -1), repeat=len(diffs)):
            tot += 1
            hits += abs(sum(s * d for s, d in zip(signs, diffs))) >= obs - 1e-12
        return hits / tot
    rng = random.Random(seed)
    hits = sum(abs(sum(d if rng.random() < 0.5 else -d for d in diffs)) >= obs - 1e-12
               for _ in range(n_flips))
    return (hits + 1) / (n_flips + 1)


def cell_of(r):
    return r.get("resume_context") if r.get("kind") == "restart" else r.get("failure_mode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--base", required=True, help="baseline cell (failure_mode or resume_context)")
    ap.add_argument("--model", default="", help="substring filter on model")
    a = ap.parse_args()
    rows = {}
    for line in open(a.raw):
        line = line.strip()
        if line:
            r = json.loads(line)
            rows[r["episode_id"]] = r          # last row per episode, as released
    rows = [r for r in rows.values() if not r.get("refusal") and a.model in r.get("model", "")]
    by = defaultdict(lambda: defaultdict(list))   # (model, cell) -> unit -> [dup]
    meta = defaultdict(lambda: {"look": 0, "done": 0, "omit": 0, "n": 0, "tmpl": {}})
    for r in rows:
        key = (r["model"], cell_of(r))
        unit = r.get("task_id") or r.get("graph_id")
        dup = 1 if (r.get("givr_hit") or (r.get("violations") or 0) > 0) else 0
        by[key][unit].append(dup)
        m = meta[key]
        m["n"] += 1
        m["look"] += 1 if (r.get("readback_after_failure") or r.get("session2_lookup")) else 0
        m["done"] += 1 if r.get("achieved") else 0
        m["omit"] += 1 if r.get("omission") else 0
        m["tmpl"][unit] = r.get("template") or r.get("topology") or r.get("family")
    models = sorted({k[0] for k in by})
    for model in models:
        print(f"== {model}")
        base = by.get((model, a.base))
        for (mm, cell), units in sorted(by.items()):
            if mm != model:
                continue
            k = sum(sum(u) for u in units.values())
            n = sum(len(u) for u in units.values())
            lo, hi = wilson_cluster(units)
            m = meta[(mm, cell)]
            line = (f"  {cell:40} dup {k}/{n} = {k / n:.2f} [{lo:.2f}, {hi:.2f}]  "
                    f"looked {m['look']}/{n}  complete {m['done']}/{n}  omission {m['omit']}/{n}")
            common = sorted(set(units) & set(base)) if base is not None else []
            if common and cell != a.base:
                diffs = [sum(units[u]) / len(units[u]) - sum(base[u]) / len(base[u]) for u in common]
                d = sum(diffs) / len(diffs) if diffs else float("nan")
                p = signflip(diffs)
                tm = defaultdict(list)
                for u, df in zip(common, diffs):
                    tm[m["tmpl"].get(u)].append(df)
                tdiffs = [sum(v) / len(v) for v in tm.values()]
                pt = signflip(tdiffs) if len(tdiffs) > 2 else float("nan")
                line += f"  | vs {a.base}: d={d:+.2f} p={p:.4f} (units {len(common)}); template-level p={pt:.3f} ({len(tdiffs)} groups)"
            print(line)


if __name__ == "__main__":
    main()
