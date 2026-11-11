"""Registered analysis of R5 (prereg_r5.md). Reads only the R5 out_dir.

  python3 v32-experiments/src/analysis_r5.py [--config ...]
  -> <out_dir>/analysis/{r5_real_errors.json, r5_cells.csv, r5_tests.csv, r5_refusals.csv}

Units: the task template (8) is the unit of the registered tests (exact
sign-flip over the 2^8 sign patterns of the per-template mean differences);
the task (32) test is reported next to it (paired sign-flip permutation,
10,000 flips, seed 20261025, p = (hits + 1)/(N + 1), the R2/R3 scheme).

Class rates are means over the class's texts of each text's rate, so every
text weighs the same. With no data it writes {"status": "no data"} and exits 0.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.dirname(HERE)
REPO = os.path.dirname(V32)
for d in (os.path.join(REPO, "src"), HERE):
    if d not in sys.path:
        sys.path.insert(0, d)
import r5_texts                                  # noqa: E402

E500, W3 = "error_500_after_commit", "status_unknown_after_commit"


# ---------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def signflip_exact(diffs, alternative):
    """Exact sign-flip over unit means; alternative 'less' (mean < 0) or 'two-sided'."""
    obs = sum(diffs) / len(diffs)
    cnt = tot = 0
    for s in itertools.product((1, -1), repeat=len(diffs)):
        v = sum(a * b for a, b in zip(s, diffs)) / len(diffs)
        tot += 1
        cnt += (v <= obs + 1e-12) if alternative == "less" else (abs(v) >= abs(obs) - 1e-12)
    return obs, cnt / tot


def signflip_mc(diffs, alternative, n=10000, seed=20261025):
    obs = sum(diffs)
    rng = random.Random(seed)
    hit = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs)
        hit += (s <= obs + 1e-12) if alternative == "less" else (abs(s) >= abs(obs) - 1e-12)
    return obs / len(diffs), (hit + 1) / (n + 1)


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


# ---------------------------------------------------------------- data
def read_rows(raw_path):
    by = {}
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                by[r["episode_id"]] = r          # last row wins (tier1 rule)
    return list(by.values())


def rate_by(eps, key):
    acc = defaultdict(list)
    for e in eps:
        acc[e[key]].append(1.0 if e["violations"] > 0 else 0.0)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def class_unit_rates(cells, fms, key):
    """{unit: mean over texts (present on that unit) of the text's unit rate}."""
    per_text = [rate_by(cells.get(fm, []), key) for fm in fms]
    units = set().union(*[set(p) for p in per_text]) if per_text else set()
    out = {}
    for u in units:
        xs = [p[u] for p in per_text if u in p]
        if xs:
            out[u] = sum(xs) / len(xs)
    return out


def contrast(cells, a_fms, b_fms, alternative, perm_n, seed):
    """d = rate(A) - rate(B); 'less' tests A < B."""
    out = {}
    for key, unit in (("template", "template"), ("task", "task_id")):
        a = class_unit_rates(cells, a_fms, unit)
        b = class_unit_rates(cells, b_fms, unit)
        shared = sorted(set(a) & set(b))
        diffs = [a[u] - b[u] for u in shared]
        if not diffs:
            out[key] = {"d": None, "p": None, "units": 0}
            continue
        if key == "template":
            d, p = signflip_exact(diffs, alternative)
        else:
            d, p = signflip_mc(diffs, alternative, perm_n, seed)
        out[key] = {"d": round(d, 4), "p": round(p, 6), "units": len(diffs),
                    "plus": sum(1 for x in diffs if x > 1e-12), "minus": sum(1 for x in diffs if x < -1e-12)}
    return out


def cell_stats(eps, planned):
    n = len(eps)
    k = sum(1 for e in eps if e["violations"] > 0)
    T = len({e["task_id"] for e in eps})
    p = k / n if n else None
    fired = [e for e in eps if e.get("failure_fired")]
    return {"n": n, "planned": planned, "k": k, "ivr": None if p is None else round(p, 4),
            "ci_episode": [round(x, 4) for x in wilson(k, n)],
            "ci_task": [round(x, 4) for x in (wilson(p * T, T) if n else (0.0, 1.0))],
            "T": T, "templates": len({e["template"] for e in eps}),
            "readback_after_failure": round(sum(1 for e in fired if e.get("readback_after_failure")) / len(fired), 4) if fired else None,
            "retried": round(sum(1 for e in fired if e.get("retried")) / len(fired), 4) if fired else None,
            "success": round(sum(1 for e in eps if e.get("achieved")) / n, 4) if n else None,
            "truncated": sum(1 for e in eps if e.get("truncated")),
            "turn_cap_hit": sum(1 for e in eps if e.get("turn_cap_hit"))}


def analyse(cfg) -> dict:
    r5 = cfg["r5"]
    cat = r5_texts.load(r5["catalogue"] if os.path.isabs(r5["catalogue"]) else os.path.join(REPO, r5["catalogue"]))
    ids = r5_texts.by_id(cat)
    rows = read_rows(os.path.join(cfg["out_dir"], "raw.jsonl"))
    if not rows:
        return {"status": "no data"}
    refusals = [r for r in rows if r.get("refusal")]
    rows = [r for r in rows if not r.get("refusal")]
    planned = int(cfg.get("expect_per_cell", 64))
    min_n = math.ceil(float(r5["min_fraction"]) * planned)
    seed, perm_n, alpha, floor = int(r5["perm_seed"]), int(r5["perm_n"]), float(r5["alpha"]), float(r5["floor_ivr"])
    inj = [r5_texts.fm_of(i) for i in r5["injected"]]
    cls = {fm: ids[r5_texts.id_of(fm)]["class"] for fm in inj}
    U = [fm for fm in inj if cls[fm] == "U"]
    G = [fm for fm in inj if cls[fm] == "G"]
    N = [fm for fm in inj if cls[fm] == "N"]
    R = [fm for fm in inj if cls[fm] == "R"]
    U_nocheck = [fm for fm in U if not ids[r5_texts.id_of(fm)].get("instructs_check")]
    out = {"status": "ok", "exp": cfg["exp"], "planned_per_cell": planned, "min_n": min_n,
           "classes": {"U": U, "G": G, "N": N, "R": R}, "models": {}, "refusals": len(refusals)}
    tests_csv, cells_csv = [], []
    for m in cfg["models"]:
        eps = [r for r in rows if r["model"] == m]
        refused = defaultdict(int)
        for r in refusals:
            if r["model"] == m:
                refused[r["failure_mode"]] += 1
        cells = defaultdict(list)
        for e in eps:
            cells[e["failure_mode"]].append(e)
        stats = {}
        for fm in inj + [E500, W3]:
            st = cell_stats(cells.get(fm, []), planned)
            st["refusals"] = refused.get(fm, 0)
            st["ok90"] = st["n"] + st["refusals"] >= min_n
            st["class"] = cls.get(fm, r5_texts.ANCHORS.get(fm))
            stats[fm] = st
            cells_csv.append({"model": m, "failure_mode": fm, **{k: (json.dumps(v) if isinstance(v, list) else v)
                                                                   for k, v in st.items()}})
        role = "primary" if m in r5["primary_models"] else "secondary"
        res = {"role": role, "cells": stats}

        def class_rate(fms):
            xs = [stats[f]["ivr"] for f in fms if stats[f]["ivr"] is not None]
            return round(sum(xs) / len(xs), 4) if xs else None
        res["class_ivr"] = {"U": class_rate(U), "G": class_rate(G), "N": class_rate(N), "R": class_rate(R),
                            "U_no_check": class_rate(U_nocheck)}
        res["spread"] = {c: (round(max(stats[f]["ivr"] for f in fs) - min(stats[f]["ivr"] for f in fs), 4)
                             if all(stats[f]["ivr"] is not None for f in fs) else None)
                         for c, fs in (("U", U), ("G", G))}
        # ---- primary: R5-P1 (U < G), template level, one-sided
        p1_ok = all(stats[f]["ok90"] for f in U + G)
        t = contrast(cells, U, G, "less", perm_n, seed) if p1_ok else None
        if not p1_ok:
            verdict = "not evaluable"
        else:
            verdict = "supported" if t["template"]["p"] is not None and t["template"]["p"] < alpha else "not supported"
        reading = None
        if p1_ok and res["class_ivr"]["G"] is not None and res["class_ivr"]["G"] <= floor:
            reading = (f"floor: generic-class IVR {res['class_ivr']['G']} <= {floor}; "
                       f"'not supported' is read as no duplication to remove")
        res["P1"] = {"test": "U < G (template-level exact sign-flip, one-sided)", "contrast": t,
                     "verdict": verdict if role == "primary" else f"{verdict} (secondary model, no confirmatory claim)",
                     "reading": reading}
        tests_csv.append({"model": m, "test": "P1 U<G", "family": "primary", "alternative": "less",
                          **_flat(t), "holm_p": None, "verdict": res["P1"]["verdict"]})
        # ---- secondary block (Holm within model over the template-level p's)
        sec = []
        for fm in U:
            sec.append((f"S1 {r5_texts.id_of(fm)} vs anchor 500", [fm], [E500], "two-sided"))
        sec.append(("S2 G class vs anchor 500", G, [E500], "two-sided"))
        sec.append(("S3 U class vs anchor W3", U, [W3], "two-sided"))
        sec.append(("S4 N text vs G class", N, G, "two-sided"))
        sec.append(("S5 R text vs G class", R, G, "two-sided"))
        sec.append(("S6 U without check-instructing text vs G (sensitivity)", U_nocheck, G, "less"))
        block = []
        for name, a_fms, b_fms, alt in sec:
            ok = all(stats[f]["ok90"] for f in a_fms + b_fms)
            c = contrast(cells, a_fms, b_fms, alt, perm_n, seed) if ok else None
            block.append({"test": name, "alternative": alt, "evaluable": ok, "contrast": c})
        ev = [b for b in block if b["evaluable"] and b["contrast"]["template"]["p"] is not None]
        adj = holm([b["contrast"]["template"]["p"] for b in ev])
        for b, h in zip(ev, adj):
            b["holm_p_template"] = round(h, 6)
        res["secondary"] = block
        for b in block:
            tests_csv.append({"model": m, "test": b["test"], "family": "secondary", "alternative": b["alternative"],
                              **_flat(b["contrast"]), "holm_p": b.get("holm_p_template"),
                              "verdict": None if b["evaluable"] else "not evaluable"})
        out["models"][m] = res
    out["_csv"] = (cells_csv, tests_csv, refusals)
    return out


def _flat(c):
    if not c:
        return {"d_template": None, "p_template": None, "d_task": None, "p_task": None}
    return {"d_template": c["template"]["d"], "p_template": c["template"]["p"],
            "templates_plus": c["template"].get("plus"), "templates_minus": c["template"].get("minus"),
            "d_task": c["task"]["d"], "p_task": c["task"]["p"]}


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(V32, "configs", "r5_real_errors.json"))
    a = ap.parse_args(argv)
    import tier1
    cfg = tier1.load_config(a.config)
    out = analyse(cfg)
    adir = os.path.join(cfg["out_dir"], "analysis")
    os.makedirs(adir, exist_ok=True)
    csvs = out.pop("_csv", None)
    with open(os.path.join(adir, "r5_real_errors.json"), "w") as f:
        json.dump(out, f, indent=1)
    if csvs:
        _write_csv(os.path.join(adir, "r5_cells.csv"), csvs[0])
        _write_csv(os.path.join(adir, "r5_tests.csv"), csvs[1])
        _write_csv(os.path.join(adir, "r5_refusals.csv"),
                   [{k: r.get(k) for k in ("episode_id", "model", "failure_mode", "refusal_error")} for r in csvs[2]])
    if out["status"] != "ok":
        print(f"R5 analysis: {out['status']} -> {adir}")
        return 0
    for m, res in out["models"].items():
        p1 = res["P1"]
        c = p1["contrast"]
        tl = c["template"] if c else {}
        print(f"{m} [{res['role']}] class IVR {res['class_ivr']}  P1: d={tl.get('d')} p={tl.get('p')} -> {p1['verdict']}"
              + (f"  ({p1['reading']})" if p1["reading"] else ""))
    print(f"-> {adir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
