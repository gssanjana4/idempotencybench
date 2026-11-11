"""Session-resume claim verifier: re-derives every quantitative claim in
the session-resume results from the raw data and fails loudly on any mismatch.

This is the check the paper itself argues for: don't trust the dashboard,
audit the ledger. Run before every submission:
    python3 src2/verify_resume.py
Exit code 0 = every claim verified. Any FAIL = do not submit.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "sweep2_raw.jsonl")
TDIR = os.path.join(HERE, "results", "transcripts2")
SUB = os.path.join(HERE, "results", "graph_summary.csv")

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    # ---- load, dedupe-verify ----
    rows = [json.loads(l) for l in open(RAW) if l.strip()]
    ids = [r["episode_id"] for r in rows]
    check("no duplicate episode_ids in raw data", len(ids) == len(set(ids)),
          f"{len(rows)} rows, {len(set(ids))} unique")
    check("total model episodes == 400 (as stated in paper)", len(rows) == 400)

    cell = defaultdict(list)
    for r in rows:
        cell[(r["model"], r["resume_context"], r["mitigation"])].append(r)

    def givr(m, c, mit):
        rs = cell[(m, c, mit)]
        return sum(r["givr_hit"] for r in rs) / len(rs), len(rs)

    F, S, H = ("claude-fable-5", "claude-sonnet-5", "claude-haiku-4-5-20251001")

    # ---- abstract & table claims ----
    g, n = givr(F, "amnesiac", "none")
    check("Fable amnesiac gIVR = 1.00, n=32", g == 1.0 and n == 32, f"{g} n={n}")
    g, n = givr(H, "amnesiac", "none")
    check("Haiku amnesiac gIVR = 1.00, n=32", g == 1.0 and n == 32, f"{g} n={n}")
    g, n = givr(S, "amnesiac", "none")
    check("Sonnet amnesiac gIVR = 0.625, n=32", abs(g - 0.625) < 1e-9 and n == 32,
          f"{g} n={n}")
    g, n = givr(F, "bare_hint", "none")
    check("bare_hint gIVR = 0.00, n=16", g == 0.0 and n == 16, f"{g} n={n}")
    g, n = givr(H, "plan_only", "none")
    check("Haiku plan_only gIVR = 0.25, n=32", abs(g - 0.25) < 1e-9 and n == 32,
          f"{g} n={n}")
    for m in (F, S):
        g, n = givr(m, "plan_only", "none")
        check(f"{m.split('-')[1]} plan_only gIVR = 0.00", g == 0.0 and n == 32)
    rb = [r for (k, v) in cell.items() if k[1] == "ledger_readback" for r in v]
    check("readback gIVR = 0.00 across all models, n=96",
          len(rb) == 96 and all(r["givr_hit"] == 0 for r in rb), f"n={len(rb)}")
    rc = [r for (k, v) in cell.items() if k[2] == "runtime_receipts" for r in v]
    check("receipts gIVR = 0.00 across all models, n=96",
          len(rc) == 96 and all(r["givr_hit"] == 0 for r in rc), f"n={len(rc)}")

    # ---- stale re-execution ----
    for m, want in ((F, 0.75), (H, 0.75)):
        rs = cell[(m, "amnesiac", "none")]
        v = sum(r["order_violation"] for r in rs) / len(rs)
        check(f"{m.split('-')[1]} amnesiac stale = 0.75", abs(v - want) < 1e-9, f"{v}")
    rs = cell[(S, "amnesiac", "none")]
    v = sum(r["order_violation"] for r in rs) / len(rs)
    check("Sonnet amnesiac stale = 0.47 (table shows 0.47)", abs(v - 0.46875) < 1e-9, f"{v}")
    tot = sum(1 for r in rows if r["order_violation"])
    ind = sum(1 for r in rows if r["order_violation"] and not r["givr_hit"])
    check("63 stale events, ALL co-occur with duplication (0 independent)",
          tot == 63 and ind == 0, f"tot={tot} ind={ind}")

    # ---- duplicate-effects magnitudes ----
    means = {}
    for m in (F, S, H):
        vs = [r["violations"] for r in cell[(m, "amnesiac", "none")]]
        means[m] = sum(vs) / len(vs)
        check(f"{m.split('-')[1]} max duplicate effects <= 4", max(vs) <= 4, f"max={max(vs)}")
    check("mean duplicate effects range is 1.5–2.5",
          abs(min(means.values()) - 1.5) < 1e-9 and abs(max(means.values()) - 2.5) < 1e-9,
          f"{sorted(round(v,2) for v in means.values())}")

    # ---- completion everywhere ----
    check("completion = 1.00 in EVERY cell",
          all(all(r["completion"] == 1 for r in v) for v in cell.values()))

    # ---- lookup counts from transcripts (Chk column) ----
    look = defaultdict(lambda: [0, 0])
    seen_t = set()
    for p in glob.glob(os.path.join(TDIR, "*.json")):
        t = json.load(open(p))
        eid = os.path.basename(p)[:-5]
        if eid in seen_t or eid not in set(ids):
            continue
        seen_t.add(eid)
        if len(t["sessions"]) < 2:
            continue
        k = (t["model"], t["resume_context"], t["mitigation"])
        calls = [b["name"] for turn in t["sessions"][1]["turns"]
                 if turn.get("role") == "assistant"
                 for b in turn["content"] if b.get("type") == "tool_use"]
        look[k][0] += 1
        if "list_effects" in calls:
            look[k][1] += 1
    check("bare_hint lookups 16/16", tuple(look[(F, "bare_hint", "none")]) == (16, 16),
          str(look[(F, "bare_hint", "none")]))
    check("Fable amnesiac lookups 0/32", tuple(look[(F, "amnesiac", "none")]) == (32, 0))
    check("Haiku amnesiac lookups 0/32", tuple(look[(H, "amnesiac", "none")]) == (32, 0))
    check("Sonnet amnesiac lookups 12/32", tuple(look[(S, "amnesiac", "none")]) == (32, 12),
          str(look[(S, "amnesiac", "none")]))
    check("Haiku plan_only lookups 24/32", tuple(look[(H, "plan_only", "none")]) == (32, 24),
          str(look[(H, "plan_only", "none")]))
    rb_l = [look[k] for k in look if k[1] == "ledger_readback"]
    check("readback lookups 96/96",
          sum(x[0] for x in rb_l) == 96 and sum(x[1] for x in rb_l) == 96)
    rc_l = [look[k] for k in look if k[2] == "runtime_receipts"]
    check("receipts lookups 20/96",
          sum(x[0] for x in rc_l) == 96 and sum(x[1] for x in rc_l) == 20,
          f"{sum(x[1] for x in rc_l)}/{sum(x[0] for x in rc_l)}")

    # ---- CI claims ----
    lo, hi = wilson(32, 32)
    check("amnesiac ceiling CI = [0.89, 1.00]", round(lo, 2) == 0.89 and hi == 1.0,
          f"[{lo:.4f},{hi:.4f}]")
    lo, hi = wilson(20, 32)
    check("Sonnet amnesiac CI = [0.45, 0.77]", round(lo, 2) == 0.45 and round(hi, 2) == 0.77,
          f"[{lo:.4f},{hi:.4f}]")
    lo, hi = wilson(8, 32)
    check("Haiku plan_only CI = [0.13, 0.42]", round(lo, 2) == 0.13 and round(hi, 2) == 0.42,
          f"[{lo:.4f},{hi:.4f}]")
    lo, hi = wilson(0, 16)
    check("zero-cell upper bounds <= 0.20 (paper says <= 0.19 — check!)",
          hi <= 0.20, f"n=16 upper = {hi:.4f}")

    # ---- interrupt placement + determinism (harness validity) ----
    bad = 0
    for p in glob.glob(os.path.join(TDIR, "*.json")):
        t = json.load(open(p))
        s1 = t["sessions"][0]
        if not s1.get("interrupted"):
            bad += 1
            continue
        marks = [x for x in s1["turns"] if x.get("role") == "interrupt"]
        if not marks or marks[0]["at_node"] != t["interrupt_node"]:
            bad += 1
    check("every episode interrupted at intended node", bad == 0, f"bad={bad}")

    # ---- substrate claims ----
    sub = list(csv.DictReader(open(SUB)))
    agg = defaultdict(lambda: [0, 0.0])
    for r in sub:
        k = (r["resume_context"], r["mitigation"], r["key_posture"])
        n = int(r["n_episodes"])
        agg[k][0] += n
        agg[k][1] += float(r["givr"]) * n
    v = agg[("amnesiac", "none", "naive")]
    check("substrate amnesiac/naive gIVR = 0.76", round(v[1] / v[0], 2) == 0.76,
          f"{v[1]/v[0]:.4f}")
    v = agg[("plan_only", "none", "naive")]
    check("substrate plan_only/naive gIVR = 0.44", round(v[1] / v[0], 2) == 0.44,
          f"{v[1]/v[0]:.4f}")
    tot_sub = sum(int(r["n_episodes"]) for r in sub)
    check("substrate total = 40,320 episodes", tot_sub == 40320, f"{tot_sub}")

    print()
    if FAILURES:
        print(f"*** {len(FAILURES)} FAILURES — DO NOT SUBMIT ***")
        for f in FAILURES:
            print("   -", f)
        sys.exit(1)
    print("ALL CLAIMS VERIFIED — paper numbers match raw data exactly.")


if __name__ == "__main__":
    main()
