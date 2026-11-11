"""SENSITIVITY (added 2026-10-26, after the E1-6 results were seen; not
confirmatory): E1-6 compares October W1/W3 runs with July
(v2) bare baselines. This re-tests the E1-6 contrasts against an October
rerun of the bare baselines made through the same Tier-1 harness/SDK path
(configs/e1_6_oct.json -> results/tier1/e1_6_oct/raw.jsonl).

Run by analysis/run_all_v3.sh. If the October data does not exist, this
writes nothing and removes a stale results/analysis_v3/sens_oct.json, so
numbers.json is exactly what it was without it.

Keys (all under sens.oct., never mixed with the registered e1_6.* keys):
  sens.oct.<m>.base.none.{e500,toA}:ivr / :ivr_ci_ep / :readback   October cells
  sens.oct.<m>.w3_minus_oct_e500:{d,p}       IVR(W3) - IVR(Oct bare 500)
  sens.oct.<m>.w1_minus_oct_e500:{d,p}       IVR(W1) - IVR(Oct bare 500)
  sens.oct.<m>.oct_e500_minus_oct_toA:{d,p}  IVR(Oct 500) - IVR(Oct timeout)
  sens.oct.<m>.drift.{e500,toA}:{d,p}        IVR(Oct) - IVR(July v2), same cell
Same protocol as the registered tests (task-paired sign-flip permutation,
10,000 perms, seed 20261025; task-level Wilson CI). `reading` gives the
direction/significance only; there is no registered verdict.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (OUT, RES, Numbers, rate_stats, perm_test, task_rates, short_model,
                       read_jsonl, dedupe, rel, f, fp, prefer_v2)
import v3_load

OCT_RAW = prefer_v2(os.path.join(RES, "tier1", "e1_6_oct", "raw.jsonl"))   # raw_v2: C1 re-score
E16_RAW = prefer_v2(os.path.join(RES, "tier1", "e1_6", "raw.jsonl"))
OUT_JSON = os.path.join(OUT, "sens_oct.json")
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]
STATUS = "sensitivity (added 2026-10-26 after seeing results; not confirmatory)"
EXPECT = 64


def cells_from(path, src):
    """Cells keyed by (model, mitigation, failure_mode), prompt == "base" only.

    Keyed internally by (model, prompt, mitigation, failure_mode) so prompts are never
    pooled: results/sweep_raw_v2.jsonl also holds prompt=careful for mitigation=none, and
    pooling it into the July baseline made the drift tests compare a base+careful mix
    (fixed 2026-10-26; regression check: analysis/v3_sens_oct_fix.py)."""
    full = {}
    for r in dedupe(read_jsonl(path)).values():
        full.setdefault((r["model"], r["prompt"], r["mitigation"], r["failure_mode"]), []).append(
            v3_load._ep(r, src))
    return {(m, mit, fm): eps for (m, pr, mit, fm), eps in full.items() if pr == "base"}


def reading(r):
    if r["p"] is None:
        return "not testable"
    s = "significant" if r["p"] < 0.05 else "null (p >= 0.05)"
    return s if r["p"] >= 0.05 else f"{s}, {'positive' if r['d'] > 0 else 'negative'}"


def main():
    if not os.path.exists(OCT_RAW) or not read_jsonl(OCT_RAW):
        if os.path.exists(OUT_JSON):
            os.remove(OUT_JSON)
        print("no October baseline data (results/tier1/e1_6_oct/raw.jsonl): sensitivity rows skipped")
        return
    N = Numbers()
    octc = cells_from(OCT_RAW, "e1_6_oct")
    e16 = cells_from(E16_RAW, "e1_6")
    v2 = cells_from(os.path.join(RES, "sweep_raw_v2.jsonl"), "v2")
    so, s16, sv2 = rel(OCT_RAW), rel(E16_RAW), "results/sweep_raw_v2.jsonl"
    for m in MODELS:
        sm = short_model(m)
        o500 = octc.get((m, "none", "error_500_after_commit"), [])
        oto = octc.get((m, "none", "timeout_after_commit"), [])
        for tag, eps in (("e500", o500), ("toA", oto)):
            st = rate_stats(eps)
            nm = f"sens.oct.{sm}.base.none.{tag}"
            if not eps:
                continue
            part = None if st["n"] >= EXPECT else f"partial ({st['n']}/{EXPECT})"
            N.cell(nm, st, so, status=part or STATUS)
            rb = rate_stats(eps, "rb")
            N.put(nm + ":readback", rb["p"], rb["ci_task"] if rb["n"] else None,
                  "wilson95_task(p*T,T)", rb["n"], so, k=rb["k"], status=STATUS)
            print(f"  {nm:34s} IVR {f(st['p'])} (n={st['n']}, T={st['T']})  read-back {f(rb['p'])}")
        tests = [("w3_minus_oct_e500", o500, e16.get((m, "none", "status_unknown_after_commit"), []),
                  "IVR(W3) - IVR(October bare 500)", [s16, so]),
                 ("w1_minus_oct_e500", o500, e16.get((m, "none", "error_500_body_after_commit"), []),
                  "IVR(W1) - IVR(October bare 500)", [s16, so]),
                 ("oct_e500_minus_oct_toA", oto, o500,
                  "IVR(October bare 500) - IVR(October timeout)", [so]),
                 ("drift.e500", v2.get((m, "none", "error_500_after_commit"), []), o500,
                  "IVR(October bare 500) - IVR(July v2 bare 500)", [sv2, so]),
                 ("drift.toA", v2.get((m, "none", "timeout_after_commit"), []), oto,
                  "IVR(October timeout) - IVR(July v2 timeout)", [sv2, so])]
        for tag, x, y, label, src in tests:
            if not x or not y:
                continue
            r = perm_test(task_rates(x), task_rates(y))
            pre = f"sens.oct.{sm}.{tag}"
            N.test(pre, r, sorted(src), status=STATUS, note=label)
            N.d[pre + ":p"]["reading"] = reading(r)
            print(f"  {pre:34s} d={f(r['d'])} p={fp(r['p'])} T={r['T']}  {label}")
    N.write(OUT_JSON)
    print(f"sens_oct.json: {len(N.d)} values")


if __name__ == "__main__":
    main()
