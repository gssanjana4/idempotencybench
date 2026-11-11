"""FIX for the sens.oct.*.drift.* keys written by analysis/v3_sens_oct.py
(added 2026-10-26; SENSITIVITY, not confirmatory).

Bug: v3_sens_oct.cells_from() keys cells by (model, mitigation, failure_mode)
and drops `prompt`. The October rerun and E1-6 contain only prompt=base, so
their cells are unaffected, but results/sweep_raw_v2.jsonl has prompt=base
AND prompt=careful for mitigation=none. The July "baseline" in the drift
tests was therefore base+careful pooled (128 episodes), not the base cell
(64 episodes) that October replicates. Example, Haiku bare 500: July
base 44/64 = 0.6875, base+careful 47/128 = 0.367; October 44/64 = 0.6875;
reported drift +0.32 = 0.6875 - 0.367, a prompt mix, not a month effect.

This script recomputes the drift tests with the July v2 cell restricted to
prompt=base (the cell Table 8 reports), with the same protocol (task-paired
sign-flip permutation, 10,000 perms, seed 20261025; d = Oct - July). It also
reports episode-level concordance (same episode_id in both months).
The other sens.oct.* keys (W3/W1 vs October, October 500 vs timeout) only
use base-only files and are unchanged; they are re-derived here as a check.

Writes results/analysis_v3/sens_oct_fix.json with keys
  sens.oct_fix.<m>.drift.{e500,toA}:{d,p}
  sens.oct_fix.<m>.july.base.none.{e500,toA}:ivr
  sens.oct_fix.<m>.concord.{e500,toA}    {both, oct_only, july_only, neither}
  sens.oct_fix.<m>.check.<test>:d        (must equal the sens.oct.* value)
Existing analysis files and numbers.json are not modified.
Run: python3 analysis/v3_sens_oct_fix.py
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (OUT, RES, Numbers, rate_stats, perm_test, task_rates, short_model,
                       read_jsonl, dedupe, rel, f, fp, prefer_v2)
import v3_load

OCT_RAW = prefer_v2(os.path.join(RES, "tier1", "e1_6_oct", "raw.jsonl"))   # raw_v2: C1 re-score
E16_RAW = prefer_v2(os.path.join(RES, "tier1", "e1_6", "raw.jsonl"))
V2_RAW = os.path.join(RES, "sweep_raw_v2.jsonl")
OUT_JSON = os.path.join(OUT, "sens_oct_fix.json")
MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5"]
STATUS = "sensitivity (added 2026-10-26 after seeing results; not confirmatory); corrected drift"
FMS = {"e500": "error_500_after_commit", "toA": "timeout_after_commit"}


def cells(path, src):
    out = {}
    for r in dedupe(read_jsonl(path)).values():
        out.setdefault((r["model"], r["prompt"], r["mitigation"], r["failure_mode"]), []).append(
            v3_load._ep(r, src))
    return out


def main():
    if not os.path.exists(OCT_RAW):
        print("no October data"); return
    octc, e16, v2 = cells(OCT_RAW, "e1_6_oct"), cells(E16_RAW, "e1_6"), cells(V2_RAW, "v2")
    old = {}
    p_old = os.path.join(OUT, "sens_oct.json")
    if os.path.exists(p_old):
        old = json.load(open(p_old))
    so, s16, sv2 = rel(OCT_RAW), rel(E16_RAW), rel(V2_RAW)
    N = Numbers()
    for m in MODELS:
        sm = short_model(m)
        for tag, fm in FMS.items():
            o = octc.get((m, "base", "none", fm), [])
            j = v2.get((m, "base", "none", fm), [])
            jp = j + v2.get((m, "careful", "none", fm), [])
            st = rate_stats(j)
            N.cell(f"sens.oct_fix.{sm}.july.base.none.{tag}", st, sv2, status=STATUS)
            r = perm_test(task_rates(j), task_rates(o))
            pre = f"sens.oct_fix.{sm}.drift.{tag}"
            N.test(pre, r, sorted([sv2, so]), status=STATUS,
                   note=f"IVR(October base) - IVR(July v2 base), {fm}")
            jo = {e["eid"]: e["v"] for e in j}
            c = {"both": 0, "oct_only": 0, "july_only": 0, "neither": 0, "unmatched": 0}
            for e in o:
                if e["eid"] not in jo:
                    c["unmatched"] += 1
                    continue
                a, b = e["v"], jo[e["eid"]]
                c["both" if a and b else "oct_only" if a else "july_only" if b else "neither"] += 1
            N.put(f"sens.oct_fix.{sm}.concord.{tag}", c, n=len(o), source=sorted([sv2, so]), status=STATUS,
                  note="episode-level: violation in October / July for the same episode_id")
            rbug = perm_test(task_rates(jp), task_rates(o))
            print(f"  {sm:6s} {tag:4s} Oct {rate_stats(o)['k']}/{len(o)}  July base {st['k']}/{st['n']} "
                  f"(base+careful {rate_stats(jp)['k']}/{len(jp)})  corrected drift d={f(r['d'])} "
                  f"p={fp(r['p'])} (+{r['plus']}/-{r['minus']})  buggy d={f(rbug['d'])}  concord {c}")
        o500 = octc.get((m, "base", "none", FMS["e500"]), [])
        oto = octc.get((m, "base", "none", FMS["toA"]), [])
        for tag, x, y in (("w3_minus_oct_e500", o500, e16.get((m, "base", "none", "status_unknown_after_commit"), [])),
                          ("w1_minus_oct_e500", o500, e16.get((m, "base", "none", "error_500_body_after_commit"), [])),
                          ("oct_e500_minus_oct_toA", oto, o500)):
            r = perm_test(task_rates(x), task_rates(y))
            ok = old.get(f"sens.oct.{sm}.{tag}:d", {}).get("value")
            same = ok is not None and abs(round(r["d"], 6) - ok) < 1e-9
            N.put(f"sens.oct_fix.{sm}.check.{tag}:d", r["d"], n=r["T"], source=sorted([s16, so]),
                  status=STATUS, matches_sens_oct=same, p=r["p"])
            print(f"  {sm:6s} check {tag:24s} d={f(r['d'])} p={fp(r['p'])} matches sens.oct: {same}")
    N.write(OUT_JSON)
    print(f"{rel(OUT_JSON)}: {len(N.d)} values")


if __name__ == "__main__":
    main()
