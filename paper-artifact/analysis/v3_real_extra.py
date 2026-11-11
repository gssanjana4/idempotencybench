"""Exploratory comparisons for the real-API slice (Stripe test mode,
preregistration R1). Added 2026-10-30, AFTER the R1 results were seen.
Everything here is EXPLORATORY / POST HOC: no entry is a registered test, and
none changes the registered R1-H1 or R1-H2 result (those are in
results/analysis_v3/real_stripe.json, written by analysis/real_stripe.py).

  python3 analysis/v3_real_extra.py     # stdlib only, no network

Why: the real slice uses payment tasks only (16 single payments; 16
two-payment plans for the restart cells), while the registered simulator
cells pool 8 templates (retry) or 16 four-node graphs (restart). This script
restricts each simulator cell to its closest payment subset and puts a
task-level interval on it, so a real-vs-simulator difference that comes from
the task mix can be told apart from one that does not.

Reads (read-only): results/analysis_v3/real_stripe.json (real values and the
simulator cell keys), results/analysis_v3/{tier1_prereg,bare_hint}.json (the
simulator cells' source raw files), the simulator raw files themselves, results/real_stripe/raw.jsonl,
the October/cross-provider sweep2 raw files and transcripts (restart cells,
via v3_bare_hint.load, which handles refusals). Writes
results/analysis_v3/real_extra.json, loaded by paper-merged/make_numbers.py
under the macro prefix "nj:xr." (not merged into numbers.json, so the
registered outputs stay byte-identical).

Keys (value, k, n, T or G, ci = task/graph-level Wilson(p*T, T), ...):
  xr.pay.<m>.<cell>:ivr     simulator cell restricted to its payment tasks
      (task_id "payments-*": 4 tasks x 2 samples per cell); real = the R1
      value; within = real point estimate inside this interval; d = real - sim
  xr.pay.within             count of retry cells with within == True
  xr.rst.<m>.<ctx>:givr     closest simulator restart graphs: the chain4
      graphs cut after node 0 (the payment) commits (2 graphs x 2 samples).
      No simulator graph is a two-payment plan.
  xr.rst.within             count of restart cells with within == True
  xr.real.<m>.e500_minus_toA:d / :p   real 500 minus real timeout, paired
      over the 16 rpay tasks (sign-flip permutation, 10,000 perms, seed
      20261025), the real-API version of the E1-1 contrast
  xr.real.<m>.keys.e500:keyfirst   first payment carried a model key, k/n
      (descriptive; Stripe deduplicates only on a forwarded key)
  xr.real.hygiene           Stripe errors, refusals, truncations, turn-cap
      hits and fired failures over the scored episodes (descriptive)
  xr.h1.outside             R1-H1 cells outside the simulator interval, with
      how many lie above and below it (descriptive)
  xr.cost.r1                logged R1 model spend (sum of cost_usd over the
      scored episodes; Stripe test mode itself is free)
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (OUT, RES, ROOT, Numbers, wilson, perm_test, read_jsonl, dedupe, rel,
                       prefer_v2, short_model)
import v3_bare_hint as bh

N = Numbers()
STATUS = "exploratory, post hoc (added 2026-10-30 after the R1 results); not registered"
REAL_RAW = os.path.join(RES, "real_stripe", "raw.jsonl")
REAL_JSON = os.path.join(OUT, "real_stripe.json")
FM = {"toA": "timeout_after_commit", "e500": "error_500_after_commit",
      "w3": "status_unknown_after_commit"}
MIT = {"none": "none", "keys": "idem_keys"}


def put_cell(key, k, n, T, real, src, unit, note):
    p = k / n if n else None
    ci = wilson(p * T, T) if n else (0.0, 1.0)
    rv = real.get("value")
    N.put(key, p, ci, f"wilson95_{unit}(p*{unit[0].upper()},{unit[0].upper()})", n, src, k=k,
          **{unit[0].upper(): T}, ci_ep=list(wilson(k, n)) if n else None,
          real=rv, real_kn=f"{real.get('k')}/{real.get('n')}",
          within=(ci[0] <= rv <= ci[1]) if rv is not None and n else None,
          d=(rv - p) if rv is not None and p is not None else None,
          status=STATUS, note=note)
    return N.d[key]["within"] if n else None


def payments():
    real = json.load(open(REAL_JSON))
    sim = {}   # the registered simulator cells (same files real_stripe.py reads)
    for part in ("tier1_prereg.json", "bare_hint.json"):
        sim.update(json.load(open(os.path.join(OUT, part))))
    within = []
    for key, e in sorted(real.items()):
        if not key.endswith(":ivr") or not e.get("sim_key"):
            continue
        _, m, prompt, mit, cell = key.split(":")[0].split(".")
        srcs = sorted(sim[e["sim_key"]].get("source") or [])
        rows = []
        for s in srcs:
            rows += read_jsonl(os.path.join(ROOT, s))
        rows = [r for r in dedupe(rows).values()
                if r.get("model") == e["model"] and r.get("prompt") == prompt
                and r.get("mitigation") == MIT[mit] and r.get("failure_mode") == FM[cell]
                and str(r.get("task_id", "")).startswith("payments-") and not r.get("refusal")]
        k = sum(1 for r in rows if (r.get("violations") or 0) > 0)
        T = len({r["task_id"] for r in rows})
        w = put_cell(f"xr.pay.{m}.{mit}.{cell}:ivr", k, len(rows), T, e, srcs, "task",
                     f"payments-only subset of {e['sim_key']}")
        within.append(w)
    N.put("xr.pay.within", sum(1 for w in within if w), n=len(within), status=STATUS)


def restart():
    real = json.load(open(REAL_JSON))
    oct_raw = prefer_v2(os.path.join(bh.OCT, "sweep2_raw.jsonl"))
    xp_raw = prefer_v2(os.path.join(bh.XP, "sweep2_raw.jsonl"))
    oct_tr, xp_tr = bh.transcripts(bh.OCT), bh.transcripts(bh.XP)
    within = []
    for model, raw, trs in (("claude-haiku-4-5-20251001", oct_raw, oct_tr),
                            ("claude-sonnet-5", oct_raw, oct_tr),
                            ("openrouter:qwen/qwen3.6-27b-20260422", xp_raw, xp_tr)):
        m = short_model(model)
        for ctx in ("amnesiac", "bare_hint"):
            eps = [x for x in bh.load(raw, trs, model, ctx)
                   if x["task"].startswith("chain4-") and "__k0__" in x["eid"] and not x["refusal"]]
            k = sum(x["v"] for x in eps)
            G = len({x["task"] for x in eps})
            w = put_cell(f"xr.rst.{m}.{ctx}:givr", k, len(eps), G,
                         real[f"real.{m}.restart.{ctx}:givr"], [rel(raw)], "graph",
                         "closest simulator graphs: chain4 cut after node 0 (send_payment) commits; "
                         "no simulator graph is a two-payment plan")
            within.append(w)
    N.put("xr.rst.within", sum(1 for w in within if w), n=len(within), status=STATUS)


def real_contrast():
    rows = list(dedupe(read_jsonl(REAL_RAW)).values())
    for model in sorted({r["model"] for r in rows}):
        m = short_model(model)
        by = {}
        for fm in ("timeout_after_commit", "error_500_after_commit"):
            d = defaultdict(list)
            for r in rows:
                if r["model"] == model and r["kind"] == "retry" and r["mitigation"] == "none" \
                        and r["failure_mode"] == fm and not r.get("refusal"):
                    d[r["task_id"]].append(1.0 if r["ivr_stripe"] else 0.0)
            by[fm] = {t: sum(v) / len(v) for t, v in d.items()}
        keyed = [r for r in rows if r["model"] == model and r["kind"] == "retry"
                 and r["mitigation"] == "idem_keys" and not r.get("refusal")]
        N.put(f"xr.real.{m}.keys.e500:keyfirst", None, n=len(keyed), source=[rel(REAL_RAW)],
              k=sum(1 for r in keyed if r.get("key_attached_first")), status="descriptive",
              note="episodes whose first send_payment carried a model key")
        N.d[f"xr.real.{m}.keys.e500:keyfirst"]["value"] = \
            f"{N.d[f'xr.real.{m}.keys.e500:keyfirst']['k']}/{len(keyed)}"
        res = perm_test(by["timeout_after_commit"], by["error_500_after_commit"])
        N.put(f"xr.real.{m}.e500_minus_toA:d", res["d"], n=res["T"], source=[rel(REAL_RAW)],
              unit="task", plus=res["plus"], minus=res["minus"], status=STATUS,
              stat="mean paired per-task difference (500 - timeout), Stripe ground truth")
        N.put(f"xr.real.{m}.e500_minus_toA:p", res["p"], n=res["T"], source=[rel(REAL_RAW)],
              status=STATUS, stat="two-sided paired sign-flip permutation over the 16 rpay tasks, "
              "10000 perms, seed 20261025")


def cost():
    rows = list(dedupe(read_jsonl(REAL_RAW)).values())
    a = sum(r.get("cost_usd") or 0 for r in rows if not r["model"].startswith("openrouter:"))
    o = sum(r.get("cost_usd") or 0 for r in rows if r["model"].startswith("openrouter:"))
    N.put("xr.cost.r1", a + o, n=len(rows), anthropic=a, openrouter=o, source=[rel(REAL_RAW)],
          note="sum of cost_usd over the scored R1 episodes (Anthropic at list prices, "
               "OpenRouter usage.cost); Stripe test mode is free")


def outside_direction():
    """Registered R1 'other way' clause: is there a consistent direction among
    the cells outside the simulator interval? Counts only (descriptive)."""
    real = json.load(open(REAL_JSON))
    out = [e for k, e in real.items() if (k.endswith(":ivr") or k.endswith(":givr"))
           and e.get("within_sim_ci") is False]
    above = sum(1 for e in out if e["value"] > e["sim_value"])
    N.put("xr.h1.outside", len(out), above=above, below=len(out) - above,
          source=[rel(REAL_JSON)], status="descriptive",
          note="R1-H1 cells outside the simulator interval, by sign of real - simulator")


def hygiene():
    rows = list(dedupe(read_jsonl(REAL_RAW)).values())
    retry = [r for r in rows if r["kind"] == "retry"]
    N.put("xr.real.hygiene", None, n=len(rows), source=[rel(REAL_RAW)],
          stripe_errors=sum(r.get("stripe_errors") or 0 for r in rows),
          refusals=sum(1 for r in rows if r.get("refusal")),
          truncated=sum(1 for r in rows if r.get("truncated")),
          turn_cap=sum(1 for r in rows if r.get("turn_cap_hit")),
          failure_fired=f"{sum(1 for r in retry if r.get('failure_fired'))}/{len(retry)}",
          note="descriptive counts over the scored R1 episodes")
    N.d["xr.real.hygiene"]["value"] = len(rows)


def main():
    if not os.path.exists(REAL_RAW) or not os.path.exists(REAL_JSON):
        print("v3_real_extra: no real-API data yet - skipped")
        return 0
    payments()
    restart()
    real_contrast()
    cost()
    hygiene()
    outside_direction()
    path = os.path.join(OUT, "real_extra.json")
    N.write(path)
    for k, v in sorted(N.d.items()):
        print(k, {a: b for a, b in v.items() if a not in ("source", "stat", "note", "status", "ci_method")})
    print(f"real_extra.json: {len(N.d)} values -> {rel(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
