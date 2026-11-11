"""R4: outcome unknown BEFORE commit (analysis/preregistration.md, section R4,
registered 2026-11-01 before any R4 data). Every number the paper may print
about R4.

Run:  python3 analysis/r4_precommit.py [--dir results/r4_precommit]
Reads (read-only): results/r4_precommit/raw.jsonl (raw_v2.jsonl if a C1-style
  re-score exists), results/r4_precommit/transcripts/ (refusal stops), and
  configs/r4_precommit.json (planned n per cell).
Writes: results/r4_precommit/analysis/r4_precommit.json (keys r4.<model>.*,
  r4.total:*), r4_precommit_cells.csv, r4_precommit_tests.csv,
  r4_precommit_refusals.csv (when any). The output stays inside
  results/r4_precommit/, which paper-merged/make_numbers.py skips, so no
  printed number changes until R4 is wired in. Not wired into run_all_v3.sh.
  With no data it writes "status": "no data" and exits 0.

Primary outcome. OMISSION = the scorer's `omission` field (world.Ledger.outcome,
  stored per episode by model_agent._score_and_log): at the end of the
  episode at least one of the task's goal effects has NO committed effect in
  the ledger with the goal's tool and canonical arguments (normalize_value).
  Under the R4 failure modes the failed call commits nothing, so the action
  is done only if the agent re-issues it. omission = 1 - achieved - misorder
  (exactly one of success / omission / misorder holds).
Statistics (R2/R3 helpers): per-task rate over the r = 2 samples; the
  primary R4-H1 is a one-sided NON-INFERIORITY test per model, margin
  DELTA = 0.10 on the mean per-task difference
  d = mean_t [omission_unknown(t) - omission_timeout(t)],
  H0: d >= DELTA vs H1: d < DELTA, by the paired sign-flip permutation of R3
  (analysis/r2_wording.perm_test_one_sided, 10,000 permutations, seed
  20261025, p = (hits + 1)/(N + 1)) applied to the shifted differences
  d_t - DELTA. One primary test per model, so Holm within the model's primary
  family leaves p unchanged; verdict "supported" at p < 0.05. A one-sided
  95% task-cluster bootstrap upper bound of d is reported next to it
  (descriptive). Secondary contrasts are two-sided paired sign-flip tests,
  Holm within each model, no verdict. Refusals are excluded from every
  estimate and counted; a cell is tested only at >= 90% of its planned n
  (refusals count toward it); a model's verdict is "not evaluable" if either
  cell of R4-H1 is below the rule.
"""
import json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (ROOT, RES, PERM_N, PERM_SEED, Numbers, wilson, perm_test, holm,
                       read_jsonl, dedupe, prefer_v2, write_csv, rel, f, fp, short_model)
from r2_wording import perm_test_one_sided, refusal_eids, ALPHA, BOOT_N

CFG = os.path.join(ROOT, "configs", "r4_precommit.json")
DIR = os.path.join(RES, "r4_precommit")
DELTA = 0.10                       # registered non-inferiority margin (per-task difference)
TO, E500, UNK = "timeout_before_commit", "error_500_before_commit", "status_unknown_before_commit"
CONDS = [TO, E500, UNK]
SHORT = {TO: "toB", E500: "e500B", UNK: "w3B"}
MEASURES = {"omission": "om", "ivr": "v", "readback_after_failure": "rb"}


def planned(cfg):
    """planned n per (model, condition) from the config (32 tasks x samples)."""
    return {(m, c[2]): 32 * cfg["samples"] for g in cfg["groups"] for m in g["models"] for c in g["cells"]}


def rates(eps, key):
    """per-task mean of a 0/1 field over the episodes where it is defined."""
    d = defaultdict(list)
    for e in eps:
        if e[key] is not None:
            d[e["unit"]].append(e[key])
    return {u: sum(v) / len(v) for u, v in d.items()}


def block(eps, key):
    vals = [e[key] for e in eps if e[key] is not None]
    n = len(vals); k = sum(vals)
    T = len({e["unit"] for e in eps if e[key] is not None})
    p = k / n if n else None
    return {"n": n, "k": k, "T": T, "p": p, "ci_ep": wilson(k, n),
            "ci_unit": wilson(p * T, T) if n else (0.0, 1.0)}


def ni_test(x_by, y_by, delta=DELTA):
    """One-sided non-inferiority, H1: mean_t(y - x) < delta: the R3 one-sided
    sign-flip on the shifted differences (y - x) - delta. Returns d (unshifted
    mean difference), p, T, and the one-sided 95% task-bootstrap upper bound."""
    units = sorted(set(x_by) & set(y_by))
    if not units:
        return {"d": None, "p": None, "T": 0, "plus": 0, "minus": 0, "ub95_boot": None}
    r = perm_test_one_sided({u: x_by[u] + delta for u in units}, {u: y_by[u] for u in units})
    diffs = [y_by[u] - x_by[u] for u in units]
    rng = random.Random(PERM_SEED)
    bs = sorted(sum(diffs[rng.randrange(len(diffs))] for _ in diffs) / len(diffs) for _ in range(BOOT_N))
    return {"d": sum(diffs) / len(diffs), "p": r["p"], "T": r["T"],
            "plus": sum(1 for v in diffs if v > 0), "minus": sum(1 for v in diffs if v < 0),
            "ub95_boot": bs[int(0.95 * len(bs)) - 1]}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--out", default=None, help="default: <dir>/analysis")
    ap.add_argument("--config", default=CFG)
    a = ap.parse_args(argv)
    out_dir = a.out or os.path.join(a.dir, "analysis")
    with open(a.config) as fh:
        cfg = json.load(fh)
    plan = planned(cfg)
    raw = prefer_v2(os.path.join(a.dir, "raw.jsonl"))
    N = Numbers()
    meta = {"analysis": "analysis/r4_precommit.py", "preregistration": "analysis/preregistration.md (section R4)",
            "raw": rel(raw), "perm_n": PERM_N, "perm_seed": PERM_SEED, "alpha": ALPHA, "margin": DELTA,
            "primary_outcome": "omission (world.Ledger.outcome: some goal effect never committed)",
            "primary_test": f"one-sided non-inferiority, H0: mean per-task omission difference "
                            f"(status_unknown - timeout, before commit) >= {DELTA}; paired sign-flip on the "
                            f"shifted differences; supported if p < {ALPHA}",
            "planned_episodes": sum(plan.values())}
    out_json = os.path.join(out_dir, "r4_precommit.json")
    if not os.path.exists(raw):
        meta["status"] = "no data"
        os.makedirs(out_dir, exist_ok=True)
        with open(out_json, "w") as fh:
            json.dump({"meta": meta, "numbers": {}}, fh, indent=1)
        print(f"r4_precommit: no data at {rel(raw)}; wrote {rel(out_json)} (status: no data)")
        return 0
    os.makedirs(out_dir, exist_ok=True)
    rows = [r for r in dedupe(read_jsonl(raw)).values() if r.get("kind") != "restart"]
    refused_tr = refusal_eids(os.path.join(a.dir, "transcripts"))
    cells = defaultdict(list); refusals = defaultdict(int); refusal_rows = []
    for r in rows:
        key = (r["model"], r["failure_mode"])
        if r.get("refusal") or r["episode_id"] in refused_tr:
            refusals[key] += 1
            refusal_rows.append({"model": r["model"], "condition": r["failure_mode"], "unit": r["task_id"],
                                 "sample": r["sample"], "episode_id": r["episode_id"],
                                 "error": (r.get("refusal_error") or "stop_reason refusal")[:120]})
            continue
        cells[key].append({"unit": r["task_id"], "sample": r["sample"], "om": int(bool(r.get("omission"))),
                           "v": int((r.get("violations") or 0) > 0),
                           "rb": None if not r.get("failure_fired") else int(bool(r.get("readback_after_failure"))),
                           "retried": int(bool(r.get("retried"))), "ok": int(bool(r.get("achieved"))),
                           "trunc": bool(r.get("truncated")), "served": r.get("model_served")})
    crows, trows = [], []
    incomplete = set()
    for key in sorted(set(cells) | set(refusals) | set(plan), key=str):
        model, cond = key
        eps = cells.get(key, [])
        nm = f"r4.{short_model(model)}.{SHORT.get(cond, cond)}"
        pl = plan.get(key)
        n = len(eps)
        status = None
        if pl and n + refusals[key] < 0.9 * pl:
            status = f"incomplete: {n + refusals[key]}/{pl} episodes (< 90%)"
            incomplete.add(key)
        served = sorted({str(e["served"]) for e in eps if e["served"]})
        row = {"model": model, "condition": cond, "n": n, "planned": pl, "refusals": refusals[key],
               "model_served": ";".join(served), "status": status or ""}
        for meas, fld in MEASURES.items():
            st = block(eps, fld)
            N.put(f"{nm}:{meas}", st["p"], st["ci_unit"] if st["n"] else None, "wilson95_task(p*T,T)", st["n"],
                  rel(raw), k=st["k"], T=st["T"], planned=pl, refusals=refusals[key], status=status,
                  **({"model_served": served or None} if meas == "omission" else {}))
            if st["n"]:
                N.put(f"{nm}:{meas}_ci_ep", st["p"], st["ci_ep"], "wilson95_episode", st["n"], rel(raw))
            row[meas] = f(st["p"], 4); row[f"{meas}_k"] = st["k"]
            row[f"{meas}_ci_task"] = "[%s, %s]" % (f(st["ci_unit"][0], 4), f(st["ci_unit"][1], 4))
        if n:
            for fld, label in (("retried", "retry_rate"), ("ok", "success")):
                N.put(f"{nm}:{label}", sum(e[fld] for e in eps) / n, n=n, source=rel(raw), k=sum(e[fld] for e in eps))
                row[label] = f(sum(e[fld] for e in eps) / n, 4)
            N.put(f"{nm}:truncated", sum(1 for e in eps if e["trunc"]), n=n, source=rel(raw))
        row["truncated"] = sum(1 for e in eps if e["trunc"])
        crows.append(row)

    def emit(t, holm_p=None, verdict=None):
        nm = f"r4.{t['sm']}.{t['meas']}.{SHORT[t['cond']]}_vs_{SHORT[t['base']]}"
        N.put(nm + ":d", t["d"], n=t["T"], source=rel(raw), unit="task", status=t["class"],
              stat=f"mean paired per-task difference in {t['meas']} ({t['cond']} - {t['base']})",
              plus=t["plus"], minus=t["minus"], **({"ub95_boot": t["ub95_boot"]} if "ub95_boot" in t else {}))
        N.put(nm + ":p", t["p"], n=t["T"], source=rel(raw), status=t["class"], stat=t["stat"],
              holm_p=holm_p, verdict=verdict)

    for model in sorted({k[0] for k in plan} | {k[0] for k in cells}):
        sm = short_model(model)
        # ---- primary R4-H1: omission(unknown) not higher than omission(timeout) by >= DELTA
        hk = f"r4.{sm}.H1:supported"
        if (model, TO) in incomplete or (model, UNK) in incomplete or not cells.get((model, TO)) \
                or not cells.get((model, UNK)):
            N.put(hk, "not evaluable", source=rel(raw),
                  status="incomplete: a cell of R4-H1 is below 90% of its planned n (refusals counted)")
        else:
            r = ni_test(rates(cells[(model, TO)], "om"), rates(cells[(model, UNK)], "om"))
            t = {"model": model, "sm": sm, "meas": "omission", "base": TO, "cond": UNK, "class": "primary R4-H1",
                 "one_sided": True, "margin": DELTA,
                 "stat": f"one-sided non-inferiority (margin {DELTA}): paired sign-flip on per-task differences "
                         f"minus {DELTA}, {PERM_N} perms, seed {PERM_SEED}", **r}
            hp = holm([t["p"]])[0]                    # a family of one: Holm leaves p unchanged
            sup = hp < ALPHA
            t["holm_p"] = hp; t["supported"] = sup
            trows.append(t)
            emit(t, holm_p=hp, verdict="supported" if sup else "not supported")
            N.put(hk, "supported" if sup else "not supported", n=t["T"], source=rel(raw), status="primary",
                  criterion=f"one-sided non-inferiority p < {ALPHA}: mean per-task omission difference "
                            f"(status_unknown_before_commit - timeout_before_commit) below the margin {DELTA}",
                  d=r["d"], p=r["p"], ub95_boot=r["ub95_boot"], margin=DELTA)
        # ---- secondary, two-sided, Holm within the model, no verdict
        sec = [("omission", "om", E500, TO), ("omission", "om", UNK, E500),
               ("ivr", "v", UNK, TO), ("ivr", "v", E500, TO),
               ("readback_after_failure", "rb", UNK, TO), ("readback_after_failure", "rb", E500, TO)]
        ts = []
        for meas, fld, cond, base in sec:
            if (model, cond) in incomplete or (model, base) in incomplete:
                continue
            x, y = cells.get((model, base), []), cells.get((model, cond), [])
            if not x or not y:
                continue
            r = perm_test(rates(x, fld), rates(y, fld))
            if r["p"] is None:
                continue
            ts.append({"model": model, "sm": sm, "meas": meas, "base": base, "cond": cond,
                       "class": "secondary", "one_sided": False,
                       "stat": f"two-sided paired sign-flip permutation over tasks, {PERM_N} perms, seed {PERM_SEED}",
                       **r})
        for t, h in zip(ts, holm([t["p"] for t in ts]) if ts else []):
            t["holm_p"] = h
            trows.append(t)
            emit(t, holm_p=h)
    N.put("r4.total:episodes", len(rows), source=rel(raw), planned=meta["planned_episodes"])
    N.put("r4.total:refusals", sum(refusals.values()), source=rel(raw),
          by_cell={f"{short_model(k[0])}.{SHORT.get(k[1], k[1])}": v for k, v in refusals.items() if v} or None)
    N.put("r4.total:truncated", sum(1 for eps in cells.values() for e in eps if e["trunc"]), source=rel(raw))
    N.put("r4.total:cost_usd", round(sum(r.get("cost_usd") or 0 for r in rows), 4), source=rel(raw))
    N.put("r4.total:model_served", sorted({str(r.get("model_served")) for r in rows if r.get("model_served")}),
          source=rel(raw))
    meta["status"] = ("complete" if len(rows) >= meta["planned_episodes"]
                      else f"partial: {len(rows)}/{meta['planned_episodes']} episodes")
    meta["n_numbers"] = len(N.d)
    with open(out_json, "w") as fh:
        json.dump({"meta": meta, "numbers": dict(sorted(N.d.items()))}, fh, indent=1)
    write_csv(os.path.join(out_dir, "r4_precommit_cells.csv"), crows)
    write_csv(os.path.join(out_dir, "r4_precommit_tests.csv"),
              [{k: (f(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in trows])
    if refusal_rows:
        write_csv(os.path.join(out_dir, "r4_precommit_refusals.csv"), refusal_rows)
    for r in crows:
        print(r["model"][:34], r["condition"], f"{r['omission_k']}/{r['n']}", "omission", r["omission"],
              r["omission_ci_task"], "ivr", r["ivr"], "rb", r["readback_after_failure"], "ref", r["refusals"], r["status"])
    for t in trows:
        print(t["sm"], t["meas"], f"{SHORT[t['cond']]} - {SHORT[t['base']]}", t["class"][:16], "d", f(t["d"]),
              "p", fp(t["p"]), "holm", fp(t.get("holm_p")),
              ("supported" if t.get("supported") else "not supported") if t["class"].startswith("primary") else "")
    for model in sorted({k[0] for k in plan}):
        print(f"R4-H1 {short_model(model)}: {N.d[f'r4.{short_model(model)}.H1:supported']['value']}")
    print(f"wrote {rel(out_json)} ({len(N.d)} numbers; {meta['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
