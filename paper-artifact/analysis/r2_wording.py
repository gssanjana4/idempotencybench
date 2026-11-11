"""R2: wording robustness (analysis/prereg_r2_draft.md, registered 2026-10-31
before any R2 data). Every number the paper may print about R2.

Run:  python3 analysis/r2_wording.py [--dir results/r2_wording]
Reads (read-only): results/r2_wording/raw.jsonl (raw_v2.jsonl if a C1-style
  re-score exists), results/r2_wording/transcripts/ (refusals, read-back),
  configs/r2_wording.json (planned n), results/sweep2_raw_v2.jsonl (July
  chain plan_only, Part C comparison), results/claude_batch_2026-10/
  sweep2_raw.jsonl (October fan-out plan_only, Part C context).
Writes: results/analysis_v3/r2_wording.json, r2_wording_cells.csv,
  r2_wording_tests.csv. Not wired into run_all_v3.sh yet. With no data it
  writes a JSON whose "status" is "no data" and exits 0.

Statistics (v3_common, common protocol): IVR / gIVR = P(episode has >= 1
duplicate semantic effect); primary CI = task-level (Part B) or graph-level
(Parts A, C) Wilson(p*T, T); contrasts = paired sign-flip permutation over
tasks/graphs, 10,000 permutations, seed 20261025, p = (hit + 1)/(N + 1).
R2-H1 / R2-H2 are ONE-SIDED (paraphrase lower than its same-batch baseline),
Holm over the three paraphrases within each model; a model's hypothesis is
"supported" when >= 2 of 3 paraphrases are significant after Holm at 0.05.
Secondary contrasts are two-sided. Refusals (an assistant turn with
stop_reason "refusal") are excluded from every estimate and counted;
truncated episodes are kept and counted.
"""
import glob, json, os, random, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (ROOT, RES, OUT, PERM_N, PERM_SEED, Numbers, wilson, perm_test,
                       holm, read_jsonl, dedupe, prefer_v2, write_csv, rel, f, fp, short_model)

CFG = os.path.join(ROOT, "configs", "r2_wording.json")
JUL_RAW = os.path.join(RES, "sweep2_raw_v2.jsonl")
OCT_RAW = os.path.join(RES, "claude_batch_2026-10", "sweep2_raw.jsonl")
ALPHA = 0.05
BOOT_N = 10000
HINTS = ["bare_hint", "hint_p1", "hint_p2", "hint_p3"]
PARA_A = ["hint_p1", "hint_p2", "hint_p3"]
UNKNOWN = ["status_unknown_after_commit", "status_unknown_p1_after_commit",
           "status_unknown_p2_after_commit", "status_unknown_p3_after_commit"]
PARA_B = UNKNOWN[1:]
FM_SHORT = {"timeout_after_commit": "toA", "error_500_after_commit": "e500",
            "status_unknown_after_commit": "w3", "status_unknown_p1_after_commit": "unknown_p1",
            "status_unknown_p2_after_commit": "unknown_p2", "status_unknown_p3_after_commit": "unknown_p3"}


def perm_test_one_sided(x_by, y_by):
    """Paired sign-flip permutation over the units present in both arms,
    H1: mean(y - x) < 0. Same flipping scheme, N and seed as
    v3_common.perm_test; p = (#perms with sum <= observed sum + 1)/(N + 1)."""
    units = sorted(set(x_by) & set(y_by))
    diffs = [y_by[u] - x_by[u] for u in units]
    if not diffs:
        return {"d": None, "p": None, "T": 0, "plus": 0, "minus": 0}
    obs = sum(diffs)
    rng = random.Random(PERM_SEED)
    hit = 0
    for _ in range(PERM_N):
        s = sum(dv if rng.random() < 0.5 else -dv for dv in diffs)
        if s <= obs + 1e-12:
            hit += 1
    return {"d": obs / len(diffs), "p": (hit + 1) / (PERM_N + 1), "T": len(units),
            "plus": sum(1 for v in diffs if v > 0), "minus": sum(1 for v in diffs if v < 0)}


def unit_rates(eps):
    d = defaultdict(list)
    for e in eps:
        d[e["unit"]].append(e["v"])
    return {u: sum(v) / len(v) for u, v in d.items()}


def rate_block(eps):
    n = len(eps); k = sum(e["v"] for e in eps)
    T = len({e["unit"] for e in eps})
    p = k / n if n else None
    return {"n": n, "k": k, "T": T, "p": p, "ci_ep": wilson(k, n),
            "ci_unit": wilson(p * T, T) if n else (0.0, 1.0)}


def spread_boot(cells):
    """max - min of the cell rates over a set of wordings, with a
    unit-cluster bootstrap percentile interval (resample units with
    replacement, all wordings re-computed on the same resample)."""
    units = sorted({e["unit"] for eps in cells.values() for e in eps})
    by = {w: defaultdict(list) for w in cells}
    for w, eps in cells.items():
        for e in eps:
            by[w][e["unit"]].append(e["v"])
    def rates(sample):
        out = {}
        for w in cells:
            vals = [v for u in sample for v in by[w].get(u, [])]
            out[w] = sum(vals) / len(vals) if vals else None
        return out
    obs = rates(units)
    if any(v is None for v in obs.values()) or not units:
        return {"spread": None, "ci": None, "max": None, "min": None}
    rng = random.Random(PERM_SEED)
    sp = []
    for _ in range(BOOT_N):
        r = rates([units[rng.randrange(len(units))] for _ in units])
        if all(v is not None for v in r.values()):
            sp.append(max(r.values()) - min(r.values()))
    sp.sort()
    return {"spread": max(obs.values()) - min(obs.values()),
            "ci": [sp[int(0.025 * len(sp))], sp[int(0.975 * len(sp)) - 1]] if sp else None,
            "max": max(obs, key=obs.get), "min": min(obs, key=obs.get)}


def refusal_eids(tdir):
    """episode ids of retry transcripts with a refusal stop (restart rows
    carry their own `refusal` flag)."""
    out = set()
    for p in glob.glob(os.path.join(tdir, "*.json")):
        try:
            t = json.load(open(p))
        except Exception:
            continue
        turns = t.get("turns") or [x for s in t.get("sessions", []) for x in s["turns"]]
        if any(x.get("stop_reason") == "refusal" for x in turns if x.get("role") == "assistant"):
            out.add(t.get("episode_id"))
    return out


def planned(cfg):
    """planned n per (part, model, condition) from the config."""
    out = {}
    for g in cfg["groups"]:
        for m in g["models"]:
            for c in g.get("cells", []):
                out[("B", m, c[2])] = 32 * cfg["samples"]
            for c in g.get("restart_cells", []):
                n_graphs = 16 if c[2] == "all" else 8
                out[(g["part"], m, c[0])] = n_graphs * cfg.get("restart_samples", cfg["samples"])
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(RES, "r2_wording"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--config", default=CFG)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    N = Numbers()
    cfg = json.load(open(a.config))
    plan = planned(cfg)
    raw = prefer_v2(os.path.join(a.dir, "raw.jsonl"))
    meta = {"analysis": "analysis/r2_wording.py", "preregistration": "analysis/preregistration.md (sections R2 / R3)",
            "raw": rel(raw), "perm_n": PERM_N, "perm_seed": PERM_SEED, "alpha": ALPHA,
            "primary_test": "one-sided paired sign-flip permutation (paraphrase < same-batch baseline), "
                            "Holm over the 3 paraphrases within a model; supported if >= 2 of 3 significant",
            "planned_episodes": sum(plan.values())}
    out_json = os.path.join(a.out, "r2_wording.json")
    if not os.path.exists(raw):
        meta["status"] = "no data"
        json.dump({"meta": meta, "numbers": {}}, open(out_json, "w"), indent=1)
        print(f"r2_wording: no data at {rel(raw)}; wrote {rel(out_json)} (status: no data)")
        return 0
    rows = list(dedupe(read_jsonl(raw)).values())
    refused = refusal_eids(os.path.join(a.dir, "transcripts"))
    cells = defaultdict(list)       # (part, model, condition) -> episodes
    refusals = defaultdict(int)
    for r in rows:
        if r.get("kind") == "restart":
            part = "C" if r["resume_context"] == "plan_only" else "A"
            key = (part, r["model"], r["resume_context"])
            e = {"unit": r["graph_id"], "sample": r["sample"], "v": int(r["givr_hit"] > 0),
                 "viol": r["violations"], "trunc": bool(r.get("truncated")),
                 "lookup": r.get("session2_lookup"), "ovr": r.get("order_violation"),
                 "completion": r.get("completion"), "eid": r["episode_id"]}
            ref = bool(r.get("refusal"))
        else:
            key = ("B", r["model"], r["failure_mode"])
            e = {"unit": r["task_id"], "sample": r["sample"], "v": int(r["violations"] > 0),
                 "viol": r["violations"], "trunc": bool(r.get("truncated")),
                 "rb": None if not r.get("failure_fired") else int(bool(r.get("readback_after_failure"))),
                 "retried": r.get("retried"), "eid": r["episode_id"]}
            ref = r["episode_id"] in refused
        if ref:
            refusals[key] += 1
        else:
            cells[key].append(e)
    crows, trows = [], []
    incomplete = set()
    for key in sorted(set(cells) | set(refusals) | set(plan), key=str):
        part, model, cond = key
        eps = cells.get(key, [])
        st = rate_block(eps)
        nm = f"r2.{part}.{short_model(model)}.{FM_SHORT.get(cond, cond)}"
        metric = "ivr" if part == "B" else "givr"
        unit = "task" if part == "B" else "graph"
        pl = plan.get(key)
        status = None
        if pl and st["n"] + refusals[key] < 0.9 * pl:
            status = f"incomplete: {st['n'] + refusals[key]}/{pl} episodes (< 90%)"
            incomplete.add(key)
        N.put(f"{nm}:{metric}", st["p"], st["ci_unit"] if st["n"] else None,
              f"wilson95_{unit}(p*T,T)", st["n"], rel(raw), k=st["k"], T=st["T"], planned=pl,
              refusals=refusals[key], status=status)
        if st["n"]:
            N.put(f"{nm}:{metric}_ci_ep", st["p"], st["ci_ep"], "wilson95_episode", st["n"], rel(raw))
            N.put(f"{nm}:truncated", sum(1 for e in eps if e["trunc"]), n=st["n"], source=rel(raw))
            N.put(f"{nm}:violation_depth", sum(e["viol"] for e in eps) / st["n"], n=st["n"], source=rel(raw))
            if part == "B":
                rb = [e for e in eps if e["rb"] is not None]
                if rb:
                    N.put(f"{nm}:readback_after_failure", sum(e["rb"] for e in rb) / len(rb), n=len(rb),
                          source=rel(raw), k=sum(e["rb"] for e in rb))
                N.put(f"{nm}:retry_rate", sum(1 for e in eps if e["retried"]) / st["n"], n=st["n"], source=rel(raw))
            else:
                lk = [e for e in eps if e["lookup"] is not None]
                if lk:
                    N.put(f"{nm}:session2_lookup", sum(1 for e in lk if e["lookup"]) / len(lk), n=len(lk),
                          source=rel(raw), k=sum(1 for e in lk if e["lookup"]))
                N.put(f"{nm}:ovr", sum(1 for e in eps if e["ovr"]) / st["n"], n=st["n"], source=rel(raw))
                N.put(f"{nm}:completion", sum(1 for e in eps if e["completion"]) / st["n"], n=st["n"], source=rel(raw))
        crows.append({"part": part, "model": model, "condition": cond, "n": st["n"], "planned": pl,
                      "refusals": refusals[key], "units": st["T"], "k": st["k"], metric: f(st["p"], 4),
                      f"ci_{unit}": "[%s, %s]" % (f(st["ci_unit"][0], 4), f(st["ci_unit"][1], 4)),
                      "ci_ep": "[%s, %s]" % (f(st["ci_ep"][0], 4), f(st["ci_ep"][1], 4)),
                      "truncated": sum(1 for e in eps if e["trunc"]), "status": status or ""})

    def contrast(part, model, base, cond, one_sided, cls):
        # the registration reports a cell only at >= 90% of its planned n: no test on an incomplete cell
        if (part, model, base) in incomplete or (part, model, cond) in incomplete:
            return None
        x, y = cells.get((part, model, base), []), cells.get((part, model, cond), [])
        if not x or not y:
            return None
        r = (perm_test_one_sided if one_sided else perm_test)(unit_rates(x), unit_rates(y))
        t = {"part": part, "model": model, "sm": short_model(model), "base": base, "cond": cond,
             "class": cls, "one_sided": one_sided, **r}
        trows.append(t)
        return t

    def emit_test(t, holm_p=None, verdict=None):
        nm = f"r2.{t['part']}.{t['sm']}.{FM_SHORT.get(t['cond'], t['cond'])}_vs_{FM_SHORT.get(t['base'], t['base'])}"
        unit = "task" if t["part"] == "B" else "graph"
        N.put(nm + ":d", t["d"], n=t["T"], source=rel(raw), unit=unit, status=t["class"],
              stat=f"mean paired per-{unit} difference ({t['cond']} - {t['base']})", plus=t["plus"], minus=t["minus"])
        N.put(nm + ":p", t["p"], n=t["T"], source=rel(raw), status=t["class"],
              stat=(f"{'one-sided (paraphrase < baseline)' if t['one_sided'] else 'two-sided'} paired sign-flip "
                    f"permutation over {unit}s, {PERM_N} perms, seed {PERM_SEED}"),
              holm_p=holm_p, verdict=verdict)

    # ---- primary: R2-H1 (Part A, per model) and R2-H2 (Part B, per model)
    for part, base, paras, hyp in (("A", "amnesiac", PARA_A, "H1"), ("B", "error_500_after_commit", PARA_B, "H2")):
        for model in sorted({k[1] for k in cells if k[0] == part}):
            ts = [contrast(part, model, base, c, True, f"primary R2-{hyp}") for c in paras]
            ts = [t for t in ts if t]
            if len(ts) < 3:     # a verdict needs all three paraphrases complete
                N.put(f"r2.{part}.{short_model(model)}.{hyp}:supported", "not evaluable", n=len(ts), source=rel(raw),
                      status=f"incomplete: {len(ts)} of 3 paraphrase cells complete (>= 90% of planned n)")
                continue
            hp = holm([t["p"] for t in ts])
            sig = [h < ALPHA for h in hp]
            for t, h, s in zip(ts, hp, sig):
                t["holm_p"] = h; t["significant_holm"] = s
                emit_test(t, holm_p=h, verdict="significant (Holm, one-sided)" if s else "not significant")
            N.put(f"r2.{part}.{short_model(model)}.{hyp}:supported", sum(sig) >= 2, n=len(ts), source=rel(raw),
                  criterion=f">= 2 of 3 paraphrases with Holm-adjusted one-sided p < {ALPHA} vs same-batch {base}",
                  n_significant=sum(sig), status="primary" if len(ts) == 3 else f"incomplete: {len(ts)} of 3 paraphrases")
    # ---- secondary
    def emit_block(ts):
        """registered: secondary tests are reported with Holm within each block (one block per part and model)"""
        ts = [t for t in ts if t]
        for t, h in zip(ts, holm([t["p"] for t in ts]) if ts else []):
            t["holm_p"] = h
            emit_test(t, holm_p=h)

    def spread_cells(part, model, wordings):
        """S1 spread only over complete cells; with any wording incomplete it is not reported (deviation 13)"""
        if any((part, model, w) in incomplete or not cells.get((part, model, w)) for w in wordings):
            N.put(f"r2.{part}.{short_model(model)}.spread:maxmin", "not evaluable", source=rel(raw), status="secondary",
                  note="at least one wording cell is below the registered 90% rule")
            return None
        return {w: cells[(part, model, w)] for w in wordings}

    for model in sorted({k[1] for k in cells if k[0] == "A"}):
        emit_block([contrast("A", model, "bare_hint", c, False, "secondary") for c in PARA_A])
        sc = spread_cells("A", model, HINTS)
        sp = spread_boot(sc) if sc else {"spread": None}
        if sp["spread"] is not None:
            N.put(f"r2.A.{short_model(model)}.spread:maxmin", sp["spread"], sp["ci"],
                  f"graph_cluster_bootstrap_percentile_{BOOT_N}", source=rel(raw), status="secondary",
                  max_wording=sp["max"], min_wording=sp["min"], wordings=HINTS)
    for model in sorted({k[1] for k in cells if k[0] == "B"}):
        emit_block([contrast("B", model, "timeout_after_commit", c, False, "secondary") for c in UNKNOWN]
                   + [contrast("B", model, "error_500_after_commit", "status_unknown_after_commit", False,
                               "secondary (same-batch W3 vs 500)"),
                      contrast("B", model, "timeout_after_commit", "error_500_after_commit", False,
                               "secondary (same-batch 500 vs timeout)")])
        sc = spread_cells("B", model, UNKNOWN)
        sp = spread_boot(sc) if sc else {"spread": None}
        if sp["spread"] is not None:
            N.put(f"r2.B.{short_model(model)}.spread:maxmin", sp["spread"], sp["ci"],
                  f"task_cluster_bootstrap_percentile_{BOOT_N}", source=rel(raw), status="secondary",
                  max_wording=FM_SHORT[sp["max"]], min_wording=FM_SHORT[sp["min"]],
                  wordings=[FM_SHORT[w] for w in UNKNOWN])
    # ---- Part C, descriptive: same-month chain plan_only vs July chain plan_only (and October fan-out)
    for src, path, topo, tag in ((JUL_RAW, JUL_RAW, "chain4", "jul"), (OCT_RAW, prefer_v2(OCT_RAW), "fanout", "oct")):
        if not os.path.exists(path):
            continue
        for r in dedupe(read_jsonl(path)).values():
            if r["resume_context"] == "plan_only" and r["mitigation"] == "none" and r["topology"] == topo \
                    and r.get("key_posture", "model") == "model":
                cells[(f"C_{tag}", r["model"], "plan_only")].append(
                    {"unit": r["graph_id"], "sample": r["sample"], "v": int(r["givr_hit"] > 0),
                     "viol": r["violations"], "trunc": False, "eid": r["episode_id"]})
        for model in sorted({k[1] for k in cells if k[0] == f"C_{tag}"}):
            st = rate_block(cells[(f"C_{tag}", model, "plan_only")])
            N.put(f"r2.C.{short_model(model)}.plan_only_{topo}_{tag}:givr", st["p"], st["ci_unit"],
                  "wilson95_graph(p*T,T)", st["n"], rel(path), k=st["k"], T=st["T"], status="descriptive comparison cell")
            if tag == "jul" and cells.get(("C", model, "plan_only")):
                r = perm_test(unit_rates(cells[("C_jul", model, "plan_only")]), unit_rates(cells[("C", model, "plan_only")]))
                trows.append({"part": "C", "model": model, "sm": short_model(model), "base": "plan_only (July chain4)",
                              "cond": "plan_only", "class": "descriptive (Part C vs July chain plan_only)",
                              "one_sided": False, **r})
                N.put(f"r2.C.{short_model(model)}.plan_only_chain4_vs_jul:d", r["d"], n=r["T"], source=[rel(raw), rel(path)],
                      unit="graph", status="descriptive, no verdict", stat="mean paired per-graph difference (R2 - July)",
                      plus=r["plus"], minus=r["minus"])
                N.put(f"r2.C.{short_model(model)}.plan_only_chain4_vs_jul:p", r["p"], n=r["T"], source=[rel(raw), rel(path)],
                      status="descriptive, no verdict",
                      stat=f"two-sided paired sign-flip permutation over graphs, {PERM_N} perms, seed {PERM_SEED}")
    # ---- totals, exclusions
    N.put("r2.total:episodes", len(rows), source=rel(raw), planned=meta["planned_episodes"])
    N.put("r2.total:refusals", sum(refusals.values()), source=rel(raw))
    N.put("r2.total:truncated", sum(1 for eps in cells.values() for e in eps if e["trunc"]), source=rel(raw))
    N.put("r2.total:cost_usd", round(sum(r.get("cost_usd") or 0 for r in rows), 4), source=rel(raw))
    meta["status"] = "complete" if len(rows) >= meta["planned_episodes"] else f"partial: {len(rows)}/{meta['planned_episodes']} episodes"
    meta["n_numbers"] = len(N.d)
    json.dump({"meta": meta, "numbers": dict(sorted(N.d.items()))}, open(out_json, "w"), indent=1)
    write_csv(os.path.join(a.out, "r2_wording_cells.csv"), crows)
    write_csv(os.path.join(a.out, "r2_wording_tests.csv"),
              [{k: (f(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in trows])
    for r in crows:
        print(r["part"], r["model"][:26], r["condition"], f"{r['k']}/{r['n']}", r.get("ivr") or r.get("givr"),
              r.get("ci_task") or r.get("ci_graph"), "ref", r["refusals"], r["status"])
    for t in trows:
        print(t["part"], t["sm"], f"{FM_SHORT.get(t['cond'], t['cond'])} - {FM_SHORT.get(str(t['base']), t['base'])}",
              t["class"][:22], "d", f(t["d"]), "p", fp(t["p"]), "holm", fp(t.get("holm_p")))
    print(f"wrote {rel(out_json)} ({len(N.d)} numbers; {meta['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
