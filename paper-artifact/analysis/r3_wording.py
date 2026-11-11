"""R3: cross-provider wording (analysis/prereg_r3_draft.md, registered before
any R3 data). Every number the paper may print about R3.

Run:  python3 analysis/r3_wording.py [--dir results/r3_wording]
Reads (read-only): results/r3_wording/raw.jsonl (raw_v2.jsonl if a C1-style
  re-score exists), results/r3_wording/transcripts/ (refusal stops), and
  configs/r3_wording.json (planned n per cell).
Writes: results/analysis_v3/r3_wording.json (keys r3.A.*, r3.B.*,
  r3.total:*), r3_wording_cells.csv, r3_wording_tests.csv. Not wired into
  run_all_v3.sh. With no data it writes "status": "no data" and exits 0.

Statistics: the R2 helpers (analysis/r2_wording.py, analysis/v3_common.py):
IVR / gIVR = P(episode has >= 1 duplicate semantic effect); primary CI =
task-level (R3-B) / graph-level (R3-A) Wilson(p*T, T); contrasts = paired
sign-flip permutation over tasks / graphs, 10,000 permutations, seed
20261025, p = (hit + 1)/(N + 1). Primary tests are ONE-SIDED (wording lower
than its same-batch baseline), Holm within each model over the wordings of
that model's hypothesis; the hypothesis is "supported" for a model when at
least 2 of its wordings are significant after Holm at 0.05 (3 wordings
planned; with 2 both must be). Secondary contrasts are two-sided, Holm
within each (part, model) block, no verdict. Refusals (a provider block
recorded as a refusal row, or an assistant stop_reason "refusal") are
excluded from every estimate and counted; truncated episodes are kept and
counted; a cell is tested only at >= 90% of its planned n (refusals count
toward the planned n, as in R2).
"""
import json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (ROOT, RES, OUT, PERM_N, PERM_SEED, Numbers, perm_test, holm,
                       read_jsonl, dedupe, prefer_v2, write_csv, rel, f, fp, short_model)
from r2_wording import (perm_test_one_sided, unit_rates, rate_block, refusal_eids, planned,
                        FM_SHORT, ALPHA)

CFG = os.path.join(ROOT, "configs", "r3_wording.json")
NEED_SIG = 2                       # wordings significant after Holm for "supported"
BASE_A, BASE_B, TIMEOUT = "amnesiac", "error_500_after_commit", "timeout_after_commit"
NOTES = ["bare_hint", "hint_p1", "hint_p2", "hint_p3"]
UNKNOWN = ["status_unknown_after_commit", "status_unknown_p1_after_commit",
           "status_unknown_p2_after_commit", "status_unknown_p3_after_commit"]


def served_ids(rows):
    return sorted({str(r.get("model_served")) for r in rows if r.get("model_served")})


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(RES, "r3_wording"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--config", default=CFG)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    N = Numbers()
    with open(a.config) as fh:
        cfg = json.load(fh)
    plan = planned(cfg)
    raw = prefer_v2(os.path.join(a.dir, "raw.jsonl"))
    meta = {"analysis": "analysis/r3_wording.py", "preregistration": "analysis/preregistration.md (sections R2 / R3)",
            "raw": rel(raw), "perm_n": PERM_N, "perm_seed": PERM_SEED, "alpha": ALPHA,
            "primary_test": "one-sided paired sign-flip permutation (wording < same-batch baseline), "
                            f"Holm over a model's wordings; supported if >= {NEED_SIG} significant",
            "planned_episodes": sum(plan.values())}
    out_json = os.path.join(a.out, "r3_wording.json")
    if not os.path.exists(raw):
        meta["status"] = "no data"
        with open(out_json, "w") as fh:
            json.dump({"meta": meta, "numbers": {}}, fh, indent=1)
        print(f"r3_wording: no data at {rel(raw)}; wrote {rel(out_json)} (status: no data)")
        return 0
    rows = list(dedupe(read_jsonl(raw)).values())
    refused_tr = refusal_eids(os.path.join(a.dir, "transcripts"))
    cells = defaultdict(list)        # (part, model, condition) -> scored episodes
    refusals = defaultdict(int)
    refusal_rows = []
    for r in rows:
        if r.get("kind") == "restart":
            key = ("A", r["model"], r["resume_context"])
            e = {"unit": r["graph_id"], "sample": r["sample"], "v": int((r.get("givr_hit") or 0) > 0),
                 "viol": r.get("violations") or 0, "trunc": bool(r.get("truncated")),
                 "lookup": r.get("session2_lookup"), "ovr": r.get("order_violation"),
                 "completion": r.get("completion"), "served": r.get("model_served"), "eid": r["episode_id"]}
        else:
            key = ("B", r["model"], r["failure_mode"])
            e = {"unit": r["task_id"], "sample": r["sample"], "v": int((r.get("violations") or 0) > 0),
                 "viol": r.get("violations") or 0, "trunc": bool(r.get("truncated")),
                 "rb": None if not r.get("failure_fired") else int(bool(r.get("readback_after_failure"))),
                 "retried": r.get("retried"), "served": r.get("model_served"), "eid": r["episode_id"]}
        if r.get("refusal") or r["episode_id"] in refused_tr:
            refusals[key] += 1
            refusal_rows.append({"part": key[0], "model": r["model"], "condition": key[2], "unit": e["unit"],
                                 "sample": r["sample"], "episode_id": r["episode_id"],
                                 "error": (r.get("refusal_error") or "stop_reason refusal")[:120]})
        else:
            cells[key].append(e)
    crows, trows = [], []
    incomplete = set()
    for key in sorted(set(cells) | set(refusals) | set(plan), key=str):
        part, model, cond = key
        eps = cells.get(key, [])
        st = rate_block(eps)
        nm = f"r3.{part}.{short_model(model)}.{FM_SHORT.get(cond, cond)}"
        metric = "ivr" if part == "B" else "givr"
        unit = "task" if part == "B" else "graph"
        pl = plan.get(key)
        status = None
        if pl and st["n"] + refusals[key] < 0.9 * pl:
            status = f"incomplete: {st['n'] + refusals[key]}/{pl} episodes (< 90%)"
            incomplete.add(key)
        served = sorted({str(e["served"]) for e in eps if e["served"]})
        N.put(f"{nm}:{metric}", st["p"], st["ci_unit"] if st["n"] else None,
              f"wilson95_{unit}(p*T,T)", st["n"], rel(raw), k=st["k"], T=st["T"], planned=pl,
              refusals=refusals[key], status=status, model_served=served or None)
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
                      "truncated": sum(1 for e in eps if e["trunc"]), "model_served": ";".join(served),
                      "status": status or ""})

    def contrast(part, model, base, cond, one_sided, cls):
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
        nm = f"r3.{t['part']}.{t['sm']}.{FM_SHORT.get(t['cond'], t['cond'])}_vs_{FM_SHORT.get(t['base'], t['base'])}"
        unit = "task" if t["part"] == "B" else "graph"
        N.put(nm + ":d", t["d"], n=t["T"], source=rel(raw), unit=unit, status=t["class"],
              stat=f"mean paired per-{unit} difference ({t['cond']} - {t['base']})", plus=t["plus"], minus=t["minus"])
        N.put(nm + ":p", t["p"], n=t["T"], source=rel(raw), status=t["class"],
              stat=(f"{'one-sided (wording < baseline)' if t['one_sided'] else 'two-sided'} paired sign-flip "
                    f"permutation over {unit}s, {PERM_N} perms, seed {PERM_SEED}"),
              holm_p=holm_p, verdict=verdict)

    # the wordings each hypothesis covers, from the config (p3 may have been cut)
    def planned_conds(part, model):
        return [k[2] for k in plan if k[0] == part and k[1] == model]

    # ---- primary: R3-H1 (R3-A, per model: notes vs amnesiac) and R3-H2 (R3-B, per model: unknown wordings vs 500)
    for part, base, pool, hyp in (("A", BASE_A, NOTES, "H1"), ("B", BASE_B, UNKNOWN, "H2")):
        for model in sorted({k[1] for k in plan if k[0] == part} | {k[1] for k in cells if k[0] == part}):
            words = [w for w in pool if w in planned_conds(part, model)] or \
                    [w for w in pool if (part, model, w) in cells]
            ts = [contrast(part, model, base, w, True, f"primary R3-{hyp}") for w in words]
            ts = [t for t in ts if t]
            nmh = f"r3.{part}.{short_model(model)}.{hyp}:supported"
            if not words or len(ts) < len(words):
                N.put(nmh, "not evaluable", n=len(ts), source=rel(raw),
                      status=f"incomplete: {len(ts)} of {len(words)} wording cells complete (>= 90% of planned n)")
                continue
            hp = holm([t["p"] for t in ts])
            sig = [h < ALPHA for h in hp]
            for t, h, s_ in zip(ts, hp, sig):
                t["holm_p"] = h; t["significant_holm"] = s_
                emit_test(t, holm_p=h, verdict="significant (Holm, one-sided)" if s_ else "not significant")
            N.put(nmh, sum(sig) >= NEED_SIG, n=len(ts), source=rel(raw),
                  criterion=f">= {NEED_SIG} of {len(words)} wordings with Holm-adjusted one-sided p < {ALPHA} vs same-batch {base}",
                  n_significant=sum(sig), wordings=[FM_SHORT.get(w, w) for w in words], status="primary")

    # ---- secondary, two-sided, Holm within each (part, model) block, no verdict
    def emit_block(ts):
        ts = [t for t in ts if t]
        for t, h in zip(ts, holm([t["p"] for t in ts]) if ts else []):
            t["holm_p"] = h
            emit_test(t, holm_p=h)

    for model in sorted({k[1] for k in cells if k[0] == "B"}):
        words = [w for w in UNKNOWN if (("B", model, w) in cells)]
        emit_block([contrast("B", model, TIMEOUT, w, False, "secondary (unknown wording vs same-batch timeout)") for w in words]
                   + [contrast("B", model, TIMEOUT, BASE_B, False, "secondary (same-batch 500 vs timeout)")])
    for model in sorted({k[1] for k in cells if k[0] == "A"}):
        notes = [w for w in NOTES[1:] if ("A", model, w) in cells]
        emit_block([contrast("A", model, "bare_hint", w, False, "secondary (paraphrase vs same-batch bare_hint)") for w in notes])
    # ---- totals, exclusions
    N.put("r3.total:episodes", len(rows), source=rel(raw), planned=meta["planned_episodes"])
    N.put("r3.total:refusals", sum(refusals.values()), source=rel(raw),
          by_cell={f"{short_model(k[1])}.{FM_SHORT.get(k[2], k[2])}": v for k, v in refusals.items() if v} or None)
    N.put("r3.total:truncated", sum(1 for eps in cells.values() for e in eps if e["trunc"]), source=rel(raw))
    N.put("r3.total:cost_usd", round(sum(r.get("cost_usd") or 0 for r in rows), 4), source=rel(raw))
    N.put("r3.total:model_served", served_ids(rows), source=rel(raw))
    meta["status"] = "complete" if len(rows) >= meta["planned_episodes"] else f"partial: {len(rows)}/{meta['planned_episodes']} episodes"
    meta["n_numbers"] = len(N.d)
    with open(out_json, "w") as fh:
        json.dump({"meta": meta, "numbers": dict(sorted(N.d.items()))}, fh, indent=1)
    write_csv(os.path.join(a.out, "r3_wording_cells.csv"), crows)
    write_csv(os.path.join(a.out, "r3_wording_tests.csv"),
              [{k: (f(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in trows])
    if refusal_rows:
        write_csv(os.path.join(a.out, "r3_wording_refusals.csv"), refusal_rows)
    for r in crows:
        print(r["part"], r["model"][:34], r["condition"], f"{r['k']}/{r['n']}", r.get("ivr") or r.get("givr"),
              r.get("ci_task") or r.get("ci_graph"), "ref", r["refusals"], r["status"])
    for t in trows:
        print(t["part"], t["sm"], f"{FM_SHORT.get(t['cond'], t['cond'])} - {FM_SHORT.get(t['base'], t['base'])}",
              t["class"][:30], "d", f(t["d"]), "p", fp(t["p"]), "holm", fp(t.get("holm_p")))
    print(f"wrote {rel(out_json)} ({len(N.d)} numbers; {meta['status']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
