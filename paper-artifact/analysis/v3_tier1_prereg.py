"""Round-4 (v3) analysis: every pre-registered test/estimate of
analysis/preregistration.md (E1-0 ... E1-6), plus labelled exploratory extras.

Run: python3 analysis/v3_tier1_prereg.py   (or analysis/run_all_v3.sh)
Reads (read-only): results/sweep_raw_v2.jsonl, results/sweep_raw.jsonl,
  results/sweep_transcripts/ (strict-matcher replay only),
  results/crossprovider/sweep_raw.jsonl + run.log,
  results/tier1/e1_*/raw.jsonl, imported.jsonl, run.log
Writes: results/analysis_v3/cells.csv, tests.csv, refusals.csv,
  refusal_detail.csv, e1_4_pooled.csv, e1_5_by_k.csv, e1_2_mechanism.csv,
  tier1_prereg.json (name -> value; merged into numbers.json by run_all_v3.sh)

Conventions (preregistration "Common protocol"):
  IVR = P(episode has >= 1 duplicate semantic effect), v2/run-time matcher.
  Primary CI = task-level Wilson(p*T, T). Episode Wilson and a task-cluster
  bootstrap (percentile, 10,000 resamples, seed 20261025) are also given.
  Contrasts = two-sided paired sign-flip permutation over tasks, 10,000
  permutations, seed 20261025. P hypotheses: no correction. S hypotheses
  with a p-value: Holm within the experiment. S hypotheses that are
  threshold statements (no test pre-specified) get no p-value.
  Refusals (provider blocks, never scored): own outcome; primary estimates
  exclude them; bounds code every refusal as violation / as no violation.
"""
import json, os, random, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (OUT, PERM_SEED, Numbers, rate_stats, perm_test, task_rates,
                       holm, wilson, short_model, write_csv, rel, f, fp, MIT_SHORT, FM_SHORT)
import v3_load

BOOT_N = 10000
HAIKU, SONNET = "claude-haiku-4-5-20251001", "claude-sonnet-5"
OR_MODELS = ["openrouter:qwen/qwen3.6-27b-20260422", "openrouter:openai/gpt-5.5-20260423",
             "openrouter:google/gemini-4-argon-20261013"]
SRCFILE = {}
N = Numbers()
TESTS = []
LOG = []


def say(s=""):
    LOG.append(s)
    print(s)


def boot(eps):
    by = defaultdict(list)
    for e in eps:
        by[e["task"]].append(e["v"])
    ts = sorted(by)
    if not ts:
        return (None, None)
    rng = random.Random(PERM_SEED)
    vals = []
    for _ in range(BOOT_N):
        smp = [ts[rng.randrange(len(ts))] for _ in ts]
        k = sum(sum(by[t]) for t in smp); n = sum(len(by[t]) for t in smp)
        vals.append(k / n)
    vals.sort()
    return (vals[int(0.025 * BOOT_N)], vals[int(0.975 * BOOT_N) - 1])


def key(src, model, prompt, mit, fm):
    return (src, model, prompt, mit, fm)


def name(src, model, prompt, mit, fm):
    return ".".join([src, short_model(model), prompt, MIT_SHORT.get(mit, mit), FM_SHORT.get(fm, fm)])


def impute(eps, refs, value):
    """Episodes + refusals coded as `value` (bound analyses)."""
    return eps + [{"task": t, "sample": s, "v": value, "eid": f"refusal:{t}:{s}"}
                  for t, s, _ in refs]


def cell_block(cells, refs, k):
    eps = cells.get(k, [])
    st = rate_stats(eps)
    r = refs.get(k, [])
    src = SRCFILE[k[0]]
    nm = name(*k)
    N.cell(nm, st, src)
    bci = boot(eps)
    N.put(nm + ":ivr_ci_boot", st["p"], bci, "task_cluster_bootstrap_percentile_10000", st["n"], src)
    stl = rate_stats(impute(eps, r, 0)); sth = rate_stats(impute(eps, r, 1))
    if r:
        N.put(nm + ":refusals", len(r), n=st["n"] + len(r), source=[src, rel(v3_load.LOGS["tier1" if k[0].startswith("e1") else "crossprovider"])])
        N.put(nm + ":ivr_bound", [stl["p"], sth["p"]], n=st["n"] + len(r), source=src,
              note="refusal coded as no violation / as violation")
    strict = [e for e in eps if e.get("strict") is not None]
    rb = [e for e in eps if e.get("rb") is not None]
    sigs = defaultdict(list)
    for e in eps:
        sigs[e["task"]].append(e.get("sig_calls"))
    pairs = [v for v in sigs.values() if len(v) >= 2 and v[0] is not None]
    trunc = sum(1 for e in eps if e.get("truncated"))
    tany = defaultdict(bool)
    for e in eps:
        tany[e["task"]] |= bool(e["v"])
    if tany:
        N.put(nm + ":ivr_task_any", sum(1 for v in tany.values() if v) / len(tany), n=len(tany), source=src,
              stat="P(any sample of the task violates)")
    if strict:
        ks = sum(e["strict"] for e in strict)
        N.put(nm + ":ivr_strict", ks / len(strict), wilson(ks, len(strict)), "wilson95_episode", len(strict), src,
              k=ks, stat="raw-string matcher (secondary)",
              note="released cells: replayed through the current World (stored v1 ledger not used)" if k[0] == "v2" else None)
    if rb:
        kr = sum(e["rb"] for e in rb)
        N.put(nm + ":readback_after_failure", kr / len(rb), n=len(rb), source=src, k=kr)
    row = {"cell": nm, "src": k[0], "model": k[1], "prompt": k[2], "mitigation": k[3],
           "failure_mode": k[4], "n": st["n"], "refusals": len(r), "T": st["T"], "k": st["k"],
           "ivr": f(st["p"], 4), "ci_task_lo": f(st["ci_task"][0], 4), "ci_task_hi": f(st["ci_task"][1], 4),
           "ci_ep_lo": f(st["ci_ep"][0], 4), "ci_ep_hi": f(st["ci_ep"][1], 4),
           "ci_boot_lo": f(bci[0], 4), "ci_boot_hi": f(bci[1], 4),
           "ivr_task_any": f(sum(1 for v in tany.values() if v) / len(tany), 4) if tany else "NA",
           "ivr_if_refusal_no": f(stl["p"], 4) if r else "", "ivr_if_refusal_yes": f(sth["p"], 4) if r else "",
           "ivr_strict": f(sum(e["strict"] for e in strict) / len(strict), 4) if strict else "NA",
           "readback_after_failure": f(sum(e["rb"] for e in rb) / len(rb), 4) if rb else "NA",
           "retry_rate": f(sum(1 for e in eps if e.get("retried")) / len(eps), 4) if eps else "NA",
           "key_attach_first": f(sum(1 for e in eps if e.get("key_first")) / len(eps), 4) if eps else "NA",
           "truncated": trunc, "truncated_gt_2pct": trunc / max(1, len(eps)) > 0.02,
           "turn_cap_hit": sum(1 for e in eps if e.get("turn_cap")),
           "identical_call_pairs": f"{sum(1 for v in pairs if v[0] == v[1])}/{len(pairs)}" if pairs else "NA",
           "source": src}
    return eps, st, row


def contrast(cells, refs, kx, ky, prefix, label, hyp_class, verdict_fn=None, field="v"):
    """Paired task-level permutation, y - x. Returns result dict."""
    ex, ey = cells.get(kx, []), cells.get(ky, [])
    r = perm_test(task_rates(ex, field), task_rates(ey, field))
    rx, ry = refs.get(kx, []), refs.get(ky, [])
    out = {"test": prefix, "label": label, "class": hyp_class, "x": name(*kx), "y": name(*ky),
           "field": field, "T": r["T"], "d": r["d"], "p": r["p"], "plus": r["plus"], "minus": r["minus"],
           "refusals_x": len(rx), "refusals_y": len(ry)}
    if field == "v" and (rx or ry):
        bnds = []
        for vx, vy in ((1, 0), (0, 1), (0, 0), (1, 1)):
            b = perm_test(task_rates(impute(ex, rx, vx)), task_rates(impute(ey, ry, vy)))
            bnds.append((b["d"], b["p"]))
        out["d_bound"] = [min(b[0] for b in bnds), max(b[0] for b in bnds)]
        out["p_bound"] = [min(b[1] for b in bnds), max(b[1] for b in bnds)]
    TESTS.append(out)
    return out


def put_test(t, verdict=None, holm_p=None, status=None, note=None):
    src = sorted({SRCFILE[t["x"].split(".")[0]], SRCFILE[t["y"].split(".")[0]]})
    N.test(t["test"], {"d": t["d"], "p": t["p"], "T": t["T"], "plus": t["plus"], "minus": t["minus"]},
           src, holm_p=holm_p, status=status or t["class"], verdict=verdict, note=note)
    if "d_bound" in t:
        N.put(t["test"] + ":d_bound", t["d_bound"], n=t["T"], source=src,
              note="refusals coded to minimise/maximise d (all 4 codings)")
        N.put(t["test"] + ":p_bound", t["p_bound"], n=t["T"], source=src)
    t["verdict"] = verdict
    t["holm_p"] = holm_p


def verdict_dir(t, want, thr=None):
    """want: '+' (y > x) or '-' (y < x). thr: minimum |d| also required."""
    if t["p"] is None:
        return "not testable"
    sig = t["p"] < 0.05
    good = (t["d"] > 0) if want == "+" else (t["d"] < 0)
    if sig and good and (thr is None or abs(t["d"]) >= thr):
        return "supported"
    if sig and good:
        return f"not supported (significant, |d| = {abs(t['d']):.3f} < {thr})"
    if sig and not good:
        return "not supported (reversed, significant)"
    return "null (p >= 0.05)"


def main():
    os.makedirs(OUT, exist_ok=True)
    cells, meta = v3_load.load_all()
    SRCFILE.update(meta["sources"])
    SRCFILE["xp_first"] = meta["sources"]["xp"]
    refs, detail = v3_load.refusals(cells)
    rows = []
    for k in sorted(cells):
        if k[0] == "xp_first":
            continue
        rows.append(cell_block(cells, refs, k)[2])
    write_csv(os.path.join(OUT, "cells.csv"), rows)
    N.put("hygiene.xp.rows", meta["xp_rows"], source=meta["sources"]["xp"])
    N.put("hygiene.xp.unique", meta["xp_unique"], source=meta["sources"]["xp"])
    N.put("hygiene.xp.dup_ids", meta["xp_dup_ids"], source=meta["sources"]["xp"],
          note="all " + ", ".join(meta["xp_dup_models"]) + "; last row kept (e78e23d)")
    N.put("hygiene.xp.dup_disagree", meta["xp_dup_disagree"], source=meta["sources"]["xp"])
    for x in v3_load.EXPS:
        N.put(f"hygiene.{x}.rows", meta[f"{x}_rows"], source=meta["sources"][x])
        N.put(f"hygiene.{x}.dup_ids", meta[f"{x}_rows"] - meta[f"{x}_unique"], source=meta["sources"][x])
    nt = sum(1 for r in rows if r["truncated_gt_2pct"])
    N.put("hygiene.cells_truncated_gt_2pct", nt, n=len(rows), source="results/analysis_v3/cells.csv",
          note="truncated-excluded sensitivity analysis is triggered only for these cells")
    N.put("hygiene.turn_cap_hits_tier1", sum(r["turn_cap_hit"] for r in rows if r["src"].startswith("e1")),
          source="results/analysis_v3/cells.csv")
    low = [r["cell"] for r in rows if r["src"] in ("xp",) + tuple(v3_load.EXPS)
           and r["n"] < (0.9 * (32 if r["src"] == "e1_5" else 64))]
    N.put("hygiene.cells_below_90pct", len(low), source="results/analysis_v3/cells.csv", note=";".join(low))

    # ---------------- refusals
    rref = []
    for k, lst in sorted(refs.items()):
        rref.append({"cell": name(*k), "src": k[0], "model": k[1], "prompt": k[2], "mitigation": k[3],
                     "failure_mode": k[4], "refusals": len(lst), "scored": len(cells.get(k, [])),
                     "tasks": ";".join(sorted({t for t, _, _ in lst})),
                     "attempts": ";".join(str(a) for _, _, a in lst)})
    write_csv(os.path.join(OUT, "refusals.csv"), rref)
    write_csv(os.path.join(OUT, "refusal_detail.csv"), detail)
    for grp, srcs in (("tier1", ("e1_1", "e1_2")), ("e1_1", ("e1_1",)), ("e1_2", ("e1_2",)), ("xp", ("xp",))):
        N.put(f"refusal.{grp}.episodes", sum(len(v) for k, v in refs.items() if k[0] in srcs),
              source=rel(v3_load.LOGS["crossprovider" if grp == "xp" else "tier1"]))
    N.put("refusal.all.tasks", sorted({t for v in refs.values() for t, _, _ in v}),
          source=[rel(p) for p in v3_load.LOGS.values()])
    N.put("refusal.nonblock_missing", sum(1 for d in detail if not d["all_attempts_provider_block"]),
          source=[rel(p) for p in v3_load.LOGS.values()],
          note="missing episodes whose logged failures were not all provider blocks")

    say("=" * 70); say("E1-0 confirmation rerun (Haiku, base/idem_keys/timeout_after_commit)")
    kr, kv2 = key("e1_0", HAIKU, "base", "idem_keys", "timeout_after_commit"), key("v2", HAIKU, "base", "idem_keys", "timeout_after_commit")
    st_r, st_2 = rate_stats(cells[kr]), rate_stats(cells[kv2])
    v1 = v3_load.load_v1_cell(HAIKU, "base", "idem_keys", "timeout_after_commit")
    k1 = sum(1 for r in v1 if r["violations"] > 0)
    N.put("e1_0.v1_buggy.ivr", k1 / len(v1), wilson(k1, len(v1)), "wilson95_episode", len(v1), "results/sweep_raw.jsonl", k=k1)
    diff = st_r["p"] - st_2["p"]
    okP = abs(diff) <= 0.06
    N.put("e1_0.P.diff_rerun_minus_v2", diff, n=st_r["n"], source=[SRCFILE["e1_0"], SRCFILE["v2"]],
          criterion="|diff| <= 0.06", verdict="supported" if okP else "not supported")
    t = contrast(cells, refs, kv2, kr, "e1_0.X.rerun_vs_v2", "rerun - v2 (exploratory)", "exploratory")
    put_test(t, verdict=None, note="not pre-registered for E1-0; reported for the 'If not' clause")
    ka = sum(1 for e in cells[kr] if e["key_first"])
    N.put("e1_0.S1.key_attach_first", f"{ka}/{st_r['n']}", n=st_r["n"], source=SRCFILE["e1_0"],
          criterion="0/64", verdict="supported" if ka == 0 else "not supported")
    raw = v3_load.read_jsonl(v3_load.prefer_v2(os.path.join(v3_load.T1, "e1_0", "raw.jsonl")))
    summ = list(__import__("csv").DictReader(open(os.path.join(v3_load.T1, "e1_0", "summary.csv"))))
    rc = int(summ[0]["released_calls_identical"]); rm = int(summ[0]["released_matched"])
    agree = int(summ[0]["verdict_agrees_v2"]); ri = int(summ[0]["released_identical"])
    N.put("e1_0.S2.released_calls_identical", f"{rc}/{rm}", n=rm, source=rel(os.path.join(v3_load.T1, "e1_0", "summary.csv")),
          criterion="> 32/64", verdict="supported" if rc > 32 else "not supported")
    N.put("e1_0.released_identical_full", f"{ri}/{rm}", n=rm, source=rel(os.path.join(v3_load.T1, "e1_0", "summary.csv")))
    N.put("e1_0.verdict_agrees_v2", f"{agree}/{rm}", n=rm, source=rel(os.path.join(v3_load.T1, "e1_0", "summary.csv")))
    say(f"  rerun IVR {f(st_r['p'])} task CI [{f(st_r['ci_task'][0])}, {f(st_r['ci_task'][1])}] ep CI [{f(st_r['ci_ep'][0])}, {f(st_r['ci_ep'][1])}]")
    say(f"  v2 {f(st_2['p'])} task CI [{f(st_2['ci_task'][0])}, {f(st_2['ci_task'][1])}]; v1 (buggy) {f(k1/len(v1))}")
    say(f"  P |rerun - v2| = {abs(diff):.3f} vs 0.06 -> {'supported' if okP else 'NOT supported'}; paired d={f(t['d'])} p={fp(t['p'])} (exploratory)")
    say(f"  S1 key attach first {ka}/64; S2 calls-identical {rc}/{rm} (> 32 needed); verdict agrees v2 {agree}/{rm}")

    say("=" * 70); say("E1-1 cross-provider core")
    p_e11 = []
    for m in OR_MODELS + [HAIKU, SONNET, "claude-fable-5"]:
        src = "xp" if m.startswith("openrouter") else "v2"
        sm = short_model(m)
        kt, ke = key(src, m, "base", "none", "timeout_after_commit"), key(src, m, "base", "none", "error_500_after_commit")
        t = contrast(cells, refs, kt, ke, f"e1_1.P.H2.{sm}", "IVR(500) - IVR(timeout_after)",
                     "P" if src == "xp" else "baseline (released v2)")
        v = verdict_dir(t, "+", 0.2)
        put_test(t, verdict=v)
        say(f"  H2 {sm:7s} d={f(t['d'])} p={fp(t['p'])} T={t['T']} -> {v}"
            + (f"  [refusal bounds d {f(t['d_bound'][0])}..{f(t['d_bound'][1])}, p {fp(t['p_bound'][0])}..{fp(t['p_bound'][1])}]" if "d_bound" in t else ""))
        if sm == "qwen":   # duplicate-id sensitivity: FIRST rows
            kt1, ke1 = key("xp_first", m, "base", "none", "timeout_after_commit"), key("xp_first", m, "base", "none", "error_500_after_commit")
            t1 = contrast(cells, {}, kt1, ke1, "e1_1.P.H2.qwen.first_rows", "H2 with first-row dedupe (sensitivity)", "sensitivity")
            put_test(t1, verdict=verdict_dir(t1, "+", 0.2))
            say(f"     qwen first-row sensitivity d={f(t1['d'])} p={fp(t1['p'])} -> {verdict_dir(t1, '+', 0.2)}")
            for c in (("base", "none", "error_500_after_commit"), ("careful", "none", "error_500_after_commit"),
                      ("base", "none", "timeout_before_commit")):
                a, b = rate_stats(cells[key("xp", m, *c)]), rate_stats(cells[key("xp_first", m, *c)])
                N.put(f"sens.first_row.{name('xp', m, *c)}:ivr", b["p"], b["ci_task"], "wilson95_task(p*T,T)", b["n"],
                      SRCFILE["xp"], k=b["k"], last_row_ivr=a["p"], last_row_k=a["k"])
                say(f"     {c}: last {a['k']}/{a['n']}  first {b['k']}/{b['n']}")
        # S thresholds (no test pre-specified)
        if src == "xp":
            for tag, c in (("H1.none", ("base", "none", "none")), ("H1.toB", ("base", "none", "timeout_before_commit"))):
                st = rate_stats(cells[key(src, m, *c)])
                ok = st["ci_task"][1] <= 0.10
                N.put(f"e1_1.S.{tag}.{sm}", st["p"], st["ci_task"], "wilson95_task(p*T,T)", st["n"], SRCFILE[src],
                      criterion="task-CI upper <= 0.10", verdict="supported" if ok else "not supported", k=st["k"])
                say(f"  S {tag} {sm}: {st['k']}/{st['n']} upper {f(st['ci_task'][1])} -> {'ok' if ok else 'NOT'}")
            for tag, c in (("H3.careful_toA", ("careful", "none", "timeout_after_commit")),
                           ("H3.careful_e500", ("careful", "none", "error_500_after_commit"))):
                st = rate_stats(cells[key(src, m, *c)])
                ok = st["p"] <= 0.10
                N.put(f"e1_1.S.{tag}.{sm}", st["p"], st["ci_task"], "wilson95_task(p*T,T)", st["n"], SRCFILE[src],
                      criterion="point IVR <= 0.10 (task-CI upper also reported)", verdict="supported" if ok else "not supported",
                      k=st["k"], ci_upper_le_010=st["ci_task"][1] <= 0.10)
                say(f"  S {tag} {sm}: {st['k']}/{st['n']} = {f(st['p'])} (upper {f(st['ci_task'][1])}) -> {'ok' if ok else 'NOT'}")
            # H4: keys
            for fm, srck in (("timeout_after_commit", "xp"), ("error_500_after_commit", "e1_1")):
                eps = cells[key(srck, m, "base", "idem_keys", fm)]
                n = len(eps); att = sum(1 for e in eps if e["key_first"]) / n
                ret = [e for e in eps if e["retried"]]
                reuse = sum(1 for e in ret if e["key_cat"] == "same_key") / len(ret) if ret else None
                keyed = [e for e in eps if e["key_first"] and e["retried"] and e["key_cat"] == "same_key"]
                st = rate_stats(eps)
                N.put(f"e1_1.S.H4.{sm}.{FM_SHORT[fm]}", st["p"], st["ci_task"], "wilson95_task(p*T,T)", n, SRCFILE[srck],
                      k=st["k"], key_attach_first=att, key_reuse_rate=reuse,
                      attach_x_reuse=None if reuse is None else att * reuse,
                      retry_rate=len(ret) / n,
                      ivr_keyed_and_reused=(sum(e["v"] for e in keyed) / len(keyed)) if keyed else None,
                      n_keyed_and_reused=len(keyed), retry_key_categories=dict(Counter(e["key_cat"] for e in ret)),
                      verdict="descriptive")
                say(f"  S H4 {sm} keys/{FM_SHORT[fm]}: IVR {f(st['p'])} attach {f(att)} reuse {f(reuse)} cats {dict(Counter(e['key_cat'] for e in ret))}")
    # Claude cell 8 from E1-3 shown in E1-1 table (no test)

    say("=" * 70); say("E1-2 at-least-once redelivery")
    alo = {}
    for m in [HAIKU, SONNET] + OR_MODELS:
        for c in (("base", "none", "none"), ("base", "idem_keys", "none"), ("careful", "none", "none")):
            eps = cells[key("e1_2", m, *c)]
            st = rate_stats(eps)
            n = len(eps)
            mech = {"model": m, "cell": "/".join(c), "n": n, "refusals": len(refs.get(key("e1_2", m, *c), [])),
                    "ivr": f(st["p"], 4), "d2_readback_first": f(sum(1 for e in eps if e.get("d2_readback_first")) / n, 4),
                    "d2_reexecuted": f(sum(1 for e in eps if e.get("d2_reexecuted")) / n, 4),
                    "d1_spontaneous_dup": f(sum(1 for e in eps if e.get("d1_violations")) / n, 4)}
            for cat, cnt in sorted(Counter(e.get("d2_key_category") or "" for e in eps).items()):
                mech[f"d2_{cat or 'none'}"] = cnt
            alo[(m, c)] = mech
            nm = name("e1_2", m, *c)
            N.put(nm + ":d2_readback_first", float(mech["d2_readback_first"]), n=n, source=SRCFILE["e1_2"])
            N.put(nm + ":d1_spontaneous_dup", float(mech["d1_spontaneous_dup"]), n=n, source=SRCFILE["e1_2"])
            N.put(nm + ":d2_key_category", {k[3:]: v for k, v in mech.items() if k.startswith("d2_") and k not in ("d2_readback_first", "d2_reexecuted")}, n=n, source=SRCFILE["e1_2"])
    write_csv(os.path.join(OUT, "e1_2_mechanism.csv"), list(alo.values()))
    t = contrast(cells, refs, key("e1_2", HAIKU, "base", "none", "none"), key("e1_2", SONNET, "base", "none", "none"),
                 "e1_2.P.H2.sonnet_minus_haiku", "ALO IVR Sonnet - Haiku (base/none)", "P")
    v = verdict_dir(t, "-")
    put_test(t, verdict=v)
    say(f"  P H2 Sonnet - Haiku d={f(t['d'])} p={fp(t['p'])} T={t['T']} -> {v}")
    say(f"     d2 readback-first: Haiku {alo[(HAIKU, ('base','none','none'))]['d2_readback_first']}  Sonnet {alo[(SONNET, ('base','none','none'))]['d2_readback_first']}")
    st = rate_stats(cells[key("e1_2", HAIKU, "base", "none", "none")])
    N.put("e1_2.S.H1.haiku_base_none", st["p"], st["ci_task"], "wilson95_task(p*T,T)", st["n"], SRCFILE["e1_2"],
          k=st["k"], criterion="point IVR >= 0.85", verdict="supported" if st["p"] >= 0.85 else "not supported")
    say(f"  S H1 Haiku base/none ALO {st['k']}/{st['n']} = {f(st['p'])} -> {'ok' if st['p'] >= 0.85 else 'NOT'}")
    fam = []
    for m in [HAIKU, SONNET] + OR_MODELS:
        sm = short_model(m)
        t3 = contrast(cells, refs, key("e1_2", m, "base", "none", "none"), key("e1_2", m, "base", "idem_keys", "none"),
                      f"e1_2.S.H3.{sm}", "ALO IVR keys - none", "S")
        t4 = contrast(cells, refs, key("e1_2", m, "base", "none", "none"), key("e1_2", m, "careful", "none", "none"),
                      f"e1_2.S.H4.{sm}", "ALO IVR careful - base", "S")
        fam += [t3, t4]
    adj = holm([x["p"] for x in fam])
    for x, a in zip(fam, adj):
        sm = x["test"].split(".")[-1]
        if ".H3." in x["test"]:
            if sm == "haiku":
                v = "supported (unchanged: Holm p >= 0.05)" if a >= 0.05 else ("not supported (changed, Holm p < 0.05)")
            elif sm == "sonnet":
                v = "supported (reduced, Holm p < 0.05)" if (a < 0.05 and x["d"] < 0) else "not supported"
            else:
                v = "robustness set: " + ("reduced (Holm p < 0.05)" if (a < 0.05 and x["d"] < 0) else
                                          "increased (Holm p < 0.05)" if a < 0.05 else "no change detected")
        else:
            v = "not supported (careful reduced ALO IVR, Holm p < 0.05)" if (a < 0.05 and x["d"] < 0) else \
                "supported (no reduction detected)"
        put_test(x, verdict=v, holm_p=a)
        say(f"  {x['test']:24s} d={f(x['d'])} p={fp(x['p'])} holm={fp(a)} -> {v}"
            + (f" [bounds p {fp(x['p_bound'][0])}..{fp(x['p_bound'][1])}]" if "p_bound" in x else ""))
    say("  H3 'Fable-class' arm: no Fable model in E1-2 (Tester N5); not evaluable.")

    say("=" * 70); say("E1-3 keys under explicit 500")
    t = contrast(cells, refs, key("v2", SONNET, "base", "none", "error_500_after_commit"),
                 key("e1_3", SONNET, "base", "idem_keys", "error_500_after_commit"),
                 "e1_3.P.sonnet_keys_minus_none", "Sonnet IVR(keys,500) - IVR(none,500)", "P")
    v = verdict_dir(t, "-"); put_test(t, verdict=v)
    say(f"  P Sonnet keys500 - none500 d={f(t['d'])} p={fp(t['p'])} -> {v}")
    th = contrast(cells, refs, key("v2", HAIKU, "base", "none", "error_500_after_commit"),
                  key("e1_3", HAIKU, "base", "idem_keys", "error_500_after_commit"),
                  "e1_3.S.haiku_keys_minus_none", "Haiku IVR(keys,500) - IVR(none,500)", "S")
    a = holm([th["p"]])[0]
    v = "supported (unchanged: p >= 0.05)" if a >= 0.05 else "not supported (changed)"
    put_test(th, verdict=v, holm_p=a)
    say(f"  S Haiku keys500 - none500 d={f(th['d'])} p={fp(th['p'])} -> {v}")
    for m in (HAIKU, SONNET):
        eps = cells[key("e1_3", m, "base", "idem_keys", "error_500_after_commit")]
        ret = [e for e in eps if e["retried"]]
        cats = dict(Counter(e["key_cat"] for e in ret))
        reuse = cats.get("same_key", 0) / len(ret) if ret else None
        N.put(f"e1_3.{short_model(m)}.key_mech", sum(1 for e in eps if e["key_first"]) / len(eps), n=len(eps),
              source=SRCFILE["e1_3"], stat="key_attach_first", key_reuse_rate=reuse,
              retry_rate=len(ret) / len(eps), retry_key_categories=cats)
        say(f"  {short_model(m)}: attach {f(sum(1 for e in eps if e['key_first']) / len(eps))} reuse {f(reuse)} cats {cats}")

    say("=" * 70); say("E1-4 task-pool expansion")
    pooled = []
    for m in (HAIKU, SONNET):
        for fm in ("timeout_after_commit", "error_500_after_commit"):
            new = cells[key("e1_4", m, "base", "none", fm)]
            old = [e for e in cells[key("v2", m, "base", "none", fm)] if e["sample"] == 0]
            kn, nn = sum(e["v"] for e in new), len(new)
            ko, no = sum(e["v"] for e in old), len(old)
            olo, ohi = wilson(ko, no)
            inside = olo <= kn / nn <= ohi
            pool = new + old
            kp, npool = kn + ko, nn + no
            tag = f"e1_4.{short_model(m)}.{FM_SHORT[fm]}"
            N.put(tag + ".new.ivr", kn / nn, wilson(kn, nn), "wilson95 (r=1: episode = task)", nn, SRCFILE["e1_4"], k=kn)
            N.put(tag + ".released_s0.ivr", ko / no, (olo, ohi), "wilson95_episode (s0)", no, SRCFILE["v2"], k=ko)
            N.put(tag + ".P.new_within_released_ci", inside, n=nn, source=[SRCFILE["e1_4"], SRCFILE["v2"]],
                  verdict="supported" if inside else "not supported")
            N.put(tag + ".pooled96.ivr", kp / npool, wilson(kp, npool), "wilson95 (96 distinct tasks, 1 sample each)",
                  npool, [SRCFILE["e1_4"], SRCFILE["v2"]], k=kp)
            pooled.append({"model": m, "failure_mode": fm, "new": f"{kn}/{nn}", "ivr_new": f(kn / nn, 4),
                           "released_s0": f"{ko}/{no}", "released_s0_ci": f"[{f(olo, 4)}, {f(ohi, 4)}]",
                           "new_within_released_ci": inside, "pooled96": f"{kp}/{npool}",
                           "pooled96_ci": "[%s, %s]" % tuple(f(x, 4) for x in wilson(kp, npool))})
            say(f"  {short_model(m)} {FM_SHORT[fm]}: new {kn}/{nn}={f(kn/nn)} released s0 {ko}/{no} CI [{f(olo)}, {f(ohi)}] -> {'inside' if inside else 'OUTSIDE'}; pooled {kp}/{npool}")
            bt = defaultdict(list)
            for e in new:
                bt[e["template"]].append(e["v"])
            N.put(tag + ".S.by_template", {k: f"{sum(v)}/{len(v)}" for k, v in sorted(bt.items())}, n=nn,
                  source=SRCFILE["e1_4"], verdict="descriptive", templates=len(bt))
    write_csv(os.path.join(OUT, "e1_4_pooled.csv"), pooled)

    say("=" * 70); say("E1-5 long chains, failure at call k")
    byk = []
    for m in (HAIKU, SONNET):
        for fm in ("timeout_after_commit", "error_500_after_commit"):
            eps = cells[key("e1_5", m, "base", "none", fm)]
            st = rate_stats(eps)
            tag = f"e1_5.{short_model(m)}.{FM_SHORT[fm]}"
            new = cells[key("e1_4", m, "base", "none", fm)]
            old = [e for e in cells[key("v2", m, "base", "none", fm)] if e["sample"] == 0]
            kp, npool = sum(e["v"] for e in new + old), len(new + old)
            plo, phi = wilson(kp, npool)
            v2s = rate_stats(cells[key("v2", m, "base", "none", fm)])
            in_pool = plo <= st["p"] <= phi
            in_v2 = v2s["ci_task"][0] <= st["p"] <= v2s["ci_task"][1]
            N.cell(tag, st, SRCFILE["e1_5"])
            N.put(tag + ".P.within_e1_4_pooled96_ci", in_pool, ci=(plo, phi), ci_method="wilson95 (E1-4 pooled 96 tasks)",
                  n=st["n"], source=[SRCFILE["e1_5"], SRCFILE["e1_4"], SRCFILE["v2"]],
                  verdict="supported" if in_pool else "not supported", reading="primary reading (see report)")
            N.put(tag + ".P.within_v2_task_ci", in_v2, ci=v2s["ci_task"], ci_method="wilson95_task (v2, 32 tasks)",
                  n=st["n"], source=[SRCFILE["e1_5"], SRCFILE["v2"]],
                  verdict="supported" if in_v2 else "not supported", reading="alternative reading")
            fired = [e for e in eps if True]
            stale = sum(1 for e in eps if e.get("stale")) / len(eps)
            om = sum(1 for e in eps if e.get("omission")) / len(eps)
            mo = sum(1 for e in eps if e.get("misorder")) / len(eps)
            N.put(tag + ".S.stale_reexec", stale, wilson(sum(1 for e in eps if e.get('stale')), len(eps)),
                  "wilson95_episode (r=1: = task)", len(eps), SRCFILE["e1_5"], criterion="<= 0.10",
                  verdict="supported" if stale <= 0.10 else "not supported")
            N.put(tag + ".S.misorder", mo, n=len(eps), source=SRCFILE["e1_5"],
                  verdict="consistent (no misorder at all; claim not testable beyond that)" if mo == 0 else "see report")
            N.put(tag + ".S.omission", om, n=len(eps), source=SRCFILE["e1_5"], verdict="descriptive")
            N.put(tag + ".turn_cap_hit", sum(1 for e in eps if e.get("turn_cap")), n=len(eps), source=SRCFILE["e1_5"])
            say(f"  {short_model(m)} {FM_SHORT[fm]}: long {st['k']}/{st['n']}={f(st['p'])}; E1-4 pooled CI [{f(plo)}, {f(phi)}] -> {'in' if in_pool else 'OUT'}; "
                f"v2 task CI [{f(v2s['ci_task'][0])}, {f(v2s['ci_task'][1])}] -> {'in' if in_v2 else 'OUT'}; stale {f(stale)} omission {f(om)} misorder {f(mo)}")
            bk = defaultdict(list)
            for e in eps:
                bk[(e["template"], e["fail_on_call"])].append(e["v"])
            for (tpl, kk), v in sorted(bk.items()):
                byk.append({"model": m, "failure_mode": fm, "template": tpl, "k": kk, "n": len(v), "violated": sum(v),
                            "ivr": f(sum(v) / len(v), 3)})
            ak = defaultdict(list)
            for e in eps:
                ak[e["fail_on_call"]].append(e["v"])
            N.put(tag + ".by_k", {str(k): f"{sum(v)}/{len(v)}" for k, v in sorted(ak.items())}, n=len(eps),
                  source=SRCFILE["e1_5"], verdict="descriptive only (not powered per k)")
    write_csv(os.path.join(OUT, "e1_5_by_k.csv"), byk)

    say("=" * 70); say("E1-6 failure-signal wording")
    for m in (HAIKU, SONNET):
        sm = short_model(m)
        kb = key("v2", m, "base", "none", "error_500_after_commit")
        k1 = key("e1_6", m, "base", "none", "error_500_body_after_commit")
        k3 = key("e1_6", m, "base", "none", "status_unknown_after_commit")
        kt = key("v2", m, "base", "none", "timeout_after_commit")
        t1 = contrast(cells, refs, kb, k1, f"e1_6.P1.{sm}", "IVR(W1) - IVR(bare 500)", "P")
        v = verdict_dir(t1, "+"); put_test(t1, verdict=v)
        say(f"  P1 {sm} W1 - 500 d={f(t1['d'])} p={fp(t1['p'])} -> {v}")
        t3 = contrast(cells, refs, kb, k3, f"e1_6.P2.{sm}", "IVR(W3) - IVR(bare 500)", "P")
        t3rb = contrast(cells, refs, kb, k3, f"e1_6.P2.{sm}.readback", "read-back(W3) - read-back(bare 500)", "P (mediator)", field="rb")
        v3 = verdict_dir(t3, "-"); vrb = verdict_dir(t3rb, "+")
        put_test(t3, verdict=v3); put_test(t3rb, verdict=vrb)
        say(f"  P2 {sm} W3 - 500 d={f(t3['d'])} p={fp(t3['p'])} -> {v3};   read-back d={f(t3rb['d'])} p={fp(t3rb['p'])} -> {vrb}")
        for kk in (kt, kb, k1, k3):
            eps = cells[kk]
            rb = [e for e in eps if e["rb"] is not None]
            st = rate_stats(eps, "rb")
            N.put(name(*kk) + ":readback", st["p"], st["ci_task"], "wilson95_task(p*T,T)", st["n"], SRCFILE[kk[0]],
                  k=st["k"], verdict="descriptive (S)")
            # mediation, descriptive: IVR by read-back status (exploratory)
            yes = [e["v"] for e in rb if e["rb"]]; no = [e["v"] for e in rb if not e["rb"]]
            N.put(name(*kk) + ":ivr_given_readback", f"{sum(yes)}/{len(yes)}" if yes else "0/0",
                  source=SRCFILE[kk[0]], status="exploratory", ivr_given_no_readback=f"{sum(no)}/{len(no)}")
            say(f"     {name(*kk):40s} IVR {f(rate_stats(eps)['p'])} read-back {f(st['p'])} | IVR|rb {sum(yes)}/{len(yes)}  IVR|no rb {sum(no)}/{len(no)}")
        # exploratory: W3 vs plain timeout (same error class)
        tx = contrast(cells, refs, kt, k3, f"e1_6.X.{sm}.w3_minus_timeout", "IVR(W3) - IVR(timeout) (exploratory)", "exploratory")
        put_test(tx, verdict=None, note="exploratory; not pre-registered")
        say(f"  X {sm} W3 - timeout d={f(tx['d'])} p={fp(tx['p'])} (exploratory)")
    # Sensitivity (not a deviation): the preregistration has "one P per
    # experiment" but E1-1 (3 models) and E1-6 (P1, P2 x 2 models) carry
    # several P tests. Primary inference is uncorrected, as registered; Holm
    # across the P tests of the experiment is reported to show whether any
    # verdict depends on that.
    for exp, pref in (("e1_1", "e1_1.P.H2."), ("e1_6", "e1_6.P")):
        fam = [t for t in TESTS if t["test"].startswith(pref) and t["class"] == "P"]
        for t, a in zip(fam, holm([t["p"] for t in fam])):
            N.put(t["test"] + ":holm_sensitivity", a, n=t["T"], source=N.d[t["test"] + ":p"]["source"],
                  note=f"Holm over the {len(fam)} P tests of {exp} (sensitivity only)")
            say(f"  Holm sensitivity {t['test']}: raw {fp(t['p'])} -> {fp(a)}")
    rows_t = []
    for t in TESTS:
        rows_t.append({k: (f(v, 4) if isinstance(v, float) else v) for k, v in t.items()})
    write_csv(os.path.join(OUT, "tests.csv"), rows_t)
    N.write(os.path.join(OUT, "tier1_prereg.json"))
    with open(os.path.join(OUT, "tier1_prereg.log"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
