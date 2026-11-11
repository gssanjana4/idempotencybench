"""bare_hint vs SAME-MONTH baselines, per Claude model.

Status: NOT covered by analysis/preregistration.md (which covers E1-0..E1-6
only). The hypothesis is the one in the experiment specification E2-1
("bare_hint gIVR <= 0.1 on every model"); the contrast (paired by graph,
same permutation test and seed as the pre-registration) is the one coded in
paper-merged/make_numbers.py emit_tests at 89ba289, committed while the
October baseline run was in progress. Reported here as SECONDARY /
pre-specified-outside-the-preregistration, not as a confirmatory test.

Primary contrast (per model): October 2026 amnesiac vs October 2026
bare_hint, both from results/claude_batch_2026-10/sweep2_raw.jsonl, same
harness, same settings (provider-default temperature, logged null), same 16
benchmark graphs (8 chain4 + 8 fanout), r = 2.
Also: October ledger_readback vs bare_hint (is the bare sentence as good as
the explicit read-back instruction?), and, SECONDARY ONLY, July amnesiac
(results/sweep2_raw_v2.jsonl) vs October bare_hint, which crosses a change
in sampling (Haiku T=0 in July) and possibly weights (undated aliases).

Unit: the graph (G = 16 clusters). CI: graph-level Wilson(p*G, G).
Test: two-sided paired sign-flip permutation over graphs, 10,000 perms,
seed 20261025. Multiplicity: none pre-specified; Holm over the three Claude
models is given as an extra column.
Refusals: a session that ended with stop_reason == "refusal" is its own
outcome; primary estimates exclude it; bounds code it as violation / not.
"""
import glob, json, os, sys
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (OUT, RES, Numbers, wilson, perm_test, holm, read_jsonl, dedupe, write_csv, rel, f, fp, prefer_v2)

OCT = os.path.join(RES, "claude_batch_2026-10")
XP = os.path.join(RES, "crossprovider")
JUL_RAW = os.path.join(RES, "sweep2_raw_v2.jsonl")
MODELS = [("claude-haiku-4-5-20251001", "haiku"), ("claude-sonnet-5", "sonnet"), ("claude-fable-5", "fable")]
ORM = [("openrouter:openai/gpt-5.5-20260423", "gpt"), ("openrouter:google/gemini-4-argon-20261013", "gemini"),
       ("openrouter:qwen/qwen3.6-27b-20260422", "qwen")]
N = Numbers()


def transcripts(d):
    out = {}
    for p in glob.glob(os.path.join(d, "transcripts2", "*.json")):
        t = json.load(open(p))
        out[(t["model"], t["resume_context"], t["mitigation"], t["graph_id"], t["interrupt_node"], t["sample"])] = (t, p)
    return out


def load(raw, trs, model, ctx):
    rows = [r for r in dedupe(read_jsonl(raw)).values()
            if r["model"] == model and r["resume_context"] == ctx and r["mitigation"] == "none"
            and r.get("key_posture", "model") == "model" and r.get("interrupt_timing", "after_commit") == "after_commit"
            and r["topology"] in ("chain4", "fanout")]
    eps = []
    for r in rows:
        tp = trs.get((model, ctx, "none", r["graph_id"], r["interrupt_node"], r["sample"]))
        refusal = False
        looked = None
        if tp:
            t = tp[0]
            refusal = any(turn.get("stop_reason") == "refusal" for s in t["sessions"] for turn in s["turns"]
                          if turn.get("role") == "assistant")
            if len(t["sessions"]) >= 2:
                looked = int(any(b.get("name") == "list_effects" for turn in t["sessions"][1]["turns"]
                                 if turn.get("role") == "assistant" for b in turn["content"]
                                 if b.get("type") == "tool_use"))
        eps.append({"task": r["graph_id"], "sample": r["sample"], "v": int(r["givr_hit"] > 0),
                    "refusal": refusal, "lookup": looked, "eid": r["episode_id"]})
    return eps


def stats(eps):
    sc = [e for e in eps if not e["refusal"]]
    n = len(sc); k = sum(e["v"] for e in sc)
    G = len({e["task"] for e in sc})
    p = k / n if n else None
    return {"n": n, "k": k, "G": G, "p": p, "ci_g": wilson(p * G, G) if n else (0, 1), "ci_ep": wilson(k, n),
            "refusals": sum(1 for e in eps if e["refusal"]),
            "lookups": f"{sum(e['lookup'] or 0 for e in sc)}/{sum(1 for e in sc if e['lookup'] is not None)}"}


def rates(eps, refusal_as=None):
    d = defaultdict(list)
    for e in eps:
        if e["refusal"]:
            if refusal_as is None:
                continue
            d[e["task"]].append(refusal_as)
        else:
            d[e["task"]].append(e["v"])
    return {g: sum(v) / len(v) for g, v in d.items()}


def main():
    oct_tr, jul_tr, xp_tr = transcripts(OCT), transcripts(RES), transcripts(XP)
    oct_raw = prefer_v2(os.path.join(OCT, "sweep2_raw.jsonl"))
    rows, tests = [], []
    for model, sm in MODELS:
        cells = {"oct.amnesiac": load(oct_raw, oct_tr, model, "amnesiac"),
                 "oct.ledger_readback": load(oct_raw, oct_tr, model, "ledger_readback"),
                 "oct.bare_hint": load(oct_raw, oct_tr, model, "bare_hint"),
                 "jul.amnesiac": load(JUL_RAW, jul_tr, model, "amnesiac")}
        dates = {}
        for c, eps in cells.items():
            st = stats(eps)
            pre = f"bh.{sm}.{c}"
            src = rel(oct_raw if c.startswith("oct") else JUL_RAW)
            N.put(pre + ":givr", st["p"], st["ci_g"], "wilson95_graph(p*G,G)", st["n"], src, k=st["k"], G=st["G"],
                  refusals=st["refusals"], session2_lookups=st["lookups"])
            N.put(pre + ":givr_ci_ep", st["p"], st["ci_ep"], "wilson95_episode", st["n"], src)
            rows.append({"model": model, "cell": c, "n": st["n"], "refusals": st["refusals"], "G": st["G"],
                         "k": st["k"], "givr": f(st["p"], 4), "ci_graph": "[%s, %s]" % (f(st["ci_g"][0], 4), f(st["ci_g"][1], 4)),
                         "ci_ep": "[%s, %s]" % (f(st["ci_ep"][0], 4), f(st["ci_ep"][1], 4)),
                         "session2_lookups": st["lookups"], "source": src})
        hint = cells["oct.bare_hint"]
        for base, cls in (("oct.amnesiac", "post hoc (same month)"), ("oct.ledger_readback", "post hoc, vs ledger readback (same month)"),
                          ("jul.amnesiac", "post hoc (July baseline; crosses sampling/weights)")):
            r = perm_test(rates(cells[base]), rates(hint))
            b = [perm_test(rates(cells[base], a), rates(hint, h)) for a in (0, 1) for h in (0, 1)]
            any_ref = any(e["refusal"] for e in cells[base] + hint)
            tests.append({"model": model, "sm": sm, "contrast": f"bare_hint - {base}", "class": cls,
                          "G": r["T"], "d": r["d"], "p": r["p"], "plus": r["plus"], "minus": r["minus"],
                          "d_bound": [min(x["d"] for x in b), max(x["d"] for x in b)] if any_ref else None,
                          "p_bound": [min(x["p"] for x in b), max(x["p"] for x in b)] if any_ref else None,
                          "base": base})
        st = stats(hint)
        N.put(f"bh.{sm}.E2-1.hint_le_0.10", st["p"] <= 0.10, n=st["n"], source=rel(oct_raw),
              criterion="bare_hint gIVR <= 0.1 (experiment_specs E2-1)", ci=st["ci_g"], ci_method="wilson95_graph(p*G,G)",
              ci_upper_le_010=st["ci_g"][1] <= 0.10)
    prim = [t for t in tests if t["base"] == "oct.amnesiac"]
    for t, a in zip(prim, holm([t["p"] for t in prim])):
        t["holm_over_3_models"] = a
    for t in tests:
        tag = {"oct.amnesiac": "vs_oct_amnesiac", "oct.ledger_readback": "vs_oct_ledger_readback",
               "jul.amnesiac": "vs_jul_amnesiac"}[t["base"]]
        src = [rel(oct_raw)] + ([rel(JUL_RAW)] if t["base"].startswith("jul") else [])
        N.put(f"bh.{t['sm']}.{tag}:d", t["d"], n=t["G"], source=src, unit="graph", status=t["class"],
              stat="mean paired per-graph difference (bare_hint - baseline)", plus=t["plus"], minus=t["minus"],
              d_bound=t["d_bound"])
        N.put(f"bh.{t['sm']}.{tag}:p", t["p"], n=t["G"], source=src, status=t["class"],
              stat="two-sided paired sign-flip permutation over graphs, 10000 perms, seed 20261025",
              holm_p=t.get("holm_over_3_models"), p_bound=t["p_bound"])
    # pooled same-month (3 Claude models) + OpenRouter same-batch for context
    pool_a = [e for m, _ in MODELS for e in load(oct_raw, oct_tr, m, "amnesiac") if not e["refusal"]]
    pool_h = [e for m, _ in MODELS for e in load(oct_raw, oct_tr, m, "bare_hint") if not e["refusal"]]
    for nm, pool in (("amnesiac", pool_a), ("bare_hint", pool_h)):
        k, n = sum(e["v"] for e in pool), len(pool)
        G = 3 * 16
        N.put(f"bh.claude_pooled.oct.{nm}:givr", k / n, wilson(k / n * G, G), "wilson95_graph(p*G,G), G = 3 models x 16",
              n, rel(oct_raw), k=k)
    for model, sm in ORM:
        raw = prefer_v2(os.path.join(XP, "sweep2_raw.jsonl"))
        for ctx in ("amnesiac", "ledger_readback", "bare_hint"):
            st = stats(load(raw, xp_tr, model, ctx))
            N.put(f"bh.{sm}.xp.{ctx}:givr", st["p"], st["ci_g"], "wilson95_graph(p*G,G)", st["n"], rel(raw),
                  k=st["k"], refusals=st["refusals"], session2_lookups=st["lookups"])
            rows.append({"model": model, "cell": f"xp.{ctx}", "n": st["n"], "refusals": st["refusals"], "G": st["G"],
                         "k": st["k"], "givr": f(st["p"], 4), "ci_graph": "[%s, %s]" % (f(st["ci_g"][0], 4), f(st["ci_g"][1], 4)),
                         "ci_ep": "[%s, %s]" % (f(st["ci_ep"][0], 4), f(st["ci_ep"][1], 4)),
                         "session2_lookups": st["lookups"], "source": rel(raw)})
    # run windows (same-month evidence)
    win = defaultdict(list)
    for (m, ctx, *_), (t, p) in oct_tr.items():
        win[(m, ctx)].append(t.get("started_utc"))
    for (m, ctx), v in sorted(win.items()):
        sm = dict(MODELS).get(m, m)
        N.put(f"bh.{sm}.oct.{ctx}:utc_window", [min(v), max(v)], source=rel(os.path.join(OCT, "transcripts2")))
    write_csv(os.path.join(OUT, "bare_hint_cells.csv"), rows)
    write_csv(os.path.join(OUT, "bare_hint_tests.csv"), [{k: (f(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in tests])
    N.write(os.path.join(OUT, "bare_hint.json"))
    for r in rows:
        print(r["model"][:22], r["cell"], f"{r['k']}/{r['n']}", r["givr"], r["ci_graph"], "ref", r["refusals"], "look", r["session2_lookups"])
    for t in tests:
        print(t["sm"], t["contrast"], t["class"][:9], "d", f(t["d"]), "p", fp(t["p"]), "holm", fp(t.get("holm_over_3_models")), "bounds", t["d_bound"], t["p_bound"])


if __name__ == "__main__":
    main()
