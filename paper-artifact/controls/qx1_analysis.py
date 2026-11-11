#!/usr/bin/env python3
"""QX1: content-free note control for restarts. Analysis fixed by
QX1_PLAN_content-free-note.md (pushed before the first episode).

Reads  results/response_v31/qx1_note_control/raw.jsonl and transcripts/ only.
Writes results/response_v31/qx1_note_control/analysis/{qx1_cells.csv,
       qx1_tests.csv, qx1_numbers.json}; with --tex PATH also a file of
       \\defnum{qx1:...}{...} lines for the paper.

Per model, cells amnesiac, bare_hint, note_null, note_irrelevant
(built exactly as analysis/r2_wording.py builds its restart cells):
  k/n   resumed sessions that repeat a committed step (givr_hit > 0), with
        the graph-level 95% Wilson interval; provider refusals are excluded
        and counted
  looked            sessions with any list_effects call (session2_lookup)
  looked first      sessions whose first list_effects call comes before
                    their first effectful call (from the transcript)
  completion        the row's completion flag
  keys honoured     k with session-1 keys honoured: every call replayed
                    through the harness's World with mitigation "idem_keys",
                    as analysis_v27/v31_numbers.py does (secondary)

Tests, per model:
  H1  gIVR(note_null)       > gIVR(bare_hint)
  H2  gIVR(note_irrelevant) > gIVR(bare_hint)
      one-sided paired sign-flip permutation over graphs, 10,000 flips,
      seed 20261025 (the scheme of analysis/v3_common.perm_test), Holm over
      the two. A cell under 90% of its planned episodes is not tested.
  Descriptive: each control against amnesiac, two-sided, same scheme.
Reading (from the plan): the information effect is supported for a model
if H1 and H2 both hold after Holm.

  python3 controls/qx1_analysis.py            (from the artifact root)
  python3 controls/qx1_analysis.py --raw R --controls c1,c2   (self-test)
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("QX1_ROOT") or os.path.abspath(os.path.join(HERE, ".."))
for d in ("analysis", "src", "src2"):
    sys.path.insert(0, os.path.join(ROOT, d))
from v3_common import PERM_N, PERM_SEED, wilson, perm_test, holm, read_jsonl, dedupe  # noqa: E402
from world import World, ToolCallError, EFFECTFUL_TOOLS  # noqa: E402
from graph import _chain4, _fanout  # noqa: E402
from resume import _node_commits, score_graph  # noqa: E402

ALPHA = 0.05
PLANNED = 32                      # 16 graphs x 2 samples per cell
SCORE_KEYS = ("violations", "givr_hit", "completion", "order_violation")
GRAPHS = {}
for _i in range(8):
    for _g in (_chain4(_i), _fanout(_i)):
        GRAPHS[_g.graph_id] = _g


def one_sided_greater(base_by, ctrl_by):
    """H1: mean(ctrl - base) > 0 over the graphs present in both arms.
    Flipping scheme, N and seed of v3_common.perm_test;
    p = (#flips with sum >= observed sum + 1) / (N + 1)."""
    units = sorted(set(base_by) & set(ctrl_by))
    diffs = [ctrl_by[u] - base_by[u] for u in units]
    if not diffs:
        return {"d": None, "p": None, "T": 0, "plus": 0, "minus": 0}
    obs = sum(diffs)
    rng = random.Random(PERM_SEED)
    hit = 0
    for _ in range(PERM_N):
        s = sum(dv if rng.random() < 0.5 else -dv for dv in diffs)
        if s >= obs - 1e-12:
            hit += 1
    return {"d": obs / len(diffs), "p": (hit + 1) / (PERM_N + 1), "T": len(units),
            "plus": sum(1 for v in diffs if v > 0), "minus": sum(1 for v in diffs if v < 0)}


def unit_rates(eps, field="v"):
    d = defaultdict(list)
    for e in eps:
        d[e["unit"]].append(e[field])
    return {u: sum(v) / len(v) for u, v in d.items()}


def replay(t, mitigation):
    """Re-issue every tool call of a restart transcript through a fresh World
    (as analysis_v27/v31_numbers.py)."""
    g = GRAPHS[t["graph_id"]]
    k = t["interrupt_node"]
    w = World(mitigation=mitigation, failure_mode="none")

    def call(b):
        a = dict(b.get("input") or {})
        key = a.pop("idempotency_key", "") or ""
        try:
            w.call(b["name"], a, str(key), g.graph_id)
        except ToolCallError:
            pass

    s1 = t["sessions"][0]
    cut = False
    for turn in s1["turns"]:
        if turn["role"] != "assistant":
            continue
        for b in turn.get("content") or []:
            if b.get("type") != "tool_use":
                continue
            call(b)
            if _node_commits(w, g, k):
                cut = True
                break
        if cut:
            break
    for s in t["sessions"][1:]:
        for turn in s["turns"]:
            if turn["role"] != "assistant":
                continue
            for b in turn.get("content") or []:
                if b.get("type") == "tool_use":
                    call(b)
    row = score_graph(w, g, interrupted=cut)
    return {x: row[x] for x in SCORE_KEYS}


def looked_first(t):
    """True if session 2 calls list_effects before any effectful tool; False if an
    effectful call comes first; None if session 2 makes no tool call."""
    for turn in t["sessions"][1]["turns"]:
        if turn["role"] != "assistant":
            continue
        for b in turn.get("content") or []:
            if b.get("type") != "tool_use":
                continue
            if b["name"] == "list_effects":
                return True
            if b["name"] in EFFECTFUL_TOOLS:
                return False
    return None


def kn(k, n):
    return f"{k}/{n}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(ROOT, "results", "response_v31", "qx1_note_control", "raw.jsonl"))
    ap.add_argument("--controls", default="note_null,note_irrelevant")
    ap.add_argument("--tex", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    controls = a.controls.split(",")
    conds = ["amnesiac", "bare_hint"] + controls
    tdir = os.path.join(os.path.dirname(a.raw), "transcripts")
    out = a.out or os.path.join(os.path.dirname(a.raw), "analysis")
    os.makedirs(out, exist_ok=True)

    rows = [r for r in dedupe(read_jsonl(a.raw)).values()
            if r.get("kind") == "restart" and r.get("resume_context") in conds
            and (r.get("mitigation") or "none") == "none"]
    cells, refusals = defaultdict(list), defaultdict(int)
    n_replayed = 0
    for r in rows:
        key = (r["model"], r["resume_context"])
        if r.get("refusal"):
            refusals[key] += 1
            continue
        e = {"unit": r["graph_id"], "v": int(r["givr_hit"] > 0), "lookup": r.get("session2_lookup"),
             "completion": r.get("completion"), "eid": r["episode_id"], "hon": None, "first": None}
        tf = os.path.join(tdir, r["episode_id"].replace("/", "_") + ".json")
        if os.path.exists(tf):
            t = json.load(open(tf))
            if len(t["sessions"]) >= 2:
                asrun = replay(t, "none")
                assert asrun == {x: r[x] for x in SCORE_KEYS}, (r["episode_id"], asrun)
                e["hon"] = int(replay(t, "idem_keys")["givr_hit"] > 0)
                assert e["hon"] <= e["v"]
                e["first"] = looked_first(t)
                n_replayed += 1
        cells[key].append(e)

    models = sorted({m for m, _ in list(cells) + list(refusals)})
    crows, trows, num = [], [], {}
    for m in models:
        tested = {}
        for c in conds:
            eps = cells.get((m, c), [])
            n = len(eps); k = sum(e["v"] for e in eps); T = len({e["unit"] for e in eps})
            ref = refusals[(m, c)]
            ci = wilson(k / n * T, T) if n else (0.0, 1.0)
            lk = [e for e in eps if e["lookup"] is not None]
            hon = [e for e in eps if e["hon"] is not None]
            fst = [e for e in eps if e["first"] is not None]
            complete = (n + ref) >= 0.9 * PLANNED
            tested[c] = complete
            row = {"model": m, "cell": c, "n": n, "planned": PLANNED, "refusals": ref, "graphs": T, "k": k,
                   "givr": round(k / n, 4) if n else "", "ci_graph_lo": round(ci[0], 4), "ci_graph_hi": round(ci[1], 4),
                   "looked": sum(1 for e in lk if e["lookup"]), "looked_n": len(lk),
                   "looked_first": sum(1 for e in fst if e["first"]), "looked_first_n": len(fst),
                   "completion": sum(1 for e in eps if e["completion"]),
                   "k_keys_honoured": sum(e["hon"] for e in hon), "keys_n": len(hon),
                   "status": "ok" if complete else f"incomplete: {n + ref}/{PLANNED} (< 90%), not tested"}
            crows.append(row)
            num[f"{m}|{c}"] = row
        base = unit_rates(cells.get((m, "bare_hint"), []))
        amn = unit_rates(cells.get((m, "amnesiac"), []))
        ts = []
        for c in controls:
            cu = unit_rates(cells.get((m, c), []))
            ok = tested.get(c) and tested.get("bare_hint") and base and cu
            r1 = one_sided_greater(base, cu) if ok else {"d": None, "p": None, "T": 0, "plus": 0, "minus": 0}
            ts.append({"model": m, "contrast": f"{c} > bare_hint", "kind": "H (one-sided)", **r1})
        ps = [t["p"] for t in ts if t["p"] is not None]
        hp = holm(ps) if ps else []
        it = iter(hp)
        for t in ts:
            t["holm_p"] = next(it) if t["p"] is not None else None
            t["significant_holm"] = (t["holm_p"] is not None and t["holm_p"] < ALPHA)
        verdict = ("supported" if ts and all(t["significant_holm"] for t in ts)
                   else "not tested" if any(t["p"] is None for t in ts) else "not supported")
        for t in ts:
            t["information_effect"] = verdict
        trows += ts
        for c in controls + ["bare_hint"]:
            cu = unit_rates(cells.get((m, c), []))
            ok = tested.get(c) and tested.get("amnesiac") and amn and cu
            r2 = perm_test(amn, cu) if ok else {"d": None, "p": None, "T": 0, "plus": 0, "minus": 0}
            trows.append({"model": m, "contrast": f"{c} vs amnesiac", "kind": "descriptive (two-sided)", **r2,
                          "holm_p": None, "significant_holm": None, "information_effect": verdict})
        # the same one-sided contrasts with session-1 keys honoured (secondary, descriptive)
        hb = unit_rates([e for e in cells.get((m, "bare_hint"), []) if e["hon"] is not None], "hon")
        for c in controls:
            hc = unit_rates([e for e in cells.get((m, c), []) if e["hon"] is not None], "hon")
            ok = tested.get(c) and tested.get("bare_hint") and hb and hc
            r3 = one_sided_greater(hb, hc) if ok else {"d": None, "p": None, "T": 0, "plus": 0, "minus": 0}
            trows.append({"model": m, "contrast": f"{c} > bare_hint, keys honoured", "kind": "secondary (one-sided)", **r3,
                          "holm_p": None, "significant_holm": None, "information_effect": verdict})
        num[f"{m}|verdict"] = verdict

    def wcsv(name, rs):
        if not rs:
            return
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
    wcsv("qx1_cells.csv", crows)
    wcsv("qx1_tests.csv", trows)
    json.dump({"meta": {"raw": os.path.relpath(a.raw, ROOT), "perm_n": PERM_N, "perm_seed": PERM_SEED,
                        "alpha": ALPHA, "planned_per_cell": PLANNED, "controls": controls,
                        "replayed_transcripts": n_replayed},
               "cells": crows, "tests": trows}, open(os.path.join(out, "qx1_numbers.json"), "w"), indent=1)

    for m in models:
        print(f"== {m}   information effect: {num[f'{m}|verdict']}")
        for c in conds:
            r = num[f"{m}|{c}"]
            print(f"  {c:16} dup {r['k']}/{r['n']} [{r['ci_graph_lo']:.2f}, {r['ci_graph_hi']:.2f}]  "
                  f"looked {r['looked']}/{r['looked_n']}  first {r['looked_first']}/{r['looked_first_n']}  "
                  f"complete {r['completion']}/{r['n']}  keys honoured {r['k_keys_honoured']}/{r['keys_n']}  "
                  f"refusals {r['refusals']}  {r['status']}")
        for t in trows:
            if t["model"] == m:
                d = "n/a" if t["d"] is None else f"{t['d']:+.3f}"
                p = "n/a" if t["p"] is None else f"{t['p']:.4f}"
                h = "" if t["holm_p"] is None else f"  Holm {t['holm_p']:.4f}"
                print(f"    {t['contrast']:44} d {d}  p {p}{h}  ({t['kind']})")

    if a.tex:
        short = lambda m: ("haiku" if "haiku" in m else "fable" if "fable" in m else "sonnet" if "sonnet" in m
                           else "gpt" if "gpt" in m else "gemini" if "gemini" in m else "qwen" if "qwen" in m else m)
        cs = {"amnesiac": "amn", "bare_hint": "note", "note_null": "null", "note_irrelevant": "irr"}
        L = ["% generated by controls/qx1_analysis.py; do not edit"]
        def put(k, v):
            L.append("\\defnum{qx1:%s}{%s}" % (k, v))
        def fp(p):
            return "n/a" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")
        tot = defaultdict(lambda: [0, 0])
        for r in crows:
            s, c = short(r["model"]), cs.get(r["cell"], r["cell"])
            put(f"{s}:{c}:kn", kn(r["k"], r["n"]))
            put(f"{s}:{c}:givr", f"{r['givr']:.2f}" if r["n"] else "n/a")
            put(f"{s}:{c}:ci", f"[{r['ci_graph_lo']:.2f}, {r['ci_graph_hi']:.2f}]")
            put(f"{s}:{c}:look", kn(r["looked"], r["looked_n"]))
            put(f"{s}:{c}:first", kn(r["looked_first"], r["looked_first_n"]))
            put(f"{s}:{c}:done", kn(r["completion"], r["n"]))
            put(f"{s}:{c}:hon", kn(r["k_keys_honoured"], r["keys_n"]))
            put(f"{s}:{c}:ref", r["refusals"])
            tot[c][0] += r["k"]; tot[c][1] += r["n"]
            tot[c + ":look"][0] += r["looked"]; tot[c + ":look"][1] += r["looked_n"]
        for c, (k, n) in tot.items():
            put(f"all:{c}:kn" if ":" not in c else f"all:{c}", kn(k, n))
        for t in trows:
            s = short(t["model"])
            if t["kind"].startswith("H "):
                c = (lambda x: cs.get(x, x))(t["contrast"].split(" > ")[0])
                put(f"{s}:{c}:p", fp(t["p"])); put(f"{s}:{c}:holm", fp(t["holm_p"]))
                put(f"{s}:{c}:d", "n/a" if t["d"] is None else f"{t['d']:+.2f}")
            elif t["kind"].startswith("descriptive"):
                c = (lambda x: cs.get(x, x))(t["contrast"].split(" vs ")[0])
                put(f"{s}:{c}:amn:p", fp(t["p"])); put(f"{s}:{c}:amn:d", "n/a" if t["d"] is None else f"{t['d']:+.2f}")
            else:
                c = (lambda x: cs.get(x, x))(t["contrast"].split(" > ")[0].replace(", keys honoured", ""))
                put(f"{s}:{c}:hon:p", fp(t["p"]))
        for m in models:
            put(f"{short(m)}:verdict", num[f"{m}|verdict"])
        put("episodes", sum(r["n"] + r["refusals"] for r in crows))
        put("models:n", len(models))
        open(a.tex, "w").write("\n".join(L) + "\n")
        print("wrote", a.tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
