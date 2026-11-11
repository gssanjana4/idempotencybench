"""v27 re-analysis. Reads results/ only and writes
  paper-v27/numbers_v27.tex   (\\defnum{v27:...}{...} macros and generated table rows)
  analysis_v27/out/*.csv       (every computed row, for audit)

Sections
  T  template-level (cluster) tests next to the task-level tests for every headline contrast
  K  key-honouring rescore of the R2/R3 retry cells (a retry that reused the first call's key
     would have been dropped by a deduplicating server)
  M  simulated minimum detectable effects (80% power) for the null results
  R4 misorders in the registered pre-commit run
  P  primary-test bookkeeping: tests vs containment checks; E1-1 pre-existing rows; all
     registered experiments; a Bonferroni view across them
  H  Holm sensitivity of the E1-1 Qwen reversal under the first-run rule
  S  same-month Haiku E1-3 comparison
  F  per-cell two-sample test of the Stripe cells against their registered simulator cells
  B  Holm block sizes of the registered secondary tests (R2-R4)
  X  key reuse with different arguments; Gemini's blocked task; one example task per template
Run:  python3 analysis_v27/v27_numbers.py
"""
import collections
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v27_common import (RES, ROOT, OUT, DISP, load, path, short, template_of, dedupe_last, dedupe_first,
                        paired_contrast, signflip_mc, holm, fisher_two_sided, mde_sim, f2, pfmt, pe, dsg,
                        word, tex_escape, Macros)

os.makedirs(OUT, exist_ok=True)
M = Macros()
EXTRA = []


def prefer_v2(p):
    q = p.replace("raw.jsonl", "raw_v2.jsonl")
    return q if os.path.exists(q) else p


def ok(r):
    return not r.get("refusal")


def kn(k, n):
    return f"{k}/{n}"


def write_csv(name, rows):
    if not rows:
        return
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)


IVR = lambda r: 1.0 if r["violations"] > 0 else 0.0
OMIT = lambda r: 1.0 if r["omission"] else 0.0

# ------------------------------------------------------------------ load retry runs
def cells_of(p, filt=None):
    rows = dedupe_last([r for r in load(prefer_v2(p)) if ok(r)])
    by = collections.defaultdict(list)
    for r in rows:
        if filt and not filt(r):
            continue
        by[(short(r["model"]), r["failure_mode"])].append(r)
    return by

R2 = cells_of(path("r2_wording", "raw.jsonl"), lambda r: r.get("failure_mode") not in (None, "none"))
R3 = cells_of(path("r3_wording", "raw.jsonl"), lambda r: r.get("failure_mode") not in (None, "none"))
R4 = cells_of(path("r4_precommit", "raw.jsonl"), lambda r: r.get("failure_mode") not in (None, "none"))
A, B5, TO = "status_unknown_after_commit", "error_500_after_commit", "timeout_after_commit"
UNK = [A, "status_unknown_p1_after_commit", "status_unknown_p2_after_commit", "status_unknown_p3_after_commit"]
SHORTFM = {A: "status", "status_unknown_p1_after_commit": "p1", "status_unknown_p2_after_commit": "p2",
           "status_unknown_p3_after_commit": "p3", B5: "500", TO: "timeout",
           "status_unknown_before_commit": "status", "error_500_before_commit": "500",
           "timeout_before_commit": "timeout"}

# sanity: the paper's headline cells
assert int(sum(IVR(r) for r in R2[("Haiku", B5)])) == 42 and len(R2[("Haiku", B5)]) == 64

# ------------------------------------------------------------------ T: template-level tests
trows = []


def T(label, key, cell, ref, outcome=IVR, unit="task_id", tmpl_of=None, run=""):
    r = paired_contrast(cell, ref, outcome, unit=unit, tmpl_of=tmpl_of)
    r.update(label=label, key=key, run=run)
    trows.append(r)
    M.put(f"t:{key}:kn", f"{kn(r['k_cell'], r['n_cell'])} vs {kn(r['k_ref'], r['n_ref'])}")
    M.put(f"t:{key}:d", dsg(r["d"]))
    M.put(f"t:{key}:ptask", pfmt(r["p_task"]))
    M.put(f"t:{key}:ptmpl", pfmt(r["p_tmpl"]))
    M.put(f"t:{key}:ptmpl:pe", pe(r["p_tmpl"]))
    M.put(f"t:{key}:signs", f"{r['tmpl_pos']}+/{r['tmpl_neg']}$-$")
    M.put(f"t:{key}:sig", "yes" if r["p_tmpl"] < 0.05 else "no")
    return r


for m, cells, run in (("Haiku", R2, "R2"), ("Sonnet", R2, "R2"), ("Qwen", R3, "R3"), ("Fable", R3, "R3")):
    km = m.lower()
    T(f"{DISP[m]}: status vs bare 500", f"{km}.w3_vs_500", cells[(m, A)], cells[(m, B5)], run=run)
    for w in UNK[1:]:
        if (m, w) in cells:
            T(f"{DISP[m]}: {SHORTFM[w]} vs bare 500", f"{km}.{SHORTFM[w]}_vs_500", cells[(m, w)], cells[(m, B5)], run=run)
    T(f"{DISP[m]}: bare 500 vs timeout", f"{km}.500_vs_to", cells[(m, B5)], cells[(m, TO)], run=run)
# Fable: unknown wordings against the same-run timeout
for w in UNK:
    if ("Fable", w) in R3:
        T(f"Fable~5: {SHORTFM[w]} vs timeout", f"fable.{SHORTFM[w]}_vs_to", R3[("Fable", w)], R3[("Fable", TO)], run="R3")

# summary: template-level p range of the status/paraphrase-vs-500 tests per model
for m in ("Haiku", "Sonnet", "Qwen", "Fable"):
    ps = [r["p_tmpl"] for r in trows if r["key"].startswith(m.lower() + ".") and r["key"].endswith("_vs_500")
          and not r["key"].startswith(m.lower() + ".500")]
    nsig = sum(1 for p in ps if p < 0.05)
    M.put(f"t:{m.lower()}.unk_vs_500:pmax", pfmt(max(ps)))
    M.put(f"t:{m.lower()}.unk_vs_500:pmin", pfmt(min(ps)))
    M.put(f"t:{m.lower()}.unk_vs_500:nsig", nsig)
    M.put(f"t:{m.lower()}.unk_vs_500:n", len(ps))
allp = [r["p_tmpl"] for r in trows if r["key"].endswith("_vs_500") and ".500_" not in r["key"]
        and r["key"].split(".")[0] in ("haiku", "sonnet", "qwen")]
M.put("t:hsq.unk_vs_500:pmax", pfmt(max(allp)))
M.put("t:hsq.unk_vs_500:allsig", "yes" if max(allp) < 0.05 else "no")

# R4 omission
SB, EB, TB = "status_unknown_before_commit", "error_500_before_commit", "timeout_before_commit"
T("Qwen3.6-27B (R4): omission, status vs timeout", "r4.qwen.status_vs_to", R4[("Qwen", SB)], R4[("Qwen", TB)], OMIT, run="R4")
T("Haiku~4.5 (R4): omission, 500 vs timeout", "r4.haiku.500_vs_to", R4[("Haiku", EB)], R4[("Haiku", TB)], OMIT, run="R4")
T("Qwen3.6-27B (R4): omission, 500 vs timeout", "r4.qwen.500_vs_to", R4[("Qwen", EB)], R4[("Qwen", TB)], OMIT, run="R4")
T("Qwen3.6-27B (R4): omission, status vs 500", "r4.qwen.status_vs_500", R4[("Qwen", SB)], R4[("Qwen", EB)], OMIT, run="R4")

# E1-2 redelivery
e12 = dedupe_last([r for r in load(prefer_v2(path("tier1", "e1_2", "raw.jsonl"))) if ok(r)])
E12 = collections.defaultdict(list)
for r in e12:
    E12[(short(r["model"]), r["prompt"], r["mitigation"])].append(r)
T("Sonnet~5 (E1-2): check-first vs base, redelivery", "e12.sonnet.careful",
  E12[("Sonnet", "careful", "none")], E12[("Sonnet", "base", "none")], run="E1-2")
T("Sonnet~5 (E1-2): keys vs base, redelivery", "e12.sonnet.keys",
  E12[("Sonnet", "base", "idem_keys")], E12[("Sonnet", "base", "none")], run="E1-2")

# sessions: bare_hint vs amnesiac (graph level and shape level), October grid and R2/R3 note cells
GIVR = lambda r: 1.0 if (r.get("givr_hit") or r.get("violations", 0) > 0) else 0.0
sess_rows = []
for p in (path("claude_batch_2026-10", "sweep2_raw.jsonl"), path("crossprovider", "sweep2_raw.jsonl")):
    sess_rows += [r for r in load(p) if r.get("resume_context") and r.get("interrupt_timing", "after_commit") == "after_commit"
                  and r.get("mitigation", "none") == "none"]
sess_rows = [r for r in dedupe_last(sess_rows) if ok(r)]
SESS = collections.defaultdict(list)
for r in sess_rows:
    SESS[(short(r["model"]), r["resume_context"])].append(r)
shape_p = []
for m in ("Haiku", "Sonnet", "Fable", "GPT", "Gemini", "Qwen"):
    topo = {r["graph_id"]: r["topology"] for r in SESS[(m, "amnesiac")]}
    r = T(f"{DISP[m]} (Oct.): bare\\_hint vs amnesiac", f"sess.{m.lower()}.hint",
          SESS[(m, "bare_hint")], SESS[(m, "amnesiac")], GIVR, unit="graph_id", tmpl_of=topo, run="Oct.")
    shape_p.append(r["p_tmpl"])
M.put("t:sess:nshapes", word(trows[-1]["n_tmpl"]))
M.put("t:minp", pfmt(2 / 2 ** 8))
M.put("t:sess:shape_pmin", f2(min(shape_p)))

# template-level table rows (appendix)
sel = ["haiku.w3_vs_500", "haiku.p2_vs_500", "haiku.500_vs_to", "sonnet.w3_vs_500", "sonnet.500_vs_to",
       "qwen.w3_vs_500", "qwen.p1_vs_500", "qwen.500_vs_to", "fable.w3_vs_500", "fable.p1_vs_500", "fable.p3_vs_500",
       "fable.500_vs_to", "fable.p3_vs_to", "r4.qwen.status_vs_to", "r4.haiku.500_vs_to", "r4.qwen.500_vs_to",
       "e12.sonnet.careful", "sess.sonnet.hint", "sess.haiku.hint"]
byk = {r["key"]: r for r in trows}
rows_tex = []
for k in sel:
    r = byk[k]
    flag = "" if r["p_tmpl"] < 0.05 else r"$^{\dagger}$"
    rows_tex.append(" & ".join([r["run"], r["label"], f"{r['k_cell']}/{r['n_cell']} vs {r['k_ref']}/{r['n_ref']}",
                                dsg(r["d"]), pfmt(r["p_task"]), f"{r['tmpl_pos']}/{r['tmpl_neg']} of {r['n_tmpl']}",
                                pfmt(r["p_tmpl"]) + flag]) + r" \\")
EXTRA.append("\\def\\vtwosevenTmplRows{%%\n%s}\n" % "\n".join(rows_tex))
write_csv("template_tests.csv", [{k: v for k, v in r.items()} for r in trows])

# ------------------------------------------------------------------ K: key-honouring rescore
krows = []
HON = lambda r: 1.0 if (r["violations"] > 0 and r.get("retry_key_category") != "same_key") else 0.0
for m, cells in (("Haiku", R2), ("Sonnet", R2), ("Fable", R3), ("Qwen", R3), ("GPT", R3), ("Gemini", R3)):
    for fm in [TO, B5] + UNK:
        if (m, fm) not in cells:
            continue
        rows = cells[(m, fm)]
        k = int(sum(IVR(r) for r in rows)); kh = int(sum(HON(r) for r in rows)); n = len(rows)
        same = sum(1 for r in rows if r["violations"] > 0 and r.get("retry_key_category") == "same_key")
        krows.append(dict(model=m, cell=SHORTFM[fm], k=k, k_honoured=kh, same_key_dups=same, n=n))
        M.put(f"k:{m.lower()}.{SHORTFM[fm]}:kn", kn(k, n))
        M.put(f"k:{m.lower()}.{SHORTFM[fm]}:hkn", kn(kh, n))
        M.put(f"k:{m.lower()}.{SHORTFM[fm]}:same", same)
    # range over the unknown wordings, honoured
    hv = [r for r in krows if r["model"] == m and r["cell"] in ("status", "p1", "p2", "p3")]
    M.put(f"k:{m.lower()}.unk:hmin", kn(min(r["k_honoured"] for r in hv), hv[0]["n"]))
    M.put(f"k:{m.lower()}.unk:hmax", kn(max(r["k_honoured"] for r in hv), hv[0]["n"]))
    M.put(f"k:{m.lower()}.unk:same_all", "yes" if all(r["same_key_dups"] == r["k"] for r in hv) else "no")
    tot_d = sum(r["k"] for r in krows if r["model"] == m)
    tot_s = sum(r["same_key_dups"] for r in krows if r["model"] == m)
    M.put(f"k:{m.lower()}.share_same", f"{tot_s}/{tot_d}")
    # status vs 500, keys honoured: task and template level
    if (m, A) in cells:
        r = paired_contrast(cells[(m, A)], cells[(m, B5)], HON)
        M.put(f"k:{m.lower()}.w3_vs_500:d", dsg(r["d"]))
        M.put(f"k:{m.lower()}.w3_vs_500:ptask", pe(r["p_task"]))
        M.put(f"k:{m.lower()}.w3_vs_500:ptmpl", pe(r["p_tmpl"]))
        M.put(f"k:{m.lower()}.w3_vs_500:sig", "yes" if r["p_task"] < 0.05 else "no")
        krows[-1].update()  # keep csv flat
# share of all R2/R3 retry duplicates (after a commit) that were same-key
tot_d = sum(r["k"] for r in krows); tot_s = sum(r["same_key_dups"] for r in krows)
M.put("k:all.same_share", f"{tot_s}/{tot_d}")
M.put("k:all.same_pct", f"{round(100 * tot_s / tot_d)}\\%")
# for each model: how much of the bare-500 vs status gap survives with keys honoured
for m, cells in (("Haiku", R2), ("Sonnet", R2), ("Fable", R3), ("Qwen", R3)):
    g = (sum(IVR(r) for r in cells[(m, B5)]) - sum(IVR(r) for r in cells[(m, A)])) / len(cells[(m, A)])
    gh = (sum(HON(r) for r in cells[(m, B5)]) - sum(HON(r) for r in cells[(m, A)])) / len(cells[(m, A)])
    M.put(f"k:{m.lower()}.gap", f2(g)); M.put(f"k:{m.lower()}.hgap", f2(gh))
    M.put(f"k:{m.lower()}.gap_kept", f"{round(100 * gh / g)}\\%" if g > 0 else "--")
write_csv("key_honouring.csv", krows)

# ------------------------------------------------------------------ M: simulated MDE
mrows = []


def MDE(key, label, p0, sign, T_, S=2):
    a = mde_sim(p0, sign, T_, S, rho=0)
    b = mde_sim(p0, sign, T_, S, rho=1)
    mrows.append(dict(key=key, label=label, p0=round(p0, 4), sign=sign, T=T_, S=S, mde_indep=a, mde_identical=b))
    sa = f2(a) if a is not None else "--"
    sb = f2(b) if b is not None else "n/a"
    M.put(f"m:{key}:lo", sa)
    M.put(f"m:{key}:hi", sb)
    M.put(f"m:{key}", sa if b is None else (sa if sa == sb else f"{sa}--{sb}"))
    return a, b


def rate(rows, f=IVR):
    return sum(f(r) for r in rows) / len(rows)

MDE("fable.status_vs_to", "Fable R3 status vs timeout (+)", rate(R3[("Fable", TO)]), +1, 32)
MDE("qwen.500_vs_to", "Qwen R3 500 vs timeout (-)", rate(R3[("Qwen", TO)]), -1, 32)
MDE("gemini.500_vs_to", "Gemini R3 500 vs timeout (+)", rate(R3[("Gemini", TO)]), +1, 31)
MDE("gpt.500_vs_to", "GPT R3 500 vs timeout (+)", rate(R3[("GPT", TO)]), +1, 32)
MDE("haiku.500_vs_to", "Haiku R2 500 vs timeout (-)", rate(R2[("Haiku", TO)]), -1, 32)
MDE("sonnet.hint", "Sonnet Oct bare_hint vs amnesiac (-), graphs", rate(SESS[("Sonnet", "amnesiac")], GIVR), -1, 16)
import numpy as _np
def power_full_drop(p0, T_, S=2, rho=0, nsim=2000, nflip=1000, seed=7):
    rng = _np.random.default_rng(seed); flips = rng.choice([-1.0, 1.0], size=(nflip, T_))
    x0 = (rng.binomial(S, p0, size=(nsim, T_)) / S) if rho == 0 else (rng.random((nsim, T_)) < p0).astype(float)
    d = -x0; obs = _np.abs(d.sum(axis=1)); null = _np.abs(d @ flips.T)
    pv = ((null >= obs[:, None] - 1e-12).sum(axis=1) + 1) / (nflip + 1)
    return float((pv < 0.05).mean())
pw = power_full_drop(rate(SESS[("Sonnet", "amnesiac")], GIVR), 16)
M.put("m:sonnet.hint:power_to_zero", f"{round(100 * pw)}\\%")
MDE("e12.sonnet.keys", "Sonnet E1-2 keys vs base (-)", rate(E12[("Sonnet", "base", "none")]), -1, 32)
MDE("r4.qwen.status_vs_500", "Qwen R4 omission status vs 500 (+)", rate(R4[("Qwen", EB)], OMIT), +1, 32)
# check-first under redelivery, every model (reduction)
lo = []; hi = []
for m in ("Haiku", "Sonnet", "GPT", "Gemini", "Qwen"):
    a, b = MDE(f"e12.{m.lower()}.careful", f"{m} E1-2 check-first vs base (-)", rate(E12[(m, "base", "none")]), -1, 32)
    lo.append(a); hi.append(b)
M.put("m:e12.careful:min", f2(min(x for x in lo if x)))
M.put("m:e12.careful:max", f2(max(x for x in hi if x)))

# E1-3 Haiku keys (July 500 baseline) and E1-6 P1 Sonnet (500 with body vs July 500)
jul = dedupe_last(load(path("sweep_raw_v2.jsonl")))
JUL = collections.defaultdict(list)
for r in jul:
    JUL[(short(r["model"]), r["prompt"], r["mitigation"], r["failure_mode"])].append(r)
assert int(sum(IVR(r) for r in JUL[("Haiku", "base", "none", B5)])) == 44
MDE("e13.haiku.keys", "Haiku E1-3 keys vs July 500 (-)", rate(JUL[("Haiku", "base", "none", B5)]), -1, 32)
MDE("e16.sonnet.body", "Sonnet E1-6 500 body vs July 500 (+)", rate(JUL[("Sonnet", "base", "none", B5)]), +1, 32)

# main-grid timeout vs 500 (the contrasts of Section 4): per model at its timeout rate
xp = [r for r in load(path("crossprovider", "sweep_raw.jsonl"))]
XPC = collections.defaultdict(list)
for r in dedupe_last(xp):
    XPC[(short(r["model"]), r["prompt"], r["mitigation"], r["failure_mode"])].append(r)
grid = {}
for m in ("Haiku", "Sonnet", "Fable"):
    grid[m] = JUL[(m, "base", "none", TO)]
for m in ("GPT", "Gemini", "Qwen"):
    grid[m] = XPC[(m, "base", "none", TO)]
assert int(sum(IVR(r) for r in grid["Qwen"])) == 53 and len(grid["Gemini"]) == 62
gm = {}
for m, rows in grid.items():
    p0 = rate(rows)
    T_ = len({r["task_id"] for r in rows})
    gm[m] = MDE(f"grid.{m.lower()}", f"{m} main grid 500 vs timeout", p0, +1 if p0 <= 0.5 else -1, T_)
los = {m: v[0] for m, v in gm.items() if v[0]}
his = {m: v[1] for m, v in gm.items() if v[1]}
mlo = min(los, key=los.get); mhi = max(his, key=his.get)
M.put("m:grid:min", f2(los[mlo])); M.put("m:grid:minname", DISP[mlo])
M.put("m:grid:max", f2(his[mhi])); M.put("m:grid:maxname", DISP[mhi])
M.put("m:grid:indepmax", f2(max(los.values())))

nulls = [m for m in mrows if m["key"] in ("fable.status_vs_to", "qwen.500_vs_to", "e13.haiku.keys", "e12.sonnet.keys",
                                          "r4.qwen.status_vs_500") or m["key"].startswith("e12.") and m["key"].endswith(".careful")]
M.put("m:nulls:min", f2(min(m["mde_indep"] for m in nulls)))
M.put("m:nulls:max", f2(max(m["mde_identical"] for m in nulls)))
write_csv("mde.csv", mrows)


# ------------------------------------------------------------------ A: abstract ranges
for m, cells in (("Haiku", R2), ("Qwen", R3), ("Sonnet", R2), ("Fable", R3)):
    rs = [rate(cells[(m, w)]) for w in UNK if (m, w) in cells]
    M.put(f"a:{m.lower()}:unk:min", f2(min(rs))); M.put(f"a:{m.lower()}:unk:max", f2(max(rs)))
    M.put(f"a:{m.lower()}:nwordings", word(len(rs)))
hq = [rate(R2[("Haiku", w)]) for w in UNK] + [rate(R3[("Qwen", w)]) for w in UNK if ("Qwen", w) in R3]
M.put("a:hq:unk:min", f2(min(hq))); M.put("a:hq:unk:max", f2(max(hq)))
maj = [m for m in ("Haiku", "Sonnet", "Fable", "GPT", "Gemini", "Qwen") if rate(SESS[(m, "amnesiac")], GIVR) > 0.5]
M.put("a:amn:majority:n", len(maj)); M.put("a:amn:majority:w", word(len(maj)))
M.put("a:amn:minority", ", ".join(DISP[m] for m in ("Haiku", "Sonnet", "Fable", "GPT", "Gemini", "Qwen") if m not in maj))

# ------------------------------------------------------------------ R4 misorders
r4all = dedupe_last([r for r in load(path("r4_precommit", "raw.jsonl")) if ok(r)])
mis = [r for r in r4all if r.get("misorder")]
M.put("r4mis:total", len(mis))
M.put("r4mis:templates", ", ".join(sorted({template_of(r) for r in mis})))
M.put("r4mis:one_template", "yes" if len({template_of(r) for r in mis}) == 1 else "no")
two = [r for r in r4all if template_of(r) == "multi_step"]
M.put("r4mis:twostep_n", len(two))
for m in ("Haiku", "Sonnet", "Qwen"):
    mm = [r for r in mis if short(r["model"]) == m]
    M.put(f"r4mis:{m.lower()}", len(mm))
    for fm in (SB, EB, TB):
        M.put(f"r4mis:{m.lower()}.{SHORTFM[fm]}", sum(1 for r in mm if r["failure_mode"] == fm))
M.put("r4mis:dup", sum(1 for r in mis if r["violations"] > 0))
# same turn: both effectful calls in the first assistant turn (from the transcripts)
same = 0; seen = 0
for r in mis:
    pat = os.path.join(RES, "r4_precommit", "transcripts",
                       f"*_{r['prompt']}_{r['mitigation']}_{r['failure_mode']}__s{r['sample']}__{r['task_id']}.json")
    cands = [c for c in glob.glob(pat) if short(json.load(open(c)).get("model", "")) == short(r["model"])]
    if not cands:
        continue
    t = json.load(open(cands[0]))
    seen += 1
    first = next((tu for tu in t["turns"] if tu.get("role") == "assistant" and any(
        b.get("type") == "tool_use" for b in tu.get("content", []))), None)
    calls = [b for b in (first or {}).get("content", []) if b.get("type") == "tool_use" and b["name"] != "list_effects"]
    same += len(calls) >= 2
M.put("r4mis:sameturn", f"{same}/{seen}")
M.put("r4mis:sameturn_all", "yes" if seen == len(mis) and same == seen else "no")

# ------------------------------------------------------------------ P: primary-test bookkeeping
NJ = json.load(open(path("analysis_v3", "numbers.json")))["numbers"]


def sup(v):
    return str(v).startswith("supported")

tests = []   # (label, supported, pre_existing)
for m in ("gpt", "gemini", "qwen"):
    tests.append((f"E1-1 {m}", sup(NJ[f"e1_1.P.H2.{m}:p"]["verdict"]), m in ("gpt", "qwen")))
tests.append(("E1-2", sup(NJ["e1_2.P.H2.sonnet_minus_haiku:p"]["verdict"]), False))
tests.append(("E1-3", sup(NJ["e1_3.P.sonnet_keys_minus_none:p"]["verdict"]), False))
for m in ("haiku", "sonnet"):
    tests.append((f"E1-6 P1 {m}", sup(NJ[f"e1_6.P1.{m}:p"]["verdict"]), False))
    tests.append((f"E1-6 P2 {m}", sup(NJ[f"e1_6.P2.{m}:p"]["verdict"]) and sup(NJ[f"e1_6.P2.{m}.readback:p"]["verdict"]), False))
cont = [("E1-0", sup(NJ["e1_0.P.diff_rerun_minus_v2"]["verdict"]))]
for m in ("haiku", "sonnet"):
    for f in ("toA", "e500"):
        cont.append((f"E1-4 {m} {f}", sup(NJ[f"e1_4.{m}.{f}.P.new_within_released_ci"]["verdict"])))
        cont.append((f"E1-5 {m} {f}", sup(NJ[f"e1_5.{m}.{f}.P.within_e1_4_pooled96_ci"]["verdict"])))
M.put("p:tests:n", len(tests)); M.put("p:tests:sup", sum(s for _, s, _ in tests))
M.put("p:tests:n:w", word(len(tests))); M.put("p:tests:sup:w", word(sum(s for _, s, _ in tests)))
M.put("p:cont:n", len(cont)); M.put("p:cont:sup", sum(s for _, s in cont))
M.put("p:cont:n:w", word(len(cont))); M.put("p:cont:sup:w", word(sum(s for _, s in cont)))
M.put("p:all:n", len(tests) + len(cont)); M.put("p:all:sup", sum(s for _, s, _ in tests) + sum(s for _, s in cont))
pre = [t for t in tests if t[2]]
M.put("p:nopre:n", len(tests) + len(cont) - len(pre))
M.put("p:nopre:sup", sum(s for _, s, p in tests if not p) + sum(s for _, s in cont))
M.put("p:pre:n", len(pre))

# every registered primary verdict, all experiments
verd = []   # (exp, label, supported, deciding p or None)
for lab, s, _ in tests:
    k = {"E1-2": "e1_2.P.H2.sonnet_minus_haiku:p", "E1-3": "e1_3.P.sonnet_keys_minus_none:p"}.get(lab)
    if lab.startswith("E1-1"):
        k = f"e1_1.P.H2.{lab.split()[1]}:p"
    if lab.startswith("E1-6 P1"):
        k = f"e1_6.P1.{lab.split()[2]}:p"
    if lab.startswith("E1-6 P2"):
        mm = lab.split()[2]
        p = max(NJ[f"e1_6.P2.{mm}:p"]["value"], NJ[f"e1_6.P2.{mm}.readback:p"]["value"])
    else:
        p = NJ[k]["value"]
    verd.append(("E1", lab, s, p))
for lab, s in cont:
    verd.append(("E1", lab, s, None))
real = json.load(open(path("analysis_v3", "real_stripe.json")))
ag = real["real.agreement.ledger_vs_stripe"]
verd.append(("R1", "R1-H2", ag["k"] == ag["n"], None))
for exp, fn, pref in (("R2", path("analysis_v3", "r2_wording.json"), "r2"), ("R3", path("analysis_v3", "r3_wording.json"), "r3")):
    J = json.load(open(fn))["numbers"]
    for k, v in J.items():
        if k.endswith(":supported") and v.get("status") == "primary":
            part, mm, hyp = k.split(".")[1], k.split(".")[2], k.split(".")[3].split(":")[0]
            hp = sorted(e["holm_p"] for kk, e in J.items() if kk.startswith(f"{pref}.{part}.{mm}.") and kk.endswith(":p")
                        and str(e.get("status", "")).startswith(f"primary {exp}-{hyp}"))
            verd.append((exp, f"{exp}-{hyp} {mm}", v["value"] is True, hp[1] if len(hp) > 1 else None))
J4 = json.load(open(path("r4_precommit", "analysis", "r4_precommit.json")))["numbers"]
for k, v in J4.items():
    if k.endswith("H1:supported"):
        verd.append(("R4", f"R4-H1 {k.split('.')[1]}", v["value"] == "supported", v.get("p")))
nv = len(verd); ns = sum(1 for v in verd if v[2])
M.put("p:allexp:n", nv); M.put("p:allexp:sup", ns)
ntest = [v for v in verd if v[3] is not None]
M.put("p:allexp:ntest", len(ntest))
bonf = sum(1 for v in ntest if v[2] and v[3] * len(ntest) < 0.05)
M.put("p:allexp:bonf_sup", bonf)
M.put("p:allexp:test_sup", sum(1 for v in ntest if v[2]))
lost = [v[1] for v in ntest if v[2] and v[3] * len(ntest) >= 0.05]
M.put("p:allexp:bonf_lost", ", ".join(lost) if lost else "none")
write_csv("primary_verdicts.csv", [dict(exp=a, label=b, supported=c, deciding_p=d) for a, b, c, d in verd])

# ------------------------------------------------------------------ H: Holm under the first-run rule
pg, pm = NJ["e1_1.P.H2.gpt:p"]["value"], NJ["e1_1.P.H2.gemini:p"]["value"]
hk = holm([pg, pm, NJ["e1_1.P.H2.qwen:p"]["value"]])[2]
hf = holm([pg, pm, NJ["e1_1.P.H2.qwen.first_rows:p"]["value"]])[2]
M.put("h:qwen.kept", pe(hk)); M.put("h:qwen.first", pe(hf))
M.put("h:qwen.first_survives", "yes" if hf < 0.05 else "no"); M.put("h:qwen.kept_survives", "yes" if hk < 0.05 else "no")

# ------------------------------------------------------------------ S: same-month Haiku E1-3
oct6 = dedupe_last([r for r in load(prefer_v2(path("tier1", "e1_6_oct", "raw.jsonl"))) if ok(r)])
oct500 = [r for r in oct6 if short(r["model"]) == "Haiku" and r["failure_mode"] == B5]
e13 = dedupe_last([r for r in load(path("tier1", "e1_3", "raw_v2.jsonl")) if ok(r)])
e13h = [r for r in e13 if short(r["model"]) == "Haiku"]
r = paired_contrast(e13h, oct500, IVR)
M.put("s:haiku.e13.oct500:kn", kn(r["k_ref"], r["n_ref"])); M.put("s:haiku.e13.keys:kn", kn(r["k_cell"], r["n_cell"]))
M.put("s:haiku.e13:d", dsg(r["d"])); M.put("s:haiku.e13:p", pe(r["p_task"]))
M.put("s:haiku.e13:sig", "yes" if r["p_task"] < 0.05 else "no")

# ------------------------------------------------------------------ F: Stripe cells vs registered simulator cells
frows = []
nd = 0
for k, v in real.items():
    if not isinstance(v, dict) or "within_sim_ci" not in v:
        continue
    sk = v.get("sim_key")
    s = NJ.get(sk, {})
    if "k" not in s:
        continue
    p = fisher_two_sided(v["k"], v["n"], s["k"], s["n"])
    frows.append(dict(cell=k, real=f"{v['k']}/{v['n']}", sim=f"{s['k']}/{s['n']}", p_fisher=round(p, 4), within=v["within_sim_ci"]))
    nd += p < 0.05
M.put("f:n", len(frows)); M.put("f:differ", nd); M.put("f:differ:w", word(nd))
diff_names = []
for fr in frows:
    if fr["p_fisher"] < 0.05:
        parts = fr["cell"].split(":")[0].split(".")
        lab = {"base.none.toA": "timeout", "base.none.e500": "server error", "base.none.w3": "outcome unknown",
               "base.keys.e500": "keys", "restart.amnesiac": "amnesiac", "restart.bare_hint": "bare hint"}[".".join(parts[2:])]
        diff_names.append(f"{DISP[short(parts[1])]} {lab}")
M.put("f:differ:cells", "; ".join(diff_names))
write_csv("stripe_fisher.csv", frows)

# ------------------------------------------------------------------ B: Holm block sizes (secondary tests)
blocks = collections.Counter()
for exp, fn in (("R2", path("analysis_v3", "r2_wording.json")), ("R3", path("analysis_v3", "r3_wording.json"))):
    J = json.load(open(fn))["numbers"]
    for kk, e in J.items():
        if kk.endswith(":p") and str(e.get("status", "")).startswith("secondary"):
            parts = kk.split(".")
            blocks[(exp, parts[1], parts[2])] += 1
for kk, e in J4.items():
    if kk.endswith(":p") and e.get("holm_p") is not None and str(e.get("status", "")).startswith("secondary"):
        blocks[("R4", "-", kk.split(".")[1])] += 1
bsz = collections.defaultdict(set)
for (exp, part, m), n in blocks.items():
    bsz[(exp, part)].add(n)
for (exp, part), s in bsz.items():
    M.put(f"b:{exp}.{part}", "--".join(str(x) for x in sorted(s)))
write_csv("holm_blocks.csv", [dict(exp=a, part=b, model=c, n_tests=n) for (a, b, c), n in sorted(blocks.items())])

# ------------------------------------------------------------------ X: key reuse with other arguments
sys.path.insert(0, os.path.join(ROOT, "src"))
from world import normalize_value, EFFECTFUL_TOOLS  # noqa: E402  (read-only use of the scorer's rule)
tdirs = glob.glob(os.path.join(RES, "**", "sweep_transcripts"), recursive=True) + \
    glob.glob(os.path.join(RES, "tier1", "e1_*", "transcripts")) + [os.path.join(RES, "real_stripe", "transcripts")]
reuse = clash = keyed_eps = 0
for d in tdirs:
    for fn in glob.glob(os.path.join(d, "*.json")):
        try:
            t = json.load(open(fn))
        except Exception:
            continue
        if t.get("mitigation") != "idem_keys":
            continue
        keyed_eps += 1
        turns = t.get("turns") or [x for dd in t.get("deliveries", []) for x in dd.get("turns", [])]
        seen = {}
        for tu in turns:
            if tu.get("role") != "assistant":
                continue
            for b in tu.get("content", []):
                if b.get("type") != "tool_use" or b["name"] not in EFFECTFUL_TOOLS:
                    continue
                key = (b.get("input") or {}).get("idempotency_key")
                if not key or not isinstance(key, str):
                    continue
                sid = tuple(normalize_value((b["input"] or {}).get(f)) for f in EFFECTFUL_TOOLS[b["name"]][1])
                kk = (b["name"], key)
                if kk in seen:
                    reuse += 1
                    clash += seen[kk] != sid
                else:
                    seen[kk] = sid
M.put("x:keyreuse:calls", reuse); M.put("x:keyreuse:clash", clash); M.put("x:keyreuse:eps", keyed_eps)

# Gemini's blocked task
ref = list(csv.DictReader(open(path("analysis_v3", "refusals.csv"))))
gt = sorted({r["tasks"] for r in ref if "gemini" in r["model"]})
M.put("x:gemini_blocked_task", "\\ttt{" + tex_escape(";".join(gt)) + "}")
M.put("x:gemini_blocked_ntasks", len(gt))

# one example instruction per template (R2 Part B, timeout cell, Haiku, sample 0, first task id)
ex = {}
for fn in sorted(glob.glob(os.path.join(RES, "r2_wording", "transcripts", "*_base_none_timeout_after_commit__s0__*.json"))):
    t = json.load(open(fn))
    if short(t["model"]) != "Haiku":
        continue
    tm = t["task_id"].rsplit("-", 1)[0]
    ex.setdefault(tm, t["instruction"])
for tm, s in ex.items():
    M.put(f"x:ex:{tm}", tex_escape(s))
M.put("x:ex:n", len(ex))

# ------------------------------------------------------------------ write
M.write(os.path.join(ROOT, "paper-v27", "numbers_v27.tex"), "".join(EXTRA))
print("wrote", len(M.d), "macros")
for k, v in M.d.items():
    print(k, "=", v)
