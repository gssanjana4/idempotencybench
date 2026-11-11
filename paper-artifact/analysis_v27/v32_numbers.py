"""v32: independent recomputation of every R5 (real API error texts) number the paper prints.

Reads only
  v32-experiments/results/r5_real_errors/raw.jsonl        (2,560 episode rows, last row per id wins)
  v32-experiments/results/r5_real_errors/transcripts/     (one JSON per episode)
  v32-experiments/results/spend_ledger.jsonl              (the cumulative v32 spend ledger)
  v32-experiments/error_texts/catalogue.json              (texts, classes, flags)
  v32-experiments/configs/r5_real_errors.json             (cells, models, registered constants)
  v32-experiments/classification_check/*.json             (blind second coder)
  results/r2_wording/raw.jsonl, results/r3_wording/raw.jsonl  (anchor drift, task set)
and writes
  paper-v32/numbers_v32.tex        (\\defnum{v32:...}{...}; never redefines another key;
                                    V32_NUMBERS_TEX overrides the path)
  analysis_v27/out_v32/*.csv|txt   (audit rows and the comparison report; V32_OUT overrides)

It does NOT import v32-experiments/src/analysis_r5.py (the registered analysis): the statistics
are re-implemented here from the registration text (prereg_r5.md), and the result is compared,
key by key, with the registered analysis output analysis/r5_real_errors.json. Any mismatch is
printed and makes the script exit 1. Standard library only.

Registered definitions re-implemented (prereg_r5.md, "Primary hypothesis" and "Secondary"):
  IVR      share of episodes with >= 1 duplicate (row field violations > 0)
  class    rate on a unit = mean over the class's texts of the text's rate on that unit
  R5-P1    d_t = IVR_U(t) - IVR_G(t) over the 8 templates; exact one-sided sign-flip over the
           2^8 sign patterns, p = #{s: mean(s.d) <= mean(d)} / 256; supported if p < 0.05;
           not evaluable if any U or G cell is below 90% of its planned 64
  task     paired sign-flip over the 32 tasks, 10,000 flips, seed 20261025, p = (hits+1)/(N+1)
           (the R2/R3 scheme; reported without a verdict)
  S1-S6    eight template-level exact tests (S1 x3 two-sided, S2-S5 two-sided, S6 one-sided),
           Holm step-down within each model
"""
import collections
import csv
import itertools
import json
import math
import os
import random
import re
import sys
from decimal import Decimal, ROUND_HALF_UP

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
V32 = os.path.join(ROOT, "v32-experiments")
RD = os.path.join(V32, "results", "r5_real_errors")
TEX = os.environ.get("V32_NUMBERS_TEX", os.path.join(ROOT, "paper-v32", "numbers_v32.tex"))
OUT = os.environ.get("V32_OUT", os.path.join(ROOT, "analysis_v27", "out_v32"))
os.makedirs(OUT, exist_ok=True)

CFG = json.load(open(os.path.join(V32, "configs", "r5_real_errors.json")))
R5 = CFG["r5"]
CAT = json.load(open(os.path.join(V32, "error_texts", "catalogue.json")))
TEXTS = {e["id"]: e for e in CAT["texts"]}
PLANNED = int(CFG["expect_per_cell"])
MIN_N = math.ceil(float(R5["min_fraction"]) * PLANNED)
SEED, PERM_N, ALPHA, FLOOR = int(R5["perm_seed"]), int(R5["perm_n"]), float(R5["alpha"]), float(R5["floor_ivr"])

E500, W3 = "error_500_after_commit", "status_unknown_after_commit"
FM = {i: f"real_{i}_after_commit" for i in R5["injected"]}
INJ = [FM[i] for i in R5["injected"]]
CLS = {FM[i]: TEXTS[i]["class"] for i in R5["injected"]}
U = [f for f in INJ if CLS[f] == "U"]
G = [f for f in INJ if CLS[f] == "G"]
N = [f for f in INJ if CLS[f] == "N"]
R = [f for f in INJ if CLS[f] == "R"]
U_NC = [f for f in U if not TEXTS[f[5:-13]]["instructs_check"]]
CELLS = INJ + [E500, W3]
MODELS = {"claude-haiku-4-5-20251001": "haiku", "openrouter:qwen/qwen3.6-27b-20260422": "qwen",
          "claude-sonnet-5": "sonnet", "claude-fable-5": "fable"}
SHORT_TEXT = {"azure_storage_operation_timed_out": "azure", "gcp_rpc_deadline_exceeded": "gcp",
              "stripe_500_indeterminate": "stripe", "github_rest_timeout_server_error": "github",
              "openai_500": "openai", "shopify_503": "shopify", "aws_s3_service_unavailable": "s3unavail",
              "aws_s3_internal_error": "s3internal"}
SHORT_FM = {FM[i]: s for i, s in SHORT_TEXT.items()}
SHORT_FM.update({E500: "e500", W3: "w3"})
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen"]

NUMS = collections.OrderedDict()
MISMATCH = []
REPORT = []


def put(k, v):
    k = "v32:" + k
    assert k not in NUMS, k
    NUMS[k] = str(v)


def say(*a):
    s = " ".join(str(x) for x in a)
    REPORT.append(s)
    print(s)


ROUNDING = []


def check(label, mine, theirs):
    """Exact match, or a rounding-only difference: the registered script rounds each text's rate to
    4 decimals before averaging a class and each p to 6 decimals before Holm, so its value can
    differ from the exact one by < 1e-4 (rates) or < 1e-5 (Holm p). Those are listed apart;
    anything else is a mismatch."""
    if mine == theirs:
        return True
    if isinstance(mine, float) and isinstance(theirs, float) and abs(mine - theirs) < 1e-4:
        ROUNDING.append(f"{label}: exact {mine!r}, registered script {theirs!r} (rounded before aggregating)")
        return True
    MISMATCH.append(f"{label}: recomputed {mine!r} vs registered analysis {theirs!r}")
    return False


def f2(x):
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def f3(x):
    return str(Decimal(repr(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def sd(x):
    """signed two decimals, as the paper prints d (+0.55, $-$0.22)"""
    s = f2(abs(x))
    return ("$-$" + s) if x < -1e-12 else ("+" + s if x > 1e-12 else "0.00")


def pfmt(p):
    return "<0.001" if p < 0.001 else f3(p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def ci(lo, hi):
    return f"[{f2(lo)}, {f2(hi)}]"


# ------------------------------------------------------------------ data
rows = {}
with open(os.path.join(RD, "raw.jsonl")) as fh:
    nlines = 0
    for line in fh:
        nlines += 1
        r = json.loads(line)
        rows[r["episode_id"]] = r
rows = list(rows.values())
say(f"raw.jsonl: {nlines} lines, {len(rows)} distinct episodes")
refusals = [r for r in rows if r.get("refusal")]
rows = [r for r in rows if not r.get("refusal")]

# expected design: 4 models x 10 cells x 32 tasks x 2 samples, the 32 tasks of R2 Part B / R3-B
def tasks_of(path, fm=E500):
    s = set()
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("failure_mode") == fm and r.get("prompt") == "base" and r.get("mitigation") == "none":
                s.add(r["task_id"])
    return s

r2_tasks = tasks_of(os.path.join(ROOT, "results", "r2_wording", "raw.jsonl"))
r3_tasks = tasks_of(os.path.join(ROOT, "results", "r3_wording", "raw.jsonl"))
r5_tasks = {r["task_id"] for r in rows}
assert r2_tasks == r3_tasks == r5_tasks and len(r5_tasks) == 32, "R5 task set differs from R2/R3 Part B"
expected = {(m, fm, t, s) for m in MODELS for fm in CELLS for t in r5_tasks for s in (0, 1)}
got = collections.Counter((r["model"], r["failure_mode"], r["task_id"], r["sample"]) for r in rows)
missing = expected - set(got)
extra = set(got) - expected
dup_rows = [k for k, v in got.items() if v > 1]
say(f"design: expected {len(expected)} (model, cell, task, sample) units; missing {len(missing)}, "
    f"unexpected {len(extra)}, duplicated {len(dup_rows)}; refusals {len(refusals)}")
put("r5:episodes", f"{len(rows):,}")
put("r5:planned", f"{len(expected):,}")
put("r5:missing", len(missing))
put("r5:refusals", len(refusals))
put("r5:cells", len(MODELS) * len(CELLS))
put("r5:tasks", len(r5_tasks))
put("r5:templates", len({r["template"] for r in rows}))
put("r5:texts", len(INJ))
put("r5:per_cell", PLANNED)
put("r5:floor", f2(FLOOR))
put("r5:tests", len(re.findall(r"^\s*def test_", open(os.path.join(V32, "tests", "test_r5.py")).read(), re.M)))
put("r5:all_full", "yes" if not missing and not extra and not dup_rows and not refusals else "no")
trunc = [r for r in rows if r.get("truncated")]
put("r5:truncated", len(trunc))
put("r5:truncated:cells", "; ".join(sorted(f"{MODELS[r['model']]}/{SHORT_FM[r['failure_mode']]}" for r in trunc)) or "none")
put("r5:turncap", sum(1 for r in rows if r.get("turn_cap_hit")))
put("r5:invalid", sum(1 for r in rows if r.get("invalid_calls")))
put("r5:fired_all", "yes" if all(r.get("failure_fired") for r in rows) else "no")
put("r5:first_utc", min(r["started_utc"] for r in rows))
put("r5:last_utc", max(r["finished_utc"] for r in rows))
# truncation sensitivity is registered for cells where truncated episodes exceed 2%
tr_cells = collections.Counter((r["model"], r["failure_mode"]) for r in trunc)
put("r5:trunc_over_2pct", sum(1 for c, v in tr_cells.items() if v / PLANNED > 0.02))

by_cell = collections.defaultdict(list)
for r in rows:
    by_cell[(r["model"], r["failure_mode"])].append(r)


def dup(r):
    return 1.0 if r["violations"] > 0 else 0.0


def unit_rates(eps, key):
    acc = collections.defaultdict(list)
    for e in eps:
        acc[e[key]].append(dup(e))
    return {u: sum(v) / len(v) for u, v in acc.items()}


def class_units(model, fms, key):
    per = [unit_rates(by_cell[(model, f)], key) for f in fms]
    units = set().union(*per)
    return {u: sum(p[u] for p in per if u in p) / sum(1 for p in per if u in p) for u in units}


def exact_signflip(d, alternative):
    n = len(d)
    obs = sum(d) / n
    hit = 0
    for signs in itertools.product((1, -1), repeat=n):
        v = sum(s * x for s, x in zip(signs, d)) / n
        if alternative == "less":
            hit += v <= obs + 1e-12
        else:
            hit += abs(v) >= abs(obs) - 1e-12
    return obs, hit / 2 ** n


def mc_signflip(d, alternative):
    # the R2/R3 scheme (one uniform draw per unit in sorted unit order, Random(20261025))
    rng = random.Random(SEED)
    obs = sum(d)
    hit = 0
    for _ in range(PERM_N):
        s = sum(x if rng.random() < 0.5 else -x for x in d)
        hit += (s <= obs + 1e-12) if alternative == "less" else (abs(s) >= abs(obs) - 1e-12)
    return obs / len(d), (hit + 1) / (PERM_N + 1)


def contrast(model, a, b, alternative):
    res = {}
    for lvl, key in (("template", "template"), ("task", "task_id")):
        ra, rb = class_units(model, a, key), class_units(model, b, key)
        units = sorted(set(ra) & set(rb))
        d = [ra[u] - rb[u] for u in units]
        dm, p = (exact_signflip if lvl == "template" else mc_signflip)(d, alternative)
        res[lvl] = {"d": dm, "p": p, "units": len(d), "plus": sum(x > 1e-12 for x in d),
                    "minus": sum(x < -1e-12 for x in d)}
    return res


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


REG = json.load(open(os.path.join(RD, "analysis", "r5_real_errors.json")))
check("status", "ok", REG["status"])
check("planned_per_cell", PLANNED, REG["planned_per_cell"])
check("min_n", MIN_N, REG["min_n"])
check("classes", {"U": U, "G": G, "N": N, "R": R}, REG["classes"])
check("refusals", len(refusals), REG["refusals"])

audit_cells, audit_tests = [], []
succ_all, rb = [], {}
for model, ms in MODELS.items():
    reg = REG["models"][model]
    role = "primary" if model in R5["primary_models"] else "secondary"
    check(f"{ms} role", role, reg["role"])
    st = {}
    for fm in CELLS:
        eps = by_cell[(model, fm)]
        n = len(eps)
        k = sum(1 for e in eps if e["violations"] > 0)
        T = len({e["task_id"] for e in eps})
        p = k / n
        lo, hi = wilson(k, n)
        tlo, thi = wilson(p * T, T)
        fired = [e for e in eps if e.get("failure_fired")]
        rbk = sum(1 for e in fired if e.get("readback_after_failure"))
        ret = sum(1 for e in fired if e.get("retried"))
        succ = sum(1 for e in eps if e.get("achieved"))
        st[fm] = {"n": n, "k": k, "p": p}
        rr = reg["cells"][fm]
        tag = f"{ms}/{SHORT_FM[fm]}"
        check(tag + " n", n, rr["n"]); check(tag + " k", k, rr["k"])
        check(tag + " ivr", round(p, 4), rr["ivr"])
        check(tag + " ci_episode", [round(lo, 4), round(hi, 4)], rr["ci_episode"])
        check(tag + " ci_task", [round(tlo, 4), round(thi, 4)], rr["ci_task"])
        check(tag + " readback", round(rbk / len(fired), 4), rr["readback_after_failure"])
        check(tag + " retried", round(ret / len(fired), 4), rr["retried"])
        check(tag + " success", round(succ / n, 4), rr["success"])
        check(tag + " truncated", sum(1 for e in eps if e.get("truncated")), rr["truncated"])
        check(tag + " ok90", n >= MIN_N, rr["ok90"])
        q = f"{ms}.{SHORT_FM[fm]}"
        put(q + ":ivr", f2(p)); put(q + ":kn", f"{k}/{n}"); put(q + ":tci", ci(tlo, thi))
        put(q + ":rb", f2(rbk / len(fired))); put(q + ":succ", f2(succ / n))
        succ_all.append(succ / n)
        rb[(ms, fm)] = (rbk, len(fired))
        audit_cells.append({"model": model, "cell": fm, "class": CLS.get(fm, "anchor"), "n": n, "k": k,
                            "ivr": round(p, 4), "tci": ci(tlo, thi), "readback": round(rbk / len(fired), 4),
                            "retried": round(ret / len(fired), 4), "success": round(succ / n, 4)})
    # class rates (mean of the texts' rates; every text has n = 64) and a task-level interval
    for c, fms in (("U", U), ("G", G), ("N", N), ("R", R), ("Unc", U_NC)):
        rate = sum(st[f]["p"] for f in fms) / len(fms)
        tu = class_units(model, fms, "task_id")
        T = len(tu)
        tlo, thi = wilson(rate * T, T)
        kk, nn = sum(st[f]["k"] for f in fms), sum(st[f]["n"] for f in fms)
        put(f"{ms}.{c}:ivr", f2(rate)); put(f"{ms}.{c}:tci", ci(tlo, thi)); put(f"{ms}.{c}:kn", f"{kk}/{nn}")
        rbk = sum(rb[(ms, f)][0] for f in fms); rbn = sum(rb[(ms, f)][1] for f in fms)
        put(f"{ms}.{c}:rb", f2(rbk / rbn))
        regkey = {"Unc": "U_no_check"}.get(c, c)
        check(f"{ms} class {c}", round(rate, 4), reg["class_ivr"][regkey])
    for c, fms in (("U", U), ("G", G)):
        spread = max(st[f]["p"] for f in fms) - min(st[f]["p"] for f in fms)
        check(f"{ms} spread {c}", round(spread, 4), reg["spread"][c])
        put(f"{ms}.{c}:spread", f2(spread))
    # R5-P1
    ok = all(st[f]["n"] >= MIN_N for f in U + G)
    t = contrast(model, U, G, "less")
    verdict = ("supported" if t["template"]["p"] < ALPHA else "not supported") if ok else "not evaluable"
    rc = reg["P1"]["contrast"]
    for lvl in ("template", "task"):
        check(f"{ms} P1 {lvl} d", round(t[lvl]["d"], 4), rc[lvl]["d"])
        check(f"{ms} P1 {lvl} p", round(t[lvl]["p"], 6), rc[lvl]["p"])
        for x in ("units", "plus", "minus"):
            check(f"{ms} P1 {lvl} {x}", t[lvl][x], rc[lvl][x])
    regv = reg["P1"]["verdict"]
    check(f"{ms} P1 verdict", verdict if role == "primary" else f"{verdict} (secondary model, no confirmatory claim)", regv)
    floor = sum(st[f]["p"] for f in G) / len(G) <= FLOOR
    check(f"{ms} P1 floor reading", None if not floor else "floor", None if reg["P1"]["reading"] is None else "floor")
    put(f"{ms}.P1:d", sd(t["template"]["d"])); put(f"{ms}.P1:dabs", f2(abs(t["template"]["d"])))
    put(f"{ms}.P1:ptmpl", pfmt(t["template"]["p"])); put(f"{ms}.P1:ptask", pfmt(t["task"]["p"]))
    put(f"{ms}.P1:verdict", verdict)
    put(f"{ms}.P1:tmpl_lower", f"{t['template']['minus']} of {t['template']['units']}")
    put(f"{ms}.P1:tmpl_ties", t["template"]["units"] - t["template"]["minus"] - t["template"]["plus"])
    put(f"{ms}.P1:tmpl_higher", t["template"]["plus"])
    put(f"{ms}.P1:floor", "yes" if floor else "no")
    audit_tests.append({"model": model, "test": "P1 U<G", "d_template": round(t["template"]["d"], 4),
                        "p_template": round(t["template"]["p"], 6), "d_task": round(t["task"]["d"], 4),
                        "p_task": round(t["task"]["p"], 6), "holm_p": "", "verdict": verdict})
    # secondary block S1 x3, S2..S6, Holm over the eight template-level p's
    sec = [(f"S1 {f[5:-13]} vs anchor 500", [f], [E500], "two-sided") for f in U]
    sec += [("S2 G class vs anchor 500", G, [E500], "two-sided"), ("S3 U class vs anchor W3", U, [W3], "two-sided"),
            ("S4 N text vs G class", N, G, "two-sided"), ("S5 R text vs G class", R, G, "two-sided"),
            ("S6 U without check-instructing text vs G (sensitivity)", U_NC, G, "less")]
    res = [contrast(model, a, b, alt) for _, a, b, alt in sec]
    adj = holm([x["template"]["p"] for x in res])
    for (name, _, _, alt), x, h, rs in zip(sec, res, adj, reg["secondary"]):
        check(f"{ms} {name} name", name, rs["test"])
        check(f"{ms} {name} alternative", alt, rs["alternative"])
        for lvl in ("template", "task"):
            check(f"{ms} {name} {lvl} d", round(x[lvl]["d"], 4), rs["contrast"][lvl]["d"])
            check(f"{ms} {name} {lvl} p", round(x[lvl]["p"], 6), rs["contrast"][lvl]["p"])
        check(f"{ms} {name} holm", round(h, 6), rs["holm_p_template"])
        audit_tests.append({"model": model, "test": name, "d_template": round(x["template"]["d"], 4),
                            "p_template": round(x["template"]["p"], 6), "d_task": round(x["task"]["d"], 4),
                            "p_task": round(x["task"]["p"], 6), "holm_p": round(h, 6), "verdict": ""})
    keys = ["S1azure", "S1gcp", "S1stripe", "S2", "S3", "S4", "S5", "S6"]
    for kname, x, h in zip(keys, res, adj):
        put(f"{ms}.{kname}:d", sd(x["template"]["d"])); put(f"{ms}.{kname}:holm", pfmt(h))
        put(f"{ms}.{kname}:ptask", pfmt(x["task"]["p"]))
        put(f"{ms}.{kname}:sig", "yes" if h < ALPHA else "no")
    put(f"{ms}.S:nsig", sum(1 for h in adj if h < ALPHA))
    put(f"{ms}.S:holm_min", pfmt(min(adj)))

with open(os.path.join(RD, "summary.csv")) as fh:
    for r in csv.DictReader(fh):
        e = by_cell[(r["model"], r["failure_mode"])]
        k = sum(1 for x in e if x["violations"] > 0)
        check(f"summary.csv {MODELS[r['model']]}/{SHORT_FM[r['failure_mode']]} n", len(e), int(r["n_episodes"]))
        check(f"summary.csv {MODELS[r['model']]}/{SHORT_FM[r['failure_mode']]} ivr", round(k / len(e), 4), float(r["ivr"]))
put("succ:min", f2(min(succ_all)))
for nm, fn in (("primary", R5["primary_models"]), ("secondary", R5["secondary_models"])):
    put(f"{nm}:P1:all_supported", "yes" if all(NUMS[f"v32:{MODELS[m]}.P1:verdict"] == "supported" for m in fn) else "no")
# largest U-class rate and smallest G-class rate over the four models (abstract/summary sentence)
put("U:max", f2(max(float(NUMS[f"v32:{s}.U:ivr"]) for s in MODELS.values())))
put("G:min", f2(min(float(NUMS[f"v32:{s}.G:ivr"]) for s in MODELS.values())))
put("G:max", f2(max(float(NUMS[f"v32:{s}.G:ivr"]) for s in MODELS.values())))
# per-text extremes used in the text
hz ={f: float(NUMS[f"v32:haiku.{SHORT_FM[f]}:ivr"]) for f in U}
put("haiku:U:argmax", SHORT_FM[max(hz, key=hz.get)])
put("U:all_below_G:allmodels", "yes" if all(
    max(float(NUMS[f"v32:{s}.{SHORT_FM[f]}:ivr"]) for f in U) < min(float(NUMS[f"v32:{s}.{SHORT_FM[f]}:ivr"]) for f in G)
    for s in MODELS.values()) else "no")
put("R:above_G:n", sum(1 for s in MODELS.values() if float(NUMS[f"v32:{s}.R:ivr"]) > float(NUMS[f"v32:{s}.G:ivr"])))
put("N:below_G:n", sum(1 for s in MODELS.values() if float(NUMS[f"v32:{s}.N:ivr"]) < float(NUMS[f"v32:{s}.G:ivr"])))
# U class vs W3 anchor and G class vs bare-500 anchor: none significant after Holm
# with 8 templates and 8 tests, a two-sided test's smallest Holm-adjusted p is 7 x 2/256 (after a
# one-sided test at 1/256), so no two-sided secondary can pass Holm at the template level
put("S:twosided_min_holm", pfmt(7 * 2 / 256))
def fnum(k):
    return float(NUMS["v32:" + k].replace("$-$", "-").replace("+", "").replace("<", ""))
for t_ in ("S2", "S3"):
    put(f"{t_}:dabs_max", f2(max(abs(fnum(f"{s}.{t_}:d")) for s in MODELS.values())))
    put(f"{t_}:ptask_min", pfmt(min(fnum(f"{s}.{t_}:ptask") for s in MODELS.values())))
put("S23:ptask_min", pfmt(min(fnum(f"{s}.{t_}:ptask") for s in MODELS.values() for t_ in ("S2", "S3"))))
put("U:rb:min", f2(min(fnum(f"{s}.U:rb") for s in MODELS.values())))
put("G:rb:max", f2(max(fnum(f"{s}.G:rb") for s in MODELS.values())))
put("R:ivr:min", f2(min(fnum(f"{s}.R:ivr") for s in MODELS.values())))
put("S2:sig:n", sum(1 for s in MODELS.values() if NUMS[f"v32:{s}.S2:sig"] == "yes"))
put("S3:sig:n", sum(1 for s in MODELS.values() if NUMS[f"v32:{s}.S3:sig"] == "yes"))
put("S2:holm_min", pfmt(min(float(NUMS[f"v32:{s}.S2:holm"].replace("<", "")) for s in MODELS.values())))
put("S3:holm_min", pfmt(min(float(NUMS[f"v32:{s}.S3:holm"].replace("<", "")) for s in MODELS.values())))

# ------------------------------------------------------------------ anchor drift against R2/R3
def cell_kn(path, model, fm):
    seen = {}
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("model") == model and r.get("failure_mode") == fm and r.get("prompt") == "base" \
                    and r.get("mitigation") == "none" and not r.get("refusal"):
                seen[r["episode_id"]] = r
    v = list(seen.values())
    return sum(1 for r in v if r["violations"] > 0), len(v)

SRC = {"haiku": "r2_wording", "sonnet": "r2_wording", "qwen": "r3_wording", "fable": "r3_wording"}
for model, ms in MODELS.items():
    for fm in (E500, W3):
        k, n = cell_kn(os.path.join(ROOT, "results", SRC[ms], "raw.jsonl"), model, fm)
        put(f"{ms}.{SHORT_FM[fm]}:prev_kn", f"{k}/{n}")
        put(f"{ms}.{SHORT_FM[fm]}:prev_ivr", f2(k / n) if n else "--")
NT0 = open(os.path.join(ROOT, "paper-merged", "numbers.tex")).read()
for ms, run in (("haiku", "r2"), ("sonnet", "r2"), ("qwen", "r3"), ("fable", "r3")):
    for a in ("e500", "w3"):
        m = re.search(r"\\defnum\{nj:%s\.B\.%s\.%s:ivr:kn\}\{([0-9/]+)\}" % (run, ms, a), NT0)
        check(f"{ms} {a} earlier run vs numbers.tex", NUMS[f"v32:{ms}.{a}:prev_kn"], m.group(1))
put("drift:max", f2(max(abs(float(NUMS[f"v32:{s}.{a}:ivr"]) - float(NUMS[f"v32:{s}.{a}:prev_ivr"]))
                       for s in MODELS.values() for a in ("e500", "w3"))))

# ------------------------------------------------------------------ catalogue summary
texts = CAT["texts"]
cc = collections.Counter(e["class"] for e in texts)
for c in "UNRG":
    put(f"cat:{c}", cc[c])
put("cat:n", len(texts))
put("cat:providers", len({e["provider"] for e in texts}))
put("cat:providers:w", WORDS[len({e["provider"] for e in texts})])
put("cat:borderline", sum(1 for e in texts if e["borderline"]))
put("cat:raw", sum(1 for e in texts if e["retrieval"] == "raw"))
put("cat:webfetch", sum(1 for e in texts if e["retrieval"] == "webfetch"))
put("cat:accessed", CAT["accessed"])
sk = collections.Counter(e["source_kind"] for e in texts)
put("cat:returned", sk["returned_message"]); put("cat:statusdesc", sk["status_description"]); put("cat:guidance", sk["doc_guidance"])
cond = collections.Counter(e["condition"] for e in texts)
put("cat:timeout", cond["timeout"]); put("cat:5xx", cond["5xx"]); put("cat:ambiguous", cond["ambiguous"])
put("cat:vendor_named", sum(1 for e in texts if e["vendor_name_in_text"]))

def tex_escape(x):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in x)

KIND = {"returned_message": "returned message", "status_description": "status description",
        "doc_guidance": "guidance"}
for i, short in SHORT_TEXT.items():
    e = TEXTS[i]
    put(f"text:{short}", tex_escape(e["text"]))
    put(f"text:{short}:provider", tex_escape(e["provider"]))
    put(f"text:{short}:status", tex_escape(e["status"]))
    put(f"text:{short}:kind", KIND[e["source_kind"]])
    put(f"text:{short}:retrieval", e["retrieval"])
    put(f"text:{short}:class", e["class"])

# the registered selection rule, re-implemented from the catalogue's selection_rule text
def select(classes):
    pord, kord = CAT["provider_order"], CAT["source_kind_order"]
    quota = {"U": 3, "G": 3, "N": 1, "R": 1}
    eligible = [e for e in texts if not e["borderline"] and not e["vendor_name_in_text"]
                and e["condition"] in ("timeout", "5xx", "ambiguous")]
    per_prov, chosen = collections.Counter(), []
    for c in ("U", "N", "R", "G"):
        cand = sorted((e for e in eligible if classes[e["id"]] == c),
                      key=lambda e: (kord.index(e["source_kind"]), pord.index(e["provider"]), e["id"]))
        got_c = 0
        for e in cand:
            if got_c == quota[c]:
                break
            if per_prov[e["provider"]] >= 2:
                continue
            chosen.append(e["id"]); per_prov[e["provider"]] += 1; got_c += 1
    return chosen

reg_cls = {e["id"]: e["class"] for e in texts}
sel = select(reg_cls)
selection_ok = sorted(sel) == sorted(R5["injected"])
put("sel:reproduced", "yes" if selection_ok else "no")
say(f"selection rule (re-implemented): {sorted(sel)} == registered injected: {selection_ok}")

# ------------------------------------------------------------------ second coder
sc = json.load(open(os.path.join(V32, "classification_check", "second_coder.json")))
key = json.load(open(os.path.join(V32, "classification_check", "key_blind_ids.json")))
pairs = [(reg_cls[key[x["id"]]], x["class"], key[x["id"]]) for x in sc]
agree = sum(1 for a, b, _ in pairs if a == b)
n = len(pairs)
po = agree / n
ca, cb = collections.Counter(a for a, _, _ in pairs), collections.Counter(b for _, b, _ in pairs)
pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / n ** 2
kappa = (po - pe) / (1 - pe)
dis = [(i, a, b) for a, b, i in pairs if a != b]
put("cc:agree", agree); put("cc:n", n); put("cc:kn", f"{agree}/{n}"); put("cc:kappa", f2(kappa))
put("cc:disagree", len(dis))
put("cc:disagree:injected", sum(1 for i, _, _ in dis if i in R5["injected"]))
put("cc:disagree:borderline", "yes" if all(TEXTS[i]["borderline"] for i, _, _ in dis) else "no")
put("cc:disagree:list", "; ".join(f"\\ttt{{{i.replace('_', chr(92) + '_')}}} {a}$\\to${b}" for i, a, b in dis))
# sensitivity: R5-P1 with the second coder's classes (the injected texts' classes as coded blind)
sc_cls = {key[x["id"]]: x["class"] for x in sc}
same = all(sc_cls[i] == reg_cls[i] for i in R5["injected"])
put("cc:injected_same", "yes" if same else "no")
sens = {}
for model, ms in MODELS.items():
    U2 = [FM[i] for i in R5["injected"] if sc_cls[i] == "U"]
    G2 = [FM[i] for i in R5["injected"] if sc_cls[i] == "G"]
    t = contrast(model, U2, G2, "less")
    sens[ms] = t
    put(f"{ms}.P1cc:ptmpl", pfmt(t["template"]["p"]))
    put(f"{ms}.P1cc:same", "yes" if pfmt(t["template"]["p"]) == NUMS[f"v32:{ms}.P1:ptmpl"]
        and sd(t["template"]["d"]) == NUMS[f"v32:{ms}.P1:d"] else "no")
# the selection rule run on the second coder's classes (exploratory; not registered)
sel2 = select(sc_cls)
put("cc:selection_same", "yes" if sorted(sel2) == sorted(R5["injected"]) else "no")
say(f"second coder: {agree}/{n} agree, kappa {kappa:.4f}; disagreements {dis}; injected classes same: {same}; "
    f"selection with second-coder classes: {sorted(sel2)}")

# ------------------------------------------------------------------ spend (ledger)
res_, settle, charge = {}, {}, []
bad_lines = 0
with open(os.path.join(V32, "results", "spend_ledger.jsonl")) as fh:
    for line in fh:
        try:
            x = json.loads(line)
        except ValueError:
            bad_lines += 1
            continue
        if x["type"] == "reserve":
            res_[x["id"]] = x
        elif x["type"] == "settle":
            settle[x["id"]] = x
        elif x["type"] == "charge":
            charge.append(x)
open_res = [v for k, v in res_.items() if k not in settle]
actual = sum(v["usd"] for v in settle.values()) + sum(c["usd"] for c in charge)
held = sum(v["usd"] for v in open_res)
by_prov = collections.Counter()
for v in settle.values():
    by_prov[v["provider"]] += v["usd"]
for c in charge:
    by_prov[c["provider"]] += c["usd"]
ep_cost = collections.Counter()
for r in rows:
    ep_cost["openrouter" if r["model"].startswith("openrouter:") else "anthropic"] += r.get("cost_usd") or 0.0
say(f"ledger: {len(settle)} settled calls, {len(charge)} key-check charges, {len(open_res)} open reservations, "
    f"{bad_lines} unreadable lines; actual ${actual:.4f} (+${held:.4f} held); by provider "
    f"{ {k: round(v, 4) for k, v in by_prov.items()} }; episode rows sum { {k: round(v, 4) for k, v in ep_cost.items()} }")
log = open(os.path.join(RD, "run.log")).read()
mlog = re.findall(r"cumulative v32 spend \(ledger v32-experiments/results/spend_ledger\.jsonl\): \$([0-9.]+) actual", log)
check("ledger total vs last run.log line", f"{actual + held:.4f}", mlog[-1])
put("spend:r5", f"\\${actual + held:.2f}")
put("spend:r5:anthropic", f"\\${by_prov['anthropic']:.2f}")
put("spend:r5:openrouter", f"\\${by_prov['openrouter']:.2f}")
put("spend:r5:open_res", len(open_res))
put("spend:r5:cap", f"\\${R5['total_cap_usd']}")
# the paper's logged total before R5 (numbers.tex, printed in cents) plus R5's ledger total
NT = open(os.path.join(ROOT, "paper-merged", "numbers.tex")).read()
before = re.search(r"\\defnum\{cost:logged_total_r234\}\{\\\$([0-9.]+)\}", NT).group(1)
tot = Decimal(before) + Decimal(f"{actual + held:.2f}")
put("spend:total_r2345", f"\\${tot}")
put("spend:parts_add", "yes" if tot == Decimal(before) + Decimal(NUMS["v32:spend:r5"].strip("\\$")) else "no")
TOT_EP = re.search(r"\\defnum\{total:all_r234\}\{([0-9,]+)\}", NT).group(1)
put("total:all_r2345", f"{int(TOT_EP.replace(',', '')) + len(rows):,}")
FU = int(re.search(r"\\defnum\{prereg:followups:n\}\{(\d+)\}", NT).group(1))
put("prereg:followups:n", FU + 1); put("prereg:followups:n:w", WORDS[FU + 1])

# ------------------------------------------------------------------ OpenRouter hosts (Qwen)
hosts = collections.Counter(r["model_served"].split("@")[-1].strip() for r in rows if r["model"].startswith("openrouter:"))
put("qwen:hosts", len(hosts))
put("qwen:hosts:list", ", ".join(f"{h} {c}" for h, c in hosts.most_common()))
put("claude:temp_null", "yes" if all(r["temperature"] is None for r in rows if r["model"].startswith("claude")) else "no")

# ------------------------------------------------------------------ transcripts: text seen, ledger matcher
def norm(v):
    # the paper's identity normalization (Appendix "Identity Fields and Matching"), written from its
    # description: whitespace runs, surrounding quotes, case, currency symbols, thousands separators,
    # trailing zeros of numbers, and zero seconds in a date-time
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return format(Decimal(repr(v)).normalize(), "f")
    if not isinstance(v, str):
        return json.dumps(v, sort_keys=True)
    s = re.sub(r"\s+", " ", v.strip()).strip("'\"").strip()
    s = re.sub(r"(T\d\d:\d\d):00(?=$|[^\d:])", r"\1", s)
    m = re.fullmatch(r"[$\u20ac\u00a3 ]*(-?[\d,]*\.?\d+)", s)
    if m:
        return format(Decimal(m.group(1).replace(",", "")).normalize(), "f")
    return s.casefold()

tdir = os.path.join(RD, "transcripts")
by_id = {r["episode_id"]: r for r in rows}
seen_ids, text_ok, text_bad, match_ok, match_bad = set(), 0, [], 0, []
EXPECT = {FM[i]: "ERROR: " + TEXTS[i]["text"] for i in R5["injected"]}
EXPECT[E500] = "ERROR: internal server error"
EXPECT[W3] = "ERROR: The request may have been processed; its outcome is unknown."
for f in sorted(os.listdir(tdir)):
    t = json.load(open(os.path.join(tdir, f)))
    eid = t["episode_id"]
    seen_ids.add(eid)
    errs = [c["content"] for turn in t["turns"] if turn["role"] == "tool_results"
            for c in turn["content"] if c.get("is_error")]
    if errs and errs[0] == EXPECT[t["failure_mode"]]:
        text_ok += 1
    else:
        text_bad.append((eid, errs[:1]))
    groups = collections.Counter((e["tool"], tuple(sorted((k, norm(v)) for k, v in e["args"].items())))
                                 for e in t["ledger"])
    viol = sum(c - 1 for c in groups.values())
    if (viol > 0) == (by_id[eid]["violations"] > 0) == (t["score"]["violations"] > 0):
        match_ok += 1
    else:
        match_bad.append((eid, viol, by_id[eid]["violations"], t["score"]["violations"]))
say(f"transcripts: {len(seen_ids)} files, all ids in raw: {seen_ids == set(by_id)}; error text shown verbatim "
    f"(first error result == 'ERROR: ' + catalogue text): {text_ok}/{len(seen_ids)}; "
    f"ledger re-matched (any duplicate) == raw verdict: {match_ok}/{len(seen_ids)}")
for x in text_bad[:5]:
    say("   text mismatch:", x)
for x in match_bad[:10]:
    say("   matcher mismatch:", x)
put("tr:n", f"{len(seen_ids):,}")
put("tr:text_ok", "yes" if text_ok == len(seen_ids) == len(rows) else "no")
put("tr:match_ok", "yes" if match_ok == len(seen_ids) else "no")
put("tr:match_mismatch", len(match_bad))

# ------------------------------------------------------------------ PII sweep of the R5 result files
PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
PII_PHONE = re.compile(r"(?<![\w.-])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?![\w-])")
PII_URL = re.compile(r"\b(?:https?://|www\.)[^\s\"'<>\\)]+")
pf = ptr = 0
ph = collections.Counter()
for d, _, fs in os.walk(os.path.join(V32, "results")):
    for f in fs:
        if not f.endswith((".json", ".jsonl", ".csv", ".log", ".txt", ".md")):
            continue
        pf += 1
        ptr += "transcript" in os.path.basename(d)
        s = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
        ph["email"] += sum(1 for m in PII_EMAIL.finditer(s) if m.group(1).lower() != "example.com"
                           and not m.group(1).lower().endswith(".example.com"))
        ph["phone"] += len(PII_PHONE.findall(s))
        ph["url"] += len(PII_URL.findall(s))
put("pii:files", f"{pf:,}"); put("pii:transcripts", f"{ptr:,}"); put("pii:total", ph["email"] + ph["phone"] + ph["url"])
PF = int(re.search(r"\\defnum\{pii:files\}\{([0-9,]+)\}", NT).group(1).replace(",", ""))
PT = int(re.search(r"\\defnum\{pii:transcripts\}\{([0-9,]+)\}", NT).group(1).replace(",", ""))
put("pii:files_all", f"{PF + pf:,}"); put("pii:transcripts_all", f"{PT + ptr:,}")
say(f"PII sweep of v32-experiments/results: {pf} files ({ptr} transcripts); hits {dict(ph)}")

# ------------------------------------------------------------------ registration timing
reg_commit = os.environ.get("V32_REG_COMMIT_UTC", "2026-11-02T07:58:06Z")   # self-reported (commit 9b1bc7c)
reg_push = os.environ.get("V32_REG_PUSH_UTC", "2026-11-02T07:58:08Z")       # push reflog of the same commit
put("reg:commit_utc", reg_commit.replace("T", " ").replace("Z", " UTC"))
put("reg:push_utc", reg_push.replace("T", " ").replace("Z", " UTC"))
put("reg:before_data", "yes" if reg_push < min(r["started_utc"] for r in rows) else "no")

# ------------------------------------------------------------------ mismatches, outputs
put("check:mismatches", len(MISMATCH))
put("check:rounding_only", len(ROUNDING))
say(f"comparison with analysis/r5_real_errors.json: {len(MISMATCH)} mismatches; "
    f"{len(ROUNDING)} rounding-only differences (< 1e-4; no printed value or verdict affected)")
for m in MISMATCH:
    say("   MISMATCH", m)
for m in ROUNDING:
    say("   rounding-only", m)


def wcsv(name, rs):
    with open(os.path.join(OUT, name), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rs[0]))
        w.writeheader()
        w.writerows(rs)

wcsv("r5_cells_recomputed.csv", audit_cells)
wcsv("r5_tests_recomputed.csv", audit_tests)
with open(os.path.join(OUT, "run.log"), "w") as fh:
    fh.write("\n".join(REPORT) + "\n")
with open(TEX, "w") as fh:
    fh.write("% GENERATED by analysis_v27/v32_numbers.py -- do not edit by hand.\n")
    fh.write("% R5 (real API error texts): recomputed from v32-experiments/results/ and checked against the\n")
    fh.write("% registered analysis output. Keys are prefixed v32: and never redefine other keys.\n")
    for k, v in NUMS.items():
        fh.write("\\defnum{%s}{%s}\n" % (k, v))
print(f"wrote {len(NUMS)} keys -> {TEX}")
sys.exit(1 if MISMATCH else 0)
