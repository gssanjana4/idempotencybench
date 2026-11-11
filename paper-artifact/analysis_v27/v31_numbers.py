"""v31 additions to the v27/v28/v29 re-analysis. Reads results/ only and writes
  paper-v31/numbers_v31.tex   (\\defnum{v31:...}{...} and the row macro \\vthreeoneRestartKeyRows;
                               never redefines a numbers.tex, v27:, v28: or v29: key;
                               V31_NUMBERS_TEX overrides the path)
  analysis_v27/out_v31/*.csv   (computed rows, for audit; V31_OUT overrides the folder)
No other file is written.

Sections
  R  restart key rescore (post hoc), mirroring the retry rescore of v27_numbers.py (Table "keys"):
     every restart episode without a mitigation is replayed call by call through the harness's
     own World, once as run (keys ignored: mitigation "none") and once with keys honoured
     (mitigation "idem_keys": a call whose non-empty key repeats a committed call's key for the
     same tool commits nothing). The as-run replay must reproduce the released score of every
     episode (violations, gIVR hit, completion, order) or the script stops. Session 1 is cut the
     moment its interrupt node commits, as in the harness (src2/rescore2.py uses the same rule).
     Stripe (R1) restarts are rescored from the ledger stored in each transcript (Stripe's own
     records agree with it in every episode, R1-H2). The note contrasts are recomputed with keys
     honoured with the registered tests: one-sided paired sign-flip over graphs, Holm over a
     model's wordings, supported if >= 2 of 3 significant (R2-H1, R3-H1), and the two-sided
     post hoc test for the October grid; the same code on the as-run outcomes must reproduce the
     released p-values (checked against numbers.tex by \\ASSERT in the paper).
  P  power of the Sonnet 5 October note contrast for a drop to zero, with a graph's two samples
     independent and identical (the paper's MDE convention gives both ends)
  S  the single-session counts the v31 wording needs: models whose every "outcome unknown"
     wording is below the same-run timeout (R2/R3), and the largest GPT-5.5/Gemini bare-500 IVR
     after a commit (October grid and R3)
Run:  python3 analysis_v27/v31_numbers.py   (numpy for section P, as for v27_numbers.py)
"""
import collections
import csv
import json
import os
import random
import sys

sys.dont_write_bytecode = True      # imports src/ and src2/ read-only: no __pycache__ written there
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src2"))
from world import World, ToolCallError, EFFECTFUL_TOOLS, normalize_value  # noqa: E402  (read-only use)
from graph import _chain4, _fanout  # noqa: E402
from resume import _node_commits, score_graph  # noqa: E402

TEX = os.environ.get("V31_NUMBERS_TEX", os.path.join(ROOT, "paper-v31", "numbers_v31.tex"))
OUT = os.environ.get("V31_OUT", os.path.join(ROOT, "analysis_v27", "out_v31"))
os.makedirs(OUT, exist_ok=True)

# registered test constants (analysis/v3_common.py)
PERM_N, PERM_SEED, ALPHA = 10000, 20261025, 0.05
SHORT = [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("fable", "Fable"), ("gpt-5.5", "GPT"),
         ("gemini", "Gemini"), ("qwen", "Qwen")]
DISP = {"Haiku": "Haiku~4.5", "Sonnet": "Sonnet~5", "Fable": "Fable~5", "GPT": "GPT-5.5",
        "Gemini": "Gemini", "Qwen": "\\qwen{}"}
PLAIN = {"Haiku": "Haiku 4.5", "Sonnet": "Sonnet 5", "Fable": "Fable 5", "GPT": "GPT-5.5",
         "Gemini": "Gemini", "Qwen": "Qwen3.6-27B"}
ORDER = ["Haiku", "Sonnet", "Fable", "GPT", "Gemini", "Qwen"]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]

D = collections.OrderedDict()


def put(k, v):
    D["v31:" + k] = str(v)


def short(m):
    m = m.split("/")[-1]
    for k, v in SHORT:
        if k in m:
            return v
    return m


def load(p):
    with open(p) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def dedupe_last(rows):
    d = collections.OrderedDict()
    for r in rows:
        d[r["episode_id"]] = r
    return list(d.values())


def prefer_v2(p):
    q = p.replace("raw.jsonl", "raw_v2.jsonl")
    return q if os.path.exists(q) else p


def names(ms):
    ms = [PLAIN[m] for m in ORDER if m in ms]
    return ms[0] if len(ms) == 1 else ", ".join(ms[:-1]) + " and " + ms[-1]


def pfmt(p):
    return "<0.001" if p < 0.001 else (f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def pe(p):
    s = pfmt(p)
    return f"$p{s}$" if s.startswith("<") else f"$p={s}$"


def write_csv(name, rows):
    if rows:
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


# ---------------------------------------------------------------- registered tests (copied from
# analysis/r2_wording.py perm_test_one_sided and analysis/v3_common.py perm_test / holm, unchanged)
def perm_test_one_sided(x_by, y_by):
    units = sorted(set(x_by) & set(y_by))
    diffs = [y_by[u] - x_by[u] for u in units]
    obs = sum(diffs)
    rng = random.Random(PERM_SEED)
    hit = 0
    for _ in range(PERM_N):
        s = sum(dv if rng.random() < 0.5 else -dv for dv in diffs)
        if s <= obs + 1e-12:
            hit += 1
    return {"d": obs / len(diffs), "p": (hit + 1) / (PERM_N + 1), "T": len(units)}


def perm_test(x_by, y_by):
    units = sorted(set(x_by) & set(y_by))
    diffs = [y_by[u] - x_by[u] for u in units]
    obs = abs(sum(diffs))
    rng = random.Random(PERM_SEED)
    hit = 0
    for _ in range(PERM_N):
        s = sum(dv if rng.random() < 0.5 else -dv for dv in diffs)
        if abs(s) >= obs - 1e-12:
            hit += 1
    return {"d": sum(diffs) / len(diffs), "p": (hit + 1) / (PERM_N + 1), "T": len(units)}


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def unit_rates(eps, field):
    d = collections.defaultdict(list)
    for e in eps:
        d[e["unit"]].append(e[field])
    return {u: sum(v) / len(v) for u, v in d.items()}


# ================================================================ R: restart key rescore
GRAPHS = {}
for i in range(8):
    for g in (_chain4(i), _fanout(i)):
        GRAPHS[g.graph_id] = g
KEYS = ("violations", "givr_hit", "completion", "order_violation")


def replay(t, mitigation):
    """Re-issue every tool call of a released restart transcript through a fresh World."""
    g = GRAPHS[t["graph_id"]]
    k = t["interrupt_node"]
    w = World(mitigation=mitigation, failure_mode="none")

    def call(b):
        a = dict(b.get("input") or {})
        key = a.pop("idempotency_key", "") or ""
        try:
            w.call(b["name"], a, str(key), g.graph_id)
        except ToolCallError:
            pass                                    # invalid call: nothing commits

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
    assert cut == bool(s1.get("interrupted")), (t.get("episode_id"), "interrupt position")
    for s in t["sessions"][1:]:
        for turn in s["turns"]:
            if turn["role"] != "assistant":
                continue
            for b in turn.get("content") or []:
                if b.get("type") == "tool_use":
                    call(b)
    row = score_graph(w, g, interrupted=cut)
    return {x: row[x] for x in KEYS}


def stripe_hits(t):
    """(as run, keys honoured) duplicate verdicts from the transcript's ledger (R1)."""
    def dup(effs):
        seen = collections.Counter()
        for e in effs:
            _, canon = EFFECTFUL_TOOLS[e["tool"]]
            seen[(e["tool"],) + tuple(normalize_value(e["args"].get(f)) for f in canon)] += 1
        return sum(v - 1 for v in seen.values())
    led = sorted(t["ledger"], key=lambda e: e["seq"])
    kept, keys = [], set()
    for e in led:
        kk = (e["tool"], str(e.get("key") or ""))
        if kk[1] and kk in keys:
            continue                                # a key-honouring server replays it
        if kk[1]:
            keys.add(kk)
        kept.append(e)
    return dup(led), dup(kept)


def tfile(tdir, eid):
    return os.path.join(RES, tdir, eid.replace("/", "_") + ".json")


RUNS = [  # (run code, raw file, transcript folder, contexts)
    ("oct", "claude_batch_2026-10/sweep2_raw.jsonl", "claude_batch_2026-10/transcripts2", ("amnesiac", "bare_hint")),
    ("oct", "crossprovider/sweep2_raw.jsonl", "crossprovider/transcripts2", ("amnesiac", "bare_hint")),
    ("r2", "r2_wording/raw.jsonl", "r2_wording/transcripts", ("amnesiac", "bare_hint", "hint_p1", "hint_p2", "hint_p3")),
    ("r3", "r3_wording/raw.jsonl", "r3_wording/transcripts", ("amnesiac", "bare_hint", "hint_p1", "hint_p3")),
    ("real", "real_stripe/raw.jsonl", "real_stripe/transcripts", ("amnesiac", "bare_hint")),
]
EPS = collections.defaultdict(list)       # (run, model, ctx) -> [{unit, asrun, hon}]
n_replayed = n_keyed_redo = n_notresumed = 0
for run, raw, tdir, ctxs in RUNS:
    for r in dedupe_last(load(os.path.join(RES, raw))):
        if r.get("refusal") or r.get("mitigation") != "none" or r.get("resume_context") not in ctxs:
            continue
        if run == "real" and r.get("kind") != "restart":
            continue
        t = json.load(open(tfile(tdir, r["episode_id"])))
        if len(t["sessions"]) < 2:              # never resumed (session 1 ended without the cut):
            assert not r.get("givr_hit") and not r.get("violations_ledger")
            n_notresumed += 1                    # out of the cell, as in the paper's restart counts
            continue
        if run == "real":
            v, vh = stripe_hits(t)
            assert v == r["violations_ledger"], (r["episode_id"], v, r["violations_ledger"])
            asrun, hon = int(v > 0), int(vh > 0)
        else:
            a = replay(t, "none")
            assert a == {x: t["score"][x] for x in KEYS} == {x: r[x] for x in KEYS}, (r["episode_id"], a)
            h = replay(t, "idem_keys")
            asrun, hon = a["givr_hit"], h["givr_hit"]
        assert hon <= asrun
        n_replayed += 1
        n_keyed_redo += asrun - hon
        EPS[(run, short(r["model"]), r["resume_context"])].append(
            {"unit": r.get("graph_id") or t["graph_id"], "asrun": asrun, "hon": hon, "eid": r["episode_id"]})


def kn(cell, f):
    return f"{sum(e[f] for e in cell)}/{len(cell)}"


crows = []
for (run, m, ctx), cell in sorted(EPS.items()):
    crows.append(dict(run=run, model=m, note=ctx, n=len(cell), k_as_run=sum(e["asrun"] for e in cell),
                      k_keys_honoured=sum(e["hon"] for e in cell)))
    put(f"r:{run}.{m.lower()}.{ctx}:kn", kn(cell, "asrun"))
    put(f"r:{run}.{m.lower()}.{ctx}:hkn", kn(cell, "hon"))
write_csv("restart_keys_cells.csv", crows)
put("r:replayed", f"{n_replayed:,}")
put("r:notresumed", n_notresumed)
oct_notes = [e for m in ORDER for e in EPS[("oct", m, "bare_hint")]]
put("r:oct.notes:kn", kn(oct_notes, "asrun"))
put("r:oct.notes:hkn", kn(oct_notes, "hon"))

# October grid totals (the paper's 155/192 and 2/191 for the notes; Sonnet 5's refused session out)
oct_amn = [e for m in ORDER for e in EPS[("oct", m, "amnesiac")]]
put("r:oct.amn:kn", kn(oct_amn, "asrun"))
put("r:oct.amn:hkn", kn(oct_amn, "hon"))
dropped = [m for m in ORDER if sum(e["asrun"] for e in EPS[("oct", m, "amnesiac")])
           > sum(e["hon"] for e in EPS[("oct", m, "amnesiac")])]
put("r:oct.amn:changed", names(dropped))
# models whose as-run amnesiac cell duplicated in more than half of its sessions, and with keys honoured
maj = lambda f: [m for m in ORDER if 2 * sum(e[f] for e in EPS[("oct", m, "amnesiac")]) > len(EPS[("oct", m, "amnesiac")])]
put("r:oct.amn:majority:n", len(maj("asrun")))
put("r:oct.amn:hmajority:n", len(maj("hon")))
put("r:oct.amn:hmajority:w", WORDS[len(maj("hon"))])
put("r:oct.amn:hmajority:names", names(maj("hon")))
# every note cell: does any duplicate disappear with keys honoured?
note_changed = sum(sum(e["asrun"] - e["hon"] for e in c) for (run, m, ctx), c in EPS.items() if ctx != "amnesiac")
put("r:notes:changed", note_changed)

# note vs amnesiac with keys honoured: October grid (post hoc, two-sided)
trows = []
sig_asrun, sig_hon, p_hon_max, p_asrun_max = [], [], 0.0, 0.0
for m in ORDER:
    base, note = EPS[("oct", m, "amnesiac")], EPS[("oct", m, "bare_hint")]
    ra = perm_test(unit_rates(base, "asrun"), unit_rates(note, "asrun"))
    rh = perm_test(unit_rates(base, "hon"), unit_rates(note, "hon"))
    put(f"r:oct.{m.lower()}.note:pe", pe(ra["p"]))
    put(f"r:oct.{m.lower()}.note:hp", pfmt(rh["p"]))
    put(f"r:oct.{m.lower()}.note:hpe", pe(rh["p"]))
    put(f"r:oct.{m.lower()}.note:hd", ("$-$" if rh["d"] < 0 else "+") + f"{abs(rh['d']):.2f}")
    trows.append(dict(run="oct", model=m, note="bare_hint", d_as_run=ra["d"], p_as_run=ra["p"],
                      d_keys_honoured=rh["d"], p_keys_honoured=rh["p"], test="two-sided, graphs, post hoc"))
    if ra["p"] < ALPHA:
        sig_asrun.append(m); p_asrun_max = max(p_asrun_max, ra["p"])
        if rh["p"] < ALPHA:
            sig_hon.append(m)
        p_hon_max = max(p_hon_max, rh["p"])
put("r:oct.note:sig:names", names(sig_asrun))
put("r:oct.note:hsig:names", names(sig_hon))
put("r:oct.note:hsig:n:w", WORDS[len(sig_hon)])
put("r:oct.note:hsig:pmaxe", pe(p_hon_max))
put("r:oct.note:sig:pmaxe", pe(p_asrun_max))

# R2-H1 / R3-H1 with keys honoured (registered test, applied post hoc to the rescored outcomes)
reg = []
for run, models, notes in (("r2", ("Haiku", "Fable"), ("hint_p1", "hint_p2", "hint_p3")),
                           ("r3", ("GPT", "Gemini", "Qwen"), ("bare_hint", "hint_p1", "hint_p3"))):
    for m in models:
        out = {}
        for f in ("asrun", "hon"):
            base = unit_rates(EPS[(run, m, "amnesiac")], f)
            ps = [perm_test_one_sided(base, unit_rates(EPS[(run, m, w)], f))["p"] for w in notes]
            hp = holm(ps)
            nsig = sum(1 for x in hp if x < ALPHA)
            out[f] = (max(hp), nsig, "supported" if nsig >= 2 else "not supported", ps)
        for w, pa, ph in zip(notes, out["asrun"][3], out["hon"][3]):
            trows.append(dict(run=run, model=m, note=w, d_as_run="", p_as_run=pa, d_keys_honoured="",
                              p_keys_honoured=ph, test="one-sided, graphs, Holm over the model's wordings"))
        key = f"r:{run}.{m.lower()}"
        put(f"{key}:asrun:holm_pmax_e", pe(out["asrun"][0]))
        put(f"{key}:asrun:verdict", out["asrun"][2])
        put(f"{key}:holm_pmax_e", pe(out["hon"][0]))
        put(f"{key}:nsig", out["hon"][1])
        put(f"{key}:verdict", out["hon"][2])
        reg.append((m, out["hon"][2], out["hon"][0]))
put("r:reg:hsupported:n:w", WORDS[sum(1 for x in reg if x[1] == "supported")])
put("r:reg:hsupported:names", names([x[0] for x in reg if x[1] == "supported"]))
put("r:reg:holm_pmax_e", pe(max(x[2] for x in reg)))
write_csv("restart_keys_tests.csv", trows)

# table rows: run & model & amnesiac as run -> honoured & notes as run -> honoured & note p (honoured)
RUNNAME = {"oct": "Oct.", "r2": "R2", "r3": "R3", "real": "Stripe"}
rows_tex = []
for run, models in (("oct", ORDER), ("r2", ("Haiku", "Fable")), ("r3", ("GPT", "Gemini", "Qwen")),
                    ("real", ("Haiku", "Sonnet", "Qwen"))):
    for m in models:
        amn = EPS[(run, m, "amnesiac")]
        notes = [c for c in ("bare_hint", "hint_p1", "hint_p2", "hint_p3") if (run, m, c) in EPS]
        nk = sum(sum(e["asrun"] for e in EPS[(run, m, c)]) for c in notes)
        nh = sum(sum(e["hon"] for e in EPS[(run, m, c)]) for c in notes)
        nn = sum(len(EPS[(run, m, c)]) for c in notes)
        if run == "oct":
            p = D[f"v31:r:oct.{m.lower()}.note:hpe"]
        elif run in ("r2", "r3"):
            p = D[f"v31:r:{run}.{m.lower()}:holm_pmax_e"] + "$^{\\mathrm{H}}$"
        else:
            p = "--"
        rows_tex.append(f"{RUNNAME[run]} & {DISP[m]} & {kn(amn, 'asrun')}$\\to${kn(amn, 'hon')} & "
                        f"{len(notes)} & {nk}/{nn}$\\to${nh}/{nn} & {p} \\\\")
EXTRA = "\\def\\vthreeoneRestartKeyRows{%%\n%s}\n" % "\n".join(rows_tex)

# ================================================================ P: Sonnet 5 power, both ends
def power_full_drop(p0, T_, S=2, rho=0, nsim=2000, nflip=1000, seed=7):
    """v27_numbers.py's power_full_drop (numpy there; the same draws need numpy, so it is used)."""
    import numpy as _np
    rng = _np.random.default_rng(seed); flips = rng.choice([-1.0, 1.0], size=(nflip, T_))
    x0 = (rng.binomial(S, p0, size=(nsim, T_)) / S) if rho == 0 else (rng.random((nsim, T_)) < p0).astype(float)
    d = -x0; obs = _np.abs(d.sum(axis=1)); null = _np.abs(d @ flips.T)
    pv = ((null >= obs[:, None] - 1e-12).sum(axis=1) + 1) / (nflip + 1)
    return float((pv < 0.05).mean())


son = EPS[("oct", "Sonnet", "amnesiac")]
p0 = sum(e["asrun"] for e in son) / len(son)
T_ = len({e["unit"] for e in son})
pw_ind, pw_id = power_full_drop(p0, T_, rho=0), power_full_drop(p0, T_, rho=1)
put("m:sonnet.hint:power:indep", f"{round(100 * pw_ind)}\\%")
put("m:sonnet.hint:power:ident", f"{round(100 * pw_id)}\\%")
write_csv("sonnet_power.csv", [dict(p0=p0, graphs=T_, power_independent=pw_ind, power_identical=pw_id)])

# ================================================================ S: single-session counts
IVR = lambda r: 1 if r["violations"] > 0 else 0


def cells(p):
    by = collections.defaultdict(list)
    for r in dedupe_last([r for r in load(prefer_v2(p)) if not r.get("refusal")]):
        if r.get("failure_mode") in (None, "none") or "givr_hit" in r:
            continue
        if r.get("prompt", "base") != "base" or r.get("mitigation", "none") != "none":
            continue
        by[(short(r["model"]), r["failure_mode"])].append(r)
    return by


R2 = cells(os.path.join(RES, "r2_wording", "raw.jsonl"))
R3 = cells(os.path.join(RES, "r3_wording", "raw.jsonl"))
XP = cells(os.path.join(RES, "crossprovider", "sweep_raw.jsonl"))
TO, B5 = "timeout_after_commit", "error_500_after_commit"
UNK = ["status_unknown_after_commit", "status_unknown_p1_after_commit", "status_unknown_p2_after_commit",
       "status_unknown_p3_after_commit"]
rate = lambda c: sum(IVR(r) for r in c) / len(c)
below_to, below_500 = [], []
for m, cc in (("Haiku", R2), ("Sonnet", R2), ("Fable", R3), ("GPT", R3), ("Gemini", R3), ("Qwen", R3)):
    ws = [cc[(m, w)] for w in UNK if (m, w) in cc]
    assert len(ws) >= 3, (m, len(ws))
    if all(rate(c) < rate(cc[(m, TO)]) for c in ws):
        below_to.append(m)
    if all(rate(c) < rate(cc[(m, B5)]) for c in ws):
        below_500.append(m)
put("s:unk_below_to:n:w", WORDS[len(below_to)])
put("s:unk_below_to:names", names(below_to))
gg = [rate(cc[(m, B5)]) for m in ("GPT", "Gemini") for cc in (XP, R3)]
put("s:gg500:max", f"{max(gg):.2f}")

with open(TEX, "w") as fh:
    fh.write("% GENERATED by analysis_v27/v31_numbers.py -- do not edit by hand.\n")
    fh.write("% Reads results/ only. Keys are prefixed v31: and never redefine other keys.\n")
    for k, v in D.items():
        fh.write("\\defnum{%s}{%s}\n" % (k, v))
    fh.write(EXTRA)
print("replayed", n_replayed, "restart episodes; as-run scores reproduce the released ones")
print("wrote", len(D), "macros to", TEX)
for k, v in D.items():
    print(k, "=", v)
print(EXTRA)
