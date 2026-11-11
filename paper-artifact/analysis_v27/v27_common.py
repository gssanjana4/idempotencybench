"""Shared helpers for the v27 re-analysis (template-level tests, simulated MDE,
key-honouring rescore, bookkeeping counts). Reads results/ only; never writes
outside analysis_v27/out/ and paper-v27/numbers_v27.tex.

Test helpers follow the panel's independent scripts (v26_panel/B_scripts/b_common.py):
paired sign-flip permutation over tasks (Monte Carlo, 10,000 flips, seed 20261025)
and an exact sign-flip test over template (or shape) means.
"""
import collections
import itertools
import json
import math
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "analysis_v27", "out")
SEED = 20261025
NPERM = 10000

SHORT = [("haiku", "Haiku"), ("sonnet", "Sonnet"), ("fable", "Fable"), ("gpt", "GPT"),
         ("gemini", "Gemini"), ("qwen", "Qwen")]
DISP = {"Haiku": "Haiku~4.5", "Sonnet": "Sonnet~5", "Fable": "Fable~5", "GPT": "GPT-5.5",
        "Gemini": "Gemini", "Qwen": "Qwen3.6-27B"}


def path(*p):
    return os.path.join(RES, *p)


def load(p):
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def short(m):
    m = m.split("/")[-1]
    for k, v in SHORT:
        if k in m:
            return v
    return m


def template_of(r):
    return r.get("template") or r["task_id"].rsplit("-", 1)[0]


def dedupe_last(rows):
    d = collections.OrderedDict()
    for r in rows:
        d[r["episode_id"]] = r
    return list(d.values())


def dedupe_first(rows):
    d = collections.OrderedDict()
    for r in rows:
        d.setdefault(r["episode_id"], r)
    return list(d.values())


def per_unit(rows, unit_key, outcome):
    acc = collections.defaultdict(list)
    for r in rows:
        acc[r[unit_key]].append(outcome(r))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def signflip_mc(diffs, n=NPERM, seed=SEED, alternative="two-sided"):
    rng = random.Random(seed)
    obs = sum(diffs) / len(diffs)
    cnt = 0
    for _ in range(n):
        s = sum(d if rng.random() < 0.5 else -d for d in diffs) / len(diffs)
        if alternative == "two-sided":
            cnt += abs(s) >= abs(obs) - 1e-12
        elif alternative == "less":
            cnt += s <= obs + 1e-12
        else:
            cnt += s >= obs - 1e-12
    return obs, (cnt + 1) / (n + 1)


def signflip_exact(diffs, alternative="two-sided"):
    obs = sum(diffs) / len(diffs)
    cnt = tot = 0
    for signs in itertools.product([1, -1], repeat=len(diffs)):
        s = sum(a * b for a, b in zip(signs, diffs)) / len(diffs)
        tot += 1
        if alternative == "two-sided":
            cnt += abs(s) >= abs(obs) - 1e-12
        elif alternative == "less":
            cnt += s <= obs + 1e-12
        else:
            cnt += s >= obs - 1e-12
    return obs, cnt / tot


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    adj = [0.0] * len(ps)
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def fisher_two_sided(a, n1, b, n2):
    """Two-sided Fisher exact test for k1/n1 vs k2/n2 (sum of tables no more likely)."""
    K = a + b
    N = n1 + n2

    def hyp(x):
        return math.comb(n1, x) * math.comb(n2, K - x) / math.comb(N, K)
    p_obs = hyp(a)
    lo, hi = max(0, K - n2), min(K, n1)
    return min(1.0, sum(hyp(x) for x in range(lo, hi + 1) if hyp(x) <= p_obs * (1 + 1e-9)))


def paired_contrast(cell, ref, outcome, unit="task_id", tmpl_of=None):
    """Task-level MC sign-flip p and exact template-level sign-flip p (two-sided)."""
    a = per_unit(cell, unit, outcome)
    b = per_unit(ref, unit, outcome)
    shared = sorted(set(a) & set(b))
    d = [a[t] - b[t] for t in shared]
    obs, p_task = signflip_mc(d)
    tm = tmpl_of or {r[unit]: template_of(r) for r in list(cell) + list(ref)}
    tt = collections.defaultdict(list)
    for t in shared:
        tt[tm[t]].append(a[t] - b[t])
    dt = [sum(v) / len(v) for v in tt.values()]
    _, p_tmpl = signflip_exact(dt)
    return dict(d=obs, p_task=p_task, p_tmpl=p_tmpl, n_tmpl=len(dt),
                tmpl_pos=sum(1 for x in dt if x > 1e-12), tmpl_neg=sum(1 for x in dt if x < -1e-12),
                k_cell=int(round(sum(outcome(r) for r in cell))), n_cell=len(cell),
                k_ref=int(round(sum(outcome(r) for r in ref))), n_ref=len(ref))


# ---------------------------------------------------------------- simulated MDE
def mde_sim(p0, sign, T, S=2, rho=0, nsim=1000, nflip=1000, alpha=0.05, seed=7, step=0.02, power=0.8):
    """Smallest |delta| on a 0.02 grid at which a two-sided paired sign-flip test over T units
    (S samples each) rejects with probability >= power, simulated under the alternative
    p1 = p0 + sign*delta from the baseline rate p0. rho=0: samples independent given the
    unit; rho=1: a unit's samples identical. Returns None if no delta inside [0,1] reaches it."""
    import numpy as np
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(nflip, T))
    k = 1
    while True:
        delta = round(step * k, 4)
        p1 = p0 + sign * delta
        if p1 < -1e-9 or p1 > 1 + 1e-9:
            return None
        p1 = min(1.0, max(0.0, p1))
        if rho == 0:
            x0 = rng.binomial(S, p0, size=(nsim, T)) / S
            x1 = rng.binomial(S, p1, size=(nsim, T)) / S
        else:
            x0 = (rng.random((nsim, T)) < p0).astype(float)
            x1 = (rng.random((nsim, T)) < p1).astype(float)
        d = x1 - x0
        obs = np.abs(d.sum(axis=1))
        null = np.abs(d @ flips.T)
        pv = ((null >= obs[:, None] - 1e-12).sum(axis=1) + 1) / (nflip + 1)
        if (pv < alpha).mean() >= power:
            return delta
        k += 1


# ---------------------------------------------------------------- formatting
def f2(x):
    return f"{x:.2f}"


def pfmt(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def pe(p):
    """$p<0.001$ / $p=0.034$ style, as in numbers.tex"""
    s = pfmt(p)
    return f"$p{s}$" if s.startswith("<") else f"$p={s}$"


def dsg(x):
    if abs(x) < 1e-12:
        return "0.00"
    return ("+" if x > 0 else "$-$") + f2(abs(x))


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve"]


def word(n):
    return WORDS[n] if 0 <= n < len(WORDS) else str(n)


def tex_escape(s):
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
           "\u2014": "---", "\u2013": "--", "\u2019": "'", "\u2018": "`", "\u201c": "``", "\u201d": "''"}
    return "".join(rep.get(c, c) for c in s)


class Macros:
    def __init__(self):
        self.d = collections.OrderedDict()

    def put(self, k, v):
        self.d["v27:" + k] = str(v)

    def write(self, fn, extra=""):
        with open(fn, "w") as fh:
            fh.write("% GENERATED by analysis_v27/v27_numbers.py -- do not edit by hand.\n")
            fh.write("% Reads results/ only. Keys are prefixed v27: and never redefine numbers.tex keys,\n")
            fh.write("% except \\preregrows-style table macros named \\v27...rows.\n")
            for k, v in self.d.items():
                fh.write("\\defnum{%s}{%s}\n" % (k, v))
            fh.write(extra)
