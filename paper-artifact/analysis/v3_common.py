"""Shared helpers for the v3 analyses of the Tier-1 runs.

Everything here follows analysis/preregistration.md, "Common protocol":
  * episode-level Wilson 95% CI, and the PRIMARY task-level Wilson CI
    Wilson(p_hat * T, T), T = number of distinct tasks (rho = 1);
  * contrasts: two-sided paired permutation test over tasks (sign flip of
    the per-task difference of per-task mean rates), 10,000 permutations,
    seed 20261025 (a fresh RNG per test, as in paper-merged/make_numbers.py,
    so the same contrast gives the same p in both scripts);
  * Holm step-down within an experiment for secondary (S) hypotheses only.

Read-only on every input. Writes only under results/analysis_v3/.
"""
import csv, json, math, os, random
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "analysis_v3")
PERM_N = 10000
PERM_SEED = 20261025
SAMPLE_SEED = 20261026          # hand-check sample (E1-6 belief coding)

MODEL_SHORT = [("haiku", "haiku"), ("sonnet", "sonnet"), ("fable", "fable"),
               ("gpt-5.5", "gpt"), ("gemini", "gemini"), ("qwen", "qwen")]
FM_SHORT = {"none": "none", "timeout_after_commit": "toA",
            "timeout_before_commit": "toB", "error_500_after_commit": "e500",
            "error_500_body_after_commit": "w1", "status_unknown_after_commit": "w3"}
MIT_SHORT = {"none": "none", "idem_keys": "keys", "runtime_receipts": "rcpt"}


def short_model(m):
    for sub, k in MODEL_SHORT:
        if sub in m:
            return k
    return m


def rel(p):
    return os.path.relpath(p, ROOT)


def prefer_v2(path):
    """x.jsonl -> x_v2.jsonl when it exists: the C1 re-score (src/rescore_c1.py,
    ISO date-time seconds) writes changed sources there and never edits x.jsonl."""
    root, ext = os.path.splitext(path)
    p2 = root + "_v2" + ext
    return p2 if not root.endswith("_v2") and os.path.exists(p2) else path


def read_jsonl(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def dedupe(rows, keep="last"):
    """One row per episode_id. keep='last' is the rule fixed in e78e23d
    (before outcomes were compared); keep='first' is the sensitivity check."""
    d = {}
    for r in rows:
        if keep == "first":
            d.setdefault(r["episode_id"], r)
        else:
            d[r["episode_id"]] = r
    return d


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / den
    return (max(0.0, c - h), min(1.0, c + h))


def task_rates(eps, field="v"):
    d = defaultdict(list)
    for e in eps:
        if e.get(field) is not None:
            d[e["task"]].append(float(e[field]))
    return {t: sum(v) / len(v) for t, v in d.items()}


def perm_test(x_by_task, y_by_task):
    """Paired sign-flip permutation over the tasks present in both arms.
    Returns dict(d = mean(y - x), p, T, plus, minus)."""
    tasks = sorted(set(x_by_task) & set(y_by_task))
    diffs = [y_by_task[t] - x_by_task[t] for t in tasks]
    if not diffs:
        return {"d": None, "p": None, "T": 0, "plus": 0, "minus": 0}
    obs = abs(sum(diffs))
    rng = random.Random(PERM_SEED)
    hit = 0
    for _ in range(PERM_N):
        s = sum(dv if rng.random() < 0.5 else -dv for dv in diffs)
        if abs(s) >= obs - 1e-12:
            hit += 1
    return {"d": sum(diffs) / len(diffs), "p": (hit + 1) / (PERM_N + 1), "T": len(tasks),
            "plus": sum(1 for v in diffs if v > 0), "minus": sum(1 for v in diffs if v < 0)}


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def rate_stats(eps, field="v"):
    """Episode- and task-level Wilson for a 0/1 field over episodes."""
    vals = [e for e in eps if e.get(field) is not None]
    n = len(vals)
    k = sum(1 for e in vals if e[field])
    T = len({e["task"] for e in vals})
    p = k / n if n else None
    lo, hi = wilson(k, n)
    tlo, thi = wilson(p * T, T) if n else (0.0, 1.0)
    return {"k": k, "n": n, "T": T, "p": p, "ci_ep": [lo, hi], "ci_task": [tlo, thi]}


def r6(x):
    """Store 6 decimals; below 0.001 keep 6 significant digits, so a p-value such as
    0.00099990 is not stored as 0.001 and then printed as p = 0.001 instead of p < 0.001."""
    return round(x, 6) if abs(x) >= 1e-3 or x == 0 else float(f"{x:.6g}")


class Numbers:
    """name -> {value, ci, ci_method, n, k, T, source, status, note}."""
    def __init__(self):
        self.d = {}

    def put(self, name, value, ci=None, ci_method=None, n=None, source=None, **kw):
        if isinstance(value, float):
            value = r6(value)
        e = {"value": value}
        if ci is not None:
            e["ci"] = [r6(ci[0]), r6(ci[1])]
            e["ci_method"] = ci_method or "wilson95_task"
        if n is not None:
            e["n"] = n
        if source is not None:
            e["source"] = source if isinstance(source, list) else [source]
        for k, v in kw.items():
            if v is not None:
                e[k] = r6(v) if isinstance(v, float) else v
        self.d[name] = e

    def cell(self, prefix, st, source, status=None):
        """Standard cell block: IVR with the task CI (primary) + episode CI."""
        self.put(prefix + ":ivr", st["p"], st["ci_task"], "wilson95_task(p*T,T)", st["n"],
                 source, k=st["k"], T=st["T"], status=status)
        self.put(prefix + ":ivr_ci_ep", st["p"], st["ci_ep"], "wilson95_episode", st["n"],
                 source, k=st["k"])

    def test(self, prefix, r, source, holm_p=None, status=None, verdict=None, note=None):
        self.put(prefix + ":d", r["d"], n=r["T"], source=source, unit="task",
                 stat="mean paired per-task difference (second - first)",
                 plus=r["plus"], minus=r["minus"], status=status)
        self.put(prefix + ":p", r["p"], n=r["T"], source=source,
                 stat=f"two-sided paired sign-flip permutation, {PERM_N} perms, seed {PERM_SEED}",
                 holm_p=holm_p, status=status, verdict=verdict, note=note)

    def write(self, path):
        with open(path, "w") as fh:
            json.dump(self.d, fh, indent=1, sort_keys=True)


def write_csv(path, rows):
    if not rows:
        return
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def f(x, nd=3):
    return "NA" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def fp(p):
    return "NA" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")
