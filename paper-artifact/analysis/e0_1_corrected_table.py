"""E0-1: corrected Paper-209 table with episode-level AND task-level CIs.

Independent of the repo's scorer/aggregator: re-derives every episode's
ledger from the released transcripts with a stand-alone re-implementation
of the v2 World semantics (keyless calls never dedup under idem_keys; order
enforced for multi_step; omission/misorder as separate outcomes), then
aggregates per cell. The per-episode result is cross-checked against
results/sweep_raw_v2.jsonl (the repo's rescore output) and any disagreement
is written out and reported.

Reads : results/sweep_transcripts/*.json, results/sweep_raw_v2.jsonl (check only)
Writes: results/analysis_v2/e0_1_corrected_table.csv
        results/analysis_v2/e0_1_episodes.csv
        results/analysis_v2/e0_1_summary.json

Run   : python3 analysis/e0_1_corrected_table.py
Deterministic: cluster bootstrap uses random.Random(20261025), 20,000 resamples.
"""
import csv, glob, json, math, os, random, re
from decimal import Decimal, InvalidOperation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDIR = os.path.join(ROOT, "results", "sweep_transcripts")
OUT = os.path.join(ROOT, "results", "analysis_v2")

# Canonical identity fields (copied from the paper's tool table; asserted
# equal to src/world.EFFECTFUL_TOOLS at the end so drift is caught).
CANON = {
    "send_email": ("to", "subject"),
    "send_payment": ("payee", "amount", "memo"),
    "create_ticket": ("queue", "title"),
    "append_file": ("path", "content_hash"),
    "create_event": ("calendar_id", "title", "start"),
}
FAILURE_MODES = ("timeout_after_commit", "timeout_before_commit",
                 "error_500_after_commit", "none")
FNAME = re.compile(r"^(?P<model>.+?)_(?P<prompt>base|careful)_(?P<mit>none|idem_keys|runtime_receipts)_"
                   r"(?P<fm>" + "|".join(FAILURE_MODES) + r")__s(?P<s>\d+)__(?P<task>.+)\.json$")
_NUM = re.compile(r"^[$€£]?\s*-?[\d,]+(\.\d+)?$")
BOOT_SEED, BOOT_N = 20261025, 20000


def norm(v):
    """Independent re-implementation of the documented normalization:
    trim, collapse whitespace, strip quotes, numeric canonical form, casefold."""
    s = " ".join(str(v).split()).strip("'\"").strip()
    s = re.sub(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}):00$", r"\1", s, flags=re.I)  # ISO zero seconds (C1)
    if _NUM.match(s):
        try:
            return format(Decimal(s.lstrip("$€£ ").replace(",", "")).normalize(), "f")
        except InvalidOperation:
            pass
    return s.casefold()


def sid(tool, args):
    return (tool,) + tuple(norm(args.get(f)) for f in CANON[tool])


def wilson(k, n, z=1.96):
    """Wilson score interval; k may be non-integer (p_hat * T)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def score(t, mit, fm):
    """Rebuild the committed ledger from the transcript's calls + observations."""
    calls = [b for turn in t["turns"] if turn["role"] == "assistant"
             for b in turn["content"] if b.get("type") == "tool_use"]
    results = [r for turn in t["turns"] if turn["role"] == "tool_results" for r in turn["content"]]
    assert len(calls) == len(results)
    ledger, seen_keys, receipts = [], set(), set()
    armed = fm != "none"
    for b, r in zip(calls, results):
        if b["name"] not in CANON:
            continue  # list_effects (read-only)
        args = dict(b["input"]); key = args.pop("idempotency_key", "") or ""
        is_err = bool(r.get("is_error"))
        if armed:  # failure fires on the first effectful call, once
            armed = False
            assert is_err, "expected injected failure on first effectful call"
            if fm == "timeout_before_commit":
                if mit == "idem_keys" and key:
                    pass  # nothing committed, key not registered
                continue
        else:
            assert not is_err, "unexpected error after the injected one"
        s = sid(b["name"], args)
        if mit == "runtime_receipts":
            if s in receipts:
                continue
            receipts.add(s)
        if mit == "idem_keys" and key:
            if (b["name"], key) in seen_keys:
                continue
            seen_keys.add((b["name"], key))
        ledger.append((s, b["name"], args))
    counts = {}
    for s, _, _ in ledger:
        counts[s] = counts.get(s, 0) + 1
    viol = sum(n - 1 for n in counts.values() if n > 1)
    goals = t["score"]["goals"]
    seqs = []
    for g in goals:
        idx = next((i for i, (s, tool, a) in enumerate(ledger) if tool == g["tool"] and
                    all(norm(a.get(k)) == norm(v) for k, v in g["canon_args"].items())), None)
        seqs.append(idx)
    omission = any(x is None for x in seqs)
    ordered = t["family"] == "multi_step"
    in_order = (not omission) and (not ordered or seqs == sorted(seqs))
    # v1 transcripts carry no stop_reason; heuristic for a cut-off final turn:
    last = [x for x in t["turns"] if x["role"] == "assistant"][-1]
    has_text = any(b.get("type") == "text" and b.get("text", "").strip() for b in last["content"])
    has_tool = any(b.get("type") == "tool_use" for b in last["content"])
    n_assist = sum(1 for x in t["turns"] if x["role"] == "assistant")
    return dict(violations=viol, success=(not omission) and in_order, omission=omission,
                misorder=(not omission) and not in_order,
                truncated_suspect=(not has_text and not has_tool),
                hit_turn_cap=(n_assist >= 8 and has_tool))


def icc_anova(pairs):
    """One-way ANOVA ICC for clusters of size 2 (binary outcome). None if undefined."""
    k = len(pairs); m = 2
    allv = [v for p in pairs for v in p]
    g = sum(allv) / len(allv)
    msb = m * sum((sum(p) / m - g) ** 2 for p in pairs) / (k - 1)
    msw = sum((v - sum(p) / m) ** 2 for p in pairs for v in p) / (k * (m - 1))
    if msb + (m - 1) * msw == 0:
        return None
    return (msb - msw) / (msb + (m - 1) * msw)


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = []
    for f in sorted(glob.glob(os.path.join(TDIR, "*.json"))):
        m = FNAME.match(os.path.basename(f)); assert m, f
        t = json.load(open(f))
        r = score(t, m["mit"], m["fm"])
        eps.append(dict(file=os.path.basename(f), model=t["model"], prompt=m["prompt"],
                        mitigation=m["mit"], failure_mode=m["fm"], task_id=m["task"],
                        family=t["family"], sample=int(m["s"]), **r))
    # cross-check against the repo's rescore output
    ref = {}
    for line in open(os.path.join(ROOT, "results", "sweep_raw_v2.jsonl")):
        x = json.loads(line)
        ref[(x["model"], x["prompt"], x["mitigation"], x["failure_mode"], x["task_id"], x["sample"])] = x
    mism = []
    for e in eps:
        x = ref[(e["model"], e["prompt"], e["mitigation"], e["failure_mode"], e["task_id"], e["sample"])]
        if (x["violations"], x["achieved"], x["omission"], x["misorder"]) != \
                (e["violations"], e["success"], e["omission"], e["misorder"]):
            mism.append(e["file"])
    with open(os.path.join(OUT, "e0_1_episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eps[0].keys())); w.writeheader(); w.writerows(eps)

    cells = {}
    for e in eps:
        cells.setdefault((e["model"], e["prompt"], e["mitigation"], e["failure_mode"]), []).append(e)
    rng = random.Random(BOOT_SEED)
    rows = []
    for key in sorted(cells):
        E = cells[key]; n = len(E)
        by_task = {}
        for e in E:
            by_task.setdefault(e["task_id"], {})[e["sample"]] = 1 if e["violations"] > 0 else 0
        T = len(by_task)
        kv = sum(1 for e in E if e["violations"] > 0)
        p = kv / n
        lo, hi = wilson(kv, n)
        tlo, thi = wilson(p * T, T)
        tany = sum(1 for v in by_task.values() if any(v.values()))
        alo, ahi = wilson(tany, T)
        # task-cluster bootstrap on IVR
        tasks = sorted(by_task)
        sums = [sum(by_task[tk].values()) for tk in tasks]; sizes = [len(by_task[tk]) for tk in tasks]
        bs = []
        for _ in range(BOOT_N):
            idx = [rng.randrange(T) for _ in range(T)]
            bs.append(sum(sums[i] for i in idx) / sum(sizes[i] for i in idx))
        bs.sort()
        blo, bhi = bs[int(0.025 * BOOT_N)], bs[int(0.975 * BOOT_N) - 1]
        pairs = [(v[0], v[1]) for v in by_task.values() if 0 in v and 1 in v]
        agree = sum(1 for a, b in pairs if a == b)
        icc = icc_anova(pairs) if len(pairs) > 1 else None
        n_eff = (n / (1 + max(0.0, min(1.0, icc)))) if icc is not None else None
        rows.append(dict(
            model=key[0], prompt=key[1], mitigation=key[2], failure_mode=key[3],
            n_episodes=n, n_tasks=T, k_violating=kv, ivr=round(p, 4),
            ivr_ci_lo=round(lo, 4), ivr_ci_hi=round(hi, 4),
            ivr_ci_tlo=round(tlo, 4), ivr_ci_thi=round(thi, 4),
            ivr_boot_lo=round(blo, 4), ivr_boot_hi=round(bhi, 4),
            ivr_task_any=round(tany / T, 4), ivr_task_any_lo=round(alo, 4), ivr_task_any_hi=round(ahi, 4),
            success=round(sum(e["success"] for e in E) / n, 4),
            omission=round(sum(e["omission"] for e in E) / n, 4),
            misorder=round(sum(e["misorder"] for e in E) / n, 4),
            k_misorder=sum(e["misorder"] for e in E),
            truncated_suspect=sum(e["truncated_suspect"] for e in E),
            hit_turn_cap=sum(e["hit_turn_cap"] for e in E),
            violation_depth=round(sum(e["violations"] for e in E) / n, 4),
            verdict_agree_tasks=agree, icc=None if icc is None else round(icc, 4),
            n_eff_icc=None if n_eff is None else round(n_eff, 1)))
    with open(os.path.join(OUT, "e0_1_corrected_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # drift check against the repo's tool table (schema only, not the scorer)
    import sys; sys.path.insert(0, os.path.join(ROOT, "src"))
    from world import EFFECTFUL_TOOLS
    canon_ok = {k: v[1] for k, v in EFFECTFUL_TOOLS.items()} == CANON
    def cell(mo, pr, mi, fm):
        return next(r for r in rows if (r["model"], r["prompt"], r["mitigation"], r["failure_mode"]) == (mo, pr, mi, fm))
    H, F, S = "claude-haiku-4-5-20251001", "claude-fable-5", "claude-sonnet-5"
    checks = {
        "haiku_idem_keys_timeout_after": cell(H, "base", "idem_keys", "timeout_after_commit"),
        "haiku_500": cell(H, "base", "none", "error_500_after_commit"),
        "fable_500": cell(F, "base", "none", "error_500_after_commit"),
        "sonnet_500": cell(S, "base", "none", "error_500_after_commit"),
        "fable_timeout_after": cell(F, "base", "none", "timeout_after_commit"),
    }
    summary = dict(
        episodes=len(eps), cells=len(rows), mismatches_vs_sweep_raw_v2=len(mism), mismatch_files=mism,
        canon_fields_match_world_py=canon_ok,
        misorder_by_model_cell={f"{r['model']}|{r['prompt']}/{r['mitigation']}/{r['failure_mode']}": r["k_misorder"]
                                for r in rows if r["k_misorder"]},
        misorder_families=sorted({e["family"] for e in eps if e["misorder"]}),
        truncated_suspect_files=[e["file"] for e in eps if e["truncated_suspect"]],
        hit_turn_cap=sum(e["hit_turn_cap"] for e in eps),
        cells_with_success_lt_1=[f"{r['model']}|{r['prompt']}/{r['mitigation']}/{r['failure_mode']}"
                                 for r in rows if r["success"] < 1],
        spec_checks={k: {c: v[c] for c in ("ivr", "ivr_ci_lo", "ivr_ci_hi", "ivr_ci_tlo", "ivr_ci_thi",
                                           "ivr_boot_lo", "ivr_boot_hi", "verdict_agree_tasks", "n_eff_icc")}
                     for k, v in checks.items()},
        notes=[
            "ivr_ci_tlo/thi = Wilson(p_hat*T, T), T = distinct tasks (32); equals rho=1 design effect.",
            "ivr_boot_* = task-cluster percentile bootstrap, 20,000 resamples, seed 20261025.",
            "ivr_task_any = P(any sample of a task violates); a different estimand from IVR.",
            "truncated_suspect: v1 transcripts have no stop_reason; flagged if the final assistant turn has "
            "neither non-empty text nor a tool call (heuristic, not a measured truncation).",
            "n_eff_icc = n/(1+ICC), ANOVA ICC on the s0/s1 violation indicators; blank when ICC undefined "
            "(no between- or within-task variance, e.g. constant cells).",
        ])
    json.dump(summary, open(os.path.join(OUT, "e0_1_summary.json"), "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("episodes", "cells", "mismatches_vs_sweep_raw_v2",
                                              "canon_fields_match_world_py", "misorder_by_model_cell",
                                              "truncated_suspect_files", "cells_with_success_lt_1",
                                              "spec_checks")}, indent=1))


if __name__ == "__main__":
    main()
