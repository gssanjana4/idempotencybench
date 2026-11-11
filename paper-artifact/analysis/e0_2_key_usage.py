"""E0-2: idempotency-key attachment on the FIRST call vs on the RETRY, and
stable-key reuse, per model and cell (single-session study released transcripts).

Definitions (per episode):
  first call   = first effectful tool call of the episode (the injected
                 failure always fires on it; verified in E0-1).
  key_attached_first = that call carries a non-empty idempotency_key.
  retry        = the first LATER effectful call with the same semantic id
                 (same tool + normalized canonical args) as the first call.
  retry categories (exactly one per retried episode):
    same_key   : retry carries the same non-empty key as the first call
    new_key    : first call keyed, retry keyed with a different key
    dropped_key: first call keyed, retry keyless
    added_key  : first call keyless, retry keyed   (can never dedup)
    keyless    : both keyless
  key_reuse_rate = same_key / retried.
  key_any_episode = any effectful call in the episode carries a key
                 (this is the "17/64"-style count, which mixes in retries).
Cross-sample determinism: for each (model, cell, task) with both samples,
compare the key on the first call of s0 vs s1 (variant A: first-call keys,
pairs where both first calls keyed) and the first non-empty key anywhere
in the episode (variant B: pairs where both episodes used any key).

Reads : results/sweep_transcripts/*.json
Writes: results/analysis_v2/e0_2_key_usage.csv            (per model x cell)
        results/analysis_v2/e0_2_key_usage_episodes.csv   (per episode)
        results/analysis_v2/e0_2_key_determinism.csv      (per model)
        results/analysis_v2/e0_2_summary.json
Run   : python3 analysis/e0_2_key_usage.py
No sampling; fully deterministic.
"""
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_1_corrected_table import FNAME, CANON, sid  # independent matcher (not src/)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDIR = os.path.join(ROOT, "results", "sweep_transcripts")
OUT = os.path.join(ROOT, "results", "analysis_v2")
CATS = ("same_key", "new_key", "dropped_key", "added_key", "keyless")


def effectful(t):
    out = []
    for turn in t["turns"]:
        if turn["role"] != "assistant":
            continue
        for b in turn["content"]:
            if b.get("type") == "tool_use" and b["name"] in CANON:
                a = dict(b["input"]); k = a.pop("idempotency_key", "") or ""
                out.append((b["name"], a, k))
    return out


def classify(t):
    calls = effectful(t)
    if not calls:
        return dict(n_effectful=0, key_attached_first=None, retried=False, category="",
                    key_any_episode=False, first_key="", first_any_key="",
                    n_keyed_calls=0)
    tool, a, k0 = calls[0]
    s0 = sid(tool, a)
    retry = next(((tl, aa, kk) for tl, aa, kk in calls[1:] if sid(tl, aa) == s0), None)
    cat = ""
    if retry:
        k1 = retry[2]
        if k0 and k1:
            cat = "same_key" if k0 == k1 else "new_key"
        elif k0 and not k1:
            cat = "dropped_key"
        elif not k0 and k1:
            cat = "added_key"
        else:
            cat = "keyless"
    keys = [kk for _, _, kk in calls if kk]
    return dict(n_effectful=len(calls), key_attached_first=bool(k0), retried=retry is not None,
                category=cat, key_any_episode=bool(keys), first_key=k0,
                first_any_key=keys[0] if keys else "", n_keyed_calls=len(keys))


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = []
    for f in sorted(glob.glob(os.path.join(TDIR, "*.json"))):
        m = FNAME.match(os.path.basename(f)); t = json.load(open(f))
        eps.append(dict(file=os.path.basename(f), model=t["model"], prompt=m["prompt"],
                        mitigation=m["mit"], failure_mode=m["fm"], task_id=m["task"],
                        family=t["family"], sample=int(m["s"]), violations=t["score"]["violations"],
                        **classify(t)))
    with open(os.path.join(OUT, "e0_2_key_usage_episodes.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(eps[0].keys())); w.writeheader(); w.writerows(eps)

    cells = {}
    for e in eps:
        cells.setdefault((e["model"], e["prompt"], e["mitigation"], e["failure_mode"]), []).append(e)
    rows = []
    for key in sorted(cells):
        E = cells[key]; n = len(E)
        ret = [e for e in E if e["retried"]]
        c = {k: sum(1 for e in ret if e["category"] == k) for k in CATS}
        # violation cross-tab: which categories ended up as ledger duplicates
        # (under idem_keys v1 transcripts store the BUGGY score; use E0-1 for v2)
        rows.append(dict(model=key[0], prompt=key[1], mitigation=key[2], failure_mode=key[3],
                         n_episodes=n,
                         key_attached_first=sum(1 for e in E if e["key_attached_first"]),
                         key_attach_rate=round(sum(1 for e in E if e["key_attached_first"]) / n, 4),
                         key_any_episode=sum(1 for e in E if e["key_any_episode"]),
                         retried=len(ret), retry_rate=round(len(ret) / n, 4),
                         **c,
                         key_reuse_rate=round(c["same_key"] / len(ret), 4) if ret else None))
    with open(os.path.join(OUT, "e0_2_key_usage.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # cross-sample determinism
    pairs = {}
    for e in eps:
        pairs.setdefault((e["model"], e["prompt"], e["mitigation"], e["failure_mode"], e["task_id"]), {})[e["sample"]] = e
    det = {}
    for (model, *_), p in pairs.items():
        if 0 not in p or 1 not in p:
            continue
        d = det.setdefault(model, dict(model=model, pairs=0, A_both_first_keyed=0, A_identical=0,
                                       B_both_any_keyed=0, B_identical=0))
        d["pairs"] += 1
        a, b = p[0], p[1]
        if a["first_key"] and b["first_key"]:
            d["A_both_first_keyed"] += 1; d["A_identical"] += a["first_key"] == b["first_key"]
        if a["first_any_key"] and b["first_any_key"]:
            d["B_both_any_keyed"] += 1; d["B_identical"] += a["first_any_key"] == b["first_any_key"]
    det_rows = [det[m] for m in sorted(det)]
    with open(os.path.join(OUT, "e0_2_key_determinism.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(det_rows[0].keys())); w.writeheader(); w.writerows(det_rows)

    # Fable residual idem_keys violations vs category (v2 scoring from E0-1)
    v2 = {}
    p1 = os.path.join(OUT, "e0_1_episodes.csv")
    if os.path.exists(p1):
        for r in csv.DictReader(open(p1)):
            v2[r["file"]] = int(r["violations"])
    resid = {}
    for e in eps:
        if e["mitigation"] == "idem_keys" and v2.get(e["file"], 0) > 0:
            resid.setdefault(e["model"], {}).setdefault(e["category"] or "no_retry", 0)
            resid[e["model"]][e["category"] or "no_retry"] += 1
    focus = [r for r in rows if r["prompt"] == "base" and (
        (r["mitigation"] == "idem_keys") or (r["mitigation"] == "none" and r["failure_mode"] == "error_500_after_commit"))]
    summary = dict(focus_cells=focus, determinism=det_rows,
                   idem_keys_v2_violations_by_retry_category=resid if v2 else
                   "run e0_1 first (needs e0_1_episodes.csv)",
                   notes=["key_any_episode counts episodes with a key on ANY call (incl. retries); "
                          "REVISION-NOTES' 'Haiku attached a key in 17/64' is this count, not first-call attachment.",
                          "Variant A determinism = keys on the first call of s0 vs s1; variant B = first non-empty key "
                          "anywhere in the episode."])
    json.dump(summary, open(os.path.join(OUT, "e0_2_summary.json"), "w"), indent=1)
    for r in focus:
        print(r)
    for r in det_rows:
        print(r)
    print(summary["idem_keys_v2_violations_by_retry_category"])


if __name__ == "__main__":
    main()
