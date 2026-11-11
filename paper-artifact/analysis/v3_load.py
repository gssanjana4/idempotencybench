"""Loaders for the v3 analyses: one episode dict per scored episode.

Episode dict fields: eid, task, sample, v (1 if >= 1 duplicate semantic
effect, the pre-registered IVR outcome), viol (count), strict (raw-string
matcher 0/1 or None), rb (read-back after the failed call, 0/1 or None if the
failure never fired), retried, key_first, key_cat, truncated, turn_cap,
template, fail_on_call, plus ALO fields where present.

Sources (cell key = (src, model, prompt, mitigation, failure_mode)):
  v2     results/sweep_raw_v2.jsonl (released July cells, v2 rescore,
         unmodified); per-episode features from the read-only re-derivation
         that src/tier1.py wrote to results/tier1/*/imported.jsonl
         (calls-only features; the ledger-derived fields of those rows are
         NOT used, Tester N2) and a strict count from a fresh World replay.
  xp     results/crossprovider/sweep_raw.jsonl, one row per episode_id,
         LAST row kept (rule fixed in e78e23d); features from
         results/tier1/e1_1/imported.jsonl (source=crossprovider), which was
         built from the transcript on disk (= the last run).
  xp_first  same file, FIRST row kept (sensitivity for the 144 Qwen ids).
  e1_0..e1_6  results/tier1/<exp>/raw.jsonl (last row per id; there are no
         duplicate ids), or raw_v2.jsonl where the C1 re-score
         (src/rescore_c1.py) changed a row.
"""
import glob, json, os, re, sys
from collections import Counter, defaultdict
from v3_common import ROOT, RES, read_jsonl, dedupe, rel, prefer_v2

sys.path.insert(0, os.path.join(ROOT, "src"))

V2_RAW = os.path.join(RES, "sweep_raw_v2.jsonl")
V1_RAW = os.path.join(RES, "sweep_raw.jsonl")
XP_RAW = prefer_v2(os.path.join(RES, "crossprovider", "sweep_raw.jsonl"))
T1 = os.path.join(RES, "tier1")
EXPS = ["e1_0", "e1_1", "e1_2", "e1_3", "e1_4", "e1_5", "e1_6"]


def _ep(r, src):
    rb = r.get("readback_after_failure")
    if r.get("failure_fired") is False:
        rb = None
    e = {"eid": r["episode_id"], "task": r["task_id"], "sample": r["sample"],
         "v": 1 if r["violations"] > 0 else 0, "viol": r["violations"],
         "strict": None if r.get("violations_strict") is None else int(r["violations_strict"] > 0),
         "rb": None if rb is None else int(bool(rb)),
         "retried": r.get("retried"), "key_first": r.get("key_attached_first"),
         "key_cat": r.get("retry_key_category"), "truncated": r.get("truncated"),
         "turn_cap": r.get("turn_cap_hit"), "template": r.get("template"),
         "fail_on_call": r.get("fail_on_call"), "family": r.get("family"),
         "achieved": r.get("achieved"), "omission": r.get("omission"),
         "misorder": r.get("misorder"), "stale": r.get("stale_reexec"),
         "src": src, "model": r["model"], "prompt": r["prompt"],
         "mit": r["mitigation"], "fm": r["failure_mode"], "sig_calls": r.get("sig_calls"),
         "model_served": r.get("model_served"), "temperature": r.get("temperature"),
         "started_utc": r.get("started_utc")}
    for k in ("d1_violations", "d2_readback_first", "d2_reexecuted", "d2_key_category"):
        if k in r:
            e[k] = r[k]
    return e


def _imported_features():
    """episode_id -> imported row (released_v2 and crossprovider roles)."""
    feats = {}
    for p in sorted(glob.glob(os.path.join(T1, "e1_*", "imported.jsonl"))):
        for r in read_jsonl(p):
            feats[(r["source"], r["episode_id"])] = r
    return feats


def _strict_v2():
    """Strict (raw-string) duplicate flag for the released transcripts,
    from a fresh replay through the current World (the stored v1 ledger is
    pre-fix and cannot be used, Tester N2). Same call sequence as
    src/rescore.replay; the canonical fields are read raw, unnormalized."""
    from world import World, EFFECTFUL_TOOLS, ToolTimeout, ToolServerError, ToolCallError
    import rescore, sweep
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, "sweep_transcripts", "*.json"))):
        m = rescore.FNAME.match(os.path.basename(f))
        if not m:
            continue
        t = json.load(open(f))
        model = t.get("model", m["model"])
        if not model.startswith("claude"):
            continue
        w = World(mitigation=m["mit"], failure_mode=m["fm"], fail_on_call=1)
        calls = [b for turn in t["turns"] if turn["role"] == "assistant"
                 for b in turn["content"] if b.get("type") == "tool_use"]
        for b in calls:
            args = dict(b["input"]); key = args.pop("idempotency_key", "") or ""
            try:
                w.call(b["name"], args, key, t["task_id"])
            except (ToolTimeout, ToolServerError, ToolCallError):
                pass
        raw = Counter((e.tool,) + tuple(str(e.args.get(fl)) for fl in EFFECTFUL_TOOLS[e.tool][1])
                      for e in w.ledger.for_task(t["task_id"]))
        eid = sweep.episode_id(model, m["prompt"], m["mit"], m["fm"], m["task"], int(m["s"]))
        out[eid] = int(any(v > 1 for v in raw.values()))
    return out


def load_all():
    """Returns (cells, meta). cells: {(src, model, prompt, mit, fm): [episode]}"""
    feats = _imported_features()
    cells = defaultdict(list)
    meta = {"sources": {}}
    # released v2
    strict = _strict_v2()
    v2 = dedupe(read_jsonl(V2_RAW))
    for eid, r in v2.items():
        f = feats.get(("released_v2", eid))
        e = _ep(dict(r, **({k: f[k] for k in ("readback_after_failure", "failure_fired",
                                               "retried", "key_attached_first",
                                               "retry_key_category", "truncated",
                                               "turn_cap_hit", "template", "sig_calls")}
                           if f else {}), violations_strict=None), "v2")
        if f is None:
            e["rb"] = None
        e["strict"] = strict.get(eid)
        cells[("v2", r["model"], r["prompt"], r["mitigation"], r["failure_mode"])].append(e)
    meta["sources"]["v2"] = rel(V2_RAW)
    # crossprovider, last and first rows
    rows = read_jsonl(XP_RAW)
    for src, keep in (("xp", "last"), ("xp_first", "first")):
        for eid, r in dedupe(rows, keep).items():
            f = feats.get(("crossprovider", eid)) if src == "xp" else None
            rr = dict(r)
            if f:
                assert (f["violations"] > 0) == (r["violations"] > 0), eid
                for k in ("readback_after_failure", "failure_fired", "retried",
                          "key_attached_first", "retry_key_category", "turn_cap_hit",
                          "template", "violations_strict", "stale_reexec", "sig_calls",
                          "model_served", "temperature", "started_utc"):
                    rr[k] = f.get(k)
            e = _ep(rr, src)
            if not f:
                e["rb"] = None
            cells[(src, r["model"], r["prompt"], r["mitigation"], r["failure_mode"])].append(e)
    cnt = Counter(r["episode_id"] for r in rows)
    meta["xp_rows"] = len(rows)
    meta["xp_unique"] = len(cnt)
    meta["xp_dup_ids"] = sum(1 for v in cnt.values() if v > 1)
    meta["xp_dup_models"] = sorted({k.split("__")[0] for k, v in cnt.items() if v > 1})
    fl, la = dedupe(rows, "first"), dedupe(rows, "last")
    meta["xp_dup_disagree"] = sum(1 for k, v in cnt.items() if v > 1
                                  and (fl[k]["violations"] > 0) != (la[k]["violations"] > 0))
    meta["sources"]["xp"] = rel(XP_RAW)
    # tier-1
    for x in EXPS:
        p = prefer_v2(os.path.join(T1, x, "raw.jsonl"))   # raw_v2.jsonl: C1 re-score
        rows = read_jsonl(p)
        meta[f"{x}_rows"] = len(rows)
        meta[f"{x}_unique"] = len({r["episode_id"] for r in rows})
        for eid, r in dedupe(rows).items():
            cells[(x, r["model"], r["prompt"], r["mitigation"], r["failure_mode"])].append(_ep(r, x))
        meta["sources"][x] = rel(p)
    return cells, meta


def load_v1_cell(model, prompt, mit, fm):
    rows = dedupe(read_jsonl(V1_RAW))
    return [r for r in rows.values() if (r["model"], r["prompt"], r["mitigation"],
                                          r["failure_mode"]) == (model, prompt, mit, fm)]


# ------------------------------------------------------------------ refusals
BLOCK = "Gemini blocked the response"
LOGS = {"tier1": os.path.join(T1, "run.log"),
        "crossprovider": os.path.join(RES, "crossprovider", "run.log")}
_FAIL = re.compile(r"^!! (?:\[\d+/\d+\]|episode) (\S+) failed, will retry on next run: (.*)$")


def refusals(cells):
    """Episodes that never produced a row and whose every logged failure is a
    provider block ("Gemini blocked the response"). Returned as
    {(src, model, prompt, mit, fm): [(task, sample, n_attempts)]}, with src
    mapped onto the cell the episode belongs to (xp for run_crossprovider.sh,
    e1_1/e1_2 for tier1)."""
    have = {e["eid"] for eps in cells.values() for e in eps}
    out = defaultdict(list)
    detail = []
    for runner, path in LOGS.items():
        fails = defaultdict(list)
        for line in open(path):
            m = _FAIL.match(line.strip())
            if m:
                fails[m.group(1)].append(m.group(2))
        for eid, msgs in sorted(fails.items()):
            if eid in have:
                continue
            blocked = all(BLOCK in x for x in msgs)
            parts = eid.split("__")
            model, prompt, mit, fm, task, s = parts[0], parts[1], parts[2], parts[3], parts[4], int(parts[5][1:])
            if runner == "tier1":
                src = "e1_1" if (mit, fm) == ("idem_keys", "error_500_after_commit") else "e1_2"
            else:
                src = "xp"
            detail.append({"runner": runner, "episode_id": eid, "src": src, "attempts": len(msgs),
                           "all_attempts_provider_block": blocked,
                           "last_error": msgs[-1][:200], "log": rel(path)})
            if blocked:
                out[(src, model, prompt, mit, fm)].append((task, s, len(msgs)))
    return out, detail
