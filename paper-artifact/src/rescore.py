"""Re-score released model transcripts against the CURRENT World semantics.

Why this is valid without new API calls: scoring depends only on the sequence
of effectful tool calls the model issued. Replaying those calls through a fresh
World reproduces the ledger exactly, provided every tool result the model saw
is unchanged by the fix. We check that per episode (error positions must match,
and every non-error result must still be an "ok-..." ack) and fail loudly if not.

  python3 src/rescore.py   ->  results/sweep_raw_v2.jsonl, results/sweep_summary_v2.csv,
                               results/rescore_diff.csv (every episode whose score changed)
"""
import csv, glob, json, os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
from world import World, EFFECTFUL_TOOLS, FAILURE_MODES, SIGNAL_VARIANTS, WORDING_VARIANTS, PRECOMMIT_VARIANTS, ToolTimeout, ToolServerError, ToolCallError
import sweep

FNAME = re.compile(r"^(?P<model>.+?)_(?P<prompt>base|careful)_(?P<mit>none|idem_keys|runtime_receipts)_"
                   r"(?P<fm>" + "|".join(sorted(FAILURE_MODES + tuple(SIGNAL_VARIANTS) + tuple(WORDING_VARIANTS) + tuple(PRECOMMIT_VARIANTS), key=len, reverse=True)) + r")__s(?P<s>\d+)__(?P<task>.+)\.json$")

def replay(t, mit, fm):
    # fail_on_call: tier-1 long chains (E1-5); released transcripts have none (= 1)
    w = World(mitigation=mit, failure_mode=fm, fail_on_call=t.get("fail_on_call", 1))
    seen_results = [r for turn in t["turns"] if turn["role"] == "tool_results" for r in turn["content"]]
    calls = [b for turn in t["turns"] if turn["role"] == "assistant"
             for b in turn["content"] if b.get("type") == "tool_use"]
    assert len(calls) == len(seen_results), "call/result count mismatch"
    for b, orig in zip(calls, seen_results):
        args = dict(b["input"]); key = args.pop("idempotency_key", "") or ""
        try:
            out = w.call(b["name"], args, key, t["task_id"]); err = False
        except (ToolTimeout, ToolServerError, ToolCallError) as e:
            out = f"ERROR: {e}"; err = True
        was_err = bool(orig.get("is_error"))
        if err != was_err:
            raise AssertionError(f"error position changed for {b['name']}")
        if not err and b["name"] in EFFECTFUL_TOOLS:
            assert str(orig["content"]).startswith("ok-") and out.startswith("ok-"), "ack form changed"
    viol = w.ledger.violations(t["task_id"])
    out = w.ledger.outcome(t["task_id"], _goals(t), ordered=(t["family"] in ("multi_step", "long_chain")))
    return viol, out["success"], out

def _goals(t):
    from world import Goal
    return [Goal(g["tool"], g["canon_args"]) for g in t["score"]["goals"]]

def main():
    rows, diffs = [], []
    files = sorted(glob.glob(os.path.join(HERE, "results", "sweep_transcripts", "*.json")))
    for f in files:
        m = FNAME.match(os.path.basename(f)); assert m, f
        t = json.load(open(f))
        viol, ach, out = replay(t, m["mit"], m["fm"])
        old_v, old_a = t["score"]["violations"], t["score"]["achieved"]
        # model id from the transcript, not the filename: filenames flatten "/"
        # in OpenRouter slugs (openrouter:openai/gpt-5.5 -> ..._gpt-5.5...)
        model = t.get("model", m["model"])
        r = dict(episode_id=sweep.episode_id(model, m["prompt"], m["mit"], m["fm"], m["task"], int(m["s"])),
                 model=model, prompt=m["prompt"], mitigation=m["mit"], failure_mode=m["fm"],
                 task_id=m["task"], family=t["family"], sample=int(m["s"]), violations=viol, achieved=ach,
                 omission=out["omission"], misorder=out["misorder"])
        rows.append(r)
        if (viol, ach) != (old_v, old_a):
            diffs.append(dict(file=os.path.basename(f), old_violations=old_v, new_violations=viol,
                              old_achieved=old_a, new_achieved=ach))
    raw2 = os.path.join(HERE, "results", "sweep_raw_v2.jsonl")
    with open(raw2, "w") as fh:
        for r in rows: fh.write(json.dumps(r) + "\n")
    with open(os.path.join(HERE, "results", "rescore_diff.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file","old_violations","new_violations","old_achieved","new_achieved"])
        w.writeheader(); w.writerows(diffs)
    sweep.RAW_PATH = raw2
    sweep.SUMMARY_PATH = os.path.join(HERE, "results", "sweep_summary_v2.csv")
    sweep.aggregate()
    print(f"replayed {len(rows)} episodes; {len(diffs)} changed score")

if __name__ == "__main__":
    main()
