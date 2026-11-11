"""Session-resume: replay released model transcripts under the corrected
acknowledgement rule (audit F6) -- no API calls.

Old rule (sweep2.run_model_session before the fix): a node counted as
acknowledged the moment its world.call returned, so when the seam node
committed mid-turn, nodes committed EARLIER IN THAT SAME TURN went into the
plan_only notes although their results were never returned to the model.
New rule: acknowledged = committed as of the end of the last turn whose
results were all delivered.

Replay: every tool call of both sessions is re-issued through a fresh World,
modelling the mid-turn cut, and every tool result the model saw is asserted
unchanged (list_effects text identical; effectful acks still "ok-..."); the
recomputed score must equal the stored one. Scores cannot change (the score
does not depend on the notes), but in plan_only episodes where the notes the
model received differ from the corrected notes, session 2 was run on a
different prompt than the paper defines -- those episodes are flagged
`notes_leaked` and the cell is also reported on the clean subset.

  python3 src2/rescore2.py -> results/sweep2_raw_v2.jsonl, results/sweep2_summary_v2.csv
"""
import csv, glob, json, os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, os.path.join(HERE, "src2"))
from world import World, EFFECTFUL_TOOLS  # noqa: E402
import sweep2  # noqa: E402
from resume import score_graph, _node_commits  # noqa: E402

KEYS = ("violations", "givr_hit", "completion", "order_violation")


def _results(turns):
    return [r for t in turns if t["role"] == "tool_results" for r in t["content"]]


def _check(name, out, orig):
    if name in EFFECTFUL_TOOLS:
        assert str(orig["content"]).startswith("ok-") and out.startswith("ok-"), "ack form changed"
    else:
        assert out == orig["content"], f"{name} result changed"


def replay(t, g, k):
    """Returns (score tuple, old acknowledged, corrected acknowledged)."""
    w = World(mitigation=t["mitigation"], failure_mode="none")
    s1 = t["sessions"][0]
    turns = s1["turns"]
    acked, cut = set(), False
    for ti, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        nxt = turns[ti + 1] if ti + 1 < len(turns) else None
        # results of this turn (absent for the cut turn: never delivered)
        seen = iter(nxt["content"] if nxt and nxt["role"] == "tool_results" else [])
        for b in turn["content"]:
            if b.get("type") != "tool_use":
                continue
            a = dict(b["input"]); key = a.pop("idempotency_key", "") or ""
            out = w.call(b["name"], a, key, g.graph_id)
            if _node_commits(w, g, k):
                cut = True; break
            if nxt and nxt["role"] == "tool_results":
                _check(b["name"], out, next(seen))
        if cut:
            assert nxt is not None and nxt["role"] == "interrupt", "cut position changed"
            break
        assert next(seen, None) is None, "unconsumed session-1 results"
        acked = {i for i in range(len(g.nodes)) if _node_commits(w, g, i)}
    assert cut == s1["interrupted"], "interrupt position changed"
    acked.discard(k)
    if len(t["sessions"]) > 1:
        s2 = t["sessions"][1]; seen = iter(_results(s2["turns"]))
        for turn in s2["turns"]:
            for b in turn.get("content", []) if turn["role"] == "assistant" else []:
                if b.get("type") == "tool_use":
                    a = dict(b["input"]); key = a.pop("idempotency_key", "") or ""
                    _check(b["name"], w.call(b["name"], a, key, g.graph_id), next(seen))
        assert next(seen, None) is None, "unconsumed session-2 results"
    row = score_graph(w, g, interrupted=cut, resume_context=t["resume_context"],
                      mitigation=t["mitigation"], key_posture="model")
    old = set(s1["acknowledged"]) - {k}
    return tuple(row[x] for x in KEYS), old, acked


def main():
    G = {g.graph_id: (g, k) for g, k in sweep2.benchmark_graphs()}
    raw = {}
    for line in open(os.path.join(HERE, "results", "sweep2_raw.jsonl")):
        if line.strip():
            r = json.loads(line); raw.setdefault(r["episode_id"], r)
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "transcripts2", "*.json"))):
        t = json.load(open(f)); eid = os.path.basename(f)[:-5]
        g, k = G[t["graph_id"]]; assert k == t["interrupt_node"]
        sc, old, new = replay(t, g, k)
        r = dict(raw[eid])
        assert sc == tuple(t["score"][x] for x in KEYS) == tuple(r[x] for x in KEYS), eid
        r["notes_leaked"] = t["resume_context"] == "plan_only" and old != new
        out.append(r)
    assert len(out) == len(raw), "transcript / raw count mismatch"
    with open(os.path.join(HERE, "results", "sweep2_raw_v2.jsonl"), "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    cells = {}
    for r in out:
        cells.setdefault((r["model"], r["resume_context"], r["mitigation"]), []).append(r)
    path = os.path.join(HERE, "results", "sweep2_summary_v2.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "resume_context", "mitigation", "n", "givr", "givr_lo", "givr_hi",
                    "n_notes_leaked", "n_clean", "givr_clean", "givr_clean_lo", "givr_clean_hi"])
        for (m, ctx, mit), rs in sorted(cells.items()):
            n = len(rs); gk = sum(r["givr_hit"] for r in rs)
            clean = [r for r in rs if not r["notes_leaked"]]
            ck = sum(r["givr_hit"] for r in clean); nc = len(clean)
            lo, hi = sweep2.wilson_ci(gk, n); clo, chi = sweep2.wilson_ci(ck, nc)
            w.writerow([m, ctx, mit, n, round(gk / n, 4), round(lo, 4), round(hi, 4),
                        n - nc, nc, round(ck / nc, 4) if nc else "", round(clo, 4), round(chi, 4)])
            if n != nc:
                print(f"{m} {ctx}: gIVR {gk}/{n}={gk/n:.4f}  leaked {n-nc} "
                      f"(gIVR hits there {gk-ck})  clean {ck}/{nc}={ck/nc:.4f} [{clo:.4f}, {chi:.4f}]")
    print(f"replayed {len(out)} episodes; all tool results and scores identical; -> {path}")


if __name__ == "__main__":
    main()
