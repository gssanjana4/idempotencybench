"""E0-3 matcher audit extended to every scored episode (6,798), plus the C1
re-score summary (final audit: ISO date-time seconds).

Same pair enumeration and classes as analysis/e0_3_matcher_audit.py (every
pair of same-tool effectful calls within an episode; Paper-2 episodes: both
sessions), over results/sweep_transcripts, results/crossprovider/
sweep_transcripts, results/tier1/e1_*/transcripts, results/transcripts2,
results/crossprovider/transcripts2, results/claude_batch_2026-10/transcripts2.
Each pair is classified under the fixed matcher and under the pre-fix one
(src/rescore_c1.OLD_NORM); pairs whose class the fix changes are listed.

Reads : the transcript dirs above, results/analysis_v3/rescore_c1.json/.csv
        (src/rescore_c1.py)
Writes: results/analysis_v3/matcher_audit_all.json (numbers, c1.*),
        results/analysis_v3/matcher_audit_all_pairs.csv (every pair that is
        not raw-identical or distinct, i.e. normalization-only or near miss)
"""
import csv, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import OUT, RES, ROOT, Numbers, rel, write_csv
import e0_1_corrected_table as e01
import e0_3_matcher_audit as e03
sys.path.insert(0, os.path.join(ROOT, "src"))
import rescore_c1  # noqa: E402

DIRS = [("jul", "sweep_transcripts"), ("xp", "crossprovider/sweep_transcripts")] + \
       [(os.path.basename(os.path.dirname(d)), os.path.relpath(d, RES))
        for d in sorted(glob.glob(os.path.join(RES, "tier1", "e1_*", "transcripts")))] + \
       [("jul2", "transcripts2"), ("xp2", "crossprovider/transcripts2"),
        ("oct2", "claude_batch_2026-10/transcripts2")]


def turns_of(t):
    if "sessions" in t:
        return [x for s in t["sessions"] for x in s["turns"]]
    if "deliveries" in t:
        return [x for d in t["deliveries"] for x in d["turns"]]
    return t["turns"]


def calls_of(t):
    out = []
    for turn in turns_of(t):
        if turn["role"] != "assistant":
            continue
        for b in turn["content"]:
            if b.get("type") == "tool_use" and b["name"] in e01.CANON:
                a = dict(b.get("input") or {}); a.pop("idempotency_key", None)
                out.append((b["name"], a))
    return out


def main():
    counts, old_counts, moved, listed, n_eps, n_calls = {}, {}, [], [], 0, 0
    for src, d in DIRS:
        for f in sorted(glob.glob(os.path.join(RES, d, "*.json"))):
            t = json.load(open(f)); n_eps += 1
            calls = calls_of(t); n_calls += len(calls)
            for i in range(len(calls)):
                for j in range(i + 1, len(calls)):
                    if calls[i][0] != calls[j][0]:
                        continue
                    e03.norm = rescore_c1.NEW_NORM
                    cls, _, _, _, sim, diff = e03.classify(calls[i][0], calls[i][1], calls[j][1])
                    e03.norm = rescore_c1.OLD_NORM
                    ocls = e03.classify(calls[i][0], calls[i][1], calls[j][1])[0]
                    e03.norm = rescore_c1.NEW_NORM
                    counts[cls] = counts.get(cls, 0) + 1
                    old_counts[ocls] = old_counts.get(ocls, 0) + 1
                    row = dict(source=src, file=os.path.join(d, os.path.basename(f)), tool=calls[i][0],
                               call_i=i, call_j=j, cls=cls, cls_before_fix=ocls,
                               min_field_similarity=sim, differing_fields=json.dumps(diff))
                    if cls != ocls:
                        moved.append(row)
                    if cls in ("normalization_only", "near_miss"):
                        listed.append(row)
    write_csv(os.path.join(OUT, "matcher_audit_all_pairs.csv"), listed)
    c1 = json.load(open(os.path.join(OUT, "rescore_c1.json")))
    assert c1["episodes"] == n_eps, (c1["episodes"], n_eps)
    N = Numbers()
    srcs = [rel(os.path.join(OUT, "matcher_audit_all_pairs.csv"))]
    N.put("c1.audit.episodes", n_eps, source=srcs)
    N.put("c1.audit.effcalls", n_calls, source=srcs)
    N.put("c1.audit.pairs", sum(counts.values()), source=srcs)
    for c in ("raw_identical", "normalization_only", "near_miss", "distinct"):
        N.put(f"c1.audit.{c}", counts.get(c, 0), source=srcs, before_fix=old_counts.get(c, 0))
    N.put("c1.audit.pairs_moved_by_fix", len(moved), source=srcs,
          note="pairs whose class the ISO-seconds fix changes (near_miss -> normalization_only)")
    N.put("c1.audit.near_miss_before_fix", old_counts.get("near_miss", 0), source=srcs)
    N.put("c1.rescore.episodes", c1["episodes"], source="results/analysis_v3/rescore_c1.json")
    N.put("c1.rescore.score_changed", c1["score_changed"], source="results/analysis_v3/rescore_c1.csv")
    N.put("c1.rescore.value_merge_episodes", c1["episodes_with_value_merge"],
          source="results/analysis_v3/rescore_c1.json")
    N.write(os.path.join(OUT, "matcher_audit_all.json"))
    print(json.dumps(dict(episodes=n_eps, calls=n_calls, counts=counts, before_fix=old_counts,
                          moved=len(moved)), indent=1))
    for r in moved:
        print(r["file"], r["differing_fields"])


if __name__ == "__main__":
    main()
