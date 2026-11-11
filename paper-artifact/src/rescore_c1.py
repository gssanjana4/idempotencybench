"""C1 (final audit): re-score every episode under the fixed matcher.

The fix (src/world.py normalize_value) treats zero seconds in an ISO
date-time as formatting ("2026-08-01T10:00:00" == "2026-08-01T10:00").
Before it, three October Haiku episodes that created the same calendar event
twice were scored clean.

For every one of the 6,798 scored episodes on disk, the stored tool calls are
replayed twice through a fresh World, once with the pre-fix normalizer
(OLD_NORM, frozen copy below) and once with the fixed one, and
  * the model-visible tool results of the two replays must be identical
    (error positions and texts, list_effects text, ack form incl. dedupe
    acks; random ack hex is masked unless the run seeded its acks), so the
    model's view is unchanged and replay is a valid re-score;
  * the old replay must reproduce the stored score (for the released July
    single-session transcripts: the v2 rescore row, src/rescore.py);
  * any score difference between the two replays is listed.
A value-level screen also lists every episode in which the fix merges two
argument values at all (a superset of the episodes whose score can change).

Tier-1 rows (results/tier1/<exp>/raw.jsonl) whose row changes are rewritten,
all fields recomputed by src/tier1.episode_row from the re-scored transcript,
into raw_v2.jsonl next to raw.jsonl (raw.jsonl is never modified); other
sources get <name>_v2.jsonl only if a row changes. Readers prefer the _v2
file when it exists (analysis/v3_common.prefer_v2).

  python3 src/rescore_c1.py  ->  results/analysis_v3/rescore_c1.csv (old -> new
                                 per changed episode), rescore_c1.json, *_v2.jsonl
No API calls; standard library only.
"""
import csv, glob, json, os, re, sys
from decimal import Decimal, InvalidOperation
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, os.path.join(HERE, "src2"))
import world  # noqa: E402
from world import World, EFFECTFUL_TOOLS, ToolTimeout, ToolServerError, ToolCallError, Goal  # noqa: E402
import rescore, sweep, tier1, sweep2, rescore2, resume  # noqa: E402

RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "analysis_v3")
NEW_NORM = world.normalize_value
_NUM = re.compile(r"^[$€£]?\s*-?[\d,]+(\.\d+)?$")


def OLD_NORM(v) -> str:
    """normalize_value as released before the C1 fix (12e68cf)."""
    s = re.sub(r"\s+", " ", str(v).strip()).strip("'\"").strip()
    if _NUM.match(s):
        try:
            return format(Decimal(s.lstrip("$€£ ").replace(",", "")).normalize(), "f")
        except InvalidOperation:
            pass
    return s.casefold()


def set_norm(fn):
    world.normalize_value = fn; tier1.normalize_value = fn; resume.normalize_value = fn


_ACK = re.compile(r"\bok-[0-9a-f]{8}\b")


def _mask(s, seeded):
    return str(s) if seeded else _ACK.sub("ok-ACK", str(s))


# ------------------------------------------------------------ single-session (P1 + Tier-1)
def replay_p1(t, mit, fm, task_id, goals, ordered, seed):
    w = World(mitigation=mit, failure_mode=fm, fail_on_call=t.get("fail_on_call", 1), seed=seed)
    seen = []
    for d in (t["deliveries"] if "deliveries" in t else [t]):
        for turn in d["turns"]:
            if turn["role"] != "assistant":
                continue
            for b in turn["content"]:
                if b.get("type") != "tool_use":
                    continue
                args = dict(b["input"]); key = args.pop("idempotency_key", "") or ""
                if not isinstance(key, str):
                    key = json.dumps(key, sort_keys=True)
                try:
                    out = w.call(b["name"], args, key, task_id)
                except (ToolTimeout, ToolServerError, ToolCallError) as e:
                    out = f"ERROR: {e}"
                seen.append(_mask(out, seed is not None))
    o = w.ledger.outcome(task_id, goals, ordered=ordered)
    sc = {"violations": w.ledger.violations(task_id), "achieved": o["success"],
          "omission": o["omission"], "misorder": o["misorder"]}
    led = [{"tool": e.tool, "args": e.args, "key": e.key, "seq": e.seq} for e in w.ledger.for_task(task_id)]
    return sc, seen, led


def values_p1(t):
    vals = []
    for d in (t["deliveries"] if "deliveries" in t else [t]):
        for turn in d["turns"]:
            if turn["role"] == "assistant":
                for b in turn["content"]:
                    if b.get("type") == "tool_use" and b["name"] in EFFECTFUL_TOOLS:
                        inp = b.get("input") or {}
                        vals += [inp.get(f) for f in EFFECTFUL_TOOLS[b["name"]][1] if f in inp]
    for g in t["score"]["goals"]:
        vals += list(g["canon_args"].values())
    return vals


def merges(vals):
    """Groups of argument values that the fix newly makes equal."""
    groups = {}
    for v in vals:
        if v is None or isinstance(v, (dict, list)):
            continue
        groups.setdefault(NEW_NORM(v), set()).add(OLD_NORM(v))
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


SC = ("violations", "achieved", "omission", "misorder")
# cost_usd is priced at row-write time from a price table that has since been
# edited (rounding); it is not a score and is carried over unchanged.
KEEP = {"cost_usd"}


def main():
    diffs, screen, summary = [], [], {}
    v2_rows = {}      # source raw path -> {episode_id: new row}
    # stored July v2 scores (the v1 scores in those transcripts are pre-fix)
    jul_v2 = {r["episode_id"]: r for r in map(json.loads, open(os.path.join(RES, "sweep_raw_v2.jsonl")))}
    # Tier-1 task objects by out_dir
    t1 = {}
    for c in sorted(glob.glob(os.path.join(HERE, "configs", "e1_*.json"))):
        if "smoke" in c:
            continue
        cfg = tier1.load_config(c)
        t1[os.path.realpath(cfg["out_dir"])] = (cfg, {x.task_id: x for x in tier1.tasks_for(cfg)})

    P1 = [("jul", os.path.join(RES, "sweep_transcripts"), os.path.join(RES, "sweep_raw_v2.jsonl")),
          ("xp", os.path.join(RES, "crossprovider", "sweep_transcripts"), os.path.join(RES, "crossprovider", "sweep_raw.jsonl"))]
    for x in sorted(glob.glob(os.path.join(RES, "tier1", "e1_*"))):
        P1.append((os.path.basename(x), os.path.join(x, "transcripts"), os.path.join(x, "raw.jsonl")))
    for src, tdir, rawp in P1:
        rows = {r["episode_id"]: r for r in map(json.loads, filter(str.strip, open(rawp)))}   # last row per id
        n = old_ok = 0
        for f in sorted(glob.glob(os.path.join(tdir, "*.json"))):
            t = json.load(open(f)); n += 1
            if src.startswith("e1_"):
                cfg, tasks = t1[os.path.realpath(os.path.dirname(tdir))]
                task = tasks[t["task_id"]]
                mit, fm, tid, goals, ordered = t["mitigation"], t["failure_mode"], task.task_id, task.goals, task.ordered
                seed = t.get("ack_seed"); eid = t["episode_id"]
            else:
                m = rescore.FNAME.match(os.path.basename(f)); assert m, f
                mit, fm, tid = m["mit"], m["fm"], t["task_id"]
                goals = [Goal(g["tool"], g["canon_args"]) for g in t["score"]["goals"]]
                ordered = t["family"] in ("multi_step", "long_chain"); seed = None
                eid = sweep.episode_id(t.get("model", m["model"]), m["prompt"], mit, fm, m["task"], int(m["s"]))
            mg = merges(values_p1(t))
            if mg:
                screen.append(dict(source=src, episode_id=eid, merged_values=json.dumps(mg)))
            set_norm(OLD_NORM); so, seen_o, led_o = replay_p1(t, mit, fm, tid, goals, ordered, seed)
            set_norm(NEW_NORM); sn, seen_n, led_n = replay_p1(t, mit, fm, tid, goals, ordered, seed)
            assert seen_o == seen_n, f"model-visible results change under the fix: {f}"
            ref = jul_v2[eid] if src == "jul" else t["score"]
            assert all(so[k] == ref[k] for k in SC), f"old replay does not reproduce stored score: {f}"
            if src.startswith("e1_"):
                old_row = rows[eid]
                set_norm(OLD_NORM)
                chk = tier1.episode_row(dict(t, score=dict(t["score"], **so), ledger=led_o), task)
                assert all(chk[k] == old_row[k] for k in old_row if k in chk and k not in KEEP), f"row not reproduced: {f}"
                set_norm(NEW_NORM)
                new_row = dict(old_row, **{k: v for k, v in tier1.episode_row(
                    dict(t, score=dict(t["score"], **sn), ledger=led_n), task).items() if k in old_row and k not in KEEP})
            else:
                old_row = rows.get(eid)
                assert old_row is not None and all(old_row[k] == ref[k] for k in SC), f
                new_row = dict(old_row, **sn)
            old_ok += 1
            if new_row != old_row:
                ch = sorted(k for k in old_row if old_row[k] != new_row[k])
                diffs.append(dict(source=src, episode_id=eid, file=os.path.relpath(f, HERE),
                                  old_violations=old_row["violations"], new_violations=new_row["violations"],
                                  old_achieved=old_row["achieved"], new_achieved=new_row["achieved"],
                                  fields_changed=" ".join(ch),
                                  old_values=json.dumps({k: old_row[k] for k in ch}),
                                  new_values=json.dumps({k: new_row[k] for k in ch})))
                v2_rows.setdefault(rawp, {})[eid] = new_row
        summary[src] = dict(episodes=n, replayed_and_reproduced=old_ok)
        assert n == len(rows), (src, n, len(rows))

    # ---- two-session (the session-resume setting) transcripts
    G = {g.graph_id: (g, k) for g, k in sweep2.benchmark_graphs()}
    P2 = [("jul2", os.path.join(RES, "transcripts2"), os.path.join(RES, "sweep2_raw_v2.jsonl")),
          ("xp2", os.path.join(RES, "crossprovider", "transcripts2"), os.path.join(RES, "crossprovider", "sweep2_raw.jsonl")),
          ("oct2", os.path.join(RES, "claude_batch_2026-10", "transcripts2"),
           os.path.join(RES, "claude_batch_2026-10", "sweep2_raw.jsonl"))]
    for src, tdir, rawp in P2:
        rows = {r["episode_id"]: r for r in map(json.loads, filter(str.strip, open(rawp)))}
        n = 0
        for f in sorted(glob.glob(os.path.join(tdir, "*.json"))):
            t = json.load(open(f)); n += 1
            g, k = G[t["graph_id"]]; assert k == t["interrupt_node"], f
            eid = sweep2.episode_id(t["model"], t["resume_context"], t["mitigation"], t["graph_id"], k, t["sample"])
            vals = [b["input"].get(fl) for s in t["sessions"] for turn in s["turns"] if turn["role"] == "assistant"
                    for b in turn["content"] if b.get("type") == "tool_use" and b["name"] in EFFECTFUL_TOOLS
                    for fl in EFFECTFUL_TOOLS[b["name"]][1] if fl in b["input"]]
            vals += [v for nd in g.nodes for v in nd.goal.canon_args.values()]
            mg = merges(vals)
            if mg:
                screen.append(dict(source=src, episode_id=eid, merged_values=json.dumps(mg)))
            # rescore2.replay asserts every model-visible result (list_effects text,
            # ack form, interrupt position) against the transcript
            set_norm(OLD_NORM); so = rescore2.replay(t, g, k)[0]
            set_norm(NEW_NORM); sn = rescore2.replay(t, g, k)[0]
            r = rows[eid]
            assert so == tuple(r[x] for x in rescore2.KEYS), f
            if sn != so:
                nr = dict(r, **dict(zip(rescore2.KEYS, sn)))
                ch = [x for x in rescore2.KEYS if r[x] != nr[x]]
                diffs.append(dict(source=src, episode_id=eid, file=os.path.relpath(f, HERE),
                                  old_violations=r["violations"], new_violations=nr["violations"],
                                  old_achieved="", new_achieved="", fields_changed=" ".join(ch),
                                  old_values=json.dumps({x: r[x] for x in ch}),
                                  new_values=json.dumps({x: nr[x] for x in ch})))
                v2_rows.setdefault(rawp, {})[eid] = nr
        summary[src] = dict(episodes=n, replayed_and_reproduced=n)
        assert n == len(rows), (src, n, len(rows))
    set_norm(NEW_NORM)

    released = {os.path.join(RES, "sweep_raw_v2.jsonl"), os.path.join(RES, "sweep2_raw_v2.jsonl")}
    assert not (set(v2_rows) & released), "a released July score changed: stop"
    written = []
    for rawp, new in sorted(v2_rows.items()):
        outp = rawp[:-len(".jsonl")] + "_v2.jsonl"
        with open(outp, "w") as fh:
            for line in open(rawp):
                if line.strip():
                    r = json.loads(line)
                    fh.write(json.dumps(new.get(r["episode_id"], r)) + "\n")
        written.append(os.path.relpath(outp, HERE))
    os.makedirs(OUT, exist_ok=True)
    flds = ["source", "episode_id", "file", "old_violations", "new_violations", "old_achieved",
            "new_achieved", "fields_changed", "old_values", "new_values"]
    with open(os.path.join(OUT, "rescore_c1.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds); w.writeheader(); w.writerows(diffs)
    tot = sum(v["episodes"] for v in summary.values())
    res = dict(episodes=tot, by_source=summary, value_merges=screen, episodes_with_value_merge=len(screen),
               rows_changed=len(diffs), score_changed=sum(1 for d in diffs if d["old_violations"] != d["new_violations"]
                                                         or d["old_achieved"] != d["new_achieved"]),
               written=written)
    json.dump(res, open(os.path.join(OUT, "rescore_c1.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "value_merges"}, indent=1))
    for d in diffs:
        print(d["source"], d["episode_id"], d["fields_changed"], d["old_values"], "->", d["new_values"])


if __name__ == "__main__":
    main()
