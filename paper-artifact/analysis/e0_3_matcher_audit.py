"""E0-3: matcher audit on the existing Paper-209 duplicates.

Enumerates EVERY pair of same-tool effectful calls within an episode (all
1,728 released transcripts) and classifies each pair with three matchers:
  strict  : raw canonical args identical as strings (no normalization)
  paper   : the paper's normalization (case/whitespace/quotes/currency/
            thousands separators/trailing zeros); independent re-implementation
  loose   : alphanumeric-only casefold, plus difflib ratio >= 0.8 per field
Pair classes:
  raw_identical        strict match
  normalization_only   paper match but not strict      -> human-check ALL
  near_miss            loose match / similarity>=0.8 but not paper match -> human-check ALL
  distinct             none of the above
Also counts non-canonical fields actually sent, and writes the canonical-
field table.

Hand-check list (results/analysis_v2/e0_3_handcheck.csv) contains every
normalization_only and near_miss pair, plus a fixed-seed random sample of 20
raw_identical pairs as a control (random.Random(20261025)).

Reads : results/sweep_transcripts/*.json, src/world.py (schema table only)
Writes: results/analysis_v2/e0_3_pairs.csv, e0_3_handcheck.csv,
        e0_3_canonical_fields.csv, e0_3_summary.json
Run   : python3 analysis/e0_3_matcher_audit.py
"""
import csv, difflib, glob, json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_1_corrected_table import FNAME, CANON, norm, wilson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDIR = os.path.join(ROOT, "results", "sweep_transcripts")
OUT = os.path.join(ROOT, "results", "analysis_v2")
SEED, N_CONTROL = 20261025, 20


def loose(v):
    return re.sub(r"[^0-9a-z]", "", str(v).casefold())


def classify(tool, a, b):
    fs = CANON[tool]
    raw = all(str(a.get(f)) == str(b.get(f)) for f in fs)
    paper = all(norm(a.get(f)) == norm(b.get(f)) for f in fs)
    sims = [difflib.SequenceMatcher(None, str(a.get(f)), str(b.get(f))).ratio() for f in fs]
    lo = all(loose(a.get(f)) == loose(b.get(f)) for f in fs)
    if raw:
        cls = "raw_identical"
    elif paper:
        cls = "normalization_only"
    elif lo or min(sims) >= 0.8:
        cls = "near_miss"
    else:
        cls = "distinct"
    diff = {f: [a.get(f), b.get(f)] for f in fs if str(a.get(f)) != str(b.get(f))}
    return cls, raw, paper, lo, round(min(sims), 4), diff


def main():
    os.makedirs(OUT, exist_ok=True)
    pairs, n_calls, noncanon = [], 0, []
    for f in sorted(glob.glob(os.path.join(TDIR, "*.json"))):
        m = FNAME.match(os.path.basename(f)); t = json.load(open(f))
        calls = []
        for turn in t["turns"]:
            if turn["role"] != "assistant":
                continue
            for b in turn["content"]:
                if b.get("type") == "tool_use" and b["name"] in CANON:
                    a = dict(b["input"]); a.pop("idempotency_key", None)
                    extra = sorted(set(a) - set(CANON[b["name"]]))
                    if extra:
                        noncanon.append(dict(file=os.path.basename(f), tool=b["name"], extra=extra))
                    calls.append((b["name"], a))
        n_calls += len(calls)
        for i in range(len(calls)):
            for j in range(i + 1, len(calls)):
                if calls[i][0] != calls[j][0]:
                    continue
                cls, raw, paper, lo, sim, diff = classify(calls[i][0], calls[i][1], calls[j][1])
                pairs.append(dict(file=os.path.basename(f), model=t["model"], prompt=m["prompt"],
                                  mitigation=m["mit"], failure_mode=m["fm"], family=t["family"],
                                  tool=calls[i][0], call_i=i, call_j=j, cls=cls,
                                  strict_match=raw, paper_match=paper, loose_match=lo,
                                  min_field_similarity=sim, differing_fields=json.dumps(diff)))
    with open(os.path.join(OUT, "e0_3_pairs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pairs[0].keys())); w.writeheader(); w.writerows(pairs)

    counts = {c: sum(1 for p in pairs if p["cls"] == c)
              for c in ("raw_identical", "normalization_only", "near_miss", "distinct")}
    # what kind of normalization did the work
    reasons = {}
    for p in pairs:
        if p["cls"] != "normalization_only":
            continue
        for fld, (x, y) in json.loads(p["differing_fields"]).items():
            sx, sy = str(x), str(y)
            if sx.casefold() == sy.casefold():
                r = "case only"
            elif " ".join(sx.split()) == " ".join(sy.split()):
                r = "whitespace only"
            elif sx.strip("'\"") == sy.strip("'\""):
                r = "quotes only"
            elif re.sub(r"[$€£,\s]", "", sx) == re.sub(r"[$€£,\s]", "", sy):
                r = "currency symbol / thousands separator"
            else:
                r = "numeric canonical form (e.g. trailing zeros) or combined"
            reasons[f"{p['tool']}.{fld}: {r}"] = reasons.get(f"{p['tool']}.{fld}: {r}", 0) + 1

    rng = random.Random(SEED)
    raw_ids = [p for p in pairs if p["cls"] == "raw_identical"]
    control = sorted(rng.sample(raw_ids, min(N_CONTROL, len(raw_ids))), key=lambda p: (p["file"], p["call_i"]))
    hand = []
    for p in pairs:
        if p["cls"] in ("normalization_only", "near_miss"):
            hand.append(dict(file=p["file"], tool=p["tool"], call_i=p["call_i"], call_j=p["call_j"],
                             reason=f"{p['cls']}: fields differ before normalization {p['differing_fields']}",
                             auto_class=p["cls"], your_verdict_same_action_Y_N=""))
    for p in control:
        hand.append(dict(file=p["file"], tool=p["tool"], call_i=p["call_i"], call_j=p["call_j"],
                         reason="control: random raw-identical pair (seed 20261025)",
                         auto_class=p["cls"], your_verdict_same_action_Y_N=""))
    with open(os.path.join(OUT, "e0_3_handcheck.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(hand[0].keys())); w.writeheader(); w.writerows(hand)

    sys.path.insert(0, os.path.join(ROOT, "src"))
    from world import EFFECTFUL_TOOLS
    import model_agent
    schema = {s["name"]: sorted(s["input_schema"]["properties"]) for s in model_agent._tool_schemas()}
    ctab = [dict(tool=k, domain=v[0], canonical_fields=" ".join(v[1]),
                 schema_properties_exposed=" ".join(schema[k]),
                 non_canonical_exposed=" ".join(sorted(set(schema[k]) - set(v[1]) - {"idempotency_key"})) or "(none)")
            for k, v in EFFECTFUL_TOOLS.items()]
    with open(os.path.join(OUT, "e0_3_canonical_fields.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ctab[0].keys())); w.writeheader(); w.writerows(ctab)

    # sensitivity: IVR per cell if the duplicate matcher were STRICT (raw string
    # equality on canonical fields). Goal matching is left on the paper's
    # normalization so only the duplicate decision changes.
    import e0_1_corrected_table as e01
    paper_sid = e01.sid
    strict_sid = lambda tool, args: (tool,) + tuple(str(args.get(f)) for f in CANON[tool])
    sens = {}
    for f in sorted(glob.glob(os.path.join(TDIR, "*.json"))):
        m = FNAME.match(os.path.basename(f)); t = json.load(open(f))
        e01.sid = paper_sid; vp = e01.score(t, m["mit"], m["fm"])["violations"]
        e01.sid = strict_sid; vs = e01.score(t, m["mit"], m["fm"])["violations"]
        k = f"{t['model']}|{m['prompt']}/{m['mit']}/{m['fm']}"
        d = sens.setdefault(k, dict(n=0, viol_paper=0, viol_strict=0))
        d["n"] += 1; d["viol_paper"] += vp > 0; d["viol_strict"] += vs > 0
    e01.sid = paper_sid
    sens_changed = {k: dict(v, ivr_paper=round(v["viol_paper"] / v["n"], 4),
                            ivr_strict=round(v["viol_strict"] / v["n"], 4))
                    for k, v in sens.items() if v["viol_paper"] != v["viol_strict"]}

    by_model = {}
    for p in pairs:
        d = by_model.setdefault(p["model"], {c: 0 for c in counts}); d[p["cls"]] += 1
    summary = dict(pairs=len(pairs), **counts, by_model=by_model, effectful_calls=n_calls,
                   calls_with_non_canonical_fields=len(noncanon), non_canonical_examples=noncanon[:10],
                   normalization_reasons=reasons,
                   paper_vs_strict_disagreements=counts["normalization_only"],
                   paper_vs_loose_disagreements=sum(1 for p in pairs if p["paper_match"] != p["loose_match"]),
                   strict_matcher_ivr_changes=sens_changed or "none",
                   normalization_only_distinct_tasks=len({p["file"].split("__", 2)[2] for p in pairs
                                                          if p["cls"] == "normalization_only"}),
                   handcheck_rows=len(hand), handcheck_control_seed=SEED,
                   fp_rate_if_all_normalization_only_were_false=dict(
                       k=counts["normalization_only"], n=counts["normalization_only"] + counts["raw_identical"],
                       wilson=[round(x, 4) for x in wilson(counts["normalization_only"],
                                                           counts["normalization_only"] + counts["raw_identical"])]),
                   notes=["False-positive count requires the hand verdicts in e0_3_handcheck.csv; "
                          "not derivable automatically.",
                          "Schemas expose only canonical fields + idempotency_key, so 0 non-canonical args is "
                          "by construction: the current matcher task is easy (confirms qkC4)."])
    json.dump(summary, open(os.path.join(OUT, "e0_3_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "non_canonical_examples"}, indent=1))
    for r in ctab:
        print(r)


if __name__ == "__main__":
    main()
