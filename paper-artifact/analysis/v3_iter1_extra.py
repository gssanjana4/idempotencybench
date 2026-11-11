"""Extra numbers requested by the iteration-1 writing review (added
2026-10-30, after all results were seen). Every entry here is DESCRIPTIVE or
POST HOC: none is a registered test, and none changes a registered verdict.

  python3 analysis/v3_iter1_extra.py     # stdlib only, no network

Reads (read-only) the same raw files and transcripts as the other v3 scripts
and writes results/analysis_v3/iter1_extra.json, which
paper-merged/make_numbers.py loads under the macro prefix "nj:x." (it is not
merged into numbers.json, so the registered outputs stay byte-identical).

Keys (same schema as numbers.json: value, k, n, ci, ...):
  x.amn_month.<m>:d / :p   October minus July amnesiac gIVR, paired over the
      16 graphs (sign-flip permutation, 10,000 perms, seed 20261025; the
      refused Sonnet episode is excluded, as in v3_bare_hint.py)
  x.misorder.<m>           misordered two-step timeout-before episodes, k/n,
      core grid (July Claude, October OpenRouter; last row per id)
  x.misorder.all           the same, summed; :all_toB_twostep says whether
      every misorder in the core grid is a two-step timeout-before episode
  x.misorder.refused_task_two_step  is any Gemini refused task a two-step task
  x.rtext.<m>.e500         July base/none/500 retries: retries, retries after
      visible text, after text naming the key, and those that also reused
      the first call's key (same rule as make_numbers.retry_text)
  x.rb500.<m>:d / :p       read-back after the failure, 500 minus timeout,
      paired over the 32 tasks (post hoc), core cells
  x.e1_4.<m>.<fm>.tmpl     per-template IVR on the 64 new E1-4 tasks (k/8) and
      on the pooled 96 tasks (new + s0 of the released 32; k/12), as the
      registration's "If not" clause asks for the appendix
"""
import glob, json, os, re, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import (OUT, RES, ROOT, Numbers, perm_test, read_jsonl, dedupe, rel, prefer_v2,
                       short_model)
import v3_bare_hint as bh

sys.path.insert(0, os.path.join(ROOT, "src"))
from world import normalize_value, EFFECTFUL_TOOLS   # noqa: E402

N = Numbers()
V2_RAW = os.path.join(RES, "sweep_raw_v2.jsonl")
XP_RAW = prefer_v2(os.path.join(RES, "crossprovider", "sweep_raw.jsonl"))
ORDER = ["haiku", "sonnet", "fable", "gpt", "gemini", "qwen"]
STATUS = "post hoc (iteration-1 review, 2026-10-30); not registered"


# ---------------------------------------------------------------- July vs October amnesiac
def amnesiac_month():
    oct_raw = prefer_v2(os.path.join(bh.OCT, "sweep2_raw.jsonl"))
    oct_tr, jul_tr = bh.transcripts(bh.OCT), bh.transcripts(RES)
    for model, sm in bh.MODELS:
        jul = bh.load(bh.JUL_RAW, jul_tr, model, "amnesiac")
        octo = bh.load(oct_raw, oct_tr, model, "amnesiac")
        r = perm_test(bh.rates(jul), bh.rates(octo))
        src = [rel(bh.JUL_RAW), rel(oct_raw)]
        N.put(f"x.amn_month.{sm}:d", r["d"], n=r["T"], source=src, unit="graph", status=STATUS,
              stat="mean paired per-graph gIVR difference (October - July)", plus=r["plus"],
              minus=r["minus"])
        N.put(f"x.amn_month.{sm}:p", r["p"], n=r["T"], source=src, status=STATUS,
              stat="two-sided paired sign-flip permutation over graphs, 10000 perms, seed 20261025")


# ---------------------------------------------------------------- misorder denominators
def core_rows():
    rows = []
    for path, claude in ((V2_RAW, True), (XP_RAW, False)):
        for r in dedupe(read_jsonl(path)).values():
            if r["model"].startswith("claude") == claude:
                rows.append(r)
    return rows


def misorder():
    rows = core_rows()
    tot_k = tot_n = 0
    every = True
    for m in ORDER:
        mine = [r for r in rows if short_model(r["model"]) == m]
        two = [r for r in mine if r.get("family") == "multi_step"
               and r["failure_mode"] == "timeout_before_commit"]
        k = sum(1 for r in two if r.get("misorder"))
        allmis = sum(1 for r in mine if r.get("misorder"))
        every &= allmis == k
        N.put(f"x.misorder.{m}", k / len(two) if two else None, n=len(two), k=k,
              source=[rel(V2_RAW), rel(XP_RAW)], all_misorders=allmis,
              note="two-step (family multi_step) timeout_before_commit episodes, core grid")
        tot_k += k
        tot_n += len(two)
    N.put("x.misorder.all", tot_k / tot_n, n=tot_n, k=tot_k, source=[rel(V2_RAW), rel(XP_RAW)],
          all_toB_twostep=every)
    refused = set()
    p = os.path.join(OUT, "refusals.csv")
    if os.path.exists(p):
        import csv
        for r in csv.DictReader(open(p)):
            refused |= set(filter(None, r["tasks"].split(";")))
    fam = {r["task_id"]: r.get("family") for r in rows}
    N.put("x.misorder.refused_task_two_step", any(fam.get(t) == "multi_step" for t in refused),
          tasks=sorted(refused), families=sorted({str(fam.get(t)) for t in refused}),
          source=[rel(p)])


# ---------------------------------------------------------------- key-naming retries
KEYTEXT = re.compile(r"idempoten", re.I)


def sid(tool, args):
    return (tool,) + tuple(normalize_value(args.get(f)) for f in EFFECTFUL_TOOLS[tool][1])


def retry_text(t):
    """Same rule as paper-merged/make_numbers.retry_text: visible assistant
    text between the first effectful call and its retry (the first later
    effectful call with the same identity)."""
    seq = []
    for turn in t.get("turns") or []:
        if turn.get("role") != "assistant":
            continue
        for b in turn["content"]:
            if b.get("type") == "tool_use":
                seq.append(("call", b))
            elif b.get("type") == "text":
                seq.append(("text", b.get("text", "")))
    eff = [i for i, (k, b) in enumerate(seq) if k == "call" and b["name"] in EFFECTFUL_TOOLS]
    if len(eff) < 2:
        return None
    strip = lambda b: {k: v for k, v in b["input"].items() if k != "idempotency_key"}
    try:
        s0 = sid(seq[eff[0]][1]["name"], strip(seq[eff[0]][1]))
    except Exception:
        return None
    i1 = None
    for i in eff[1:]:
        try:
            if sid(seq[i][1]["name"], strip(seq[i][1])) == s0:
                i1 = i
                break
        except Exception:
            pass
    if i1 is None:
        return None
    txt = " ".join(x for k, x in seq[eff[0] + 1:i1] if k == "text").strip()
    named = any(KEYTEXT.search(s) for s in re.split(r"(?<=[.!?:])\s+", txt.replace("\n", " ")))
    k0 = seq[eff[0]][1]["input"].get("idempotency_key")
    k1 = seq[i1][1]["input"].get("idempotency_key")
    return bool(txt), named, bool(k0) and k0 == k1


def key_naming():
    d = os.path.join(RES, "sweep_transcripts")
    for m, pat in (("sonnet", "claude-sonnet-5_"), ("fable", "claude-fable-5_")):
        c = defaultdict(int)
        for f in sorted(glob.glob(os.path.join(d, pat + "base_none_error_500_after_commit__s*__*.json"))):
            rt = retry_text(json.load(open(f)))
            if rt is None:
                continue
            c["retries"] += 1
            c["text"] += rt[0]
            c["key_named"] += rt[1]
            c["key_named_same_key"] += rt[1] and rt[2]
            c["same_key"] += rt[2]
        N.put(f"x.rtext.{m}.e500", c["key_named_same_key"], n=c["key_named"], k=c["key_named_same_key"],
              source=[rel(d)], retries=c["retries"], text=c["text"], same_key=c["same_key"],
              note="k = retries whose preceding text names the key AND that reuse the first key; "
                   "n = retries whose preceding text names the key")


# ---------------------------------------------------------------- read-back 500 vs timeout
def readback_contrast():
    import v3_load
    cells, _ = v3_load.load_all()
    for m in ORDER:
        src = "v2" if m in ("haiku", "sonnet", "fable") else "xp"
        arms = {}
        for (s, model, prompt, mit, fm), eps in cells.items():
            if s == src and short_model(model) == m and prompt == "base" and mit == "none" \
                    and fm in ("timeout_after_commit", "error_500_after_commit"):
                arms[fm] = eps
        if len(arms) < 2:
            continue
        rate = lambda eps: {t: sum(v) / len(v) for t, v in _by_task(eps).items()}
        r = perm_test(rate(arms["timeout_after_commit"]), rate(arms["error_500_after_commit"]))
        N.put(f"x.rb500.{m}:d", r["d"], n=r["T"], unit="task", status=STATUS,
              stat="mean paired per-task read-back difference (500 - timeout)",
              plus=r["plus"], minus=r["minus"], source=[src])
        N.put(f"x.rb500.{m}:p", r["p"], n=r["T"], status=STATUS, source=[src],
              stat="two-sided paired sign-flip permutation over tasks, 10000 perms, seed 20261025")


def _by_task(eps):
    d = defaultdict(list)
    for e in eps:
        if e.get("rb") is not None:
            d[e["task"]].append(e["rb"])
    return d


# ---------------------------------------------------------------- E1-4 per template
def template_of(task_id):
    return task_id.rsplit("-", 1)[0]


def e1_4_templates():
    new = dedupe(read_jsonl(prefer_v2(os.path.join(RES, "tier1", "e1_4", "raw.jsonl"))))
    rel_rows = dedupe(read_jsonl(V2_RAW))
    fmn = {"timeout_after_commit": "toA", "error_500_after_commit": "e500"}
    for m in ("haiku", "sonnet"):
        for fm, f in fmn.items():
            nn = [r for r in new.values() if short_model(r["model"]) == m and r["failure_mode"] == fm
                  and r["prompt"] == "base" and r["mitigation"] == "none"]
            s0 = [r for r in rel_rows.values() if short_model(r["model"]) == m and r["failure_mode"] == fm
                  and r["prompt"] == "base" and r["mitigation"] == "none" and r["sample"] == 0]
            for r in nn:   # the template name in the row must match the task id
                assert r.get("template") in (None, template_of(r["task_id"])), r["task_id"]
            out = {}
            for tag, rows in (("new", nn), ("pooled96", nn + s0)):
                by = defaultdict(lambda: [0, 0])
                for r in rows:
                    t = template_of(r["task_id"])
                    by[t][0] += r["violations"] > 0
                    by[t][1] += 1
                out[tag] = {t: f"{k}/{n}" for t, (k, n) in sorted(by.items())}
                out[tag + "_total"] = f"{sum(v[0] for v in by.values())}/{sum(v[1] for v in by.values())}"
            N.put(f"x.e1_4.{m}.{f}.tmpl", out["pooled96_total"], new=out["new"], pooled96=out["pooled96"],
                  new_total=out["new_total"], templates=len(out["new"]),
                  source=[rel(prefer_v2(os.path.join(RES, "tier1", "e1_4", "raw.jsonl"))), rel(V2_RAW)],
                  note="registration E1-4 'If not': per-template IVR goes in the appendix")


def main():
    amnesiac_month()
    misorder()
    key_naming()
    readback_contrast()
    e1_4_templates()
    path = os.path.join(OUT, "iter1_extra.json")
    N.write(path)
    for k, v in sorted(N.d.items()):
        print(k, {a: b for a, b in v.items() if a not in ("source", "stat", "note", "status")})
    print(f"iter1_extra.json: {len(N.d)} values -> {rel(path)}")


if __name__ == "__main__":
    main()
