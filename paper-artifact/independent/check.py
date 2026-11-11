#!/usr/bin/env python3
"""Independent recomputation check for paper-merged/numbers.tex.

Second, independent implementation of every statistic the paper
prints, computed straight from the raw result files (results/**) and
compared with the \\defnum macros in paper-merged/numbers.tex.

Rules (see the task notes in independent/README-less header):
  * nothing is imported from src/, src2/, src_real/, analysis/ or
    paper-merged/; only the Python standard library is used;
  * every statistic is written from its definition in
    analysis/preregistration.md ("Common protocol") and main.tex S3-S4:
    Wilson 95% interval (episode level, and the task-level variant
    Wilson(p*T, T)), the two-sided paired sign-flip permutation test
    (10,000 permutations, random.Random(20261025), one fresh RNG per test,
    per-unit differences visited in sorted unit order, p = (hits+1)/(N+1)),
    its one-sided form for the R2 primaries (registration section R2),
    Holm step-down, half-up rounding to two decimals;
  * where the pipeline encodes a convention the text does not state, the
    convention is named in a comment marked CONVENTION.

Usage:
  python3 independent/check.py [--root REPO] [--report PATH] [--quiet]
Exit status 0 when every recomputed macro matches, 1 otherwise, 2 when
results/ is incomplete (the check cannot run).
"""
import argparse, csv, datetime, glob, json, math, os, random, re, subprocess, sys, time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

# ------------------------------------------------------------------ constants
PERM_N, PERM_SEED = 10000, 20261025
Z = 1.96
MODELS = [("haiku", "haiku", "Haiku 4.5", "Anthropic"), ("sonnet", "sonnet", "Sonnet 5", "Anthropic"),
          ("fable", "fable", "Fable 5", "Anthropic"), ("gpt-5.5", "gpt", "GPT-5.5", "OpenAI"),
          ("gemini", "gemini", "Gemini", "Google"), ("qwen", "qwen", "Qwen3.6-27B", "Alibaba")]
ORDER = [m[1] for m in MODELS]
DISP = {m[1]: m[2] for m in MODELS}
PROVIDER = {m[1]: m[3] for m in MODELS}
CLAUDE = ("haiku", "sonnet", "fable")
FM = {"none": "none", "timeout_after_commit": "toA", "timeout_before_commit": "toB",
      "error_500_after_commit": "e500", "error_500_body_after_commit": "e500body",
      "error_503_retry_after_after_commit": "e503", "status_unknown_after_commit": "unknown",
      "timeout_504_after_commit": "to504"}
# the analysis scripts use w1/w3 for the E1-6 wording variants
FM_NJ = dict(FM, error_500_body_after_commit="w1", status_unknown_after_commit="w3")
MIT = {"none": "none", "idem_keys": "keys", "runtime_receipts": "rcpt"}
# canonical identity fields of the five effectful tools (paper S3, tool schema)
CANON = {"send_email": ("to", "subject"), "send_payment": ("payee", "amount", "memo"),
         "create_ticket": ("queue", "title"), "append_file": ("path", "content_hash"),
         "create_event": ("calendar_id", "title", "start")}
BENCH = {"chain4", "fanout"}
WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
         8: "eight", 9: "nine", 10: "ten"}
EXPECTED_209, EXPECTED_249 = 64, 32
MIN_FRAC = 0.9       # preregistration "Exclusions": report a cell only at >= 90% of planned n

MINE = {}            # key -> my value (plain text)
NOTE = {}            # key -> how it was computed (family tag)
HYG = {}             # hygiene checks (not macros)
WARN = []


def put(key, value, fam):
    MINE[key] = str(value)
    NOTE[key] = fam


def word(n):
    return WORDS.get(int(n), str(n))


# ------------------------------------------------------------------ statistics
def f2(x):
    """Two decimals, half-up (20/32 -> 0.63, as a reader computes it)."""
    return str(Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def wilson(k, n, z=Z):
    """Wilson score interval for k successes in n trials (k may be fractional)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    z2 = z * z
    centre = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z / (1 + z2 / n) * math.sqrt(max(0.0, p * (1 - p) / n + z2 / (4 * n * n)))
    return (max(0.0, centre - half), min(1.0, centre + half))


def wilson_task(p, T):
    """Task-level variant: the proportion applied to T task clusters (rho = 1)."""
    return wilson(p * T, T)


def ci(lo, hi):
    return "[%s, %s]" % (f2(lo), f2(hi))


def pfmt(p):
    return "<0.001" if p < 0.001 else "%.3f" % p


def pe(p):
    return "p<0.001" if p < 0.001 else "p=" + pfmt(p)


def signed(d):
    return "0.00" if f2(abs(d)) == "0.00" else ("+" if d > 0 else "-") + f2(abs(d))


def perm_test(x, y):
    """Two-sided paired sign-flip permutation test. x, y: {unit: rate}.
    Units present in both arms, visited in sorted order; observed statistic
    |sum of differences|; PERM_N sign-flip permutations from a fresh
    random.Random(PERM_SEED); p = (hits + 1) / (PERM_N + 1)."""
    units = sorted(set(x) & set(y))
    d = [y[u] - x[u] for u in units]
    if not d:
        return None
    obs = abs(sum(d))
    rng = random.Random(PERM_SEED)
    hits = 0
    for _ in range(PERM_N):
        s = 0.0
        for v in d:
            s += v if rng.random() < 0.5 else -v
        if abs(s) >= obs - 1e-12:
            hits += 1
    return {"d": sum(d) / len(d), "p": (hits + 1) / (PERM_N + 1), "T": len(units),
            "plus": sum(1 for v in d if v > 0), "minus": sum(1 for v in d if v < 0), "diffs": d}


def holm(ps):
    """Holm step-down adjusted p-values (monotone), input order."""
    m = len(ps)
    idx = sorted(range(m), key=lambda i: ps[i])
    out = [0.0] * m
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        out[i] = run
    return out


def rates(pairs):
    """{unit: mean value} from [(unit, value)]."""
    d = defaultdict(list)
    for u, v in pairs:
        if v is not None:
            d[u].append(float(v))
    return {u: sum(v) / len(v) for u, v in d.items()}


# ------------------------------------------------------------------ normalization (paper S3)
_NUM = re.compile(r"^[$€£]?\s*-?[\d,]+(\.\d+)?$")
_ISO0 = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}):00$", re.I)


def norm(v):
    """The paper's semantic normalization: trim, collapse whitespace, strip
    surrounding quotes, zero seconds of an ISO date-time dropped, numbers in
    canonical form (currency symbol and thousands separators removed,
    trailing zeros dropped), case folded."""
    s = " ".join(str(v).split()).strip("'\"").strip()
    s = _ISO0.sub(r"\1", s)
    if _NUM.match(s):
        try:
            return format(Decimal(s.lstrip("$€£ ").replace(",", "")).normalize(), "f")
        except InvalidOperation:
            pass
    return s.casefold()


def sid(tool, args):
    return (tool,) + tuple(norm(args.get(f)) for f in CANON[tool])


def strict_sid(tool, args):
    return (tool,) + tuple(str(args.get(f)) for f in CANON[tool])


def ledger_violations(ledger, strict=False):
    c = Counter((strict_sid if strict else sid)(e["tool"], e["args"]) for e in ledger
                if e["tool"] in CANON)
    return sum(v - 1 for v in c.values() if v > 1)


# ------------------------------------------------------------------ io
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


def last_by_id(rows):
    """CONVENTION: one row per episode_id, the LAST row in file order wins
    (its transcript is the one on disk; fixed before outcomes were compared)."""
    d = {}
    for r in rows:
        d[r["episode_id"]] = r
    return d


def first_by_id(rows):
    d = {}
    for r in rows:
        d.setdefault(r["episode_id"], r)
    return d


def prefer_v2(path):
    """x.jsonl -> x_v2.jsonl when it exists (C1 re-score; original kept)."""
    root, ext = os.path.splitext(path)
    p2 = root + "_v2" + ext
    return p2 if not root.endswith("_v2") and os.path.exists(p2) else path


def short_model(model):
    m = model.lower()
    for sub, key, _, _ in MODELS:
        if sub in m:
            return key
    return re.sub(r"[^a-z0-9]+", "-", m)


def names(ms):
    ns = [DISP[m] for m in ORDER if m in ms]
    return "none" if not ns else ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


# ------------------------------------------------------------------ transcript parsing
def flat_turns(t):
    if "turns" in t and t["turns"] is not None:
        return list(t["turns"])
    if "sessions" in t:
        return [x for s in t["sessions"] for x in s["turns"]]
    if "deliveries" in t:
        return [x for d in t["deliveries"] for x in d.get("turns", [])]
    return []


def seq_of(turns, invalid=()):
    """Ordered assistant events: ('text', str) and ('call', dict) where the
    call dict has name, args (without key), key, id, is_error, invalid."""
    results = {}
    for turn in turns:
        if turn.get("role") == "tool_results":
            for b in turn["content"]:
                if b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = b
    seq = []
    for turn in turns:
        if turn.get("role") != "assistant":
            continue
        for b in turn["content"]:
            if b.get("type") == "text":
                seq.append(("text", b.get("text", "")))
            elif b.get("type") == "tool_use":
                a = dict(b.get("input") or {})
                k = a.pop("idempotency_key", "") or ""
                if not isinstance(k, str):
                    k = json.dumps(k)
                r = results.get(b.get("id"), {})
                seq.append(("call", {"name": b["name"], "args": a, "key": k, "id": b.get("id"),
                                     "is_error": bool(r.get("is_error")),
                                     "invalid": b.get("id") in invalid}))
    return seq


def key_cat(k0, k1):
    return ("same_key" if k0 == k1 else "new_key") if (k0 and k1) else \
        "dropped_key" if k0 else "added_key" if k1 else "keyless"


KEYTEXT = re.compile(r"idempoten", re.I)


def features(t):
    """Per-episode transcript features.
      rb_first:  list_effects between the FIRST effectful call and the next
                 effectful call (the paper's read-back-after-the-failure for
                 cells whose failure fires on call 1)
      rb_fail:   list_effects after the first effectful call whose result was
                 an error, before the next valid effectful call (runner rule,
                 used for cells whose failure fires on call k)
      retried:   a later effectful call with the same semantic identity as
                 the first; cat: key category of that retry
      gaveup:    neither retried nor read back, and the closing text has no DONE
      rtext:     (has visible text before the retry, text names the key,
                 retry reused the first key) or None when there is no retry
    """
    invalid = {c.get("tool_use_id") for c in (t.get("invalid_calls") or []) if isinstance(c, dict)}
    seq = seq_of(flat_turns(t), invalid)
    calls = [(i, e) for i, (k, e) in enumerate(seq) if k == "call"]
    eff = [(i, e) for i, e in calls if e["name"] in CANON]
    if not eff:
        return None
    i0, c0 = eff[0]
    s0 = sid(c0["name"], c0["args"])
    later = [(i, e) for i, e in eff if i > i0]
    nxt = later[0][0] if later else len(seq)
    rb_first = any(e["name"] == "list_effects" for i, e in calls if i0 < i < nxt)
    retry = None
    rb_before_retry = False
    for i, e in later:
        if sid(e["name"], e["args"]) == s0:
            retry = (i, e)
            rb_before_retry = any(x["name"] == "list_effects" for j, x in calls if i0 < j < i)
            break
    texts = [x for k, x in seq if k == "text"] if "turns" in t and t["turns"] is not None else []
    said_done = bool(texts) and "DONE" in texts[-1]
    rtext = None
    if retry:
        txt = " ".join(x for k, x in seq[i0 + 1:retry[0]] if k == "text").strip()
        sent = ""
        for s in re.split(r"(?<=[.!?:])\s+", txt.replace("\n", " ")):
            if KEYTEXT.search(s):
                sent = s.strip()
                break
        rtext = (bool(txt), bool(sent), bool(c0["key"]) and c0["key"] == retry[1]["key"], sent)
    # runner rule: first VALID effectful call whose result was an error
    veff = [(i, e) for i, e in eff if not e["invalid"]]
    fi = next((n for n, (i, e) in enumerate(veff) if e["is_error"]), None)
    rb_fail = retried_fail = None
    cat_fail = ""
    if fi is not None:
        pi, pc = veff[fi]
        sf = sid(pc["name"], pc["args"])
        rb_fail = False
        for i, e in calls:
            if i <= pi:
                continue
            if e["name"] == "list_effects":
                rb_fail = True
                break
            if e["name"] in CANON and not e["invalid"]:
                break
        rf = next((e for i, e in veff[fi + 1:] if sid(e["name"], e["args"]) == sf), None)
        retried_fail = rf is not None
        cat_fail = key_cat(pc["key"], rf["key"]) if rf else ""
    return {"rb_first": rb_first, "key_first": bool(c0["key"]), "retried": retry is not None,
            "cat": key_cat(c0["key"], retry[1]["key"]) if retry else "",
            "rb_before_retry": rb_before_retry,
            "gaveup": retry is None and not rb_first and not said_done,
            "rtext": rtext, "rb_fail": rb_fail, "retried_fail": retried_fail, "cat_fail": cat_fail,
            "n_valid_eff": len(veff), "key_first_valid": bool(veff and veff[0][1]["key"])}


def refused(t):
    return any(turn.get("stop_reason") == "refusal" for turn in flat_turns(t)
               if turn.get("role") == "assistant")


# ------------------------------------------------------------------ data discovery
class Data:
    def __init__(self, root):
        self.root = root
        self.res = os.path.join(root, "results")

    def rel(self, p):
        return os.path.relpath(p, self.root)

    def batch_of(self, path):
        """CONVENTION (pipeline naming): jul = results root, xp = crossprovider,
        oct = claude_batch_2026-10, Tier-1 e1_0..e1_6 -> e10, e13 (E1-1 and
        E1-3 share the prefix), e12, e14, e15, e16; anything else is the
        sanitized directory name (e.g. tier1-e1-6-oct)."""
        rel = os.path.relpath(os.path.dirname(path), self.res)
        rel = "" if rel == "." else rel
        names_ = {"": "jul", "crossprovider": "xp", "claude_batch_2026-10": "oct",
                  "tier1/e1_0": "e10", "tier1/e1_1": "e13", "tier1/e1_2": "e12", "tier1/e1_3": "e13",
                  "tier1/e1_4": "e14", "tier1/e1_5": "e15", "tier1/e1_6": "e16"}
        return names_.get(rel, re.sub(r"[^A-Za-z0-9]+", "-", rel))

    def raw_files(self, root_name, pattern, want_249):
        files = [os.path.join(self.res, root_name)]
        cands = glob.glob(os.path.join(self.res, "**", pattern), recursive=True) + \
            [prefer_v2(p) for p in glob.glob(os.path.join(self.res, "tier1", "e1_*", "raw.jsonl"))]
        for p in sorted(set(cands)):
            rel = os.path.relpath(p, self.res)
            parts = rel.split(os.sep)
            if len(parts) == 1 or "smoke" in parts or p.endswith(".bak") or "analysis_v2" in parts:
                continue
            if prefer_v2(p) != p:
                continue
            if os.path.basename(p) in ("raw.jsonl", "raw_v2.jsonl"):
                try:
                    first = json.loads(open(p).readline())
                except Exception:
                    continue
                if ("resume_context" in first) != want_249:
                    continue
            files.append(p)
        return [f for f in files if os.path.exists(f)]

    def complete(self):
        need = ["sweep_raw_v2.jsonl", "sweep2_raw_v2.jsonl", "sweep_transcripts", "transcripts2",
                "crossprovider/sweep_raw.jsonl", "crossprovider/sweep2_raw.jsonl",
                "claude_batch_2026-10/sweep2_raw.jsonl", "tier1/e1_0/raw.jsonl", "tier1/e1_6/raw.jsonl",
                "real_stripe/raw.jsonl", "real_stripe/stripe_ground_truth.jsonl", "summary_by_failure.csv",
                "graph_summary.csv", "analysis_v2/e0_5_sampling_protocol.csv"]
        missing = [n for n in need if not os.path.exists(os.path.join(self.res, n))]
        if not os.path.exists(os.path.join(self.root, "paper-merged", "numbers.tex")):
            missing.append("paper-merged/numbers.tex")
        return missing


FNAME209 = re.compile(r"^(?P<model>.+?)_(?P<prompt>base|careful|informed)_"
                      r"(?P<mit>none|idem_keys|runtime_receipts)_(?P<fm>.+?)__s(?P<s>\d+)__(?P<task>.+)\.json$")


def tags209(r):
    """CONVENTION: a cell key carries retry_mode / task_set / schema only when
    they differ from the defaults (agent_retry, base-or-core, lean)."""
    out = []
    for fld, default in (("retry_mode", "agent_retry"), ("task_set", "base"), ("schema", "lean")):
        v = r.get(fld)
        if v and v != default and not (fld == "task_set" and v == "core"):
            out.append(str(v))
    return tuple(out)


def load_209(D):
    """{(batch, model, prompt, mit, fm, tags): {eid: row}} plus every row per id (dups)."""
    cells = defaultdict(dict)
    allrows = {}
    for path in D.raw_files("sweep_raw_v2.jsonl", "sweep_raw*.jsonl", False):
        b = D.batch_of(path)
        rows = read_jsonl(path)
        per = defaultdict(list)
        for r in rows:
            per[r["episode_id"]].append(r)
        allrows[b] = per
        for eid, rs in per.items():
            r = rs[-1]
            cells[(b, r["model"], r["prompt"], r["mitigation"], r["failure_mode"], tags209(r))][eid] = r
    return cells, allrows


def load_249(D):
    cells = defaultdict(dict)
    for path in D.raw_files("sweep2_raw_v2.jsonl", "sweep2_raw*.jsonl", True):
        b = D.batch_of(path)
        for eid, r in last_by_id(read_jsonl(path)).items():
            cells[(b, r["model"], r["resume_context"], r["mitigation"])][eid] = r
    return cells


def load_trans209(D):
    out = {}
    dirs = [os.path.join(D.res, "sweep_transcripts")] + \
        [d for d in glob.glob(os.path.join(D.res, "**", "sweep_transcripts"), recursive=True)
         if "smoke" not in d and os.path.dirname(d) != D.res] + \
        glob.glob(os.path.join(D.res, "tier1", "e1_*", "transcripts"))
    for d in dirs:
        b = D.batch_of(d)
        for f in glob.glob(os.path.join(d, "*.json")):
            m = FNAME209.match(os.path.basename(f))
            if not m:
                continue
            t = json.load(open(f))
            t["_file"] = os.path.basename(f)
            t["_path"] = f
            out[(b, t.get("model", m["model"]), m["prompt"], m["mit"], m["fm"], m["task"], int(m["s"]))] = t
    return out


def load_trans249(D):
    out = {}
    dirs = [os.path.join(D.res, "transcripts2")] + \
        [d for d in glob.glob(os.path.join(D.res, "**", "transcripts2"), recursive=True)
         if "smoke" not in d and os.path.dirname(d) != D.res]
    for d in dirs:
        b = D.batch_of(d)
        for f in glob.glob(os.path.join(d, "*.json")):
            t = json.load(open(f))
            t["_file"] = os.path.basename(f)
            out[(b, t["model"], t["resume_context"], t["mitigation"], t["graph_id"],
                 t["interrupt_node"], t["sample"])] = t
    return out


def t209(trans, b, r):
    return trans.get((b, r["model"], r["prompt"], r["mitigation"], r["failure_mode"], r["task_id"], r["sample"]))


def t249(trans, b, r):
    return trans.get((b, r["model"], r["resume_context"], r["mitigation"], r["graph_id"],
                      r["interrupt_node"], r["sample"]))


# ------------------------------------------------------------------ 209 cells
EPI = {}        # qual -> [(task, violated, rb_first or None)]
STALE_LEDGER = Counter()   # qual -> episodes whose stored ledger disagrees with the v2 row
FEAT = {}       # qual -> {eid: features}
COMPLETE209 = {}  # qual -> (batch, model, prompt, mit, fm, tags, rows)


def min_n(expected):
    return math.ceil(MIN_FRAC * expected)


def emit_209(D, cells, trans):
    fam = "209 cell (raw rows + transcripts)"
    seen = defaultdict(list)
    ledger_mismatch = ledger_checked = 0
    stale_cells = Counter()
    haiku_kf = [0, 0]
    for (b, model, prompt, mit, fm, tags), eps in sorted(cells.items()):
        ms = short_model(model)
        base = ":".join([ms, prompt, MIT.get(mit, mit), FM.get(fm, fm)] + list(tags))
        qual = b + ":" + base
        seen[base].append(qual)
        rows = list(eps.values())
        n = len(rows)
        exp = 32 if b == "e15" else EXPECTED_209
        if n < min_n(exp):
            continue
        COMPLETE209[qual] = (b, model, prompt, mit, fm, tags, rows)
        k = sum(1 for r in rows if r["violations"] > 0)
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_id"]].append(1 if r["violations"] > 0 else 0)
        T = len(by_task)
        p = k / n
        put(qual + ":n", n, fam); put(qual + ":T", T, fam); put(qual + ":k", k, fam)
        put(qual + ":kn", "%d/%d" % (k, n), fam); put(qual + ":ivr", f2(p), fam)
        put(qual + ":ci", ci(*wilson(k, n)), fam); put(qual + ":tci", ci(*wilson_task(p, T)), fam)
        tany = sum(1 for v in by_task.values() if any(v))
        put(qual + ":tvr", f2(tany / T), fam); put(qual + ":tvrci", ci(*wilson(tany, T)), fam)
        put(qual + ":succ", f2(sum(1 for r in rows if r["achieved"]) / n), fam)
        put(qual + ":mis", sum(1 for r in rows if r.get("misorder")), fam)
        depth = sum(r["violations"] for r in rows) / n
        put(qual + ":depth", f2(depth), fam)
        put(qual + ":depth_eq_ivr", "yes" if abs(depth - p) < 1e-9 else "no", fam)
        put(qual + ":trunc", sum(1 for r in rows if r.get("truncated")), fam)
        put(qual + ":invalid", sum(int(r.get("invalid_calls") or 0) for r in rows), fam)
        fc = Counter(r["family"] for r in rows if r["violations"] > 0)
        for f_ in ("atomic", "multi_step", "read_then_act", "ambiguous"):
            put(qual + ":fam:" + f_, fc.get(f_, 0), fam)
        put(qual + ":fams_violating", len(fc), fam)
        feats = {}
        epi = []
        rt = [0, 0, 0]
        ex = None
        gave = [0, 0]
        for r in rows:
            t = t209(trans, b, r)
            v = 1 if r["violations"] > 0 else 0
            if t is None:
                epi.append((r["task_id"], v, None))
                continue
            if "ledger" in t:
                ledger_checked += 1
                if ledger_violations(t["ledger"]) != r["violations"]:
                    ledger_mismatch += 1
                    stale_cells[qual] += 1
            fe = features(t)
            epi.append((r["task_id"], v, (1 if fe["rb_first"] else 0) if fe else None))
            if ms == "haiku" and b == "jul":
                haiku_kf[1] += 1
                haiku_kf[0] += bool(fe and fe["key_first"])
            if fe:
                feats[r["episode_id"]] = fe
                if fe["gaveup"]:
                    gave[1] += 1
                    gave[0] += bool(r["achieved"])
                if fe["rtext"] is not None:
                    rt[0] += 1
                    rt[1] += fe["rtext"][0]
                    rt[2] += fe["rtext"][1]
                    if fe["rtext"][1] and fe["rtext"][2] and v:
                        cand = (t["_file"], fe["rtext"][3])
                        if ex is None or cand[0] < ex[0]:
                            ex = cand
        EPI[qual] = epi
        FEAT[qual] = feats
        put(qual + ":rtext:retries", rt[0], fam); put(qual + ":rtext:any", rt[1], fam)
        put(qual + ":rtext:key", rt[2], fam); put(qual + ":rtext:none", rt[0] - rt[1], fam)
        put(qual + ":gaveup_succ", "%d/%d" % tuple(gave), fam)
        if ex:
            put(qual + ":rtext:ex_file", ex[0], "string (transcript example)")
            put(qual + ":rtext:ex", ex[1], "string (transcript example)")
        # CONVENTION: transcript features are emitted only when every episode
        # has a transcript, and not for E1-5 (failure on call k > 1)
        if len(feats) == n and b != "e15":
            fs = list(feats.values())
            rb = sum(f["rb_first"] for f in fs)
            put(qual + ":rb", "%d/%d" % (rb, n), fam); put(qual + ":rbrate", f2(rb / n), fam)
            put(qual + ":keyfirst", "%d/%d" % (sum(f["key_first"] for f in fs), n), fam)
            ret = [f for f in fs if f["retried"]]
            put(qual + ":retried", len(ret), fam)
            put(qual + ":rbretry", sum(1 for f in fs if f["rb_first"] and f["retried"]), fam)
            put(qual + ":norb_retry", sum(1 for f in ret if not f["rb_first"]), fam)
            put(qual + ":rb_before_retry", sum(1 for f in ret if f["rb_before_retry"]), fam)
            put(qual + ":gaveup", sum(1 for f in fs if f["gaveup"]), fam)
            for c in ("same_key", "new_key", "dropped_key", "added_key", "keyless"):
                put(qual + ":" + c, sum(1 for f in ret if f["cat"] == c), fam)
            put(qual + ":reuse", f2(sum(1 for f in ret if f["cat"] == "same_key") / len(ret)) if ret else "--", fam)
    # unqualified aliases when exactly one batch has the cell
    for base, quals in seen.items():
        if len(quals) == 1:
            q = quals[0]
            for key in [x for x in list(MINE) if x.startswith(q + ":")]:
                MINE[base + key[len(q):]] = MINE[key]
                NOTE[base + key[len(q):]] = NOTE[key]
    if haiku_kf[1]:
        put("haiku:keyfirst:all", "%d/%d" % tuple(haiku_kf), fam)
    g = MINE.get("jul:haiku:base:none:e500:gaveup_succ", "0/0").split("/")
    put("jul:haiku:base:none:e500:gaveup_allsucc", "yes" if g[0] == g[1] else "no", fam)
    # CONVENTION: the July transcripts of the corrected key cell keep the
    # pre-fix (v1) ledger; the v2 raw row holds the re-scored value
    HYG["ledger_vs_row_violations_209"] = {"checked": ledger_checked, "mismatch": ledger_mismatch,
                                           "cells_with_stale_ledger": dict(stale_cells)}
    STALE_LEDGER.update(stale_cells)


# ------------------------------------------------------------------ 249 cells
EPI249 = {}   # qual -> {(graph, k, sample): givr_hit}
CELL249 = {}  # qual -> (batch, rows)


def lookups(rows, b, trans):
    resumed = looked = looked_dup = 0
    for r in rows:
        t = t249(trans, b, r)
        if t is None or len(t["sessions"]) < 2:
            continue
        resumed += 1
        names_ = [x["name"] for turn in t["sessions"][1]["turns"] if turn.get("role") == "assistant"
                  for x in turn["content"] if x.get("type") == "tool_use"]
        if "list_effects" in names_:
            looked += 1
            looked_dup += r["givr_hit"] > 0
    return looked, resumed, looked_dup


def emit_249_cell(qual, rows, b, trans, expected):
    fam = "249 cell (raw rows + transcripts)"
    n = len(rows)
    if n < min_n(expected):
        return
    k = sum(r["givr_hit"] for r in rows)
    G = len({r["graph_id"] for r in rows})
    p = k / n
    EPI249[qual] = {(r["graph_id"], r["interrupt_node"], r["sample"]): r["givr_hit"] for r in rows}
    CELL249[qual] = (b, rows)
    st = sum(1 for r in rows if r.get("order_violation"))
    put(qual + ":n", n, fam); put(qual + ":G", G, fam); put(qual + ":k", k, fam)
    put(qual + ":kn", "%d/%d" % (k, n), fam); put(qual + ":givr", f2(p), fam)
    put(qual + ":ci", ci(*wilson(k, n)), fam); put(qual + ":gci", ci(*wilson_task(p, G)), fam)
    put(qual + ":stale", f2(st / n), fam); put(qual + ":stalekn", "%d/%d" % (st, n), fam)
    put(qual + ":stale_nodup", sum(1 for r in rows if r.get("order_violation") and not r["givr_hit"]), fam)
    pos = [r for r in rows if r["interrupt_node"] > 0]
    put(qual + ":stale_k0", sum(1 for r in rows if r["interrupt_node"] == 0 and r.get("order_violation")), fam)
    put(qual + ":stale_kpos", "%d/%d" % (sum(1 for r in pos if r.get("order_violation")), len(pos)), fam)
    comp = sum(r["completion"] for r in rows)
    put(qual + ":comp", f2(comp / n), fam); put(qual + ":compkn", "%d/%d" % (comp, n), fam)
    viol = [r["violations"] for r in rows]
    put(qual + ":meanviol", "%.1f" % (sum(viol) / n), fam); put(qual + ":maxviol", max(viol), fam)
    put(qual + ":trunc", sum(1 for r in rows if r.get("truncated")), fam)
    put(qual + ":leaked", sum(1 for r in rows if r.get("notes_leaked")), fam)
    lk, rs, lkd = lookups(rows, b, trans)
    put(qual + ":chk", "%d/%d" % (lk, rs), fam); put(qual + ":resumed", rs, fam)
    put(qual + ":chkdup", "%d/%d" % (lkd, k), fam); put(qual + ":notresumed", n - rs, fam)
    put(qual + ":dupG", len({r["graph_id"] for r in rows if r["givr_hit"]}), fam)
    ups = set()
    for r in rows:
        if not r["givr_hit"]:
            continue
        t = t249(trans, b, r)
        for turn in flat_turns(t or {}):
            up = (turn.get("provider_meta") or {}).get("upstream_provider")
            if up:
                ups.add(up)
    if ups:
        put(qual + ":dup_upstream", ", ".join(sorted(ups)), "string"); put(qual + ":dup_upstream_n", len(ups), fam)


def expected_249(b, ctx):
    # CONVENTION: the October plan_only rerun covers fan-out only (16) and the
    # July Fable bare_hint ablation ran one sample (16)
    return {("oct", "plan_only"): 16, ("jul", "bare_hint"): 16}.get((b, ctx), EXPECTED_249)


def emit_249(D, cells, trans):
    fam = "249 derived"
    seen = defaultdict(list)
    for (b, model, ctx, mit), eps in sorted(cells.items()):
        groups = defaultdict(list)
        for r in eps.values():
            groups["bench" if r["topology"] in BENCH else r["topology"]].append(r)
        for tag, rs in groups.items():
            base = ":".join([short_model(model), ctx, MIT.get(mit, mit)] + ([] if tag == "bench" else [tag]))
            qual = b + ":" + base
            seen[base].append(qual)
            emit_249_cell(qual, rs, b, trans, expected_249(b, ctx) if tag == "bench" else EXPECTED_249)
    for base, quals in seen.items():
        if len(quals) == 1:
            q = quals[0]
            for key in [x for x in list(MINE) if x.startswith(q + ":")]:
                MINE[base + key[len(q):]] = MINE[key]
                NOTE[base + key[len(q):]] = NOTE[key]
    # corrected plan_only: July chain4 + October fan-out rerun
    for (b, model, ctx, mit), eps in cells.items():
        if (b, ctx, mit) != ("oct", "plan_only", "none"):
            continue
        ms = short_model(model)
        jul = cells.get(("jul", model, "plan_only", "none"), {})
        ch = [r for r in jul.values() if r["topology"] == "chain4"]
        fan = [r for r in eps.values() if r["topology"] == "fanout"]
        if len(ch) + len(fan) < EXPECTED_249:
            continue
        rows = ch + fan
        k = sum(r["givr_hit"] for r in rows); n = len(rows); G = len({r["graph_id"] for r in rows})
        q = "pocorr:%s:plan_only:none" % ms
        l1, r1, d1 = lookups(ch, "jul", trans); l2, r2, d2 = lookups(fan, "oct", trans)
        put(q + ":chkdup", "%d/%d" % (d1 + d2, k), fam); put(q + ":n", n, fam); put(q + ":G", G, fam)
        put(q + ":kn", "%d/%d" % (k, n), fam); put(q + ":givr", f2(k / n), fam)
        put(q + ":ci", ci(*wilson(k, n)), fam); put(q + ":gci", ci(*wilson_task(k / n, G)), fam)
        put(q + ":stale", f2(sum(1 for r in rows if r.get("order_violation")) / n), fam)
        put(q + ":comp", f2(sum(r["completion"] for r in rows) / n), fam)
        put(q + ":chk", "%d/%d" % (l1 + l2, r1 + r2), fam)
        put(q + ":chain_kn", "%d/%d" % (sum(r["givr_hit"] for r in ch), len(ch)), fam)
        put(q + ":fan_kn", "%d/%d" % (sum(r["givr_hit"] for r in fan), len(fan)), fam)
        clean = [r for r in jul.values() if not r.get("notes_leaked")]
        leak = [r for r in jul.values() if r.get("notes_leaked")]
        cf = [r for r in clean if r["topology"] == "fanout"]
        put("pojulfanclean:%s:kn" % ms, "%d/%d" % (sum(r["givr_hit"] for r in cf), len(cf)), fam)
        if clean:
            kc = sum(r["givr_hit"] for r in clean)
            put("poclean:%s:kn" % ms, "%d/%d" % (kc, len(clean)), fam)
            put("poclean:%s:givr" % ms, f2(kc / len(clean)), fam)
            put("poclean:%s:ci" % ms, ci(*wilson(kc, len(clean))), fam)
            put("poleak:%s:kn" % ms, "%d/%d" % (sum(r["givr_hit"] for r in leak), len(leak)), fam)
            put("pojul:%s:kn" % ms, "%d/%d" % (sum(r["givr_hit"] for r in jul.values()), len(jul)), fam)
    core = {"jul", "xp", "oct"}
    cr = [r for (b, *_), eps in cells.items() if b in core for r in eps.values()]
    put("p2:model_topologies", len({r["topology"] for r in cr}), fam)
    put("p2:model_contexts", len({r["resume_context"] for r in cr}), fam)
    jr = [r for (b, *_), eps in cells.items() if b == "jul" for r in eps.values()]
    if jr:
        put("p2:graphs_used", len({r["graph_id"] for r in jr}), fam)
        for tp in sorted({r["topology"] for r in jr}):
            put("p2:graphs:" + tp, len({r["graph_id"] for r in jr if r["topology"] == tp}), fam)
        put("p2:nodes", "--".join(str(x) for x in sorted({r["n_nodes"] for r in jr})), fam)
        put("p2:interrupt_nodes", len({r["interrupt_node"] for r in jr}), fam)
        put("p2:k_max", max(int(r["interrupt_node"]) for r in jr), fam)
        put("p2:timing_after_only", "yes" if {r.get("interrupt_timing") for r in jr} == {"after_commit"} else "no", fam)
        put("p2:jul_episodes", len(jr), fam)
    pool = [r for (b, model, ctx, mit), eps in cells.items() if b == "oct" and ctx == "bare_hint"
            and mit == "none" and short_model(model) in CLAUDE for r in eps.values()]
    if len(pool) >= 3 * EXPECTED_249:
        k = sum(r["givr_hit"] for r in pool); G = len({(r["model"], r["graph_id"]) for r in pool})
        put("pool:bare_hint:claude:kn", "%d/%d" % (k, len(pool)), fam)
        put("pool:bare_hint:claude:G", G, fam)
        put("pool:bare_hint:claude:gci", ci(*wilson_task(k / len(pool), G)), fam)
    pool = [r for (b, model, ctx, mit), eps in cells.items() if b == "oct" and ctx == "amnesiac"
            and mit == "none" and short_model(model) in CLAUDE for r in eps.values()]
    if len(pool) >= 3 * EXPECTED_249:
        k = sum(r["givr_hit"] for r in pool); G = len({(r["model"], r["graph_id"]) for r in pool})
        put("pool:amnesiac:oct:kn", "%d/%d" % (k, len(pool)), fam)
        put("pool:amnesiac:oct:gci", ci(*wilson_task(k / len(pool), G)), fam)
    am = []
    for (b, model, ctx, mit), eps in cells.items():
        if ctx == "amnesiac" and mit == "none" and len(eps) >= min_n(EXPECTED_249):
            rs = [r for r in eps.values() if r["topology"] in BENCH]
            am.append(sum(r["givr_hit"] for r in rs) / len(rs))
    if am:
        put("range:amnesiac:min", f2(min(am)), fam); put("range:amnesiac:max", f2(max(am)), fam)
        put("range:amnesiac:nmodels", len(am), fam); put("range:amnesiac:n_at_1", sum(1 for x in am if x == 1.0), fam)
    best = {}
    for (b, model, ctx, mit), eps in cells.items():
        if ctx != "bare_hint" or mit != "none":
            continue
        rs = [r for r in eps.values() if r["topology"] in BENCH]
        if len(rs) >= expected_249(b, ctx) and len(rs) > len(best.get(short_model(model), [])):
            best[short_model(model)] = rs
    if best:
        vals = {m: sum(r["givr_hit"] for r in rs) / len(rs) for m, rs in best.items()}
        put("range:bare_hint:min", f2(min(vals.values())), fam); put("range:bare_hint:max", f2(max(vals.values())), fam)
        put("range:bare_hint:nmodels", len(vals), fam); put("range:bare_hint:n_at_0", sum(1 for v in vals.values() if v == 0), fam)
        put("range:bare_hint:episodes", sum(len(rs) for rs in best.values()), fam)
    dupc = []
    for (b, model, ctx, mit), eps in cells.items():
        rs = list(eps.values())
        if len(rs) >= expected_249(b, ctx) and any(r["givr_hit"] for r in rs):
            dupc.append(sum(r["completion"] for r in rs) / len(rs))
    if dupc:
        put("all:comp_min_duplicating", f2(min(dupc)), fam)
    per = defaultdict(lambda: [0, 0, 0])
    for (b, model, ctx, mit), eps in cells.items():
        if len(eps) < expected_249(b, ctx):
            continue
        for r in eps.values():
            for x in (b, "all"):
                per[x][2] += 1
                if r.get("order_violation"):
                    per[x][0] += 1
                    per[x][1] += r["givr_hit"] > 0
    for b, (ev, co, n) in per.items():
        put("stale:%s:events" % b, ev, fam); put("stale:%s:cooccur" % b, co, fam)
        put("stale:%s:episodes" % b, "{:,}".format(n), fam); put("stale:%s:without_dup" % b, ev - co, fam)


CORE = {"jul", "xp", "oct"}


def emit_summaries(cells209, trans209):
    fam = "209 summary"
    jul = {k: v for k, v in cells209.items() if k[0] == "jul"}
    put("jul:cells", len(jul), fam)
    put("jul:cells_succ1", sum(1 for v in jul.values() if all(r["achieved"] for r in v.values())), fam)
    put("jul:episodes", "{:,}".format(sum(len(v) for v in jul.values())), fam)
    put("jul:misorder_total", sum(1 for v in jul.values() for r in v.values() if r.get("misorder")), fam)
    tasks = {r["task_id"] for v in jul.values() for r in v.values()}
    fams = {r["family"] for v in jul.values() for r in v.values()}
    templ = {r["task_id"].rsplit("-", 1)[0] for v in jul.values() for r in v.values()}
    put("p1:tasks_used", len(tasks), fam); put("p1:families", len(fams), fam)
    put("p1:templates", len(templ), fam); put("p1:per_template", len(tasks) // max(1, len(templ)), fam)
    put("jul:depth_eq_cells", sum(1 for v in jul.values() if abs(
        sum(r["violations"] for r in v.values()) - sum(1 for r in v.values() if r["violations"] > 0)) < 1e-9), fam)
    # CONVENTION: cross-cell statements use the complete core-grid cells only
    comp = {k: v for k, v in cells209.items() if len(v) >= min_n(EXPECTED_209) and k[0] in CORE}
    viol = [v for v in comp.values() if any(r["violations"] > 0 for r in v.values())]
    if viol:
        put("all:succ_min_violating", f2(min(sum(1 for r in v.values() if r["achieved"]) / len(v) for v in viol)), fam)
    put("all:cells_violating", len(viol), fam); put("all:cells_complete", len(comp), fam)
    put("all:omission_max", f2(max(sum(1 for r in v.values() if r.get("omission")) / len(v) for v in comp.values())), fam)
    put("all:depth_eq_cells", sum(1 for v in comp.values() if sum(r["violations"] for r in v.values())
                                  == sum(1 for r in v.values() if r["violations"] > 0)), fam)
    put("all:episodes_complete", "{:,}".format(sum(len(v) for v in comp.values())), fam)
    mis = Counter()
    for k, v in comp.items():
        for r in v.values():
            if r.get("misorder"):
                mis[(short_model(k[1]), k[4])] += 1
    put("misorder:all", sum(mis.values()), fam)
    per_m = Counter()
    for (m, _), c in mis.items():
        per_m[m] += c
    put("misorder:bymodel", ", ".join("%s: %d" % (DISP[m], per_m[m]) for m in ORDER if per_m.get(m)), fam)
    put("misorder:models", len({m for m, _ in mis}), fam)
    put("misorder:only_toB", "yes" if all(f == "timeout_before_commit" for _, f in mis) else "no", fam)
    put("misorder:only_multistep", "yes" if {r["family"] for v in comp.values() for r in v.values()
                                              if r.get("misorder")} <= {"multi_step"} else "no", fam)
    put("misorder:with_violation", sum(1 for v in comp.values() for r in v.values()
                                       if r.get("misorder") and r["violations"] > 0), fam)
    rc = [v for k, v in comp.items() if k[3] == "runtime_receipts"]
    put("rcpt:cells", len(rc), fam)
    put("rcpt:allzero", "yes" if all(r["violations"] == 0 for v in rc for r in v.values()) else "no", fam)
    put("rcpt:succ_min", f2(min(sum(1 for r in v.values() if r["achieved"]) / len(v) for v in rc)), fam)
    car = {k: v for k, v in comp.items() if k[2] == "careful" and k[3] == "none"}
    nz = [k for k, v in car.items() if any(r["violations"] > 0 for r in v.values())]
    put("careful:cells", len(car), fam); put("careful:cells_nonzero", len(nz), fam)
    put("careful:cells_zero", len(car) - len(nz), fam)
    put("careful:nonzero", "; ".join("%s/%s" % (short_model(k[1]), FM[k[4]]) for k in nz) or "none", fam)
    for fm in ("timeout_after_commit", "error_500_after_commit"):
        vals = [(short_model(k[1]), sum(1 for r in v.values() if r["violations"] > 0) / len(v))
                for k, v in comp.items() if k[2] == "base" and k[3] == "none" and k[4] == fm and not k[5]]
        if vals:
            fs = FM[fm]
            put("range:%s:min" % fs, f2(min(x[1] for x in vals)), fam)
            put("range:%s:max" % fs, f2(max(x[1] for x in vals)), fam)
            put("range:%s:nmodels" % fs, len(vals), fam)
            put("range:%s:models" % fs, names({x[0] for x in vals}), fam)
    # Haiku verdict agreement between the two samples, worst July cell
    worst = None
    for k, v in jul.items():
        if short_model(k[1]) != "haiku":
            continue
        by = defaultdict(set)
        for r in v.values():
            by[r["task_id"]].add(r["violations"] > 0)
        agree = sum(1 for s in by.values() if len(s) == 1)
        if worst is None or agree < worst[0]:
            worst = (agree, len(by))
    if worst:
        put("haiku:agree_min", "%d of %d" % worst, fam)
    # misordered episodes: both effectful calls in the same assistant turn?
    same = tot = 0
    for (b, model, prompt, mit, fm, tags), v in comp.items():
        for r in v.values():
            if not r.get("misorder"):
                continue
            t = t209(trans209, b, r)
            if t is None:
                continue
            tot += 1
            for turn in t.get("turns", []):
                if turn.get("role") != "assistant":
                    continue
                nm = {x["name"] for x in turn["content"] if x.get("type") == "tool_use" and x["name"] in CANON}
                if len(nm) >= 2:
                    same += 1
                    break
    put("misorder:sameturn", "%d of %d" % (same, tot), fam)
    put("misorder:sameturn:all", "yes" if same == tot and tot else "no", fam)
    # two-step tasks: both steps issued in the first effectful turn (jul/xp complete base-task cells)
    par = defaultdict(lambda: [0, 0])
    for (b, model, prompt, mit, fm, tags), v in cells209.items():
        if b not in ("jul", "xp") or tags or len(v) < min_n(EXPECTED_209):
            continue
        for r in v.values():
            if r["family"] != "multi_step":
                continue
            t = t209(trans209, b, r)
            if t is None:
                continue
            for turn in t.get("turns", []):
                if turn.get("role") != "assistant":
                    continue
                nm = {x["name"] for x in turn["content"] if x.get("type") == "tool_use" and x["name"] in CANON}
                if nm:
                    par[short_model(model)][1] += 1
                    par[short_model(model)][0] += len(nm) >= 2
                    break
    for ms, (k, n) in par.items():
        put("par:%s:multistep" % ms, "%d/%d" % (k, n), fam)


PAIRED_MDD = {}
TESTS = {}   # key -> perm result


def emit_tests():
    fam = "paired permutation test"
    family = []
    for m in ORDER:
        for b in ("jul", "xp"):
            qa, qe = "%s:%s:base:none:toA" % (b, m), "%s:%s:base:none:e500" % (b, m)
            if qa in EPI and qe in EPI:
                break
        else:
            continue
        for metric, idx in (("ivr", 1), ("rb", 2)):
            ra = rates((e[0], e[idx]) for e in EPI[qa]); re_ = rates((e[0], e[idx]) for e in EPI[qe])
            r = perm_test(ra, re_)
            if r is None:
                continue
            k = "test:%s:toA_e500:%s" % (m, metric)
            TESTS[k] = r
            put(k + ":d", ("+" if r["d"] > 0 else "") + f2(r["d"]) if r["d"] else "0.00", fam)
            put(k + ":p", pfmt(r["p"]), fam); put(k + ":T", r["T"], fam)
            put(k + ":plus", r["plus"], fam); put(k + ":minus", r["minus"], fam)
            family.append((k, r["p"]))
            if metric == "ivr" and r["T"] > 1:
                dd = r["diffs"]
                mu = sum(dd) / len(dd)
                sd = math.sqrt(sum((x - mu) ** 2 for x in dd) / (len(dd) - 1))
                PAIRED_MDD[m] = (1.959964 + 0.841621) * sd / math.sqrt(len(dd))
    if PAIRED_MDD:
        lo = min(PAIRED_MDD, key=PAIRED_MDD.get); hi = max(PAIRED_MDD, key=PAIRED_MDD.get)
        put("power:paired:min", f2(PAIRED_MDD[lo]), fam); put("power:paired:minname", DISP[lo], fam)
        put("power:paired:max", f2(PAIRED_MDD[hi]), fam); put("power:paired:maxname", DISP[hi], fam)
    if family:
        for (k, _), a in zip(family, holm([p for _, p in family])):
            put(k + ":holm", pfmt(a), fam)
        put("test:toA_e500:family", len(family), fam)
        put("test:perm_n", "{:,}".format(PERM_N), fam)

    def graph_rates(q):
        return rates((g, v) for (g, kk, s), v in EPI249[q].items())

    def graph_test(qa, qh, k):
        r = perm_test(graph_rates(qa), graph_rates(qh))
        if r:
            TESTS[k] = r
            put(k + ":d", f2(r["d"]), fam); put(k + ":p", pfmt(r["p"]), fam); put(k + ":G", r["T"], fam)
    sm = {}
    for m in ORDER:
        b = "oct" if m in CLAUDE else "xp"
        qa, qh = "%s:%s:amnesiac:none" % (b, m), "%s:%s:bare_hint:none" % (b, m)
        if qh not in EPI249:
            continue
        if qa in EPI249:
            graph_test(qa, qh, "test:%s:hint" % m)
            sm[m] = float(MINE[qa + ":givr"])
        qj = "jul:%s:amnesiac:none" % m
        if b == "oct" and qj in EPI249:
            graph_test(qj, qh, "test:%s:hintjul" % m)
    for g, ms, b in (("cl", CLAUDE, "oct"), ("or", ("gpt", "gemini", "qwen"), "xp")):
        have = [m for m in ms if "%s:%s:bare_hint:none" % (b, m) in EPI249]
        if not have:
            continue
        put("sm:%s:names" % g, names(have), fam); put("sm:%s:n" % g, len(have), fam)
        hv = [float(MINE["%s:%s:bare_hint:none:givr" % (b, m)]) for m in have]
        put("sm:%s:hint:min" % g, f2(min(hv)), fam); put("sm:%s:hint:max" % g, f2(max(hv)), fam)
        hk = [MINE["%s:%s:bare_hint:none:kn" % (b, m)] for m in have]
        put("sm:%s:hint:kn" % g, "%d/%d" % (sum(int(x.split("/")[0]) for x in hk), sum(int(x.split("/")[1]) for x in hk)), fam)
        if all(m in sm for m in have):
            av = [sm[m] for m in have]
            put("sm:%s:amn:min" % g, f2(min(av)), fam); put("sm:%s:amn:max" % g, f2(max(av)), fam)
            ak = [MINE["%s:%s:amnesiac:none:kn" % (b, m)] for m in have]
            put("sm:%s:amn:kn" % g, "%d/%d" % (sum(int(x.split("/")[0]) for x in ak), sum(int(x.split("/")[1]) for x in ak)), fam)
            ps = [TESTS["test:%s:hint" % m]["p"] for m in have]
            put("sm:%s:pmax" % g, pfmt(max(ps)), fam)
    allm = []
    for m in ORDER:
        b = "oct" if m in CLAUDE else "xp"
        if "test:%s:hint" % m in TESTS:
            allm.append((m, MINE["%s:%s:amnesiac:none:kn" % (b, m)], MINE["%s:%s:bare_hint:none:kn" % (b, m)],
                         TESTS["test:%s:hint" % m]["p"]))
    if allm:
        put("sm:all:n", len(allm), fam)
        put("sm:all:amn:kn", "%d/%d" % (sum(int(x[1].split("/")[0]) for x in allm), sum(int(x[1].split("/")[1]) for x in allm)), fam)
        put("sm:all:hint:kn", "%d/%d" % (sum(int(x[2].split("/")[0]) for x in allm), sum(int(x[2].split("/")[1]) for x in allm)), fam)
        put("sm:all:pmaxe", pe(max(x[3] for x in allm)), fam)
        put("sm:all:sig", "yes" if all(x[3] < 0.05 for x in allm) else "no", fam)
    sig = [m for m in ORDER if "test:%s:hint" % m in TESTS and TESTS["test:%s:hint" % m]["p"] < 0.05]
    ns = [m for m in ORDER if "test:%s:hint" % m in TESTS and TESTS["test:%s:hint" % m]["p"] >= 0.05]
    put("hint:sig:names", names(sig), fam); put("hint:sig:n", len(sig), fam)
    put("hint:ns:names", names(ns), fam); put("hint:ns:n", len(ns), fam)
    if sig:
        put("hint:sig:pmaxe", pe(max(TESTS["test:%s:hint" % m]["p"] for m in sig)), fam)


def emit_contrasts():
    fam = "209 contrast summary"
    rows = []
    for m in ORDER:
        def get(cell, met):
            for b in ("jul", "xp"):
                v = MINE.get("%s:%s:base:none:%s:%s" % (b, m, cell, met))
                if v is not None:
                    return v
            return None
        ta, ea, tr, er = get("toA", "ivr"), get("e500", "ivr"), get("toA", "rbrate"), get("e500", "rbrate")
        if None in (ta, ea, tr, er):
            continue
        rows.append((m, float(ta), float(ea), float(tr), float(er)))
    if not rows:
        return
    put("contrast:nmodels", len(rows), fam)
    hi = [r for r in rows if r[2] > r[1]]; lo = [r for r in rows if r[2] < r[1]]
    put("contrast:e500_higher:n", len(hi), fam); put("contrast:e500_higher:names", names([r[0] for r in hi]), fam)
    put("contrast:e500_lower:n", len(lo), fam); put("contrast:e500_lower:names", names([r[0] for r in lo]), fam)
    chk = [r for r in rows if r[3] >= 0.5]; non = [r for r in rows if r[3] < 0.5]
    put("checkers:n", len(chk), fam); put("checkers:names", names([r[0] for r in chk]), fam)
    put("noncheckers:n", len(non), fam); put("noncheckers:names", names([r[0] for r in non]), fam)
    if chk:
        put("checkers:rb_toA:min", f2(min(r[3] for r in chk)), fam)
        put("checkers:rb_e500:max", f2(max(r[4] for r in chk)), fam)
        put("checkers:all_e500_higher", "yes" if all(r[2] > r[1] for r in chk) else "no", fam)
        put("checkers:all_rb_lower", "yes" if all(r[4] < r[3] for r in chk) else "no", fam)
        put("checkers:smallest_rise", DISP[min(chk, key=lambda r: r[2] - r[1])[0]], fam)
    if non:
        put("noncheckers:rb_max", f2(max(max(r[3], r[4]) for r in non)), fam)
        put("noncheckers:ivr:min", f2(min(min(r[1], r[2]) for r in non)), fam)
        put("noncheckers:ivr:max", f2(max(max(r[1], r[2]) for r in non)), fam)
    for k, i in (("toA", 1), ("e500", 2)):
        put("range5:%s:min" % k, f2(min(r[i] for r in rows)), fam); put("range5:%s:max" % k, f2(max(r[i] for r in rows)), fam)
    for w in ("contrast:nmodels", "checkers:n", "noncheckers:n", "contrast:e500_higher:n", "contrast:e500_lower:n"):
        put(w + ":w", word(MINE[w]), fam)
    # significance statements
    def sig(k):
        return k in TESTS and TESTS[k]["p"] < 0.05
    def sigh(k):
        return MINE.get(k + ":holm", "1").startswith("<") or float(MINE.get(k + ":holm", "1")) < 0.05
    cm = [r[0] for r in chk]; nm = [r[0] for r in non]
    if cm:
        put("test:checkers:ivr_holm_sig", "yes" if all(sigh("test:%s:toA_e500:ivr" % m) for m in cm) else "no", fam)
        put("test:checkers:rb_holm_sig", "yes" if all(sigh("test:%s:toA_e500:rb" % m) for m in cm) else "no", fam)
    if nm:
        put("test:noncheckers:ivr_raw_sig", "yes" if all(sig("test:%s:toA_e500:ivr" % m) for m in nm) else "no", fam)
        put("test:noncheckers:ivr_holm_ns", "yes" if all(not sigh("test:%s:toA_e500:ivr" % m) for m in nm) else "no", fam)
        put("test:noncheckers:rb_raw_ns", "yes" if all(not sig("test:%s:toA_e500:rb" % m) for m in nm) else "no", fam)


def emit_lists():
    fam = "249 model lists"
    for ctx in ("amnesiac", "bare_hint", "plan_only", "ledger_readback"):
        vals = {}
        for m in ORDER:
            cands = (["pocorr:%s:plan_only:none" % m] if ctx == "plan_only" else []) + \
                ["%s:%s:%s:none" % (b, m, ctx) for b in ("oct", "xp", "jul")]
            for q in cands:
                if q + ":givr" in MINE and int(MINE[q + ":n"]) >= min_n(EXPECTED_249):
                    vals[m] = (float(MINE[q + ":givr"]), q)
                    break
        if not vals:
            continue
        put("list:%s:n" % ctx, len(vals), fam); put("list:%s:names" % ctx, names(vals), fam)
        for tag, cond in (("zero", lambda x: x == 0), ("nonzero", lambda x: x > 0), ("one", lambda x: x == 1),
                          ("below1", lambda x: x < 1)):
            put("list:%s:%s" % (ctx, tag), names([m for m, x in vals.items() if cond(x[0])]), fam)
            put("list:%s:%s:n" % (ctx, tag), sum(1 for x in vals.values() if cond(x[0])), fam)
        put("list:%s:max" % ctx, f2(max(x[0] for x in vals.values())), fam)
        put("list:%s:min" % ctx, f2(min(x[0] for x in vals.values())), fam)
        for m, (v, q) in vals.items():
            put("best:%s:%s" % (m, ctx), q, "string")
        put("list:%s:allchk" % ctx, "yes" if all(MINE[q + ":chk"].split("/")[0] == MINE.get(q + ":resumed", MINE[q + ":chk"].split("/")[1])
                                                 for _, q in vals.values()) else "no", fam)
        lk = sum(int(MINE[q + ":chk"].split("/")[0]) for _, q in vals.values())
        rs = sum(int(MINE[q + ":chk"].split("/")[1]) for _, q in vals.values())
        put("list:%s:chk_total" % ctx, "%d/%d" % (lk, rs), fam)
        put("list:%s:dupchk_none" % ctx, "yes" if all(MINE.get(q + ":chkdup", "0/0").split("/")[0] == "0"
                                                      for _, q in vals.values()) else "no", fam)
    for k in [k for k in list(MINE) if k.startswith("list:") and k.endswith(":n")]:
        put(k + ":w", word(MINE[k]), fam)
    done = all("oct:%s:%s:none:givr" % (m, ctx) in MINE for ctx in ("amnesiac", "ledger_readback") for m in CLAUDE)
    put("base:oct:complete", "yes" if done else "no", fam)


def emit_totals(cells209, cells249):
    fam = "totals"
    n209 = sum(len(v) for v in cells209.values()); n249 = sum(len(v) for v in cells249.values())
    put("total:p1", "{:,}".format(n209), fam); put("total:p2", "{:,}".format(n249), fam)
    put("total:all", "{:,}".format(n209 + n249), fam)
    put("nj:c1.audit.episodes", n209 + n249, fam)   # every episode replayed by the C1 audit
    put("total:core", "{:,}".format(sum(len(v) for k, v in cells209.items() if k[0] in CORE) + n249), fam)
    models = {short_model(k[1]) for k, v in cells209.items() if len(v) >= min_n(EXPECTED_209) and k[0] in CORE} | \
        {short_model(k[1]) for k, v in cells249.items() if len(v) >= expected_249(k[0], k[2])}
    put("models:n", len(models), fam); put("models:nwords", word(len(models)), fam)
    ps = {PROVIDER.get(m, m) for m in models}
    put("providers:n", len(ps), fam); put("providers:nwords", word(len(ps)), fam)
    put("models:claude:nwords", word(len([m for m in models if m in CLAUDE])), fam)
    put("models:other:nwords", word(len([m for m in models if m not in CLAUDE])), fam)
    for b in sorted({k[0] for k in cells209} | {k[0] for k in cells249}):
        r1 = [r for k, v in cells209.items() if k[0] == b for r in v.values()]
        r2 = [r for k, v in cells249.items() if k[0] == b for r in v.values()]
        put("total:%s:episodes" % b, "{:,}".format(len(r1) + len(r2)), fam)
        put("total:%s:trunc" % b, sum(1 for r in r1 + r2 if r.get("truncated")), fam)
        put("total:%s:invalid" % b, sum(int(r.get("invalid_calls") or 0) for r in r1 + r2), fam)
    for T in (16, 32, 96):
        put("power:mdd%d" % T, f2((1.959964 + 0.841621) * math.sqrt(2 * 0.25 / T)), fam)


def emit_dups(cells209, allrows):
    """Episodes run more than once (two overlapping launches)."""
    fam = "duplicate-run rows"
    tot_conf = 0
    for b, per in sorted(allrows.items()):
        d = {e: rs for e, rs in per.items() if len(rs) > 1}
        if not d:
            continue
        put("dup:%s:rows" % b, "{:,}".format(sum(len(rs) for rs in per.values())), fam)
        put("dup:%s:unique" % b, "{:,}".format(len(per)), fam)
        put("dup:%s:ids" % b, len(d), fam); put("dup:%s:maxcopies" % b, max(len(rs) for rs in d.values()), fam)
        affected = defaultdict(list)
        for eid, rs in d.items():
            r = rs[-1]
            affected[(b, r["model"], r["prompt"], r["mitigation"], r["failure_mode"], tags209(r))].append(eid)
        put("dup:%s:cells" % b, len(affected), fam)
        put("dup:%s:models" % b, ", ".join(DISP.get(m, m) for m in sorted({short_model(k[1]) for k in affected})), fam)
        for key, eids in affected.items():
            eps = cells209[key]
            last = {e: r["violations"] > 0 for e, r in eps.items()}
            first = dict(last)
            for e in eids:
                first[e] = d[e][0]["violations"] > 0
            n = len(eps); kl, kf = sum(last.values()), sum(first.values())
            conf = sum(1 for e in eids if (d[e][0]["violations"] > 0) != (d[e][-1]["violations"] > 0))
            tot_conf += conf
            ms = short_model(key[1])
            q = "dup:%s:%s" % (b, ":".join([ms, key[2], MIT.get(key[3], key[3]), FM.get(key[4], key[4])]))
            put(q + ":rerun", len(eids), fam); put(q + ":n", n, fam)
            put(q + ":last", "%d/%d" % (kl, n), fam); put(q + ":first", "%d/%d" % (kf, n), fam)
            put(q + ":last_ivr", f2(kl / n), fam); put(q + ":first_ivr", f2(kf / n), fam); put(q + ":disagree", conf, fam)
            if key[2] == "base" and key[3] == "none" and key[4] == "error_500_after_commit":
                qa = "%s:%s:base:none:toA" % (b, ms)
                if qa in EPI:
                    ra = rates((e[0], e[1]) for e in EPI[qa])
                    rf = rates((r["task_id"], 1 if first[e] else 0) for e, r in eps.items())
                    r = perm_test(ra, rf)
                    put(q + ":first_test_p", pfmt(r["p"]), fam); put(q + ":first_test_d", f2(r["d"]), fam)
                    ld = TESTS.get("test:%s:toA_e500:ivr" % ms)
                    if ld:
                        put("dup:%s:%s:same_direction" % (b, ms), "yes" if (r["d"] < 0) == (ld["d"] < 0) else "no", fam)
    put("dup:disagree:total", tot_conf, fam)


# ------------------------------------------------------------------ analysis_v3 ("nj:" macros)
FAIL_LINE = re.compile(r"^!! (?:\[\d+/\d+\]|episode) (\S+) failed, will retry on next run: (.*)$")
BLOCK = "Gemini blocked the response"


def ledger_order(t):
    """(omission, misorder, stale_reexec, order_violation) from the committed
    ledger and the task goals, for ordered tasks (families multi_step and
    long_chain); unordered tasks never misorder or go stale."""
    goals = (t.get("score") or {}).get("goals") or []
    led = t.get("ledger") or []
    seqs = []
    for g in goals:
        seqs.append([e["seq"] for e in led if e["tool"] == g["tool"] and all(
            norm(e["args"].get(k)) == norm(v) for k, v in g["canon_args"].items())])
    ordered = t.get("family") in ("multi_step", "long_chain")
    pairs = [(i, j) for i in range(len(seqs)) for j in range(i + 1, len(seqs))] if ordered else []
    omission = any(not s for s in seqs)
    firsts = [min(s) if s else None for s in seqs]
    misorder = (not omission) and ordered and firsts != sorted(firsts)
    stale = any(len(seqs[i]) > 1 and seqs[j] and max(seqs[i]) > min(seqs[j]) for i, j in pairs)
    ov = any(seqs[i] and seqs[j] and max(seqs[i]) > min(seqs[j]) for i, j in pairs)
    return omission, misorder, stale, ov


def turn_cap_hit(t):
    mt = t.get("max_turns")
    if not mt:
        return False
    parts = t["deliveries"] if "deliveries" in t else [t]
    return any(sum(1 for x in d["turns"] if x["role"] == "assistant") >= mt and d["turns"][-1]["role"] != "assistant"
               for d in parts if d.get("turns"))


class V3:
    """Episode lists per analysis source: v2 (July, released), xp / xp_first
    (cross-provider, last / first row), e1_0..e1_6 (Tier-1), e1_6_oct."""
    SRC_BATCH = {"v2": "jul", "xp": "xp", "e1_0": "e10", "e1_1": "e13", "e1_2": "e12", "e1_3": "e13",
                 "e1_4": "e14", "e1_5": "e15", "e1_6": "e16", "e1_6_oct": "tier1-e1-6-oct"}

    def __init__(self, D, cells209, allrows, trans209):
        self.D = D
        self.cells = defaultdict(list)       # (src, model, prompt, mit, fm) -> [ep]
        self.rows = {}                       # src -> {eid: row}
        self.nrows = {}                      # src -> raw rows in the file
        for src, b in self.SRC_BATCH.items():
            if src in ("e1_1", "e1_3"):      # e13 merges E1-1 and E1-3: split them by raw file
                p = prefer_v2(os.path.join(D.res, "tier1", src, "raw.jsonl"))
                raw = read_jsonl(p) if os.path.exists(p) else []
                rows = last_by_id(raw)
                self.nrows[src] = len(raw)
            else:
                rows = {}
                per = allrows.get(b, {})
                for eid, rs in per.items():
                    rows[eid] = rs[-1]
                self.nrows[src] = sum(len(rs) for rs in per.values())
            self.rows[src] = rows
            for eid, r in rows.items():
                self.cells[(src, r["model"], r["prompt"], r["mitigation"], r["failure_mode"])].append(self.ep(r, b, trans209))
        for eid, rs in allrows.get("xp", {}).items():
            r = rs[0]
            self.cells[("xp_first", r["model"], r["prompt"], r["mitigation"], r["failure_mode"])].append(self.ep(r, "xp", trans209))

    def ep(self, r, b, trans):
        t = t209(trans, b, r)
        fe = features(t) if t else None
        e = {"task": r["task_id"], "sample": r["sample"], "v": 1 if r["violations"] > 0 else 0,
             "viol": r["violations"], "eid": r["episode_id"], "template": r["task_id"].rsplit("-", 1)[0],
             "family": r.get("family"), "fail_on_call": r.get("fail_on_call"), "truncated": r.get("truncated"),
             "rb": None, "key_first": None, "retried": None, "cat": "", "strict": None, "turn_cap": None,
             "omission": None, "misorder": None, "stale": None, "t": t}
        if fe:
            e.update(rb=fe["rb_fail"], key_first=fe["key_first_valid"], retried=fe["retried_fail"], cat=fe["cat_fail"])
        if t and "ledger" in t:
            e["strict"] = int(ledger_violations(t["ledger"], strict=True) > 0)
            e["omission"], e["misorder"], e["stale"], _ = ledger_order(dict(t, family=r.get("family")))
            e["turn_cap"] = turn_cap_hit(t)
        return e

    def cell(self, src, model, prompt, mit, fm):
        return self.cells.get((src, model, prompt, mit, fm), [])


def refusals(D, v3):
    """Episodes with no row whose every logged failure was a provider block."""
    have = {e["eid"] for eps in v3.cells.values() for e in eps}
    out = defaultdict(list)
    for runner, path in (("tier1", os.path.join(D.res, "tier1", "run.log")),
                         ("crossprovider", os.path.join(D.res, "crossprovider", "run.log"))):
        if not os.path.exists(path):
            continue
        fails = defaultdict(list)
        for line in open(path, encoding="utf-8", errors="replace"):
            m = FAIL_LINE.match(line.strip())
            if m:
                fails[m.group(1)].append(m.group(2))
        for eid, msgs in fails.items():
            if eid in have or not all(BLOCK in x for x in msgs):
                continue
            parts = eid.split("__")
            model, prompt, mit, fm, task, s = parts[0], parts[1], parts[2], parts[3], parts[4], int(parts[5][1:])
            if runner == "tier1":
                src = "e1_1" if (mit, fm) == ("idem_keys", "error_500_after_commit") else "e1_2"
            else:
                src = "xp"
            out[(src, model, prompt, mit, fm)].append((task, s))
    return out


def stat_cell(eps, field="v"):
    vals = [e for e in eps if e.get(field) is not None]
    n = len(vals); k = sum(1 for e in vals if e[field]); T = len({e["task"] for e in vals})
    return {"k": k, "n": n, "T": T, "p": k / n if n else None,
            "ci_task": wilson_task(k / n, T) if n else (0.0, 1.0), "ci_ep": wilson(k, n)}


def nj_cell(name, eps):
    fam = "analysis cell (raw rows + transcripts)"
    st = stat_cell(eps)
    if not st["n"]:
        return st
    put(name + ":ivr", f2(st["p"]), fam); put(name + ":ivr:ci", ci(*st["ci_task"]), fam)
    put(name + ":ivr:kn", "%d/%d" % (st["k"], st["n"]), fam); put(name + ":ivr:n", st["n"], fam)
    put(name + ":ivr:k", st["k"], fam); put(name + ":ivr:T", st["T"], fam)
    put(name + ":ivr:k:w", word(st["k"]) if st["k"] <= 10 else st["k"], fam)
    put(name + ":ivr_ci_ep:ci", ci(*st["ci_ep"]), fam)
    strict = [e for e in eps if e["strict"] is not None]
    if strict:
        put(name + ":ivr_strict:kn", "%d/%d" % (sum(e["strict"] for e in strict), len(strict)), fam)
    rb = [e for e in eps if e["rb"] is not None]
    if rb:
        put(name + ":readback_after_failure:kn", "%d/%d" % (sum(e["rb"] for e in rb), len(rb)), fam)
        yes = [e["v"] for e in rb if e["rb"]]; no = [e["v"] for e in rb if not e["rb"]]
        put(name + ":ivr_given_readback", "%d/%d" % (sum(yes), len(yes)), fam)
        put(name + ":ivr_given_readback:ivr_given_no_readback", "%d/%d" % (sum(no), len(no)), fam)
        st2 = stat_cell(eps, "rb")
        put(name + ":readback", f2(st2["p"]), fam); put(name + ":readback:kn", "%d/%d" % (st2["k"], st2["n"]), fam)
    return st


def verdict_dir(r, want, thr=None):
    if r is None:
        return "not testable"
    good = (r["d"] > 0) if want == "+" else (r["d"] < 0)
    if r["p"] < 0.05 and good and (thr is None or abs(r["d"]) >= thr):
        return "supported"
    if r["p"] < 0.05 and good:
        return "not supported"
    if r["p"] < 0.05:
        return "reversed"
    return "null"


def impute(eps, refs, value):
    return eps + [{"task": t, "sample": s, "v": value} for t, s in refs]


def nj_test(name, x, y, refs_x=(), refs_y=(), field="v"):
    """Paired task-level test y - x, with refusal bounds when either arm has refusals."""
    fam = "analysis contrast (paired permutation)"
    rx = rates((e["task"], e[field]) for e in x); ry = rates((e["task"], e[field]) for e in y)
    r = perm_test(rx, ry)
    if r is None:
        return None
    put(name + ":d", signed(r["d"]), fam); put(name + ":d:abs", f2(abs(r["d"])), fam)
    put(name + ":p", pfmt(r["p"]), fam); put(name + ":p:pe", pe(r["p"]), fam)
    put(name + ":d:n", r["T"], fam); put(name + ":d:plus", r["plus"], fam); put(name + ":d:minus", r["minus"], fam)
    if field == "v" and (refs_x or refs_y):
        ps, ds = [], []
        for vx, vy in ((1, 0), (0, 1), (0, 0), (1, 1)):
            b = perm_test(rates((e["task"], e["v"]) for e in impute(x, refs_x, vx)),
                          rates((e["task"], e["v"]) for e in impute(y, refs_y, vy)))
            ps.append(b["p"]); ds.append(b["d"])
        r["p_bound"] = (min(ps), max(ps)); r["d_bound"] = (min(ds), max(ds))
        put(name + ":p_bound:lo", pfmt(min(ps)), fam); put(name + ":p_bound:hi", pfmt(max(ps)), fam)
        put(name + ":p:p_bound:lo", pfmt(min(ps)), fam); put(name + ":p:p_bound:hi", pfmt(max(ps)), fam)
        put(name + ":p:p_bound", "[%s, %s]" % (pfmt(min(ps)), pfmt(max(ps))), fam)
    return r


HAIKU, SONNET, FABLE = "claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-fable-5"
ORM = {"qwen": "openrouter:qwen/qwen3.6-27b-20260422", "gpt": "openrouter:openai/gpt-5.5-20260423",
       "gemini": "openrouter:google/gemini-4-argon-20261013"}
FULL = {"haiku": HAIKU, "sonnet": SONNET, "fable": FABLE, **ORM}


def nj_name(src, model, prompt, mit, fm):
    return ".".join([src, short_model(model), prompt, MIT.get(mit, mit), FM_NJ.get(fm, fm)])


def emit_v3(D, v3, cells249, trans249):
    fam = "analysis derived"
    refs = refusals(D, v3)
    # every cell of every source
    for key, eps in sorted(v3.cells.items()):
        if key[0] == "xp_first":
            continue
        nm = "nj:" + nj_name(*key)
        st = nj_cell(nm, eps)
        if key[0] == "e1_2":
            pass
    put("nj:refusal:total", sum(len(v) for k, v in refs.items() if k[0] in ("xp", "e1_1", "e1_2")), fam)
    put("nj:refusal:tasks_n", len({t for v in refs.values() for t, _ in v}), fam)
    put("nj:hygiene.turn_cap_hits_tier1", sum(1 for k, eps in v3.cells.items() if k[0].startswith("e1_") and k[0] != "e1_6_oct"
                                              for e in eps if e["turn_cap"]), fam)
    t1 = sum(len(v3.rows[s]) for s in ("e1_0", "e1_1", "e1_2", "e1_3", "e1_4", "e1_5", "e1_6"))
    # CONVENTION: total:tier1 counts raw rows (not unique ids) of E1-0..E1-6
    put("total:tier1", "{:,}".format(sum(v3.nrows[s] for s in ("e1_0", "e1_1", "e1_2", "e1_3", "e1_4", "e1_5", "e1_6"))), fam)
    put("nj:meta:perm_n", "{:,}".format(PERM_N), "protocol constant"); put("nj:meta:perm_seed", PERM_SEED, "protocol constant")

    def C(src, m, prompt, mit, fm):
        return v3.cell(src, FULL[m], prompt, mit, fm)

    def R(src, m, prompt, mit, fm):
        return refs.get((src, FULL[m], prompt, mit, fm), [])

    # ---- E1-0
    PV = []   # verdicts of the 18 registered primary (P) tests
    r = nj_test("nj:e1_0.X.rerun_vs_v2", C("v2", "haiku", "base", "idem_keys", "timeout_after_commit"),
                C("e1_0", "haiku", "base", "idem_keys", "timeout_after_commit"))
    e10 = C("e1_0", "haiku", "base", "idem_keys", "timeout_after_commit")
    diff = stat_cell(e10)["p"] - stat_cell(C("v2", "haiku", "base", "idem_keys", "timeout_after_commit"))["p"]
    put("nj:e1_0.P.diff_rerun_minus_v2", signed(diff), fam)
    PV.append("supported" if abs(diff) <= 0.06 else "not supported")
    put("nj:e1_0.S1.key_attach_first", "%d/%d" % (sum(1 for e in e10 if e["key_first"]), len(e10)), fam)
    # ---- E1-1 timeout vs 500, per model
    ps_e11 = []
    for m in ORDER:
        src = "xp" if m in ORM else "v2"
        x, y = C(src, m, "base", "none", "timeout_after_commit"), C(src, m, "base", "none", "error_500_after_commit")
        r = nj_test("nj:e1_1.P.H2.%s" % m, x, y, R(src, m, "base", "none", "timeout_after_commit"),
                    R(src, m, "base", "none", "error_500_after_commit"))
        if r:
            put("nj:e1_1.P.H2.%s:p:v" % m, verdict_dir(r, "+", 0.2), fam)
            if src == "xp":
                ps_e11.append(("nj:e1_1.P.H2.%s" % m, r["p"]))
                PV.append(verdict_dir(r, "+", 0.2))
    for (k, _), a in zip(ps_e11, holm([p for _, p in ps_e11])):
        put(k + ":holm_sensitivity", pfmt(a), fam); put(k + ":holm_sensitivity:pe", pe(a), fam)
    put("e11:holm_keeps", "yes" if all((p < 0.05) == (a < 0.05) for (_, p), a in
                                       zip(ps_e11, holm([p for _, p in ps_e11]))) else "no", fam)
    nj_test("nj:e1_1.P.H2.qwen.first_rows", C("xp_first", "qwen", "base", "none", "timeout_after_commit"),
            C("xp_first", "qwen", "base", "none", "error_500_after_commit"))
    # ---- E1-2 redelivery
    fam2 = "E1-2 delivery features (transcripts)"
    e12 = v3.rows["e1_2"]
    d1cells = 0; d1nz = 0
    for m in ("haiku", "sonnet", "qwen", "gpt", "gemini"):
        for c in (("base", "none", "none"), ("base", "idem_keys", "none"), ("careful", "none", "none")):
            eps = C("e1_2", m, *c)
            if not eps:
                continue
            nm = "nj:" + nj_name("e1_2", FULL[m], *c)
            rbf = d1 = 0; cats = Counter()
            for e in eps:
                t = e["t"]
                if not t or "deliveries" not in t:
                    continue
                inv = {c_.get("tool_use_id") for c_ in (t.get("invalid_calls") or []) if isinstance(c_, dict)}
                s1 = [x for k, x in seq_of(t["deliveries"][0].get("turns", []), inv) if k == "call"]
                s2 = [x for k, x in seq_of(t["deliveries"][1].get("turns", []), inv) if k == "call"]
                e1 = [x for x in s1 if x["name"] in CANON and not x["invalid"]]
                e2 = [x for x in s2 if x["name"] in CANON and not x["invalid"]]
                upto = next((i for i, x in enumerate(s2) if x["name"] in CANON), len(s2))
                rbf += bool(s2) and any(x["name"] == "list_effects" for x in s2[:upto])
                if e1:
                    sf = sid(e1[0]["name"], e1[0]["args"])
                    redo = next((x for x in e2 if sid(x["name"], x["args"]) == sf), None)
                    cats[key_cat(e1[0]["key"], redo["key"]) if redo else "not_redone"] += 1
                else:
                    cats[""] += 1
                d1 += ledger_violations(t["ledger"][:t.get("d1_ledger_len", 0)]) > 0
            n = len(eps)
            put(nm + ":d2_readback_first:kn", "%d/%d" % (rbf, n), fam2)
            put(nm + ":d2_readback_first", f2(rbf / n), fam2)
            put(nm + ":d1_spontaneous_dup:kn", "%d/%d" % (d1, n), fam2)
            put(nm + ":d1_spontaneous_dup", f2(d1 / n), fam2)
            for cat, cnt in cats.items():
                put(nm + ":d2_key_category:" + (cat or "none"), cnt, fam2)
            d1cells += 1; d1nz += d1 > 0
    put("e12:d1:cells", d1cells, fam); put("e12:d1:nonzero_n", d1nz, fam)
    r = nj_test("nj:e1_2.P.H2.sonnet_minus_haiku", C("e1_2", "haiku", "base", "none", "none"), C("e1_2", "sonnet", "base", "none", "none"))
    PV.append(verdict_dir(r, "-"))
    fam12 = []
    for m in ("haiku", "sonnet", "qwen", "gpt", "gemini"):
        x = C("e1_2", m, "base", "none", "none")
        r3 = nj_test("nj:e1_2.S.H3.%s" % m, x, C("e1_2", m, "base", "idem_keys", "none"),
                     R("e1_2", m, "base", "none", "none"), R("e1_2", m, "base", "idem_keys", "none"))
        r4 = nj_test("nj:e1_2.S.H4.%s" % m, x, C("e1_2", m, "careful", "none", "none"),
                     R("e1_2", m, "base", "none", "none"), R("e1_2", m, "careful", "none", "none"))
        fam12 += [("nj:e1_2.S.H3.%s" % m, r3, m), ("nj:e1_2.S.H4.%s" % m, r4, m)]
    for (k, r, m), a in zip(fam12, holm([r["p"] for _, r, _ in fam12])):
        put(k + ":p:holm_p", pfmt(a), fam); put(k + ":p:holm_pe", pe(a), fam)
        if ".H3." in k:
            if m == "haiku":
                v = "supported" if a >= 0.05 else "not supported"
            elif m == "sonnet":
                v = "supported" if (a < 0.05 and r["d"] < 0) else "not supported"
            else:
                v = "robustness set: " + ("reduced" if (a < 0.05 and r["d"] < 0) else "increased" if a < 0.05 else "no change detected")
        else:
            v = "not supported" if (a < 0.05 and r["d"] < 0) else "supported"
        put(k + ":p:v", v, fam)
    # ---- E1-3 keys under 500
    r = nj_test("nj:e1_3.P.sonnet_keys_minus_none", C("v2", "sonnet", "base", "none", "error_500_after_commit"),
                C("e1_3", "sonnet", "base", "idem_keys", "error_500_after_commit"))
    PV.append(verdict_dir(r, "-"))
    nj_test("nj:e1_3.S.haiku_keys_minus_none", C("v2", "haiku", "base", "none", "error_500_after_commit"),
            C("e1_3", "haiku", "base", "idem_keys", "error_500_after_commit"))
    for m in ("haiku", "sonnet"):
        eps = C("e1_3", m, "base", "idem_keys", "error_500_after_commit")
        ret = [e for e in eps if e["retried"]]
        for cat, cnt in Counter(e["cat"] for e in ret).items():
            put("nj:e1_3.%s.key_mech:retry_key_categories:%s" % (m, cat), cnt, fam)
    for m in ("qwen", "gpt", "gemini"):
        for fm, src in (("timeout_after_commit", "xp"), ("error_500_after_commit", "e1_1")):
            eps = C(src, m, "base", "idem_keys", fm)
            ret = [e for e in eps if e["retried"]]
            for cat, cnt in Counter(e["cat"] for e in ret).items():
                put("nj:e1_1.S.H4.%s.%s:retry_key_categories:%s" % (m, FM[fm], cat), cnt, fam)
    # ---- E1-4 / E1-5
    e14ok = e15ok = 0
    for m in ("haiku", "sonnet"):
        for fm in ("timeout_after_commit", "error_500_after_commit"):
            f_ = FM[fm]
            new = C("e1_4", m, "base", "none", fm)
            old = [e for e in C("v2", m, "base", "none", fm) if e["sample"] == 0]
            kn, nn = sum(e["v"] for e in new), len(new); ko, no = sum(e["v"] for e in old), len(old)
            olo, ohi = wilson(ko, no)
            tag = "nj:e1_4.%s.%s" % (m, f_)
            put(tag + ".new.ivr", f2(kn / nn), fam); put(tag + ".new.ivr:kn", "%d/%d" % (kn, nn), fam); put(tag + ".new.ivr:n", nn, fam)
            put(tag + ".new.ivr:ci", ci(*wilson(kn, nn)), fam)
            put(tag + ".released_s0.ivr", f2(ko / no), fam); put(tag + ".released_s0.ivr:kn", "%d/%d" % (ko, no), fam)
            put(tag + ".released_s0.ivr:ci", ci(olo, ohi), fam)
            inside = olo <= kn / nn <= ohi
            put(tag + ".P.new_within_released_ci", "yes" if inside else "no", fam); e14ok += inside
            PV.append("supported" if inside else "not supported")
            kp, np_ = kn + ko, nn + no
            put(tag + ".pooled96.ivr", f2(kp / np_), fam); put(tag + ".pooled96.ivr:kn", "%d/%d" % (kp, np_), fam)
            put(tag + ".pooled96.ivr:n", np_, fam)
            put(tag + ".pooled96.ivr:ci", ci(*wilson(kp, np_)), fam)
            for tg, rows in (("new", new), ("pooled96", new + old)):
                by = defaultdict(lambda: [0, 0])
                for e in rows:
                    by[e["template"]][0] += e["v"]; by[e["template"]][1] += 1
                for tp, (k_, n_) in by.items():
                    put("nj:x.e1_4.%s.%s.tmpl:%s:%s" % (m, f_, tg, tp), "%d/%d" % (k_, n_), fam)
                put("nj:x.e1_4.%s.%s.tmpl%s" % (m, f_, "" if tg == "pooled96" else ":new_total"),
                    "%d/%d" % (sum(v[0] for v in by.values()), sum(v[1] for v in by.values())), fam)
            # E1-5 long chains
            eps = C("e1_5", m, "base", "none", fm)
            st = stat_cell(eps)
            tag5 = "nj:e1_5.%s.%s" % (m, f_)
            plo, phi = wilson(kp, np_)
            inp = plo <= st["p"] <= phi
            put(tag5 + ".P.within_e1_4_pooled96_ci", "yes" if inp else "no", fam); e15ok += inp
            PV.append("supported" if inp else "not supported")
            put(tag5 + ".P.within_e1_4_pooled96_ci:ci", ci(plo, phi), fam)
            n5 = len(eps)
            put(tag5 + ".S.stale_reexec:kn", "%d/%d" % (sum(1 for e in eps if e["stale"]), n5), fam)
            put(tag5 + ".S.misorder:kn", "%d/%d" % (sum(1 for e in eps if e["misorder"]), n5), fam)
            put(tag5 + ".S.omission:kn", "%d/%d" % (sum(1 for e in eps if e["omission"]), n5), fam)
    put("e14:P:cells", 4, fam); put("e14:P:supported", e14ok, fam)
    put("e15:P:cells", 4, fam); put("e15:P:supported", e15ok, fam)
    # ---- E1-6 wording
    p_tests = []
    for m in ("haiku", "sonnet"):
        kb = C("v2", m, "base", "none", "error_500_after_commit")
        kt = C("v2", m, "base", "none", "timeout_after_commit")
        k1 = C("e1_6", m, "base", "none", "error_500_body_after_commit")
        k3 = C("e1_6", m, "base", "none", "status_unknown_after_commit")
        r1 = nj_test("nj:e1_6.P1.%s" % m, kb, k1); put("nj:e1_6.P1.%s:p:v" % m, verdict_dir(r1, "+"), fam)
        r3 = nj_test("nj:e1_6.P2.%s" % m, kb, k3); put("nj:e1_6.P2.%s:p:v" % m, verdict_dir(r3, "-"), fam)
        r3r = nj_test("nj:e1_6.P2.%s.readback" % m, kb, k3, field="rb")
        put("nj:e1_6.P2.%s.readback:p:v" % m, verdict_dir(r3r, "+"), fam)
        nj_test("nj:e1_6.X.%s.w3_minus_timeout" % m, kt, k3)
        p_tests += [r1["p"], r3["p"]]
        PV.append(verdict_dir(r1, "+"))
        v1, v2 = verdict_dir(r3, "-"), verdict_dir(r3r, "+")
        PV.append(v1 if v1 == v2 else "%s; read-back %s" % (v1, v2))
    # CONVENTION: the registered-tests table counts one P row per registered
    # primary hypothesis (E1-6 P2 and its read-back mediator share one row)
    put("prereg:P:n", len(PV), fam)
    for vn in ("supported", "not supported", "reversed", "null"):
        put("prereg:P:" + vn.replace(" ", "_"), sum(1 for v in PV if v == vn), fam)
    put("prereg:P:against", sum(1 for v in PV if v != "supported"), fam)
    put("e16:holm_keeps", "yes" if all((p < 0.05) == (a < 0.05) for p, a in zip(p_tests, holm(p_tests))) else "no", fam)
    # ---- October sensitivity (e1_6_oct)
    for m in ("haiku", "sonnet"):
        for fm in ("error_500_after_commit", "timeout_after_commit"):
            o = C("e1_6_oct", m, "base", "none", fm)
            if o:
                nj_cell("nj:sens.oct.%s.base.none.%s" % (m, FM[fm]), o)
                r = nj_test("nj:sens.oct.%s.drift.%s" % (m, FM[fm]), C("v2", m, "base", "none", fm), o)
                for k in list(MINE):
                    if k.startswith("nj:sens.oct.%s.drift.%s" % (m, FM[fm])):
                        MINE[k.replace("sens.oct.", "sens.oct_fix.", 1)] = MINE[k]
                        NOTE[k.replace("sens.oct.", "sens.oct_fix.", 1)] = NOTE[k]
        o500, otoa = C("e1_6_oct", m, "base", "none", "error_500_after_commit"), C("e1_6_oct", m, "base", "none", "timeout_after_commit")
        if o500 and otoa:
            nj_test("nj:sens.oct.%s.oct_e500_minus_oct_toA" % m, otoa, o500)
            nj_test("nj:sens.oct.%s.w3_minus_oct_e500" % m, o500, C("e1_6", m, "base", "none", "status_unknown_after_commit"))
            nj_test("nj:sens.oct.%s.w1_minus_oct_e500" % m, o500, C("e1_6", m, "base", "none", "error_500_body_after_commit"))
    octrows = v3.rows.get("e1_6_oct", {})
    if octrows:
        put("sens:oct:episodes", len(octrows), fam)
        days = sorted({(r.get("started_utc") or "")[:10] for r in octrows.values()} - {""})
        if days:
            y, mo, dd = days[0].split("-")
            put("sens:oct:date", "%d %s %s" % (int(dd), datetime.date(2000, int(mo), 1).strftime("%B"), y), fam)
            put("sens:oct:days", len(days), fam)
    # ---- refusal bound flips
    flips = 0
    for k in list(MINE):
        if k.endswith(":p_bound:lo") and k.count(":p") == 1:
            base = k[:-len(":p_bound:lo")]
            p = MINE.get(base + ":p")
            if p is None:
                continue
            vals = [p, MINE[k], MINE[base + ":p_bound:hi"]]
            sig = [(v.startswith("<") or float(v) < 0.05) for v in vals]
            flips += len(set(sig)) > 1
    put("refusal:verdict_flips:partial", flips, fam)
    # ---- bare_hint vs same-month baselines (249, refusals excluded)
    def load_bh(b, model, ctx):
        eps = []
        for r in cells249.get((b, model, ctx, "none"), {}).values():
            if r.get("key_posture", "model") != "model" or r.get("interrupt_timing", "after_commit") != "after_commit" \
                    or r["topology"] not in BENCH:
                continue
            t = t249(trans249, b, r)
            eps.append({"task": r["graph_id"], "v": int(r["givr_hit"] > 0), "refusal": bool(t and refused(t)),
                        "eid": r["episode_id"], "k": r["interrupt_node"], "sample": r["sample"]})
        return eps

    def bh_rates(eps, refusal_as=None):
        pairs = []
        for e in eps:
            if e["refusal"]:
                if refusal_as is not None:
                    pairs.append((e["task"], refusal_as))
            else:
                pairs.append((e["task"], e["v"]))
        return rates(pairs)

    def bh_cell(name, eps):
        sc = [e for e in eps if not e["refusal"]]
        n = len(sc); k = sum(e["v"] for e in sc); G = len({e["task"] for e in sc})
        if n:
            put(name, f2(k / n), fam); put(name + ":ci", ci(*wilson_task(k / n, G)), fam)
            put(name + ":kn", "%d/%d" % (k, n), fam); put(name + ":n", n, fam); put(name + ":k", k, fam); put(name + ":G", G, fam)
        return k, n

    BH = {}
    ak = an = hk = hn = 0
    for m in ORDER:
        model = FULL[m]
        if m in CLAUDE:
            BH[(m, "oct.amnesiac")] = load_bh("oct", model, "amnesiac")
            BH[(m, "oct.ledger_readback")] = load_bh("oct", model, "ledger_readback")
            BH[(m, "oct.bare_hint")] = load_bh("oct", model, "bare_hint")
            BH[(m, "jul.amnesiac")] = load_bh("jul", model, "amnesiac")
            hint = BH[(m, "oct.bare_hint")]
            for base, tag in (("oct.amnesiac", "vs_oct_amnesiac"), ("oct.ledger_readback", "vs_oct_ledger_readback"),
                              ("jul.amnesiac", "vs_jul_amnesiac")):
                bs = BH[(m, base)]
                if not bs or not hint:
                    continue
                r = perm_test(bh_rates(bs), bh_rates(hint))
                nm = "nj:bh.%s.%s" % (m, tag)
                put(nm + ":d", signed(r["d"]), fam); put(nm + ":p", pfmt(r["p"]), fam); put(nm + ":p:pe", pe(r["p"]), fam)
                if any(e["refusal"] for e in bs + hint):
                    ps = [perm_test(bh_rates(bs, a), bh_rates(hint, h))["p"] for a in (0, 1) for h in (0, 1)]
                    put(nm + ":p:p_bound:lo", pfmt(min(ps)), fam); put(nm + ":p:p_bound:hi", pfmt(max(ps)), fam)
                    sig = {r["p"] < 0.05, min(ps) < 0.05, max(ps) < 0.05}
                    flips += len(sig) > 1
            # July vs October amnesiac (post hoc)
            if BH[(m, "jul.amnesiac")] and BH[(m, "oct.amnesiac")]:
                r = perm_test(bh_rates(BH[(m, "jul.amnesiac")]), bh_rates(BH[(m, "oct.amnesiac")]))
                put("nj:x.amn_month.%s:d" % m, signed(r["d"]), fam); put("nj:x.amn_month.%s:p" % m, pfmt(r["p"]), fam)
                put("nj:x.amn_month.%s:p:pe" % m, pe(r["p"]), fam)
            b = "oct"
        else:
            for ctx in ("amnesiac", "ledger_readback", "bare_hint"):
                BH[(m, "xp." + ctx)] = load_bh("xp", model, ctx)
            b = "xp"
        for c in [k2 for (m2, k2) in BH if m2 == m]:
            bh_cell("nj:bh.%s.%s:givr" % (m, c), BH[(m, c)])
        a_, h_ = BH.get((m, b + ".amnesiac")), BH.get((m, b + ".bare_hint"))
        if a_ and h_:
            k1, n1 = bh_cell("nj:bh.%s.%s.amnesiac:givr" % (m, b), a_); k2, n2 = bh_cell("nj:bh.%s.%s.bare_hint:givr" % (m, b), h_)
            ak += k1; an += n1; hk += k2; hn += n2
    put("refusal:verdict_flips", flips, fam)
    if an:
        put("smnj:all:amn:kn", "%d/%d" % (ak, an), fam); put("smnj:all:hint:kn", "%d/%d" % (hk, hn), fam)
    # ---- misorder denominators (core grid: July Claude + cross-provider OR)
    tot_k = tot_n = 0; every = True
    for m in ORDER:
        src = "v2" if m in CLAUDE else "xp"
        mine = [e for (s, model, *_), eps in v3.cells.items() if s == src and short_model(model) == m for e in eps]
        two = [e for (s, model, pr, mi, fm), eps in v3.cells.items() if s == src and short_model(model) == m
               and fm == "timeout_before_commit" for e in eps if e["family"] == "multi_step"]
        mis_all = sum(1 for e in mine if v3_misorder(e))
        k = sum(1 for e in two if v3_misorder(e))
        every &= mis_all == k
        put("nj:x.misorder.%s:kn" % m, "%d/%d" % (k, len(two)), fam); put("nj:x.misorder.%s:k" % m, k, fam)
        tot_k += k; tot_n += len(two)
    put("nj:x.misorder.all:kn", "%d/%d" % (tot_k, tot_n), fam); put("nj:x.misorder.all:k", tot_k, fam)
    put("nj:x.misorder.all:n", tot_n, fam)
    put("nj:x.misorder.all:all_toB_twostep", "yes" if every else "no", fam)
    # ---- retries whose text names the key (July base/none/500)
    for m in ("sonnet", "fable"):
        eps = C("v2", m, "base", "none", "error_500_after_commit")
        c = Counter()
        for e in eps:
            fe = features(e["t"]) if e["t"] else None
            if not fe or fe["rtext"] is None:
                continue
            c["retries"] += 1; c["text"] += fe["rtext"][0]; c["named"] += fe["rtext"][1]
            c["named_same"] += fe["rtext"][1] and fe["rtext"][2]
        put("nj:x.rtext.%s.e500" % m, c["named_same"], fam); put("nj:x.rtext.%s.e500:k" % m, c["named_same"], fam)
        put("nj:x.rtext.%s.e500:n" % m, c["named"], fam); put("nj:x.rtext.%s.e500:retries" % m, c["retries"], fam)
        put("nj:x.rtext.%s.e500:text" % m, c["text"], fam)
    # ---- read-back 500 vs timeout (post hoc)
    for m in ORDER:
        src = "xp" if m in ORM else "v2"
        x, y = C(src, m, "base", "none", "timeout_after_commit"), C(src, m, "base", "none", "error_500_after_commit")
        if x and y:
            nj_test("nj:x.rb500.%s" % m, x, y, field="rb")


def v3_misorder(e):
    """misorder from the row (raw field) cross-checked against the ledger."""
    r = e.get("misorder")
    return bool(r)


# ------------------------------------------------------------------ real-API slice (Stripe test mode, R1)
def gt_violations(gt):
    """Duplicate committed payments in Stripe's own record: succeeded
    PaymentIntents grouped by (payee, amount, memo); each extra one is a duplicate."""
    c = Counter((pi.get("payee"), pi.get("amount"), pi.get("memo")) for pi in gt.get("payment_intents", [])
                if pi.get("status") == "succeeded")
    return sum(v - 1 for v in c.values() if v > 1)


def emit_real(D, v3, cells249, trans249):
    fam = "real-API slice (Stripe ground truth + transcripts)"
    rdir = os.path.join(D.res, "real_stripe")
    raw = last_by_id(read_jsonl(os.path.join(rdir, "raw.jsonl")))
    if not raw:
        return
    gts = {}
    for g in read_jsonl(os.path.join(rdir, "stripe_ground_truth.jsonl")):
        gts[(g["episode_id"], g.get("attempt_id"))] = g
    trs = {}
    for f in glob.glob(os.path.join(rdir, "transcripts", "*.json")):
        t = json.load(open(f))
        trs[t["episode_id"]] = t
    cfg = json.load(open(os.path.join(D.root, "configs", "real_stripe.json")))
    expect = int(cfg.get("expect_per_cell") or cfg["n_tasks"] * cfg["samples"])
    eps = {}
    gt_mismatch = 0
    for eid, r in raw.items():
        g = gts.get((eid, r.get("attempt_id")))
        t = trs.get(eid)
        if g is None or t is None:
            WARN.append("real: no ground truth / transcript for %s" % eid)
            continue
        vs = gt_violations(g)
        if (vs > 0) != bool(g.get("ivr_hit")):
            gt_mismatch += 1
        vl = ledger_violations(t.get("ledger", []))
        fe = features(t)
        e = {"task": r["task_id"], "sample": r["sample"], "model": r["model"], "kind": r["kind"],
             "vs": int(vs > 0), "viol_s": vs, "vl": int(vl > 0), "viol_l": vl, "refusal": refused(t),
             "rb": fe["rb_fail"] if fe else None, "key_first": fe["key_first_valid"] if fe else False,
             "truncated": bool(t.get("truncated")), "turn_cap": turn_cap_hit(t),
             "fired": fe is not None and fe["rb_fail"] is not None,
             "cond": (r["prompt"], r["mitigation"], r["failure_mode"]) if r["kind"] == "retry"
             else ("restart", r["resume_context"]),
             "s2": (any(b.get("type") == "tool_use" and b["name"] == "list_effects" for turn in t["sessions"][1]["turns"]
                        if turn.get("role") == "assistant" for b in turn["content"])
                    if "sessions" in t and len(t["sessions"]) >= 2 else None),
             "replayed": sum(1 for ev in (t.get("stripe") or {}).get("events", []) if ev.get("replayed")),
             "stripe_errors": r.get("stripe_errors") or 0, "cost": r.get("cost_usd") or 0}
        eps[eid] = e
    HYG["stripe_ground_truth_recomputed_vs_recorded"] = {"checked": len(eps), "mismatch": gt_mismatch}
    cells = defaultdict(list)
    for e in eps.values():
        cells[(e["model"], e["cond"])].append(e)
    FMR = {"timeout_after_commit": "toA", "error_500_after_commit": "e500", "status_unknown_after_commit": "w3"}
    sims = {"base.none.timeout_after_commit": "%s.%s.base.none.toA:ivr", "base.none.error_500_after_commit": "%s.%s.base.none.e500:ivr",
            "base.none.status_unknown_after_commit": "e1_6.%s.base.none.w3:ivr",
            "base.idem_keys.error_500_after_commit": None, "restart.amnesiac": "bh.%s.%s.amnesiac:givr",
            "restart.bare_hint": "bh.%s.%s.bare_hint:givr"}
    within = []
    outside = []
    agree = agree_exact = 0
    pay_within = []
    for (model, cond), es in sorted(cells.items()):
        m = short_model(model)
        sc = [e for e in es if not e["refusal"]]
        n = len(sc); k = sum(e["vs"] for e in sc); T = len({e["task"] for e in sc})
        p = k / n
        lo, hi = wilson_task(p, T)
        if cond[0] == "restart":
            key = "nj:real.%s.restart.%s:givr" % (m, cond[1]); ckey = "restart." + cond[1]
        else:
            key = "nj:real.%s.%s.%s.%s:ivr" % (m, cond[0], MIT[cond[1]], FMR[cond[2]]); ckey = ".".join(cond)
        put(key, f2(p), fam); put(key + ":ci", ci(lo, hi), fam); put(key + ":kn", "%d/%d" % (k, n), fam)
        put(key + ":k", k, fam); put(key + ":n", n, fam); put(key + ":T", T, fam)
        agree += sum(1 for e in sc if e["vs"] == e["vl"]); agree_exact += sum(1 for e in sc if e["viol_s"] == e["viol_l"])
        put(key.rsplit(":", 1)[0] + ":agree:kn", "%d/%d" % (sum(1 for e in sc if e["vs"] == e["vl"]), n), fam)
        # simulator comparison (the registered cell; the OR Qwen cells live in xp / e1_1)
        skey = ((cfg.get("sim_compare") or {}).get(model) or {}).get(ckey)
        if skey:
            sim_v = MINE.get("nj:" + skey)
            sim_ci = MINE.get("nj:" + skey + ":ci")
            if sim_v is not None and sim_ci is not None:
                slo, shi = [float(x) for x in sim_ci.strip("[]").split(",")]
                # compare on the unrounded simulator values held in NJVAL
                sv, slo, shi = NJVAL.get("nj:" + skey, (float(sim_v), slo, shi))
                inside = slo <= p <= shi
                within.append(inside)
                put(key + ":within_sim_ci", "yes" if inside else "no", fam)
                put(key + ":dsg", signed(p - sv), fam)
                if not inside:
                    outside.append(p > sv)
        if cond[0] != "restart":
            fired = [e for e in sc if e["fired"]]
            put(key.rsplit(":", 1)[0] + ":readback_after_failure:kn", "%d/%d" % (sum(1 for e in fired if e["rb"]), len(fired)), fam)
            put(key.rsplit(":", 1)[0] + ":stripe_replayed", sum(e["replayed"] for e in sc), fam)
        else:
            put(key.rsplit(":", 1)[0] + ":s2_lookups", "%d/%d" % (sum(1 for e in sc if e["s2"]), sum(1 for e in sc if e["s2"] is not None)), fam)
        # exploratory payment-only simulator subsets (xr.pay) and chain4-k0 restart graphs (xr.rst)
        if cond[0] != "restart" and skey:
            src, sm = skey.split(".")[0], skey.split(".")[1]
            srcs = {"v2": ["v2"], "xp": ["xp"], "e1_6": ["e1_6"], "e1_3": ["e1_3"], "e1_1": ["e1_1"]}[src]
            rows = [e for s in srcs for e in v3.cell(s, model, cond[0], cond[1], cond[2]) if e["task"].startswith("payments-")]
            kk = sum(e["v"] for e in rows); nn = len(rows); TT = len({e["task"] for e in rows})
            plo, phi = wilson_task(kk / nn, TT) if nn else (0, 1)
            xk = "nj:xr.pay.%s.%s.%s:ivr" % (m, MIT[cond[1]], FMR[cond[2]])
            put(xk + ":kn", "%d/%d" % (kk, nn), fam); put(xk + ":ci", ci(plo, phi), fam); put(xk + ":T", TT, fam)
            put(xk + ":T:w", word(TT), fam)
            w = plo <= p <= phi
            put(xk + ":within", "yes" if w else "no", fam); pay_within.append(w)
        if cond[0] == "restart":
            b = "oct" if m in CLAUDE else "xp"
            rows = [r for r in cells249.get((b, model, cond[1], "none"), {}).values()
                    if r["topology"] == "chain4" and r["interrupt_node"] == 0]
            rows = [r for r in rows if not (t249(trans249, b, r) and refused(t249(trans249, b, r)))]
            kk = sum(1 for r in rows if r["givr_hit"]); nn = len(rows); G = len({r["graph_id"] for r in rows})
            plo, phi = wilson_task(kk / nn, G) if nn else (0, 1)
            xk = "nj:xr.rst.%s.%s:givr" % (m, cond[1])
            put(xk + ":kn", "%d/%d" % (kk, nn), fam); put(xk + ":ci", ci(plo, phi), fam); put(xk + ":G", G, fam)
            put(xk + ":G:w", word(G), fam)
            w = plo <= p <= phi
            put(xk + ":within", "yes" if w else "no", fam)
            RST_WITHIN.append(w)
    put("nj:real.n_episodes", len(raw), fam)
    put("nj:real.agreement.ledger_vs_stripe:k", agree, fam); put("nj:real.agreement.ledger_vs_stripe:n", len(eps), fam)
    put("nj:real.agreement.ledger_vs_stripe:kn", "%d/%d" % (agree, len(eps)), fam)
    put("nj:real.agreement.ledger_vs_stripe:k_exact", agree_exact, fam)
    put("nj:real.H1.within", sum(within), fam); put("nj:real.H1.within:n", len(within), fam)
    put("nj:real.H1.within:lt_half", "yes" if 2 * sum(within) < len(within) else "no", fam)
    put("nj:xr.h1.outside", len(outside), fam); put("nj:xr.h1.outside:above", sum(outside), fam)
    put("nj:xr.h1.outside:below", len(outside) - sum(outside), fam)
    put("nj:xr.pay.within", sum(pay_within), fam); put("nj:xr.pay.within:n", len(pay_within), fam)
    put("nj:xr.rst.within", sum(RST_WITHIN), fam); put("nj:xr.rst.within:n", len(RST_WITHIN), fam)
    put("nj:xr.rst.within:n:w", word(len(RST_WITHIN)), fam)
    # search API vs list API
    att = {eid: r.get("attempt_id") for eid, r in raw.items()}
    srch = [x for x in last_by_id(read_jsonl(os.path.join(rdir, "search_check.jsonl"))).values()
            if att.get(x["episode_id"]) == x.get("attempt_id")]
    if srch:
        put("nj:real.search_check.agree:kn", "%d/%d" % (sum(1 for x in srch if set(x["search_ids"]) == set(x["list_ids"])), len(srch)), fam)
    # real 500 minus timeout, Stripe truth, per model
    for model in sorted({e["model"] for e in eps.values()}):
        m = short_model(model)
        x = rates((e["task"], e["vs"]) for e in eps.values() if e["model"] == model and e["cond"] == ("base", "none", "timeout_after_commit") and not e["refusal"])
        y = rates((e["task"], e["vs"]) for e in eps.values() if e["model"] == model and e["cond"] == ("base", "none", "error_500_after_commit") and not e["refusal"])
        r = perm_test(x, y)
        if r:
            put("nj:xr.real.%s.e500_minus_toA:d" % m, signed(r["d"]), fam); put("nj:xr.real.%s.e500_minus_toA:p" % m, pfmt(r["p"]), fam)
            put("nj:xr.real.%s.e500_minus_toA:p:pe" % m, pe(r["p"]), fam)
        keyed = [e for e in eps.values() if e["model"] == model and e["kind"] == "retry" and e["cond"][1] == "idem_keys" and not e["refusal"]]
        kf = sum(1 for e in keyed if e["key_first"])
        put("nj:xr.real.%s.keys.e500:keyfirst" % m, "%d/%d" % (kf, len(keyed)), fam)
        put("nj:xr.real.%s.keys.e500:keyfirst:k" % m, kf, fam)
    retry = [e for e in eps.values() if e["kind"] == "retry"]
    put("nj:xr.real.hygiene:refusals", sum(1 for e in eps.values() if e["refusal"]), fam)
    put("nj:xr.real.hygiene:truncated", sum(1 for e in eps.values() if e["truncated"]), fam)
    put("nj:xr.real.hygiene:turn_cap", sum(1 for e in eps.values() if e["turn_cap"]), fam)
    put("nj:xr.real.hygiene:stripe_errors", sum(e["stripe_errors"] for e in eps.values()), "raw field passthrough")
    put("nj:xr.real.hygiene:failure_fired", "%d/%d" % (sum(1 for e in retry if e["fired"]), len(retry)), fam)
    ra = sum(e["cost"] for e in eps.values() if not e["model"].startswith("openrouter:"))
    ro = sum(e["cost"] for e in eps.values() if e["model"].startswith("openrouter:"))
    put("cost:r1", "%.2f" % (ra + ro), "cost (raw cost_usd)"); put("cost:r1_anthropic", "%.2f" % ra, "cost (raw cost_usd)")
    put("cost:r1_openrouter", "%.2f" % ro, "cost (raw cost_usd)")
    COST["r1"] = (ra, ro)
    put("real:ready", "yes" if len(raw) >= 18 * expect else "no", fam)


RST_WITHIN = []
NJVAL = {}   # unrounded simulator cell values: key -> (p, lo, hi)
COST = {}


def record_njvals(v3, cells249, trans249):
    """Unrounded values of the simulator cells the real slice is compared with."""
    for key, eps in v3.cells.items():
        if key[0] == "xp_first":
            continue
        st = stat_cell(eps)
        if st["n"]:
            NJVAL["nj:" + nj_name(*key) + ":ivr"] = (st["p"], st["ci_task"][0], st["ci_task"][1])
    for (b, model, ctx, mit), eps in cells249.items():
        if mit != "none" or b not in ("oct", "xp"):
            continue
        rows = [r for r in eps.values() if r["topology"] in BENCH]
        rows = [r for r in rows if not (t249(trans249, b, r) and refused(t249(trans249, b, r)))]
        if not rows:
            continue
        k = sum(r["givr_hit"] for r in rows); n = len(rows); G = len({r["graph_id"] for r in rows})
        lo, hi = wilson_task(k / n, G)
        NJVAL["nj:bh.%s.%s.%s:givr" % (short_model(model), b, ctx)] = (k / n, lo, hi)


# ------------------------------------------------------------------ sampling table, figures, misc
def emit_sampling(D, trans209, trans249):
    fam = "sampling (transcripts)"
    info = defaultdict(lambda: dict(temps=Counter(), dates=set(), served=Counter(), upstream=Counter(),
                                    sigs=defaultdict(dict), refusals=0, mt=[]))

    def sig(turn_lists):
        out = []
        for turns in turn_lists:
            for turn in turns:
                if turn.get("role") == "assistant":
                    for x in turn["content"]:
                        if x.get("type") == "text":
                            out.append(x.get("text", ""))
                        elif x.get("type") == "tool_use":
                            out.append((x["name"], json.dumps(x.get("input"), sort_keys=True)))
        return json.dumps(out)

    def canon(turns):
        """E0-5 text variant: assistant + tool_result turns without ids, ack
        ids masked, thinking blocks dropped."""
        out = []
        for tu in turns:
            if tu.get("role") not in ("assistant", "tool_results"):
                continue
            c = []
            for b in tu["content"]:
                b = dict(b); b.pop("id", None); b.pop("tool_use_id", None)
                if "content" in b:
                    b["content"] = re.sub(r"ok-[0-9a-f]{8}", "ok-X", str(b["content"]))
                if b.get("type") == "thinking":
                    continue
                c.append(b)
            out.append((tu["role"], c))
        return json.dumps(out, sort_keys=True)

    for (b, model, prompt, mit, fm, task, s), t in trans209.items():
        run = "209" if b == "jul" else b
        d = info[(run, short_model(model))]
        d["temps"][str(t.get("temperature", "absent"))] += 1
        if t.get("started_utc"):
            d["dates"].add(t["started_utc"][:10])
        d["served"][str(t.get("model_served"))] += 1
        for turn in t.get("turns", []):
            up = (turn.get("provider_meta") or {}).get("upstream_provider")
            if up:
                d["upstream"][up] += 1
        d["sigs"][("p1", prompt, mit, fm, task)][s] = canon(t["turns"]) if b == "jul" else sig([t.get("turns", [])])
        d["refusals"] += refused(t)
        if b == "jul":
            d["mt"].append(os.stat(t["_path"]).st_mtime)
    for (b, model, ctx, mit, gid, k, s), t in trans249.items():
        run = "249" if b == "jul" else b
        d = info[(run, short_model(model))]
        d["temps"][str(t["temperature"]) if "temperature" in t else "absent"] += 1
        if t.get("started_utc"):
            d["dates"].add(t["started_utc"][:10])
        d["served"][str(t.get("model_served"))] += 1
        for ses in t["sessions"]:
            for turn in ses["turns"]:
                up = (turn.get("provider_meta") or {}).get("upstream_provider")
                if up:
                    d["upstream"][up] += 1
        d["sigs"][("p2", ctx, mit, gid, k)][s] = json.dumps([canon(x["turns"]) for x in t["sessions"]]) if b == "jul" \
            else sig([x["turns"] for x in t["sessions"]])
        d["refusals"] += refused(t)
        if b == "jul":
            d["mt"].append(os.stat(os.path.join(D.res, "transcripts2", t["_file"])).st_mtime)
    tier0 = {}
    p0 = os.path.join(D.res, "analysis_v2", "e0_5_sampling_protocol.csv")
    if os.path.exists(p0):
        for r in csv.DictReader(open(p0)):
            tier0[(r["paper"], short_model(r["model"]))] = r
    for (run, ms), d in sorted(info.items()):
        q = "samp:%s:%s" % (run, ms)
        temps = set(d["temps"])
        lab = "0" if temps <= {"0.0", "0"} else "default" if temps == {"None"} else \
            "0 sent, unverified" if temps == {"unverified"} else None if temps == {"absent"} else "mixed"
        if lab is None:
            put(q + ":temp_logged", "inferred", fam)
            t0 = tier0.get((run, ms))
            if t0:
                put(q + ":temp", "0" if t0["temperature_effective"].startswith("0.0") else "default", "Tier-0 CSV passthrough")
        else:
            put(q + ":temp", lab, fam); put(q + ":temp_logged", "logged", fam)
        put(q + ":temp_n", sum(d["temps"].values()), fam)
        pairs = [v for v in d["sigs"].values() if 0 in v and 1 in v]
        put(q + ":ident", "%d/%d" % (sum(1 for v in pairs if v[0] == v[1]), len(pairs)), fam)
        put(q + ":ident:n", len(pairs), fam)
        if d["dates"]:
            dates = sorted(d["dates"])
            put(q + ":dates", dates[0][5:] if dates[0] == dates[-1] else "%s--%s" % (dates[0][5:], dates[-1][5:]), fam)
        elif d["mt"] and (not os.path.exists(p0) or min(d["mt"]) < os.stat(p0).st_mtime):
            # July transcripts carry no timestamp; the paper reports the file
            # modification times. They are only meaningful where the files kept
            # their original mtimes (older than the Tier-0 table derived from
            # them); a fresh checkout rewrites them all.
            u = lambda ts: datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%m-%d %H:%M")
            put(q + ":dates", "%s--%s" % (u(min(d["mt"])), u(max(d["mt"]))), "file mtimes")
        elif d["mt"]:
            WARN.append("%s:dates: transcript mtimes were rewritten here; run dates not reproducible" % q)
        if run not in ("209", "249"):
            put(q + ":refusals", d["refusals"], fam); put(q + ":episodes", sum(d["temps"].values()), fam)
            put(q + ":upstream_n", len(d["upstream"]), fam)
            sv = Counter()
            for s_, c in d["served"].items():
                sv[s_.split(" @")[0] if s_ != "None" else "not recorded"] += c
            put(q + ":served", ", ".join(sorted(sv)), "string (served id)")
            put(q + ":served_dated", "yes" if all(re.search(r"-\d{8}$", s_) for s_ in sv) else "no", fam)
            up_ep = Counter()
            for s_, c in d["served"].items():
                if " @" in s_:
                    up_ep[s_.split(" @", 1)[1]] += c
            if up_ep:
                put(q + ":served_counts", ", ".join("%s %d" % (u, c) for u, c in sorted(up_ep.items(), key=lambda x: (-x[1], x[0]))), fam)
            model_full = next((FULL[ms] for _ in [0] if ms in FULL), ms)
            put(q + ":id", model_full.replace("openrouter:", ""), "string (model id)")
            put(q + ":dated", "yes" if re.search(r"-\d{8}$", model_full) else "no", fam)
        else:
            model_full = FULL.get(ms, ms)
            put(q + ":id", model_full.replace("openrouter:", ""), "string (model id)")
            put(q + ":dated", "yes" if re.search(r"-\d{8}$", model_full) else "no", fam)
    hk = {run: MINE.get("samp:%s:haiku:temp" % run) for run in ("209", "249", "oct")}
    put("haiku:temp:jul209", hk["209"] or "?", fam); put("haiku:temp:jul249", hk["249"] or "?", fam)
    put("haiku:temp:oct", hk["oct"] or "?", fam)
    put("haiku:temp:changed", "yes" if hk["oct"] and hk["209"] and hk["oct"] != hk["209"] else "no", fam)
    put("pocorr:haiku:temp_chain", hk["249"] or "?", fam); put("pocorr:haiku:temp_fan", hk["oct"] or "?", fam)
    # July identical tool-call pairs
    by = defaultdict(dict)
    for (b, model, prompt, mit, fm, task, s), t in trans209.items():
        if b == "jul":
            by[(short_model(model), prompt, mit, fm, task)][s] = json.dumps(
                [(e["name"], json.dumps(e["args"], sort_keys=True), e["key"]) for k, e in seq_of(t["turns"]) if k == "call"])
    n, k = Counter(), Counter()
    for key, d in by.items():
        if 0 in d and 1 in d:
            n[key[0]] += 1; k[key[0]] += d[0] == d[1]
    for m in n:
        put("samp:209:%s:ident_calls" % m, "%d/%d" % (k[m], n[m]), fam)


def emit_figures(D):
    fam = "figure transcript facts"
    seam = {"am": "openrouter:qwen_qwen3.6-27b-20260422__amnesiac__none__chain4-27d504eba1__k2__s0.json",
            "bh": "openrouter:qwen_qwen3.6-27b-20260422__bare_hint__none__chain4-27d504eba1__k2__s0.json"}
    for tag, fn in seam.items():
        p = os.path.join(D.res, "crossprovider", "transcripts2", fn)
        if not os.path.exists(p):
            continue
        t = json.load(open(p))
        put("figs:%s:dups" % tag, ledger_violations(t["ledger"]) if "ledger" in t else t["score"]["violations"], fam)
        put("figs:%s:k" % tag, t["interrupt_node"], fam); put("figs:%s:committed" % tag, t["interrupt_node"] + 1, fam)
        put("figs:%s:graph" % tag, t["graph_id"], "string"); put("figs:%s:instr" % tag, t["instruction"], "string")
        for i, ses in enumerate(t["sessions"]):
            nm = [x["name"] for turn in ses["turns"] if turn.get("role") == "assistant" for x in turn["content"] if x.get("type") == "tool_use"]
            put("figs:%s:s%dcalls" % (tag, i + 1), ", ".join(nm), "string"); put("figs:%s:s%dn" % (tag, i + 1), len(nm), fam)
            put("figs:%s:s%dlooked" % (tag, i + 1), "yes" if "list_effects" in nm else "no", fam)
        put("figs:%s:sample" % tag, t.get("sample", 0), fam)
    # cross-provider launches and finish reasons
    p = os.path.join(D.res, "crossprovider", "run.log")
    if os.path.exists(p):
        starts = re.findall(r"=== start \w+ (\w+) (\d+) (\d+:\d+):\d+ (\w+) (\d{4})", open(p).read())
        for i, (mon, day, hm, tz, yr) in enumerate(starts[:2]):
            put("dup:xp:launch%d" % (i + 1), "%s %s" % (hm, tz), "log timestamp")
        if starts:
            put("dup:xp:date", "%d %s %s" % (int(starts[0][1]), datetime.datetime.strptime(starts[0][0], "%b").strftime("%B"), starts[0][4]), "log timestamp")
    for tag, dirs in (("xp", [os.path.join(D.res, "crossprovider", "sweep_transcripts"), os.path.join(D.res, "crossprovider", "transcripts2")]),
                      ("t1", glob.glob(os.path.join(D.res, "tier1", "e1_*", "transcripts")))):
        c = Counter()
        for d in dirs:
            for f in glob.glob(os.path.join(d, "*.json")):
                t = json.load(open(f))
                m = short_model(t.get("model", ""))
                for x in flat_turns(t):
                    if x.get("role") != "assistant" or not any(b.get("type") == "tool_use" for b in x["content"]):
                        continue
                    if (x.get("provider_meta") or {}).get("native_finish_reason") == "length":
                        c[(m, "length")] += 1
                    if x.get("stop_reason") == "error":
                        c[(m, "error")] += 1
        for m in ("qwen", "gemini", "gpt"):
            for r in ("length", "error"):
                put("fin:%s:%s:%s" % (tag, m, r), c[(m, r)], "finish reasons (transcripts)")


def emit_cost(D):
    fam = "cost (transcript usage x list prices; launcher logs)"
    src = open(os.path.join(D.root, "src2", "sweep2.py")).read()
    prices = {m: (float(a), float(b)) for m, a, b in re.findall(r'"(claude-[\w-]+)": \(([\d.]+), ([\d.]+)\)', src)}
    oct_ = 0.0
    for f in glob.glob(os.path.join(D.res, "claude_batch_2026-10", "transcripts2", "*.json")):
        t = json.load(open(f))
        pr = next((v for k, v in prices.items() if t.get("model", "").startswith(k)), None)
        if not pr:
            continue
        for s in t.get("sessions", []):
            u = (s.get("meta") or {}).get("usage") or {}
            oct_ += ((u.get("input") or 0) * pr[0] + (u.get("output") or 0) * pr[1]) / 1e6
    put("cost:claude_oct", "%.2f" % oct_, fam)
    a = o = None
    for log in (os.path.join(D.res, "tier1", "run.log"), os.path.join(D.res, "tier1", "e1_6_oct", "run.log")):
        if not os.path.exists(log):
            continue
        for line in open(log, encoding="utf-8", errors="replace"):
            m = re.search(r"logged spend under .*: Anthropic \$([\d.]+).*OpenRouter \$([\d.]+)", line)
            if m and (a is None or float(m.group(1)) + float(m.group(2)) >= a + o):
                a, o = float(m.group(1)), float(m.group(2))
    lim = re.search(r"\$(\d+) key limit", open(os.path.join(D.root, "run_crossprovider.sh")).read())
    if lim:
        put("cost:or_key_limit", lim.group(1), "design constant (script)")
    r2c = COST.get("r2")
    if r2c is not None:
        put("cost:r2", "%.2f" % r2c, fam)
    r3c = COST.get("r3")
    if r3c is not None:      # logged R3 spend (cost_usd per row: Anthropic list prices, OpenRouter usage.cost)
        put("cost:r3", "%.2f" % sum(r3c), fam); put("cost:r3_anthropic", "%.2f" % r3c[0], fam)
        put("cost:r3_openrouter", "%.2f" % r3c[1], fam)
    r4c = COST.get("r4")
    if r4c is not None:      # logged R4 spend, same convention as R3
        put("cost:r4", "%.2f" % sum(r4c), fam); put("cost:r4_anthropic", "%.2f" % r4c[0], fam)
        put("cost:r4_openrouter", "%.2f" % r4c[1], fam)
    if r2c is not None and COST.get("r2_completion") is not None:
        put("cost:r2_completion", "%.2f" % COST["r2_completion"], fam)
        put("cost:r2_before_completion", "%.2f" % (r2c - COST["r2_completion"]), fam)
    if a is not None:
        put("cost:tier1_anthropic", "%.2f" % a, fam); put("cost:tier1_openrouter", "%.2f" % o, fam)
        ra, ro = COST.get("r1", (0.0, 0.0))
        put("cost:logged_total", "%.2f" % (oct_ + a + o + ra + ro), fam)
        if r2c is not None:
            # CONVENTION: the total with R2 is the sum of the two printed (rounded)
            # parts, so that it equals the parts the sentence prints
            put("cost:logged_total_r2", "%.2f" % (float("%.2f" % (oct_ + a + o + ra + ro)) + float("%.2f" % r2c)), fam)
            r3c = COST.get("r3")
            if r3c is not None:
                # CONVENTION (as for R2): the total with R2 and R3 is the sum of the printed parts
                put("cost:logged_total_r23", "%.2f" % (float(MINE["cost:logged_total_r2"]) + float("%.2f" % sum(r3c))), fam)
                r4c = COST.get("r4")
                if r4c is not None:      # and R4, again as the sum of the printed parts
                    put("cost:logged_total_r234", "%.2f" % (float(MINE["cost:logged_total_r23"]) + float("%.2f" % sum(r4c))), fam)
        if lim:
            # the pre-R2 total with its composition; the clause about the
            # unlogged cross-provider runs moved into the paper's sentence
            put("cost:total", "%.2f (Claude %.2f at list prices; OpenRouter %.2f" % (oct_ + a + o + ra + ro, oct_ + a + ra, o + ro)
                + ("; the real-API check %.2f of the total, with no Stripe charge" % (ra + ro) if ra + ro else "")
                + ")", fam)


def emit_scripted(D):
    fam = "scripted-agent CSV (half-up formatting)"
    rows = list(csv.DictReader(open(os.path.join(D.res, "summary_by_failure.csv"))))
    for r in rows:
        k = ":".join(["scr1", r["agent"], r["retry_mode"], MIT[r["mitigation"]], FM[r["failure_mode"]]])
        put(k + ":ivr", f2(float(r["ivr"])), fam); put(k + ":succ", f2(float(r["success"])), fam)
    put("scr1:ntasks", rows[0]["n_tasks"], fam)
    tb = [r for r in rows if r["failure_mode"] == "timeout_before_commit"]
    put("scr1:toB:cells", len(tb), fam); put("scr1:toB:allzero", "yes" if all(float(r["ivr"]) == 0 for r in tb) else "no", fam)
    put("scr1:rcpt:allzero", "yes" if all(float(r["ivr"]) == 0 for r in rows if r["mitigation"] == "runtime_receipts") else "no", fam)
    g = list(csv.DictReader(open(os.path.join(D.res, "graph_summary.csv"))))
    put("scr2:episodes", "{:,}".format(sum(int(r["n_episodes"]) for r in g)), fam)
    for r in g:
        k = ":".join(["scr2", r["resume_context"], MIT[r["mitigation"]], r["key_posture"], r["interrupt_timing"]])
        put(k + ":givr", f2(float(r["givr"])), fam); put(k + ":stale", f2(float(r["order_violation_rate"])), fam)
        put(k + ":meanviol", "%.2f" % float(r["mean_violations"]), fam); put(k + ":n", r["n_episodes"], fam)
        put(k + ":comp", f2(float(r["completion"])), fam)
    for ctx in ("amnesiac", "plan_only"):
        for post in ("naive", "stable"):
            sel = [r for r in g if r["resume_context"] == ctx and r["mitigation"] == "none" and r["key_posture"] == post]
            n = sum(int(r["n_episodes"]) for r in sel)
            put("scr2pool:%s:%s:givr" % (ctx, post), f2(sum(float(r["givr"]) * int(r["n_episodes"]) for r in sel) / n), fam)
            put("scr2pool:%s:%s:stale" % (ctx, post), f2(sum(float(r["order_violation_rate"]) * int(r["n_episodes"]) for r in sel) / n), fam)
            put("scr2pool:%s:%s:n" % (ctx, post), "{:,}".format(n), fam)
    # v1 (July submission) CSV values, response letter only
    for r in csv.DictReader(open(os.path.join(D.res, "sweep_summary.csv"))):
        q = ":".join(["v1", short_model(r["model"]), r["prompt"], MIT[r["mitigation"]], FM[r["failure_mode"]]])
        put(q + ":ivr", f2(float(r["ivr"])), fam); put(q + ":succ", f2(float(r["success"])), fam)


# ------------------------------------------------------------------ matcher audit (E0-3) and C1 audit
def loose(v):
    return re.sub(r"[^0-9a-z]", "", str(v).casefold())


def norm_nofix(v):
    """The normalization before the C1 fix (no ISO zero-seconds rule)."""
    s = " ".join(str(v).split()).strip("'\"").strip()
    if _NUM.match(s):
        try:
            return format(Decimal(s.lstrip("$€£ ").replace(",", "")).normalize(), "f")
        except InvalidOperation:
            pass
    return s.casefold()


def classify_pair(tool, a, b, nf=norm):
    import difflib
    fs = CANON[tool]
    raw = all(str(a.get(f)) == str(b.get(f)) for f in fs)
    paper = all(nf(a.get(f)) == nf(b.get(f)) for f in fs)
    if raw:
        return "raw_identical"
    if paper:
        return "normalization_only"
    sims = [difflib.SequenceMatcher(None, str(a.get(f)), str(b.get(f))).ratio() for f in fs]
    if all(loose(a.get(f)) == loose(b.get(f)) for f in fs) or min(sims) >= 0.8:
        return "near_miss"
    return "distinct"


def emit_matcher(D, trans209, cells209):
    fam = "matcher audit (July transcripts)"
    counts = Counter(); ncalls = 0; noncanon = 0; normtasks = set()
    near_all = near_all_nofix = 0
    for (b, model, prompt, mit, fm, task, s), t in trans209.items():
        calls = [(e["name"], e["args"]) for k, e in seq_of(t.get("turns") or flat_turns(t)) if k == "call" and e["name"] in CANON]
        for i in range(len(calls)):
            for j in range(i + 1, len(calls)):
                if calls[i][0] != calls[j][0]:
                    continue
                if b == "jul":
                    c = classify_pair(calls[i][0], calls[i][1], calls[j][1])
                    counts[c] += 1
                    if c == "normalization_only":
                        normtasks.add(task)
                near_all += classify_pair(calls[i][0], calls[i][1], calls[j][1]) == "near_miss"
                near_all_nofix += classify_pair(calls[i][0], calls[i][1], calls[j][1], norm_nofix) == "near_miss"
        if b == "jul":
            ncalls += len(calls)
            noncanon += sum(1 for n_, a in calls if set(a) - set(CANON[n_]))
    put("match:pairs", sum(counts.values()), fam); put("match:raw", counts["raw_identical"], fam)
    put("match:normonly", counts["normalization_only"], fam); put("match:near", counts["near_miss"], fam)
    put("match:distinct", counts["distinct"], fam); put("match:effcalls", "{:,}".format(ncalls), fam)
    put("match:noncanon", noncanon, fam); put("match:normtasks", len(normtasks), fam)
    put("nj:c1.audit.near_miss", near_all, fam); put("nj:c1.audit.near_miss_before_fix", near_all_nofix, fam)
    put("nj:c1.audit.near_miss_before_fix:w", word(near_all_nofix), fam)
    # strict (raw-string) matcher on the committed ledger, July cells
    changed = 0
    for (b, model, prompt, mit, fm, tags), eps in cells209.items():
        q = ":".join([b, short_model(model), prompt, MIT.get(mit, mit), FM.get(fm, fm)] + list(tags))
        if b != "jul" or STALE_LEDGER.get(q):
            continue      # a stale (pre-fix) stored ledger cannot be re-matched
        ks = kp = 0; n = 0
        for r in eps.values():
            t = t209(trans209, b, r)
            if t is None or "ledger" not in t:
                continue
            n += 1
            ks += ledger_violations(t["ledger"], strict=True) > 0
            kp += r["violations"] > 0
        if n and ks != kp:
            changed += 1
        if short_model(model) == "haiku" and (prompt, mit, fm) == ("base", "none", "error_500_after_commit"):
            put("match:strict:haiku500:ivr", f2(ks / n), fam); put("match:strict:haiku500:kn", "%d/%d" % (ks, n), fam)
            put("match:paper:haiku500:kn", "%d/%d" % (kp, n), fam)
    put("match:strict:cells_changed", changed, fam); put("match:strict:cells_changed:w", word(changed), fam)
    hc = os.path.join(D.res, "analysis_v2", "e0_3_handcheck.csv")
    if os.path.exists(hc):
        rows = list(csv.DictReader(open(hc)))
        put("match:hand:controls", sum(1 for r in rows if r["auto_class"] != "normalization_only"), "Tier-0 CSV passthrough")
        # the hand verdicts (Y = same action) on the normalization-only pairs and on the controls
        fh = "hand-check CSV (E0-3 verdict column)"
        yes = lambda r: r["your_verdict_same_action_Y_N"].strip().upper() == "Y"
        nrm = [r for r in rows if r["auto_class"] == "normalization_only"]
        ctl = [r for r in rows if r["auto_class"] != "normalization_only"]
        put("match:hand:n", len(rows), fh); put("match:hand:norm:n", len(nrm), fh)
        put("match:hand:norm:same", sum(map(yes, nrm)), fh); put("match:hand:fp", sum(1 for r in nrm if not yes(r)), fh)
        put("match:hand:clause", "our hand check judged %d of these %d pairs to be the same action" % (sum(map(yes, nrm)), len(nrm)), fh)
        put("match:hand:controls:clause", ", of which it judged %d the same action" % sum(map(yes, ctl)), fh)
        for r in rows:
            m = re.search(r'"(\w+)":\s*\["([^"]*)",\s*"([^"]*)"\]', r["reason"])
            if r["auto_class"] == "normalization_only" and m:
                put("match:example:a", m.group(2), "string (example)"); put("match:example:b", m.group(3), "string (example)")
                put("match:example:norm", norm(m.group(2)), "string (example)")
                break
    # C1 re-score: episodes whose score changed between raw.jsonl and raw_v2.jsonl
    changed_eps = 0
    for p2 in glob.glob(os.path.join(D.res, "**", "*_v2.jsonl"), recursive=True):
        p1 = p2.replace("_v2.jsonl", ".jsonl")
        if not os.path.exists(p1) or os.path.basename(p1) == "sweep_raw.jsonl" and os.path.dirname(p1) == D.res \
                or os.path.basename(p1) == "sweep2_raw.jsonl" and os.path.dirname(p1) == D.res:
            continue
        old = last_by_id(read_jsonl(p1)); new = last_by_id(read_jsonl(p2))
        changed_eps += sum(1 for e, r in new.items() if e in old and old[e]["violations"] != r["violations"])
    put("nj:c1.rescore.score_changed", changed_eps, fam); put("nj:c1.rescore.score_changed:w", word(changed_eps), fam)


def emit_design(D):
    """Design constants read from the source files by regex (not imported)."""
    fam = "design constant (source regex)"

    def grab(path, pat, cast=str):
        try:
            m = re.search(pat, open(os.path.join(D.root, path)).read(), re.M)
            return cast(m.group(1)) if m else None
        except OSError:
            return None
    v = grab("src/model_agent.py", r"^MAX_TURNS\s*=\s*(\d+)"); v and put("design:maxturns1", v, fam)
    v = grab("src2/sweep2.py", r"^MAX_TURNS_SESSION\s*=\s*(\d+)"); v and put("design:maxturns2", v, fam)
    v = grab("src/model_agent.py", r'BENCH_MAX_TOKENS",\s*"(\d+)"'); v and put("design:maxtokens", v, fam)
    v = grab("run_crossprovider.sh", r"BENCH_MAX_TOKENS=(\d+)"); v and put("design:maxtokens_or", v, fam)
    v = grab("src/model_agent.py", r"^TEMPERATURE\s*=\s*([\d.]+)")
    v and put("design:temp_req", v.rstrip("0").rstrip(".") if "." in v else v, fam)
    for k, pat in (("design:schema:keydesc", r'"idempotency_key"\]\s*=\s*\{[^}]*"description":\s*"([^"]+)"'),
                   ("design:schema:tooldesc", r'"description":\s*f"(Execute \{tool\}\.[^"]*)"'),
                   ("design:schema:listdesc", r'"name":\s*"list_effects",\s*"description":\s*"([^"]+)"')):
        v = grab("src/model_agent.py", pat)
        v and put(k, v.replace("{tool}", "<tool>"), "string (schema text)")
    put("design:domains", len(CANON), fam); put("design:domains:w", word(len(CANON)), fam)
    for k, path, pat in (("design:retry_modes", "src/agents.py", r"^RETRY_MODES\s*=\s*\(([^)]*)\)"),
                         ("design:failure_modes", "src/world.py", r"^FAILURE_MODES\s*=\s*\(([^)]*)\)"),
                         ("design:families", "src/tasks.py", r"^FAMILIES\s*=\s*\(([^)]*)\)"),
                         ("design:topologies", "src2/graph.py", r"^TOPOLOGIES\s*=\s*\(([^)]*)\)"),
                         ("design:resume_contexts", "src2/resume.py", r"^RESUME_CONTEXTS\s*=\s*\(([^)]*)\)")):
        v = grab(path, pat)
        if v:
            put(k, len(re.findall(r'"\w+"', v)), fam); put(k + ":w", word(len(re.findall(r'"\w+"', v))), fam)
    v = grab("src/tasks.py", r"^TEMPLATE_COUNT\s*=.*#\s*(\d+)"); v and (put("design:templates", v, fam), put("design:templates:w", word(v), fam))
    vs = sorted({re.search(r"anthropic-([\d.]+)\.dist-info", p).group(1) for p in
                 glob.glob(os.path.join(D.root, ".venv*", "lib", "*", "site-packages", "anthropic-*.dist-info"))})
    if vs:
        put("design:sdk_version", ", ".join(vs), "string (environment)")
    try:
        cfg = json.load(open(os.path.join(D.root, "configs", "real_stripe.json")))
        put("design:stripe_version", cfg["stripe_api_version"], "string (config)")
        put("design:maxturns_long", json.load(open(os.path.join(D.root, "configs", "e1_5_long_chains.json")))["max_turns"], fam)
    except (OSError, KeyError, ValueError):
        pass
    put("design:samples", 2, fam); put("design:samples:w", "two", fam)


def emit_prereg(D, trans209):
    """Registration timing (git, read-only) and counts from the registration text."""
    fam = "registration (git dates / text counts)"
    def git(*a):
        try:
            return subprocess.run(["git", "-C", D.root, *a], capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:
            return []
    reg = git("log", "--diff-filter=A", "--format=%cI", "--", "analysis/preregistration.md")
    # CONVENTION: "Tier-1 episodes" = every results/tier1/e1_*/transcripts
    # directory, including the October sensitivity rerun e1_6_oct
    starts = [t["started_utc"] for (b, *_), t in trans209.items()
              if b in ("e10", "e13", "e12", "e14", "e15", "e16", "tier1-e1-6-oct") and t.get("started_utc")]
    if reg:
        r = reg[-1]
        put("prereg:commit_utc", r.replace("T", " ")[:16].replace("+00:00", "") + " UTC", fam); put("prereg:date", r[:10], fam)
    if starts:
        put("prereg:first_data_utc", min(starts).replace("T", " ")[:16] + " UTC", fam)
        put("prereg:last_data_utc", max(starts).replace("T", " ")[:16] + " UTC", fam)
    if reg and starts:
        a = datetime.datetime.fromisoformat(reg[-1]); z = datetime.datetime.fromisoformat(min(starts).replace("Z", "+00:00"))
        put("prereg:before_data", "yes" if a < z else "no", fam)
        mins = round((z - a).total_seconds() / 60)
        put("prereg:lead_hours", "%.1f" % ((z - a).total_seconds() / 3600), fam)
        put("prereg:lead_hm", "%d h %d min" % (mins // 60, mins % 60), fam)
    r1 = git("log", "--diff-filter=A", "--format=%cI", "--", "configs/real_stripe.json")
    rraw = os.path.join(D.res, "real_stripe", "raw.jsonl")
    rst = [r.get("started_utc") for r in read_jsonl(rraw)] if os.path.exists(rraw) else []
    rst = [x for x in rst if x]
    if r1 and rst:
        a = datetime.datetime.fromisoformat(r1[-1]).astimezone(datetime.timezone.utc)
        z = datetime.datetime.fromisoformat(min(rst).replace("Z", "+00:00"))
        put("r1:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC", fam)
        put("r1:first_data_utc", min(rst).replace("T", " ")[:16] + " UTC", fam)
        put("r1:last_data_utc", max(rst).replace("T", " ")[:16] + " UTC", fam)
        put("r1:before_data", "yes" if a < z else "no", fam)
        MON = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
        put("r1:commit_dm", "%d %s" % (a.day, MON[a.month - 1]), fam)
        put("r1:commit_hm", a.strftime("%H:%M") + " UTC", fam)
        try:
            # the R1 heading of the registration carries the committer's local date (some UTC
            # offset from -12 to +14 h); the paper states times in UTC and names that date once
            from datetime import timedelta as _td, date as _date
            _h = re.search(r"^## R1:.*registered (\d{4})-(\d{2})-(\d{2})", open(os.path.join(D.root, "analysis", "preregistration.md")).read(), re.M)
            _d = _date(int(_h.group(1)), int(_h.group(2)), int(_h.group(3))) if _h else None
            pac = _d if _d in {(a + _td(hours=h)).date() for h in range(-12, 15)} else a.date()
            put("r1:commit_local_dm", "%d %s" % (pac.day, MON[pac.month - 1]), fam)
        except Exception:
            pass
    pr = os.path.join(D.root, "analysis", "preregistration.md")
    if os.path.exists(pr):
        full = open(pr).read()
        put("prereg:deviations", len(re.findall(r"^\d+\.\s", full.split("## Deviations log", 1)[-1], re.M)), fam)
        MON = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
        for rx, sep in (("R1", ":"), ("R2", "."), ("R3", "."), ("R4", ".")):
            m = re.search(r"^## %s%s.*registered (\d{4})-(\d{2})-(\d{2})" % (rx, re.escape(sep)), full, re.M)
            if m:
                put("%s:reg_dm" % rx.lower(), "%d %s" % (int(m.group(3)), MON[int(m.group(2)) - 1]), fam)
        nexp = len(re.findall(r"^## E1-\d", full, re.M))
        put("prereg:experiments", nexp, fam); put("prereg:experiments:w", word(nexp), fam)
        def sec(name):
            m = re.search(r"^## " + re.escape(name) + r".*?(?=^## )", full, re.M | re.S)
            return m.group(0) if m else ""
        m = re.search(r"within ±([\d.]+)", sec("E1-0")); m and put("prereg:e10_tol", m.group(1), fam)
        m = re.search(r"≥ ([\d.]+), tested", sec("E1-1")); m and put("prereg:e11_margin", f2(float(m.group(1))), fam)
        m = re.search(r"is rare \(≤ ([\d.]+)\)", sec("E1-5")); m and put("prereg:e15_stale_max", f2(float(m.group(1))), fam)
        m = re.search(r"H1: controls 1–2 are ≈ 0 \(task-CI upper bound ≤ ([\d.]+)\)", sec("E1-1")); m and put("prereg:e11_ctrl_max", f2(float(m.group(1))), fam)
        m = re.search(r"≥ ([\d.]+)% of planned n", full); m and put("prereg:min_pct", m.group(1) + "%", fam)
        if reg:
            pre, tot = Counter(), Counter()
            for (b, model, prompt, mit, fm, task, s), t in trans209.items():
                if b != "xp" or prompt != "base" or mit != "none" or fm not in ("timeout_after_commit", "error_500_after_commit"):
                    continue
                ms = short_model(model); tot[ms] += 1
                st = t.get("started_utc") or ""
                if st and st.replace("Z", "+00:00") < reg[-1]:
                    pre[ms] += 1
            put("prereg:e1_1_pre", "%d/%d" % (sum(pre.values()), sum(tot.values())), fam)
            for m_ in tot:
                put("prereg:e1_1_pre:%s" % m_, "%d/%d" % (pre[m_], tot[m_]), fam)
            clean = nexp - (1 if sum(pre.values()) else 0)
            put("prereg:clean", clean, fam); put("prereg:clean:w", word(clean), fam)
            if "r1:before_data" in MINE:
                n_all = nexp + 1; n_clean = clean + (1 if MINE["r1:before_data"] == "yes" else 0)
                # R2, R3 and R4 are registered in the same file; each heading states that it precedes its data
                for rx in ("R2", "R3", "R4"):
                    if re.search(r"^## %s\..*registered \d{4}-\d{2}-\d{2}, before any %s data" % (rx, rx), full, re.M):
                        n_all += 1; n_clean += 1
                put("prereg:all:n", n_all, fam); put("prereg:all:notclean", n_all - n_clean, fam)
                put("prereg:all:notclean:w", word(n_all - n_clean), fam)
                put("prereg:followups:n", n_all - 1, fam)   # E1-0..E1-6, R2, R3 and R4: the experiments after the reviews


PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
PII_PHONE = re.compile(r"(?<![\w.-])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?![\w-])")
PII_URL = re.compile(r"\b(?:https?://|www\.)[^\s\"'<>\\)]+")
STRIPE_NON_TRANSCRIPT = set()   # non-transcript files holding the Stripe base URL
PII_SKIP = {"sweep_run2.log", "sweep_run.log", "swap3_pairs.csv", "swap3_summary.csv"}


def emit_pii(D):
    """Regex sweep of the released result files (tracked files only; the
    untracked blind hand-check CSVs are never opened)."""
    fam = "PII sweep (regex over released files)"
    try:
        out = subprocess.run(["git", "-C", D.res, "ls-files", "--others", "--exclude-standard", "--full-name", "-z", "."],
                             capture_output=True, text=True, timeout=30)
        top = subprocess.run(["git", "-C", D.res, "rev-parse", "--show-toplevel"], capture_output=True, text=True, timeout=30).stdout.strip()
        skip = {os.path.realpath(os.path.join(top, f)) for f in out.stdout.split("\0") if f} if out.returncode == 0 and top else set()
    except Exception:
        skip = set()
    files = trans = 0; hits = Counter()
    for d, dirs, fs in os.walk(D.res):
        dirs[:] = [x for x in dirs if x != "smoke"]
        for f in fs:
            if "smoke" in f or f in PII_SKIP or f.endswith(".bak") or "handcheck_model_coder" in f or \
                    not f.endswith((".json", ".jsonl", ".csv", ".log", ".txt", ".md")) or os.path.realpath(os.path.join(d, f)) in skip:
                continue
            files += 1; trans += "transcript" in os.path.basename(d)
            txt = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
            hits["email"] += sum(1 for m in PII_EMAIL.finditer(txt) if m.group(1).lower() != "example.com"
                                 and not m.group(1).lower().endswith(".example.com"))
            hits["phone"] += len(PII_PHONE.findall(txt))
            for u in PII_URL.findall(txt):
                hits["url_stripe_api" if re.match(r"https://api\.stripe\.com/?$", u) else "url"] += 1
                # CONVENTION: "the real-API-check transcripts" = the files of results/real_stripe
                # (576 transcripts plus one hit in its stripe_check.json key-check record)
                if re.match(r"https://api\.stripe\.com/?$", u):
                    hits["url_stripe_api_in_transcripts"] += "transcript" in d
                    hits["url_stripe_elsewhere"] += "real_stripe" not in os.path.relpath(d, D.res).split(os.sep)
                    if "transcript" not in d:
                        STRIPE_NON_TRANSCRIPT.add(os.path.basename(f) if isinstance(f, str) else str(f))
    put("pii:files", "{:,}".format(files), fam); put("pii:transcripts", "{:,}".format(trans), fam)
    for k in ("email", "phone", "url", "url_stripe_api"):
        put("pii:" + k, hits[k], fam)
    put("pii:total", hits["email"] + hits["phone"] + hits["url"], fam)
    HYG["pii_stripe_url"] = {"all": hits["url_stripe_api"], "in_real_stripe_transcripts": hits["url_stripe_api_in_transcripts"],
                             "outside_results_real_stripe": hits["url_stripe_elsewhere"]}
    if hits["url"] == 0:
        put("pii:url_clause", "no URL other than the Stripe API base address that the real-API-check transcripts"
            + (" and its key-check record" if STRIPE_NON_TRANSCRIPT == {"stripe_check.json"} else
               " and %d other real-API-check files" % len(STRIPE_NON_TRANSCRIPT) if STRIPE_NON_TRANSCRIPT else "")
            + " contain"
            if hits["url_stripe_api"] and not hits["url_stripe_elsewhere"] else
            "no URL other than the Stripe API base address" if hits["url_stripe_api"] else "no URL", fam)


# ------------------------------------------------------------------ R2 wording robustness (results/r2_wording)
R2_FM = {"timeout_after_commit": "toA", "error_500_after_commit": "e500", "status_unknown_after_commit": "w3",
         "status_unknown_p1_after_commit": "unknown_p1", "status_unknown_p2_after_commit": "unknown_p2",
         "status_unknown_p3_after_commit": "unknown_p3"}
R2_UNKNOWN = ["status_unknown_after_commit", "status_unknown_p1_after_commit",
              "status_unknown_p2_after_commit", "status_unknown_p3_after_commit"]
R2_HINTS = ["bare_hint", "hint_p1", "hint_p2", "hint_p3"]


def perm_test_lower(x, y):
    """One-sided paired sign-flip permutation test (registration R2, primary
    hypotheses: the paraphrase LOWER than its same-batch baseline, H1:
    mean(y - x) < 0). Same flipping scheme, N, seed and unit order as
    perm_test; p = (#permuted sums <= observed sum + 1) / (N + 1)."""
    units = sorted(set(x) & set(y))
    d = [y[u] - x[u] for u in units]
    if not d:
        return None
    obs = sum(d)
    rng = random.Random(PERM_SEED)
    hits = 0
    for _ in range(PERM_N):
        s = 0.0
        for v in d:
            s += v if rng.random() < 0.5 else -v
        if s <= obs + 1e-12:
            hits += 1
    return {"d": obs / len(d), "p": (hits + 1) / (PERM_N + 1), "T": len(units)}


def r2_planned(D):
    """Planned n per (part, model, condition). CONVENTION: the registration
    states the design (Part A 16 graphs x 2 samples, Part B 32 tasks x 2,
    Part C 8 chain graphs x 2); the cell list and the sample counts are read
    from configs/r2_wording.json, as analysis/r2_wording.py reads them."""
    cfg = json.load(open(os.path.join(D.root, "configs", "r2_wording.json")))
    out = {}
    for g in cfg["groups"]:
        for m in g["models"]:
            for c in g.get("cells", []):
                out[("B", m, c[2])] = 32 * cfg["samples"]
            for c in g.get("restart_cells", []):
                out[(g["part"], m, c[0])] = (16 if c[2] == "all" else 8) * cfg.get("restart_samples", cfg["samples"])
    return out


def emit_r2(D):
    """R2 (registration section "R2" and deviation 13): same-batch cells with
    unit-level Wilson intervals, one-sided primaries with Holm within each
    model, two-sided secondaries with Holm within each (part, model) block,
    the 90% completeness rule, refusals excluded, read-back and session-2
    lookups from the transcripts."""
    fam = "R2 cell (raw rows + transcripts)"; famt = "R2 contrast (paired permutation)"; famd = "R2 derived"
    raw = prefer_v2(os.path.join(D.res, "r2_wording", "raw.jsonl"))
    if not os.path.exists(raw):
        return
    rows = last_by_id(read_jsonl(raw))
    trans = {}
    for f in glob.glob(os.path.join(D.res, "r2_wording", "transcripts", "*.json")):
        t = json.load(open(f))
        trans[t["episode_id"]] = t
    plan = r2_planned(D)
    cells = defaultdict(list); refs = Counter()
    checked = mism_v = mism_rb = mism_lk = 0
    for eid, r in rows.items():
        t = trans.get(eid)
        if r.get("kind") == "restart":
            part = "C" if r["resume_context"] == "plan_only" else "A"
            key = (part, r["model"], r["resume_context"])
            # S3 (Part A): a list_effects call anywhere in session 2
            s2 = t["sessions"][1]["turns"] if t and len(t.get("sessions") or []) > 1 else None
            look = None if s2 is None else any(x["name"] == "list_effects" for turn in s2 if turn.get("role") == "assistant"
                                               for x in turn["content"] if x.get("type") == "tool_use")
            e = {"unit": r["graph_id"], "v": int(r["givr_hit"] > 0), "lookup": look}
            if look is not None and "session2_lookup" in r:
                mism_lk += look != bool(r["session2_lookup"])
        else:
            key = ("B", r["model"], r["failure_mode"])
            fe = features(t) if t else None
            # S3 (Part B): list_effects between the failed call and the next effectful call
            e = {"unit": r["task_id"], "v": int(r["violations"] > 0), "rb": fe["rb_fail"] if fe else None}
            if t and "ledger" in t:
                checked += 1
                mism_v += (ledger_violations(t["ledger"]) > 0) != (r["violations"] > 0)
                if fe and fe["rb_fail"] is not None:
                    mism_rb += bool(fe["rb_fail"]) != bool(r.get("readback_after_failure"))
        if t is not None and refused(t):
            refs[key] += 1
        else:
            cells[key].append(e)
    HYG["r2_transcripts_vs_rows"] = {"checked_retry": checked, "violation_mismatch": mism_v,
                                     "readback_mismatch": mism_rb, "session2_lookup_mismatch": mism_lk,
                                     "transcripts": len(trans), "rows": len(rows)}
    incomplete = set()
    for key in sorted(set(cells) | set(refs) | set(plan), key=str):
        part, model, cond = key
        eps = cells.get(key, [])
        pl = plan.get(key)
        if pl and len(eps) + refs[key] < MIN_FRAC * pl:
            incomplete.add(key)      # registration: reported only at >= 90% of planned n
            continue
        if not eps:
            continue
        nm = "nj:r2.%s.%s.%s" % (part, short_model(model), R2_FM.get(cond, cond))
        metric = "ivr" if part == "B" else "givr"
        n = len(eps); k = sum(e["v"] for e in eps); U = len({e["unit"] for e in eps})
        put(nm + ":" + metric, f2(k / n), fam); put(nm + ":%s:kn" % metric, "%d/%d" % (k, n), fam)
        put(nm + ":%s:n" % metric, n, fam); put(nm + ":%s:k" % metric, k, fam); put(nm + ":%s:T" % metric, U, fam)
        put(nm + ":%s:ci" % metric, ci(*wilson_task(k / n, U)), fam)
        put(nm + ":%s_ci_ep:ci" % metric, ci(*wilson(k, n)), fam); put(nm + ":refusals", refs[key], fam)
        if part == "B":
            rb = [e for e in eps if e["rb"] is not None]
            if rb:
                put(nm + ":readback_after_failure:kn", "%d/%d" % (sum(e["rb"] for e in rb), len(rb)), fam)
                put(nm + ":readback_after_failure", f2(sum(e["rb"] for e in rb) / len(rb)), fam)
        else:
            lk = [e for e in eps if e["lookup"] is not None]
            if lk:
                put(nm + ":session2_lookup:kn", "%d/%d" % (sum(e["lookup"] for e in lk), len(lk)), fam)
                put(nm + ":session2_lookup", f2(sum(e["lookup"] for e in lk) / len(lk)), fam)

    def rates_u(eps):
        return rates((e["unit"], e["v"]) for e in eps)

    def contrast(part, model, base, cond, one_sided):
        # no test on an incomplete cell (deviation 13)
        if (part, model, base) in incomplete or (part, model, cond) in incomplete:
            return None
        x, y = cells.get((part, model, base)), cells.get((part, model, cond))
        if not x or not y:
            return None
        r = (perm_test_lower if one_sided else perm_test)(rates_u(x), rates_u(y))
        if r is None:
            return None
        r["name"] = "nj:r2.%s.%s.%s_vs_%s" % (part, short_model(model), R2_FM.get(cond, cond), R2_FM.get(base, base))
        return r

    def emit_t(r, hp=None):
        put(r["name"] + ":d", signed(r["d"]), famt); put(r["name"] + ":d:n", r["T"], famt)
        put(r["name"] + ":d:abs", f2(abs(r["d"])), famt)
        put(r["name"] + ":p", pfmt(r["p"]), famt); put(r["name"] + ":p:pe", pe(r["p"]), famt)
        if hp is not None:
            put(r["name"] + ":p:holm_p", pfmt(hp), famt); put(r["name"] + ":p:holm_pe", pe(hp), famt)

    def models(part):
        return sorted({k[1] for k in cells if k[0] == part})

    verdicts = {}
    # primaries R2-H1 / R2-H2: one-sided, Holm over the three paraphrases within a model,
    # supported if >= 2 of 3 are significant; a verdict needs all three paraphrase cells complete
    for part, base, paras in (("A", "amnesiac", R2_HINTS[1:]), ("B", "error_500_after_commit", R2_UNKNOWN[1:])):
        for model in models(part):
            m = short_model(model)
            ts = [t for t in (contrast(part, model, base, c, True) for c in paras) if t]
            if len(ts) < 3:
                verdicts[(part, m)] = "not evaluable"
                put("r2:%s:%s:verdict" % (part, m), "not evaluable", famd)
                continue
            hp = holm([t["p"] for t in ts])
            for t, h in zip(ts, hp):
                emit_t(t, h)
            sig = sum(1 for h in hp if h < 0.05)
            verdicts[(part, m)] = "supported" if sig >= 2 else "not supported"
            put("r2:%s:%s:verdict" % (part, m), verdicts[(part, m)], famd)
            put("r2:%s:%s:nsig" % (part, m), sig, famd)
            put("r2:%s:%s:holm_pmax_e" % (part, m), pe(max(hp)), famd)
    # secondaries: two-sided, Holm within each (part, model) block
    for model in models("A"):
        m = short_model(model)
        ts = [t for t in (contrast("A", model, "bare_hint", c, False) for c in R2_HINTS[1:]) if t]
        for t, h in zip(ts, holm([t["p"] for t in ts])):
            emit_t(t, h)
        have = [cells[("A", model, c)] for c in R2_HINTS if ("A", model, c) in cells and ("A", model, c) not in incomplete]
        put("r2:A:%s:hinted_cells" % m, len(have), famd)
        put("r2:A:%s:hinted_all_zero" % m, "yes" if have and all(sum(e["v"] for e in eps) == 0 for eps in have) else "no", famd)
        lk = [[e for e in eps if e["lookup"] is not None] for eps in have]
        put("r2:A:%s:hinted_all_looked" % m, "yes" if lk and all(l and all(e["lookup"] for e in l) for l in lk) else "no", famd)
    for model in models("B"):
        m = short_model(model)
        ts = [contrast("B", model, "timeout_after_commit", c, False) for c in R2_UNKNOWN] + \
             [contrast("B", model, "error_500_after_commit", "status_unknown_after_commit", False),
              contrast("B", model, "timeout_after_commit", "error_500_after_commit", False)]
        ts = [t for t in ts if t]
        hp = holm([t["p"] for t in ts])
        for t, h in zip(ts, hp):
            emit_t(t, h)
            if t["name"].endswith(".e500_vs_toA"):
                put("r2:B:%s:e500_vs_toA:sig" % m, "yes" if h < 0.05 else "no", famd)
        toa = [(t, h) for t, h in zip(ts, hp) if t["name"].endswith("_vs_toA") and not t["name"].endswith("e500_vs_toA")]
        if len(toa) == 4:
            put("r2:B:%s:below_timeout:all" % m, "yes" if all(t["d"] < 0 and h < 0.05 for t, h in toa) else "no", famd)
            put("r2:B:%s:below_timeout:n" % m, sum(1 for t, h in toa if t["d"] < 0 and h < 0.05), famd)
            put("r2:B:%s:equal_timeout:all" % m, "yes" if all(t["d"] == 0 for t, _ in toa) else "no", famd)
        par = [cells[("B", model, c)] for c in R2_UNKNOWN[1:] if ("B", model, c) in cells and ("B", model, c) not in incomplete]
        if len(par) == 3:
            ks = sorted((sum(e["v"] for e in eps), len(eps)) for eps in par)
            put("r2:B:%s:par:min:kn" % m, "%d/%d" % ks[0], famd); put("r2:B:%s:par:max:kn" % m, "%d/%d" % ks[-1], famd)
            put("r2:B:%s:par:min:ivr" % m, f2(ks[0][0] / ks[0][1]), famd); put("r2:B:%s:par:max:ivr" % m, f2(ks[-1][0] / ks[-1][1]), famd)
            put("r2:B:%s:par:all_zero" % m, "yes" if ks[-1][0] == 0 else "no", famd); put("r2:B:%s:par:max_k" % m, ks[-1][0], famd)
        rbs = []
        for c in R2_UNKNOWN:
            eps = [e for e in cells.get(("B", model, c), []) if e["rb"] is not None]
            if ("B", model, c) in incomplete or not eps:
                break
            rbs.append((sum(e["rb"] for e in eps), len(eps)))
        if len(rbs) == 4:
            put("r2:B:%s:rb:min:kn" % m, "%d/%d" % min(rbs), famd); put("r2:B:%s:rb:max:kn" % m, "%d/%d" % max(rbs), famd)
    # S4 (Part C, descriptive): the July chain4 plan_only cell of the same model (same
    # graphs, interrupt points and samples; results/sweep2_raw_v2.jsonl) and the paired
    # per-graph difference R2 - July, two-sided, no Holm, no verdict
    jraw = os.path.join(D.res, "sweep2_raw_v2.jsonl")
    if os.path.exists(jraw):
        jul = defaultdict(list)
        for r in last_by_id(read_jsonl(jraw)).values():
            if r["resume_context"] == "plan_only" and r["mitigation"] == "none" and r["topology"] == "chain4":
                jul[r["model"]].append({"unit": r["graph_id"], "v": int(r["givr_hit"] > 0)})
        for model in sorted({k[1] for k in cells if k[0] == "C"}):
            key = ("C", model, "plan_only")
            if key in incomplete or not cells.get(key) or not jul.get(model):
                continue
            nm = "nj:r2.C.%s.plan_only" % short_model(model)
            j = jul[model]; k = sum(e["v"] for e in j)
            put(nm + "_chain4_jul:givr:kn", "%d/%d" % (k, len(j)), fam); put(nm + "_chain4_jul:givr", f2(k / len(j)), fam)
            put(nm + "_chain4_jul:givr:ci", ci(*wilson_task(k / len(j), len({e["unit"] for e in j}))), fam)
            put(nm + ":givr:refusals", refs[key], fam)
            t = perm_test(rates_u(j), rates_u(cells[key]))
            if t:
                t["name"] = nm + "_chain4_vs_jul"
                emit_t(t)
    # totals, verdict counts, Part C, data window
    put("r2:episodes", "{:,}".format(len(rows)), famd); put("r2:planned", "{:,}".format(sum(plan.values())), famd)
    put("r2:paraphrases:n", len(R2_HINTS) - 1, famd)
    put("r2:C:ran", "yes" if any(k[0] == "C" and k not in incomplete and cells.get(k) for k in cells) else "no", famd)
    vs = [verdicts.get(k) for k in (("A", "haiku"), ("A", "fable"), ("B", "haiku"), ("B", "sonnet"))]
    put("r2:verdicts:n", len(vs), famd); put("r2:verdicts:supported", sum(1 for v in vs if v == "supported"), famd)
    put("r2:verdicts:not_evaluable", sum(1 for v in vs if v == "not evaluable"), famd)
    put("r2:verdicts:evaluable", sum(1 for v in vs if v in ("supported", "not supported")), famd)
    st = sorted(r.get("started_utc") for r in rows.values() if r.get("started_utc"))
    if st:
        put("r2:first_data_utc", st[0].replace("T", " ")[:16] + " UTC", "row timestamps")
        put("r2:last_data_utc", st[-1].replace("T", " ")[:16] + " UTC", "row timestamps")
        try:   # registration commit = the commit that added the R2 runner config (git, read-only)
            reg = subprocess.run(["git", "-C", D.root, "log", "--diff-filter=A", "--format=%cI", "--", "configs/r2_wording.json"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:
            reg = []
        if reg:
            a = datetime.datetime.fromisoformat(reg[-1]).astimezone(datetime.timezone.utc)
            z = datetime.datetime.fromisoformat(st[0].replace("Z", "+00:00")).astimezone(datetime.timezone.utc)
            put("r2:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC", "git log")
            put("r2:commit_hm", a.strftime("%H:%M") + " UTC", "git log")
            put("r2:first_data_hm", z.strftime("%H:%M") + " UTC", "row timestamps")
            put("r2:before_data", "yes" if a < z else "no", "git log")
    COST["r2"] = sum(r.get("cost_usd") or 0 for r in rows.values())
    # the registration text: date of the R2 section and the registered wordings
    pr = os.path.join(D.root, "analysis", "preregistration.md")
    if os.path.exists(pr):
        full = open(pr).read()
        m = re.search(r"^## R2\..*registered (\d{4})-(\d{2})-(\d{2}), before any R2 data", full, re.M)
        if m:
            put("r2:reg_date", "%d %s %s" % (int(m.group(3)), datetime.date(2000, int(m.group(2)), 1).strftime("%B"), m.group(1)),
                "registration text")
        for p in ("p1", "p2", "p3"):
            m = re.search(r"`hint_%s` — \"([^\"]*)\"" % p, full)
            if m:
                put("design:r2:hint_" + p, m.group(1), "registration text")
            m = re.search(r"\(unknown_%s\) — \"ERROR: ([^\"]*)\"" % p, full)
            if m:
                put("design:r2:unknown_" + p, m.group(1), "registration text")


# ------------------------------------------------------------------ R3 cross-provider wording (results/r3_wording)
# registration section "R3": models, conditions and the planned n per cell
R3_B_MODELS = ["openrouter:openai/gpt-5.5-20260423", "openrouter:google/gemini-4-argon-20261013",
               "openrouter:qwen/qwen3.6-27b-20260422", "claude-fable-5"]
R3_A_MODELS = ["openrouter:openai/gpt-5.5-20260423", "openrouter:qwen/qwen3.6-27b-20260422",
               "openrouter:google/gemini-4-argon-20261013"]
R3_B_CONDS = ["timeout_after_commit", "error_500_after_commit", "status_unknown_after_commit",
              "status_unknown_p1_after_commit", "status_unknown_p3_after_commit"]
R3_UNKNOWN = ["status_unknown_after_commit", "status_unknown_p1_after_commit", "status_unknown_p3_after_commit"]
R3_A_CTX = ["amnesiac", "bare_hint", "hint_p1", "hint_p3"]
R3_NOTES = R3_A_CTX[1:]
REFUSED_LINE = re.compile(r"^\[\d+/\d+\] (\S+) REFUSED by the provider")


def r3_section(D):
    pr = os.path.join(D.root, "analysis", "preregistration.md")
    if not os.path.exists(pr):
        return ""
    m = re.search(r"^## R3\..*?(?=^## |\Z)", open(pr).read(), re.M | re.S)
    return m.group(0) if m else ""


def emit_r3(D, v3):
    """R3 (registration section "R3" and deviation 15), from results/r3_wording
    only: unit-level Wilson per cell; primaries one-sided (wording LOWER than its
    same-batch baseline), Holm over the three wordings of one model, supported
    at >= 2 of 3 significant; secondaries two-sided, Holm within each (part,
    model) block; 90% rule with refusals counted toward n; a model's verdict is
    "not evaluable" if any cell of its hypothesis is below the rule; floor rule:
    same-batch bare-500 IVR (amnesiac gIVR for R3-A) <= the registered floor."""
    fam = "R3 cell (raw rows + transcripts)"; famt = "R3 contrast (paired permutation)"; famd = "R3 derived"
    raw = os.path.join(D.res, "r3_wording", "raw.jsonl")
    if not os.path.exists(raw):
        return
    sec = r3_section(D)
    m = re.search(r"IVR is <= ([\d.]+)", sec)
    FLOOR = float(m.group(1)) if m else 0.10
    nB = re.search(r"(\d+) episodes per cell\. Prompt", sec); nA = re.search(r"(\d+) episodes per cell\. Mitigation", sec)
    nB = int(nB.group(1)) if nB else 64; nA = int(nA.group(1)) if nA else 32
    plan = {("B", mo, c): nB for mo in R3_B_MODELS for c in R3_B_CONDS}
    plan.update({("A", mo, c): nA for mo in R3_A_MODELS for c in R3_A_CTX})
    rows = last_by_id(read_jsonl(raw))
    trans = {}
    for f in glob.glob(os.path.join(D.res, "r3_wording", "transcripts", "*.json")):
        t = json.load(open(f))
        trans[t["episode_id"]] = t
    cells = defaultdict(list); refs = Counter(); ref_units = defaultdict(set); ref_ids = set()
    checked = mism_v = mism_rb = mism_lk = 0
    for eid, r in rows.items():
        t = trans.get(eid)
        if r.get("kind") == "restart":
            key = ("A", r["model"], r["resume_context"]); unit = r["graph_id"]
        else:
            key = ("B", r["model"], r["failure_mode"]); unit = r["task_id"]
        # refusal: (ii) the runner's refusal row (OpenRouter "blocked", no score, no
        # transcript) or (i) an Anthropic assistant turn with stop_reason "refusal"
        if r.get("refusal") is True or (t is not None and refused(t)):
            refs[key] += 1; ref_units[key].add(unit); ref_ids.add(eid)
            continue
        if key[0] == "A":
            s2 = t["sessions"][1]["turns"] if t and len(t.get("sessions") or []) > 1 else None
            look = None if s2 is None else any(x["name"] == "list_effects" for turn in s2 if turn.get("role") == "assistant"
                                               for x in turn["content"] if x.get("type") == "tool_use")
            if look is not None and "session2_lookup" in r:
                mism_lk += look != bool(r["session2_lookup"])
            e = {"unit": unit, "v": int(r["givr_hit"] > 0), "lookup": look}
        else:
            fe = features(t) if t else None
            e = {"unit": unit, "v": int(r["violations"] > 0), "rb": fe["rb_fail"] if fe else None}
            if t and "ledger" in t:
                checked += 1
                mism_v += (ledger_violations(t["ledger"]) > 0) != (r["violations"] > 0)
                if fe and fe["rb_fail"] is not None:
                    mism_rb += bool(fe["rb_fail"]) != bool(r.get("readback_after_failure"))
        cells[key].append(e)
    # refusals recomputed from the launcher log (each "REFUSED by the provider" line)
    log = os.path.join(D.res, "r3_wording", "run.log")
    log_ids = set()
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            mm = REFUSED_LINE.match(line.strip())
            if mm:
                log_ids.add(mm.group(1))
    HYG["r3_transcripts_vs_rows"] = {"checked_retry": checked, "violation_mismatch": mism_v,
                                     "readback_mismatch": mism_rb, "session2_lookup_mismatch": mism_lk,
                                     "transcripts": len(trans), "rows": len(rows),
                                     "refusals_rows": len(ref_ids), "refusals_runlog": len(log_ids),
                                     "refusals_rows_vs_runlog_differ": len(ref_ids ^ log_ids)}
    incomplete = set()
    for key in sorted(set(cells) | set(refs) | set(plan), key=str):
        part, model, cond = key
        eps = cells.get(key, [])
        pl = plan.get(key)
        if pl and len(eps) + refs[key] < MIN_FRAC * pl:
            incomplete.add(key)        # registration: reported only at >= 90% of planned n, refusals counted
            continue
        if not eps:
            continue
        nm = "nj:r3.%s.%s.%s" % (part, short_model(model), R2_FM.get(cond, cond))
        metric = "ivr" if part == "B" else "givr"
        n = len(eps); k = sum(e["v"] for e in eps); U = len({e["unit"] for e in eps})
        put(nm + ":" + metric, f2(k / n), fam); put(nm + ":%s:kn" % metric, "%d/%d" % (k, n), fam)
        put(nm + ":%s:n" % metric, n, fam); put(nm + ":%s:k" % metric, k, fam); put(nm + ":%s:T" % metric, U, fam)
        put(nm + ":%s:ci" % metric, ci(*wilson_task(k / n, U)), fam)
        put(nm + ":%s_ci_ep:ci" % metric, ci(*wilson(k, n)), fam)
        put(nm + ":refusals", refs[key], fam); put(nm + ":%s:refusals" % metric, refs[key], fam)
        put(nm + ":planned", pl, fam)
        if part == "B":
            rb = [e for e in eps if e["rb"] is not None]
            if rb:
                put(nm + ":readback_after_failure:kn", "%d/%d" % (sum(e["rb"] for e in rb), len(rb)), fam)
                put(nm + ":readback_after_failure", f2(sum(e["rb"] for e in rb) / len(rb)), fam)
        else:
            lk = [e for e in eps if e["lookup"] is not None]
            if lk:
                put(nm + ":session2_lookup:kn", "%d/%d" % (sum(e["lookup"] for e in lk), len(lk)), fam)
                put(nm + ":session2_lookup", f2(sum(e["lookup"] for e in lk) / len(lk)), fam)

    def ok(key):
        return key in cells and key not in incomplete

    def kn(key):
        eps = cells[key]
        return (sum(e["v"] for e in eps), len(eps))

    def contrast(part, model, base, cond, one_sided):
        if not (ok((part, model, base)) and ok((part, model, cond))):
            return None              # no test on a cell below the 90% rule
        x = rates((e["unit"], e["v"]) for e in cells[(part, model, base)])
        y = rates((e["unit"], e["v"]) for e in cells[(part, model, cond)])
        r = (perm_test_lower if one_sided else perm_test)(x, y)
        if r is not None:
            r["name"] = "nj:r3.%s.%s.%s_vs_%s" % (part, short_model(model), R2_FM.get(cond, cond), R2_FM.get(base, base))
        return r

    def emit_t(r, hp):
        put(r["name"] + ":d", signed(r["d"]), famt); put(r["name"] + ":d:n", r["T"], famt)
        put(r["name"] + ":d:abs", f2(abs(r["d"])), famt)
        put(r["name"] + ":p", pfmt(r["p"]), famt); put(r["name"] + ":p:pe", pe(r["p"]), famt)
        put(r["name"] + ":p:holm_p", pfmt(hp), famt); put(r["name"] + ":p:holm_pe", pe(hp), famt)

    def minmax(ks):
        # CONVENTION: lowest / highest rate; on a tie the first cell in registered
        # order (for GPT-5.5's all-zero notes, "at most 0/31" names bare_hint)
        return min(ks, key=lambda x: x[0] / x[1]), max(ks, key=lambda x: x[0] / x[1])

    verdict = {}
    floor = {}
    # ---- primaries
    for part, models_, base, conds in (("A", R3_A_MODELS, "amnesiac", R3_NOTES), ("B", R3_B_MODELS, "error_500_after_commit", R3_UNKNOWN)):
        for model in models_:
            m = short_model(model); q = "r3:%s:%s" % (part, m)
            if ok((part, model, base)):
                k, n = kn((part, model, base))
                floor[(part, m)] = k / n <= FLOOR + 1e-12
                put(q + ":floor", "yes" if floor[(part, m)] else "no", famd)
            ts = [contrast(part, model, base, c, True) for c in conds]
            if any(t is None for t in ts):
                verdict[(part, m)] = "not evaluable"
            else:
                hp = holm([t["p"] for t in ts])
                for t, h in zip(ts, hp):
                    emit_t(t, h)
                sig = sum(1 for h in hp if h < 0.05)
                verdict[(part, m)] = "supported" if sig >= 2 else "not supported"
                put(q + ":nsig", sig, famd); put(q + ":holm_pmax_e", pe(max(hp)), famd)
            put(q + ":verdict", verdict[(part, m)], famd)
            # planned episodes of the model's part with no row (a refusal row counts as run)
            miss = sum(max(0, plan[k] - len(cells.get(k, [])) - refs[k]) for k in plan if k[0] == part and k[1] == model)
            put(q + ":missing", miss, famd); put(q + ":missing:W", word(miss).capitalize(), famd)
            if part == "A":   # every note cell at or above the 90% rule
                put(q + ":notes_complete", "yes" if all(ok((part, model, c)) for c in conds) else "no", famd)
            have = [(part, model, c) for c in conds if ok((part, model, c))]
            put(q + (":note_cells" if part == "A" else ":unknown_cells"), len(have), famd)
            if part == "A" and have:
                lo, hi = minmax([kn(c) for c in have])
                put(q + ":hinted_max:kn", "%d/%d" % hi, famd); put(q + ":hinted_max_k", hi[0], famd)
                put(q + ":hinted_all_zero", "yes" if hi[0] == 0 else "no", famd)
                lk = [[e["lookup"] for e in cells[c] if e["lookup"] is not None] for c in have]
                put(q + ":hinted_all_looked", "yes" if all(l and all(l) for l in lk) else "no", famd)
            if part == "B" and have:
                lo, hi = minmax([kn(c) for c in have])
                put(q + ":par:min:kn", "%d/%d" % lo, famd); put(q + ":par:max:kn", "%d/%d" % hi, famd)
                put(q + ":par:min:ivr", f2(lo[0] / lo[1]), famd); put(q + ":par:max:ivr", f2(hi[0] / hi[1]), famd)
                put(q + ":par:max_k", hi[0], famd); put(q + ":par:all_zero", "yes" if hi[0] == 0 else "no", famd)
                if ok((part, model, base)):
                    b5 = kn((part, model, base))
                    put(q + ":par:max_above_500", "yes" if hi[0] / hi[1] > b5[0] / b5[1] else "no", famd)
                rbs = []
                for c in have:
                    rb = [e["rb"] for e in cells[c] if e["rb"] is not None]
                    rbs.append((sum(rb), len(rb)))
                lo, hi = minmax(rbs)
                put(q + ":rb:min:kn", "%d/%d" % lo, famd); put(q + ":rb:max:kn", "%d/%d" % hi, famd)
    put("r3:floor", f2(FLOOR), "registration text")
    m = re.search(r"\*\*OpenRouter hard stop at \$(\d+)\*\*", sec)
    if m:
        put("r3:or_cap", m.group(1), "registration text")
    put("r3:A:notes:n", len(R3_NOTES), famd); put("r3:B:wordings:n", len(R3_UNKNOWN), famd)
    # ---- secondaries: S1 + S2 (R3-B), S3 (R3-A); one two-sided Holm block per (part, model)
    for model in R3_B_MODELS:
        m = short_model(model); q = "r3:B:%s" % m
        s1 = [t for t in (contrast("B", model, "timeout_after_commit", c, False) for c in R3_UNKNOWN) if t]
        s2 = contrast("B", model, "timeout_after_commit", "error_500_after_commit", False)
        ts = s1 + ([s2] if s2 else [])
        hp = holm([t["p"] for t in ts])
        for t, h in zip(ts, hp):
            emit_t(t, h)
        h1 = hp[:len(s1)]
        if s1:
            below = all(t["d"] < 0 and h < 0.05 for t, h in zip(s1, h1))
            above = all(t["d"] > 0 and h < 0.05 for t, h in zip(s1, h1))
            put(q + ":below_timeout:all", "yes" if below else "no", famd)
            put(q + ":below_timeout:n", sum(1 for t, h in zip(s1, h1) if t["d"] < 0 and h < 0.05), famd)
            put(q + ":equal_timeout:all", "yes" if all(t["d"] == 0 for t in s1) else "no", famd)
            # CONVENTION: "at" its timeout level = no unknown wording differs from the
            # same-batch timeout after Holm
            put(q + ":timeout_level", "below" if below else "above" if above else
                "at" if all(h >= 0.05 for h in h1) else "mixed", famd)
        if s2:
            put(q + ":e500_vs_toA:dir", "above" if s2["d"] > 0 else "below" if s2["d"] < 0 else "equal to", famd)
            put(q + ":e500_vs_toA:sig", "yes" if hp[-1] < 0.05 else "no", famd)
    for model in R3_A_MODELS:
        ts = [t for t in (contrast("A", model, "bare_hint", c, False) for c in R3_NOTES[1:]) if t]
        for t, h in zip(ts, holm([t["p"] for t in ts])):
            emit_t(t, h)
    # ---- verdict tallies and names
    for part, models_ in (("A", R3_A_MODELS), ("B", R3_B_MODELS)):
        ms = [short_model(x) for x in models_]
        for lab, v in (("supported", "supported"), ("not_supported", "not supported"), ("not_evaluable", "not evaluable")):
            sel = [x for x in ms if verdict[(part, x)] == v]
            put("r3:%s:%s:n" % (part, lab), len(sel), famd); put("r3:%s:%s:names" % (part, lab), names(sel), famd)
    # floor: a "not supported" R3-H2 at a same-batch bare-500 IVR <= floor reads as no
    # duplication to remove; above the floor it is a failure of the wording
    fl = [short_model(x) for x in R3_B_MODELS if verdict[("B", short_model(x))] == "not supported" and floor.get(("B", short_model(x)))]
    fail = [short_model(x) for x in R3_B_MODELS if verdict[("B", short_model(x))] == "not supported" and not floor.get(("B", short_model(x)))]
    put("r3:B:floor:n", len(fl), famd); put("r3:B:floor:names", names(fl), famd)
    put("r3:B:fail:n", len(fail), famd); put("r3:B:fail:names", names(fail), famd)
    vs = list(verdict.values())
    put("r3:verdicts:n", len(vs), famd)
    for lab, v in (("supported", "supported"), ("not_supported", "not supported"), ("not_evaluable", "not evaluable")):
        put("r3:verdicts:" + lab, sum(1 for x in vs if x == v), famd)
    put("r3:verdicts:evaluable", sum(1 for x in vs if x != "not evaluable"), famd)
    # ---- wording claims across R2 and R3 (supported verdicts only)
    a_ok = [x for x in ORDER if MINE.get("r2:A:%s:verdict" % x) == "supported" or verdict.get(("A", x)) == "supported"]
    b_ok = [x for x in ORDER if MINE.get("r2:B:%s:verdict" % x) == "supported" or verdict.get(("B", x)) == "supported"]
    put("wording:A:supported:names", names(a_ok), famd); put("wording:A:supported:n", len(a_ok), famd)
    put("wording:B:supported:names", names(b_ok), famd); put("wording:B:supported:n", len(b_ok), famd)
    nc = [x for x in ORDER if x not in CLAUDE and (x in a_ok or x in b_ok)]
    put("wording:nonclaude:names", names(nc), famd); put("wording:nonclaude:n", len(nc), famd)
    put("wording:providers:n", len({PROVIDER[x] for x in set(a_ok) | set(b_ok)}), famd)
    put("wording:A:providers:n", len({PROVIDER[x] for x in a_ok}), famd)
    put("wording:B:providers:n", len({PROVIDER[x] for x in b_ok}), famd)
    # ---- coverage, totals, refusals
    planned = sum(plan.values())
    put("r3:episodes", "{:,}".format(len(rows)), famd); put("r3:planned", "{:,}".format(planned), famd)
    put("r3:missing", "{:,}".format(planned - len(rows)), famd)
    put("r3:coverage", ("all {:,} planned episodes".format(planned) if len(rows) >= planned else
                        "{:,} of {:,} planned episodes".format(len(rows), planned)), famd)
    if "total:all" in MINE and "r2:episodes" in MINE:
        tot = int(MINE["total:all"].replace(",", "")) + int(MINE["r2:episodes"].replace(",", ""))
        put("total:all_r2", "{:,}".format(tot), "episode counts"); put("total:all_r23", "{:,}".format(tot + len(rows)), "episode counts")
    put("r3:refusals", sum(refs.values()), famd)
    byc = {k: v for k, v in refs.items() if v}
    put("r3:refusals:cells", len(byc), famd)
    if byc:
        put("r3:refusals:per_cell:min", min(byc.values()), famd); put("r3:refusals:per_cell:max", max(byc.values()), famd)
        # the task Gemini blocked in October (cross-provider run.log, every attempt blocked)
        oct_units = {t for (src, model, *_), v in refusals(D, v3).items() if src == "xp" for t, _ in v}
        units = {u for v in ref_units.values() for u in v}
        put("r3:refusals:same_task", "yes" if units and units <= oct_units else "no", famd)
        put("r3:refusals:tasks_n", len(units), famd)
    # ---- data window and registration timing
    st = sorted(r.get("started_utc") for r in rows.values() if r.get("started_utc"))
    if st:
        put("r3:first_data_utc", st[0].replace("T", " ")[:16] + " UTC", "row timestamps")
        put("r3:last_data_utc", st[-1].replace("T", " ")[:16] + " UTC", "row timestamps")
        try:   # registration commit = the commit that added the R3 runner config (git, read-only)
            reg = subprocess.run(["git", "-C", D.root, "log", "--diff-filter=A", "--format=%cI", "--", "configs/r3_wording.json"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:
            reg = []
        if reg:
            a = datetime.datetime.fromisoformat(reg[-1]).astimezone(datetime.timezone.utc)
            z = datetime.datetime.fromisoformat(st[0].replace("Z", "+00:00")).astimezone(datetime.timezone.utc)
            put("r3:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC", "git log")
            put("r3:commit_hm", a.strftime("%H:%M") + " UTC", "git log")
            put("r3:first_data_hm", z.strftime("%H:%M") + " UTC", "row timestamps")
            put("r3:before_data", "yes" if a < z else "no", "git log")
    m = re.search(r"^## R3\..*registered (\d{4})-(\d{2})-(\d{2}), before any R3 data", sec, re.M)
    if m:
        put("r3:reg_date", "%d %s %s" % (int(m.group(3)), datetime.date(2000, int(m.group(2)), 1).strftime("%B"), m.group(1)),
            "registration text")
    # ---- logged spend: R3 rows by provider; the R2 completion = R2 rows started at or
    # after the first run_r3.sh launch (its "=== start" line), which ran the completion first
    COST["r3"] = (sum(r.get("cost_usd") or 0 for r in rows.values() if r["model"].startswith("claude")),
                  sum(r.get("cost_usd") or 0 for r in rows.values() if r["model"].startswith("openrouter:")))
    if os.path.exists(log):
        first = next((mm.group(1) for mm in (re.match(r"^=== start (\S+)", l) for l in open(log, encoding="utf-8", errors="replace")) if mm), None)
        r2raw = prefer_v2(os.path.join(D.res, "r2_wording", "raw.jsonl"))
        if first and os.path.exists(r2raw):
            r2rows = last_by_id(read_jsonl(r2raw)).values()
            COST["r2_completion"] = sum(r.get("cost_usd") or 0 for r in r2rows if (r.get("started_utc") or "") >= first)
            for l in open(log, encoding="utf-8", errors="replace"):
                mm = re.search(r"Part 1 spent \$([\d.]+)", l)
                if mm:
                    HYG["r2_completion_rows_vs_runlog_usd"] = [round(COST["r2_completion"], 4), float(mm.group(1))]


# ------------------------------------------------------------------ R4 outcome unknown before commit (results/r4_precommit)
# registration section "R4": models, the three pre-commit conditions, 64 episodes per cell
R4_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-5", "openrouter:qwen/qwen3.6-27b-20260422"]
R4_CONDS = ["timeout_before_commit", "error_500_before_commit", "status_unknown_before_commit"]
R4_FM = {"timeout_before_commit": "toB", "error_500_before_commit": "e500B", "status_unknown_before_commit": "w3B"}
R4_MEAS = ("omission", "ivr", "success", "readback_after_failure")


def r4_section(D):
    pr = os.path.join(D.root, "analysis", "preregistration.md")
    if not os.path.exists(pr):
        return ""
    m = re.search(r"^## R4\..*?(?=^## |\Z)", open(pr).read(), re.M | re.S)
    return m.group(0) if m else ""


def perm_test_noninf(x, y, margin):
    """R4-H1 non-inferiority (registration R4, "Test"): the one-sided sign-flip
    of perm_test_lower applied to the SHIFTED per-unit differences
    (y - x) - margin, H1: mean(y - x) < margin; p = (#permuted sums <=
    observed + 1) / (N + 1). Same N, seed and unit order. d is unshifted."""
    units = sorted(set(x) & set(y))
    d = [y[u] - x[u] for u in units]
    if not d:
        return None
    s_ = [v - margin for v in d]
    obs = sum(s_)
    rng = random.Random(PERM_SEED)
    hits = 0
    for _ in range(PERM_N):
        s = 0.0
        for v in s_:
            s += v if rng.random() < 0.5 else -v
        if s <= obs + 1e-12:
            hits += 1
    return {"d": sum(d) / len(d), "p": (hits + 1) / (PERM_N + 1), "T": len(units), "diffs": d}


def boot_ub95(diffs, n=PERM_N):
    """One-sided 95% task-cluster bootstrap upper bound of the mean difference
    (registration R4: 10,000 resamples, same seed; descriptive).
    CONVENTION (not stated in the text): one random.Random(PERM_SEED); each
    resample draws T tasks with replacement by rng.randrange(T) on the per-task
    differences (unshifted); the bound is the nearest-rank 95th percentile of
    the n resampled means, sorted[ceil(0.95 n) - 1]."""
    T = len(diffs)
    rng = random.Random(PERM_SEED)
    means = sorted(sum(diffs[rng.randrange(T)] for _ in range(T)) / T for _ in range(n))
    return means[int(math.ceil(0.95 * n)) - 1]


def emit_r4(D):
    """R4 (registration section "R4"), from results/r4_precommit only: omission,
    misorder and success re-derived from each transcript's committed ledger and
    goals, duplication from the ledger, read-back after the failure from the
    turns; R4-H1 one-sided non-inferiority per model (margin from the text), one
    test per model so Holm leaves p unchanged; S1-S3 six two-sided tests in one
    Holm block per model; 90% rule with refusals counted toward n."""
    fam = "R4 cell (transcript ledger + turns)"; famt = "R4 contrast (paired permutation)"; famd = "R4 derived"
    raw = os.path.join(D.res, "r4_precommit", "raw.jsonl")
    if not os.path.exists(raw):
        return
    sec = r4_section(D)
    m = re.search(r"\*\*Margin ([\d.]+)\*\*", sec)
    MARGIN = float(m.group(1)) if m else 0.10
    m = re.search(r"(\d+) episodes per cell\. Prompt", sec)
    nper = int(m.group(1)) if m else 64
    plan = {(mo, c): nper for mo in R4_MODELS for c in R4_CONDS}
    rows = last_by_id(read_jsonl(raw))
    trans = {}
    for f in glob.glob(os.path.join(D.res, "r4_precommit", "transcripts", "*.json")):
        t = json.load(open(f))
        trans[t["episode_id"]] = t
    cells = defaultdict(list); refs = Counter(); ref_ids = set()
    mism = Counter(); checked = no_trans = 0
    for eid, r in rows.items():
        key = (r["model"], r["failure_mode"]); t = trans.get(eid)
        if r.get("refusal") is True or (t is not None and refused(t)):
            refs[key] += 1; ref_ids.add(eid)
            continue
        if t is None or "ledger" not in t:
            # no transcript: the row's own fields (counted in the hygiene line)
            no_trans += 1
            cells[key].append({"unit": r["task_id"], "omission": int(bool(r["omission"])), "ivr": int(r["violations"] > 0),
                               "success": int(bool(r["achieved"])), "readback_after_failure": None})
            continue
        om, mis, _, _ = ledger_order(dict(t, family=r.get("family")))
        v = ledger_violations(t["ledger"]) > 0
        fe = features(t)
        rb = fe["rb_fail"] if fe else None
        checked += 1
        mism["omission"] += om != bool(r["omission"]); mism["misorder"] += mis != bool(r["misorder"])
        mism["success"] += (not om and not mis) != bool(r["achieved"]); mism["violation"] += v != (r["violations"] > 0)
        if rb is not None:
            mism["readback"] += bool(rb) != bool(r.get("readback_after_failure"))
        cells[key].append({"unit": r["task_id"], "omission": int(om), "ivr": int(v),
                           "success": int(not om and not mis), "readback_after_failure": None if rb is None else int(rb)})
    log = os.path.join(D.res, "r4_precommit", "run.log")
    log_ids = set()
    if os.path.exists(log):
        for line in open(log, encoding="utf-8", errors="replace"):
            mm = REFUSED_LINE.match(line.strip())
            if mm:
                log_ids.add(mm.group(1))
    HYG["r4_transcripts_vs_rows"] = {"checked": checked, "rows_without_transcript": no_trans,
                                     "omission_mismatch": mism["omission"], "misorder_mismatch": mism["misorder"],
                                     "success_mismatch": mism["success"], "violation_mismatch": mism["violation"],
                                     "readback_mismatch": mism["readback"], "transcripts": len(trans), "rows": len(rows),
                                     "refusals_rows": len(ref_ids), "refusals_runlog": len(log_ids),
                                     "refusals_rows_vs_runlog_differ": len(ref_ids ^ log_ids)}
    incomplete = set()
    for key in sorted(set(cells) | set(refs) | set(plan)):
        model, cond = key
        eps = cells.get(key, [])
        pl = plan.get(key)
        if pl and len(eps) + refs[key] < MIN_FRAC * pl:
            incomplete.add(key)
            continue
        if not eps:
            continue
        nm = "nj:r4.%s.%s" % (short_model(model), R4_FM.get(cond, cond))
        put(nm + ":refusals", refs[key], fam); put(nm + ":planned", pl, fam)
        for meas in R4_MEAS:
            vals = [(e["unit"], e[meas]) for e in eps if e[meas] is not None]
            if not vals:
                continue
            n = len(vals); k = sum(v for _, v in vals); U = len({u for u, _ in vals})
            put(nm + ":" + meas, f2(k / n), fam); put(nm + ":%s:kn" % meas, "%d/%d" % (k, n), fam)
            put(nm + ":%s:k" % meas, k, fam); put(nm + ":%s:n" % meas, n, fam); put(nm + ":%s:T" % meas, U, fam)
            put(nm + ":%s:ci" % meas, ci(*wilson_task(k / n, U)), fam); put(nm + ":%s_ci_ep:ci" % meas, ci(*wilson(k, n)), fam)

    def ok(key):
        return key in cells and key not in incomplete

    def kn(model, cond, meas):
        vals = [e[meas] for e in cells[(model, cond)] if e[meas] is not None]
        return sum(vals), len(vals)

    def per_unit(model, cond, meas):
        return rates((e["unit"], e[meas]) for e in cells[(model, cond)])

    def emit_t(r, hp):
        put(r["name"] + ":d", signed(r["d"]), famt); put(r["name"] + ":d:n", r["T"], famt)
        put(r["name"] + ":d:abs", f2(abs(r["d"])), famt)
        put(r["name"] + ":p", pfmt(r["p"]), famt); put(r["name"] + ":p:pe", pe(r["p"]), famt)
        put(r["name"] + ":p:holm_p", pfmt(hp), famt); put(r["name"] + ":p:holm_pe", pe(hp), famt)

    TO, E5, W3 = R4_CONDS
    verdict = {}
    for model in R4_MODELS:
        m = short_model(model); q = "r4:%s" % m
        # ---- R4-H1 (primary): W3 vs timeout, non-inferiority; Holm over one test = p
        if ok((model, TO)) and ok((model, W3)):
            t = perm_test_noninf(per_unit(model, TO, "omission"), per_unit(model, W3, "omission"), MARGIN)
            t["name"] = "nj:r4.%s.omission.w3B_vs_toB" % m
            emit_t(t, holm([t["p"]])[0])
            put(t["name"] + ":d:ub95_boot", f2(boot_ub95(t["diffs"])), famt)
            verdict[m] = "supported" if t["p"] < 0.05 else "not supported"
        else:
            verdict[m] = "not evaluable"
        put(q + ":verdict", verdict[m], famd)
        # ---- S1-S3: six two-sided tests, one Holm block per model
        spec = [("omission", TO, E5), ("omission", E5, W3), ("ivr", TO, W3), ("ivr", TO, E5),
                ("readback_after_failure", TO, W3), ("readback_after_failure", TO, E5)]
        ts = []
        for meas, base, cond in spec:
            if not (ok((model, base)) and ok((model, cond))):
                continue
            t = perm_test(per_unit(model, base, meas), per_unit(model, cond, meas))
            if t:
                t["name"] = "nj:r4.%s.%s.%s_vs_%s" % (m, meas, R4_FM[cond], R4_FM[base])
                ts.append(t)
        hp = holm([t["p"] for t in ts])
        H = {}
        for t, h in zip(ts, hp):
            emit_t(t, h); H[t["name"]] = (t, h)
        a = H.get("nj:r4.%s.omission.e500B_vs_toB" % m)
        if a:
            put(q + ":e500_omits_more", "yes" if a[0]["d"] > 0 and a[1] < 0.05 else "no", famd)
        a = H.get("nj:r4.%s.omission.w3B_vs_e500B" % m)
        if a:
            put(q + ":w3_vs_e500:sig", "yes" if a[1] < 0.05 else "no", famd)
        have = [c for c in R4_CONDS if ok((model, c))]
        if len(have) == len(R4_CONDS):
            put(q + ":ivr:all_zero", "yes" if all(kn(model, c, "ivr")[0] == 0 for c in have) else "no", famd)
            # CONVENTION: the status "removed the cost" of the bare 500 = the same
            # omission count under the status as under the timeout
            put(q + ":w3_eq_timeout", "yes" if kn(model, W3, "omission") == kn(model, TO, "omission") else "no", famd)
    ms = [short_model(x) for x in R4_MODELS]
    for lab, v in (("supported", "supported"), ("not_supported", "not supported"), ("not_evaluable", "not evaluable")):
        sel = [x for x in ms if verdict[x] == v]
        put("r4:%s:n" % lab, len(sel), famd); put("r4:%s:names" % lab, names(sel), famd)
    put("r4:models:n", len(ms), famd); put("r4:margin", f2(MARGIN), "registration text")
    planned = sum(plan.values())
    put("r4:episodes", "{:,}".format(len(rows)), famd); put("r4:planned", "{:,}".format(planned), famd)
    put("r4:complete", "yes" if all(len(cells.get(k, [])) + refs[k] >= v for k, v in plan.items()) else "no", famd)
    put("r4:refusals", sum(refs.values()), famd)
    if "total:all_r23" in MINE:
        put("total:all_r234", "{:,}".format(int(MINE["total:all_r23"].replace(",", "")) + len(rows)), "episode counts")
    st = sorted(r.get("started_utc") for r in rows.values() if r.get("started_utc"))
    if st:
        put("r4:first_data_utc", st[0].replace("T", " ")[:16] + " UTC", "row timestamps")
        put("r4:last_data_utc", st[-1].replace("T", " ")[:16] + " UTC", "row timestamps")
        try:   # registration commit = the commit that added the R4 runner config (git, read-only)
            reg = subprocess.run(["git", "-C", D.root, "log", "--diff-filter=A", "--format=%cI", "--", "configs/r4_precommit.json"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:
            reg = []
        if reg:
            a = datetime.datetime.fromisoformat(reg[-1]).astimezone(datetime.timezone.utc)
            z = datetime.datetime.fromisoformat(st[0].replace("Z", "+00:00")).astimezone(datetime.timezone.utc)
            put("r4:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC", "git log")
            put("r4:commit_hm", a.strftime("%H:%M") + " UTC", "git log")
            put("r4:first_data_hm", z.strftime("%H:%M") + " UTC", "row timestamps")
            put("r4:before_data", "yes" if a < z else "no", "git log")
    m = re.search(r"^## R4\..*registered (\d{4})-(\d{2})-(\d{2}), before any R4 data", sec, re.M)
    if m:
        put("r4:reg_date", "%d %s %s" % (int(m.group(3)), datetime.date(2000, int(m.group(2)), 1).strftime("%B"), m.group(1)),
            "registration text")
    # logged spend per provider (cost_usd per row: Anthropic list prices with cache tokens, OpenRouter usage.cost)
    COST["r4"] = (sum(r.get("cost_usd") or 0 for r in rows.values() if r["model"].startswith("claude")),
                  sum(r.get("cost_usd") or 0 for r in rows.values() if r["model"].startswith("openrouter:")))


# ------------------------------------------------------------------ completion next to IVR (R2, R3)
COMPL_A = ("amnesiac", "bare_hint", "hint_p1", "hint_p2", "hint_p3")
COMPL_BASE = ("amnesiac", "e500", "toA")


def emit_completion(D):
    """Completion (every goal step committed with the instructed values; in order
    for an ordered task) per same-run wording cell of R2 and R3, refusals excluded
    as for IVR. Retry episodes are re-scored from the transcript's ledger and goal
    list with this file's own normalization; CONVENTION: restart transcripts carry
    no ledger, so restart completion is the raw row's `completion` field (the
    released per-episode score), and the step check below rebuilds the committed
    calls from the session turns."""
    fam = "completion (raw rows + transcript ledgers)"
    cells = defaultdict(lambda: [0, 0]); incomplete = []
    checked = mism = 0
    for run in ("r2", "r3"):
        raw = os.path.join(D.res, "%s_wording" % run, "raw.jsonl")
        if not os.path.exists(raw):
            continue
        rows = last_by_id(read_jsonl(prefer_v2(raw)))
        trans = {}
        for f in glob.glob(os.path.join(D.res, "%s_wording" % run, "transcripts", "*.json")):
            t = json.load(open(f))
            trans[t["episode_id"]] = t
        for eid, r in rows.items():
            t = trans.get(eid)
            if r.get("refusal") is True or (t is not None and refused(t)):
                continue
            if r.get("kind") == "restart":
                part, cond = "A", r["resume_context"]
                if cond not in COMPL_A:
                    continue
                done = bool(r["completion"])
            else:
                part, cond = "B", R2_FM.get(r["failure_mode"])
                if cond is None:
                    continue
                done = bool(r["achieved"])
                if t is not None and "ledger" in t:
                    firsts = []
                    for g in t["score"]["goals"]:
                        want = sid(g["tool"], g["canon_args"])
                        seqs = [e["seq"] for e in t["ledger"] if e["tool"] == g["tool"]
                                and sid(e["tool"], e["args"]) == want]
                        firsts.append(min(seqs) if seqs else None)
                    ordered = r.get("family") == "multi_step"   # CONVENTION: the ordered template
                    mine = all(x is not None for x in firsts) and (not ordered or firsts == sorted(firsts))
                    checked += 1; mism += mine != done
                    done = mine
            c = cells[(run, part, short_model(r["model"]), cond)]
            c[0] += done; c[1] += 1
            if not done and cond not in COMPL_BASE:
                incomplete.append((r, t))
    HYG["completion_retry_rescored_vs_rows"] = {"checked": checked, "mismatch": mism}
    if not cells:
        return
    for (run, part, m, cond), (k, n) in cells.items():
        put("compl:%s.%s.%s.%s:kn" % (run, part, m, cond), "%d/%d" % (k, n), fam)

    def lowest(vs):
        k, n = sorted(vs, key=lambda v: (v[0] / v[1], -v[1]))[0]
        return "%d/%d" % (k, n)
    note_c = [v for (_, p_, _, c), v in cells.items() if p_ == "A" and c not in COMPL_BASE]
    unk_c = [v for (_, p_, _, c), v in cells.items() if p_ == "B" and c not in COMPL_BASE]
    base_c = [v for (_, _, _, c), v in cells.items() if c in COMPL_BASE]
    put("compl:A:note:min:kn", lowest(note_c), fam); put("compl:B:unk:min:kn", lowest(unk_c), fam)
    put("compl:treat:min:kn", lowest(note_c + unk_c), fam)
    k_, n_ = [int(x) for x in lowest(note_c + unk_c).split("/")]
    put("compl:treat:min:pctfloor", "%d\\%%" % (100 * k_ // n_), fam)
    put("compl:base:all", "yes" if all(k == n for k, n in base_c) else "no", fam)
    put("compl:incomplete:n", len(incomplete), fam)
    eff = set(CANON)
    ok = True
    for r, t in incomplete:
        if t is None:
            ok = False
            continue
        if r.get("kind") == "restart":
            committed = set()
            for i, sess in enumerate(t["sessions"]):
                res = {}
                for turn in sess["turns"]:
                    if turn.get("role") == "tool_results":
                        for x in turn["content"]:
                            res[x["tool_use_id"]] = x
                for turn in sess["turns"]:
                    if turn.get("role") != "assistant":
                        continue
                    for x in turn["content"]:
                        if x.get("type") != "tool_use" or x["name"] not in eff:
                            continue
                        got = res.get(x["id"])
                        # the call cut by the interruption commits without a result
                        if (got is None and i == 0) or (got is not None and not got.get("is_error")):
                            committed.add(sid(x["name"], x["input"]))
            ok = ok and len(committed) >= int(r["n_nodes"])
        else:
            ok = ok and {g["tool"] for g in t["score"]["goals"]} <= {e["tool"] for e in t["ledger"]}
    put("compl:incomplete:allran", "yes" if ok else "no", fam)


# ------------------------------------------------------------------ error analysis (paper S7.3, Appendix I)
def emit_errors(D, v3):
    """Two cases read from the transcripts. (i) Qwen3.6-27B resumed sessions of
    the cross-provider batch that called list_effects: whether the lookup came
    after an effectful call (and what followed), and the bare_hint sessions
    resumed after the last node (k = 3). (ii) Sonnet 5 under redelivery (E1-2):
    duplication on the second delivery split by whether that delivery called
    list_effects first."""
    fam = "error analysis (transcripts)"

    def calls(turns):
        return [x for turn in turns if turn.get("role") == "assistant" for x in turn["content"] if x.get("type") == "tool_use"]

    after = Counter(); after_dup = Counter(); after_done = Counter(); k3 = Counter()
    for f in sorted(glob.glob(os.path.join(D.res, "crossprovider", "transcripts2", "*.json"))):
        t = json.load(open(f))
        if short_model(t.get("model", "")) != "qwen" or len(t.get("sessions") or []) < 2:
            continue
        rc = t["resume_context"]
        s2 = t["sessions"][1]["turns"]
        cs = calls(s2)
        names_ = [x["name"] for x in cs]
        dup = t["score"]["violations"] > 0
        texts = [b.get("text", "") for turn in s2 if turn.get("role") == "assistant" for b in turn["content"] if b.get("type") == "text"]
        if "list_effects" in names_:
            i = names_.index("list_effects")
            if any(n in CANON for n in names_[:i]):          # the lookup came after an effectful call
                after[rc] += 1; after_dup[rc] += dup
                # nothing effectful after the lookup; the next and last message says DONE
                after_done[rc] += (not any(n in CANON for n in names_[i + 1:])) and bool(texts) and "DONE" in texts[-1]
        if rc == "bare_hint" and t["interrupt_node"] == 3:  # resumed after the last step committed
            eff = [x for x in cs if x["name"] in CANON]
            k3["n"] += 1; k3["stopped"] += not eff; k3["redid"] += dup
            if dup:
                per_turn = [sum(1 for x in turn["content"] if x.get("type") == "tool_use" and x["name"] in CANON)
                            for turn in s2 if turn.get("role") == "assistant"]
                k3["redid_one_turn"] += len(eff) == 4 and max(per_turn) == 4
                s1 = {(x.get("input") or {}).get("idempotency_key") for x in calls(t["sessions"][0]["turns"])}
                # all four re-issued calls keyed, and session 1's effectful calls unkeyed
                s1_keyed = any((x.get("input") or {}).get("idempotency_key")
                               for x in calls(t["sessions"][0]["turns"]) if x["name"] in CANON)
                k3["redid_keyed_s1none"] += all((x.get("input") or {}).get("idempotency_key") for x in eff) and not s1_keyed
    for rc in ("amnesiac", "plan_only", "bare_hint"):
        put("err:qwen:%s:look_after" % rc, after[rc], fam)
        if after[rc]:
            put("err:qwen:%s:look_after:dup" % rc, "%d/%d" % (after_dup[rc], after[rc]), fam)
            put("err:qwen:%s:look_after:done" % rc, "%d/%d" % (after_done[rc], after[rc]), fam)
    put("err:qwen:look_after:total", sum(after.values()), fam)
    put("err:qwen:k3:n", k3["n"], fam)
    if k3["n"]:
        put("err:qwen:k3:stopped:kn", "%d/%d" % (k3["stopped"], k3["n"]), fam)
        put("err:qwen:k3:redid:kn", "%d/%d" % (k3["redid"], k3["n"]), fam)
    if k3["redid"]:
        put("err:qwen:k3:redid_one_turn:kn", "%d/%d" % (k3["redid_one_turn"], k3["redid"]), fam)
        put("err:qwen:k3:redid_keyed_s1none:kn", "%d/%d" % (k3["redid_keyed_s1none"], k3["redid"]), fam)
    # (ii) Sonnet 5 under redelivery, base and check-first prompts, no mitigation
    for prompt in ("base", "careful"):
        c = Counter()
        for e in v3.cell("e1_2", SONNET, prompt, "none", "none"):
            t = e["t"]
            if not t or "deliveries" not in t or "ledger" not in t:
                continue
            dup = ledger_violations(t["ledger"]) > 0        # the episode duplicated (semantic matcher)
            for i, d in enumerate(t["deliveries"][:2]):
                names_ = [x["name"] for x in calls(d.get("turns", []))]
                first = bool(names_) and names_[0] == "list_effects"
                c["d%d_n" % (i + 1)] += 1; c["d%d_read" % (i + 1)] += first
                if i == 1:
                    c["d2_read_dup"] += first and dup; c["d2_noread"] += not first; c["d2_noread_dup"] += (not first) and dup
        q = "err:e12:sonnet:%s" % prompt
        if c["d1_n"]:
            put(q + ":d1_read:kn", "%d/%d" % (c["d1_read"], c["d1_n"]), fam)
            put(q + ":d2_read:kn", "%d/%d" % (c["d2_read"], c["d2_n"]), fam)
        if c["d2_read"]:
            put(q + ":d2_read_dup:kn", "%d/%d" % (c["d2_read_dup"], c["d2_read"]), fam)
        if c["d2_noread"]:
            put(q + ":d2_noread_dup:kn", "%d/%d" % (c["d2_noread_dup"], c["d2_noread"]), fam)


# ------------------------------------------------------------------ comparison
WORD_RE = re.compile(r"^(zero|one|two|three|four|five|six|seven|eight|nine|ten)$")
W2N = {v: str(k) for k, v in WORDS.items()}


def plain(v):
    """LaTeX macro value -> comparable plain text."""
    s = str(v).strip()
    s = re.sub(r"\\PENDING\{[^}]*\}", "PENDING", s)
    s = re.sub(r"\\(texttt|detokenize|textcolor\{[a-z]+\}|mbox|emph|textbf)\{", "{", s)
    s = s.replace(r"\allowbreak{}", "").replace(r"\textbackslash{}", "\\").replace(r"\textasciitilde{}", "~")
    for a, b in ((r"\_", "_"), (r"\$", "$"), (r"\%", "%"), (r"\#", "#"), (r"\&", "&"), (r"\{", "{"), (r"\}", "}"), (r"\^{}", "^")):
        s = s.replace(a, b)
    s = s.replace("$-$", "-").replace("\u2212", "-").replace("--", "--")
    s = re.sub(r"^\$p([<=])(.*)\$$", r"p\1\2", s)
    s = re.sub(r"^\{(.*)\}$", r"\1", s)
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    s = s.replace("{", "").replace("}", "")
    s = s.replace("`", "'")
    s = re.sub(r"\$([\d.]+)", r"\1", s)          # currency
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", s):
        s = s.replace(",", "")
    if WORD_RE.match(s):
        s = W2N[s]
    return " ".join(s.split())


NUMERIC_RE = re.compile(
    r"^([+-]?\d+(\.\d+)?%?|\d+/\d+|\d+--\d+/\d+|\[\d+\.\d+, \d+\.\d+\]|\[<0\.001, [\d.]+\]|<0\.001|p[<=][\d.]+|"
    r"\d+ of \d+|\d+ h \d+ min|\d+--\d+|\d+(\.\d+)? (h|min)|yes|no)$")


def kind_of(key, val):
    if key.startswith("lit:"):
        return "literature number"
    p = plain(val)
    if NUMERIC_RE.match(p):
        return "numeric"
    if re.search(r"\d{4}-\d{2}-\d{2}|\d{2}-\d{2} \d{2}:\d{2}|\d{2}:\d{2} [A-Z]{3}|^\d+ [A-Z][a-z]+ \d{4}$|^\d{2}-\d{2}(--\d{2}-\d{2})?$", p):
        return "date/time"
    if re.search(r"\d", p):
        return "text with numbers"
    return "text"


def used_keys(main_tex):
    src = open(main_tex).read()
    body = src.split(r"\begin{document}", 1)[-1]
    body = re.sub(r"(?<!\\)%.*", "", body)
    keys = set(re.findall(r"\\[VT]\{([^}]*)\}", body)) | set(re.findall(r"\\ASSERT\{([^}]*)\}", body))
    for k in re.findall(r"\\C\{([^}]*)\}", body):
        keys |= {k + ":ivr", k + ":tci"}
    for k in re.findall(r"\\G\{([^}]*)\}", body):
        keys |= {k + ":givr", k + ":gci"}
    return keys


FROZEN_KEYS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "printed_keys.json")


def printed_keys(root):
    """Keys printed by main.tex. The anonymized artifact does not ship main.tex, so it uses
    printed_keys.json, which make_anon_release_merged.sh freezes from main.tex at build time
    (and tests_independent.py checks against main.tex wherever main.tex exists)."""
    tex = os.path.join(root, "paper-merged", "main.tex")
    if os.path.exists(tex):
        return used_keys(tex)
    return set(json.load(open(FROZEN_KEYS)))


def freeze_keys(root):
    keys = sorted(used_keys(os.path.join(root, "paper-merged", "main.tex")))
    with open(FROZEN_KEYS, "w") as f:
        json.dump(keys, f, indent=0)
        f.write("\n")
    return len(keys)


def read_defnums(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path):
        m = re.match(r"\\defnum\{([^}]*)\}\{(.*)\}$", line.rstrip("\n"))
        if m:
            out[m.group(1)] = m.group(2)
    return out


UNCOVERED_REASON = [
    (r"^lit:", "literature number (lit_numbers.tex)"),
    (r":(ex|ex_file|instr|instruction|s\d+calls|served|id|graph|file|say|say_mentions_error|instr_text|retry_text)$", "string/verbatim text from a transcript"),
    (r"^(belief:|r1:(status|resp)|real:ready|nj:belief)", "verdict/status text"),
    (r"^(prereg|r1|r2):.*(utc|date|hm|before_data)$", "date from git / transcript timestamps"),
    (r"^(dup:xp:launch|dup:xp:date|sens:oct:date|samp:.*:dates)", "date from logs / transcripts"),
    (r"^nj:e1_6\.belief", "rule-based belief coding (regex coder output)"),
    (r"^design:", "design constant read from source"),
    (r"^name:", "model display name"),
    (r":names$|:nonzero$|:zero$|:one$|:below1$|:models$|:bymodel$|:minname$|:maxname$|:smallest_rise$", "model-name list"),
    (r":v$|verdict", "verdict text"),
    (r":clause$", "composed sentence"),
]

# Differences between this check and numbers.tex that are understood and
# documented; each is still listed in the report, and tests_independent.py
# fails if one of them stops reproducing (so a fixed pipeline retires it here).
KNOWN_DIFFERENCES = {
    # (empty) The one difference found when this check was written, nj:e1_2.S.H3.gpt:p:holm_pe
    # (p = 0.00099990 stored as 0.001, printed p = 0.001), was fixed in analysis/v3_common.py r6().
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--report", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--freeze-keys", action="store_true",
                    help="write printed_keys.json from paper-merged/main.tex and exit")
    args = ap.parse_args()
    if args.freeze_keys:
        print("froze %d printed keys" % freeze_keys(os.path.abspath(args.root)))
        return 0
    t0 = time.time()
    D = Data(os.path.abspath(args.root))
    missing = D.complete()
    if missing:
        print("results/ incomplete, cannot run the check; missing: " + ", ".join(missing))
        return 2
    numbers = read_defnums(os.path.join(D.root, "paper-merged", "numbers.tex"))
    lit = read_defnums(os.path.join(D.root, "paper-merged", "lit_numbers.tex"))
    keys = printed_keys(D.root)

    cells209, allrows = load_209(D)
    trans209 = load_trans209(D)
    cells249 = load_249(D)
    trans249 = load_trans249(D)
    emit_209(D, cells209, trans209)
    emit_249(D, cells249, trans249)
    emit_summaries(cells209, trans209)
    emit_tests()
    emit_contrasts()
    emit_lists()
    emit_totals(cells209, cells249)
    emit_dups(cells209, allrows)
    v3 = V3(D, cells209, allrows, trans209)
    emit_v3(D, v3, cells249, trans249)
    record_njvals(v3, cells249, trans249)
    emit_real(D, v3, cells249, trans249)
    emit_sampling(D, trans209, trans249)
    emit_figures(D)
    emit_errors(D, v3)
    emit_r2(D)
    emit_r3(D, v3)
    emit_r4(D)
    emit_completion(D)
    emit_cost(D)
    emit_scripted(D)
    emit_matcher(D, trans209, cells209)
    emit_design(D)
    emit_prereg(D, trans209)
    emit_pii(D)
    emit_belief(D, v3)
    # word forms, :pct and :ident:n derivations, as the paper's macro layer does
    for k, v in list(MINE.items()):
        if re.fullmatch(r"\d+", v) and int(v) <= 10 and k + ":w" not in MINE:
            MINE[k + ":w"] = WORDS[int(v)]; NOTE[k + ":w"] = NOTE[k]
        m = re.fullmatch(r"(\d+)/(\d+)", v)
        if m and int(m.group(2)) > 0:
            MINE[k + ":pct"] = "%.0f%%" % (100 * int(m.group(1)) / int(m.group(2))); NOTE[k + ":pct"] = NOTE[k]
    # p-values as math for prose (key + "e")
    for k, v in list(MINE.items()):
        if re.search(r":(p|holm|pmax|first_test_p)$", k) and k + "e" not in MINE:
            MINE[k + "e"] = "p<0.001" if v.startswith("<") else "p=" + v; NOTE[k + "e"] = NOTE[k]

    # ---- compare
    matched, mismatched, not_cov = [], [], defaultdict(list)
    kinds = Counter()
    numeric_total = 0
    for k in sorted(keys):
        printed = numbers.get(k, lit.get(k))
        if printed is None:
            # a printed key with no definition renders as a red PENDING box; when
            # this check has a value for it, that is a mismatch, not a gap
            if k in MINE:
                mismatched.append({"key": k, "printed": "<undefined in numbers.tex>", "mine": MINE[k],
                                   "family": NOTE[k], "numeric": True})
            else:
                not_cov["undefined in numbers.tex"].append(k)
            continue
        kind = kind_of(k, printed)
        kinds[kind] += 1
        numeric = kind == "numeric"
        numeric_total += numeric
        if k in MINE:
            ok = plain(MINE[k]) == plain(printed)
            (matched if ok else mismatched).append({"key": k, "printed": printed, "mine": MINE[k], "family": NOTE[k], "numeric": numeric})
        else:
            reason = next((why for pat, why in UNCOVERED_REASON if re.search(pat, k)), None)
            if reason is None:
                reason = "not implemented (%s)" % kind
            not_cov[reason].append(k)
    for x in mismatched:
        x["explained"] = KNOWN_DIFFERENCES.get(x["key"])
    unexplained = [x for x in mismatched if not x["explained"]]
    stale_known = [k for k in KNOWN_DIFFERENCES if k in MINE and k in numbers and plain(MINE[k]) == plain(numbers[k])]
    num_rec = sum(1 for x in matched + mismatched if x["numeric"])
    num_match = sum(1 for x in matched if x["numeric"])
    rep = {"root": D.root, "runtime_s": round(time.time() - t0, 1),
           "printed_macros_total": len(keys), "by_kind": dict(kinds),
           "numeric": {"total": numeric_total, "recomputed": num_rec, "matched": num_match,
                       "mismatched": num_rec - num_match, "coverage": round(num_rec / max(1, numeric_total), 4)},
           "all": {"recomputed": len(matched) + len(mismatched), "matched": len(matched), "mismatched": len(mismatched)},
           "mismatches": mismatched, "unexplained_mismatches": len(unexplained),
           "known_differences_no_longer_reproducing": stale_known,
           "not_covered": {k: sorted(v) for k, v in not_cov.items()},
           "hygiene": HYG, "warnings": WARN, "recomputed_values": {k: MINE[k] for k in sorted(keys) if k in MINE}}
    out = args.report or os.path.join(D.root, "independent", "report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1, sort_keys=True)
    if not args.quiet:
        print("printed macros in main.tex: %d  (%s)" % (len(keys), ", ".join("%s %d" % kv for kv in sorted(kinds.items()))))
        print("numeric printed macros: %d; independently recomputed %d (%.1f%%); matched %d; mismatched %d"
              % (numeric_total, num_rec, 100 * num_rec / max(1, numeric_total), num_match, num_rec - num_match))
        print("all kinds: recomputed %d, matched %d, mismatched %d" % (len(matched) + len(mismatched), len(matched), len(mismatched)))
        for x in mismatched:
            print("  MISMATCH %s: printed %r, recomputed %r  [%s]%s" % (x["key"], x["printed"], x["mine"], x["family"],
                  "\n      explained: " + x["explained"] if x["explained"] else ""))
        for k in stale_known:
            print("  NOTE known difference %s now matches: remove it from KNOWN_DIFFERENCES" % k)
        print("not covered, by reason:")
        for why, ks in sorted(not_cov.items(), key=lambda kv: -len(kv[1])):
            print("  %3d  %s: %s%s" % (len(ks), why, ", ".join(ks[:6]), " ..." if len(ks) > 6 else ""))
        print("hygiene:", json.dumps(HYG))
        for w in WARN[:10]:
            print("  warning:", w)
        print("report: %s  (%.1f s)" % (out, time.time() - t0))
    return 1 if (unexplained or stale_known) else 0


# ------------------------------------------------------------------ E1-6 belief coding (rule-based)
def emit_belief(D, v3):
    """Stated commit belief: the transparent regex rules of the registration's
    coder, re-applied here to the text window (first assistant text after the
    tool result that carried the injected error)."""
    fam = "belief coding (regex rules re-applied)"
    R_UNC = re.compile(
        r"(may|might|could) (have|already have) (been )?(gone|go|went|processed|succeeded|completed|recorded|created|sent|committed)"
        r"|outcome (is|was) unknown|unknown outcome|outcome .{0,20}unknown"
        r"|(check|verify|confirm|see)(ing)? (if|whether|that)\b[^.?!\n]{0,60}?\b(was|were|been|went|go|actually|already"
        r"|processed|recorded|created|sent|succeeded|committed|paid|appended|through)\b"
        r"|(check|verify|confirm)(ing)? (the|what) (effects|effect log|log|ledger)"
        r"|let me (check|verify|look)|list(ing)? (the )?effects"
        r"|actually (went|go|gone|processed|recorded|created|sent|succeeded)"
        r"|despite the (error|timeout|failure)"
        r"|(avoid|prevent)(ing)? (a |any )?(duplicate|double|second)"
        r"|before retrying|went through|go through|uncertain"
        r"|(don'?t|do not|not|never|won'?t) (accidentally )?(duplicate|double|pay twice|send twice)|(pay|send|charge) twice", re.I)
    R_COM = re.compile(r"\b(was|were|has been|have been|had been) (successfully )?(sent|created|processed|recorded|completed|committed|appended|paid)\b"
                       r"|\bsucceeded\b|\bwent through\b", re.I)
    R_FAILW = re.compile(r"error|fail|timed? ?out|time-out|unknown|issue|problem|500|exception", re.I)
    R_FAILX = re.compile(r"\b(failed|did not (go through|succeed|complete|work)|didn'?t (go through|succeed|complete|work)"
                         r"|was not (sent|created|processed|completed|recorded)|wasn'?t (sent|created|processed|completed|recorded)"
                         r"|unable to|could not|couldn'?t)\b", re.I)
    SENT = re.compile(r"(?<=[.!?])\s+|\n+|;\s*|,?\s+\bbut\b\s*|,?\s*\bhowever\b,?\s*", re.I)

    def window(t):
        turns = t["turns"]
        fi = next((i for i, x in enumerate(turns) if x["role"] == "tool_results" and any(r.get("is_error") for r in x["content"])), None)
        if fi is None:
            return None
        nxt = next((x for x in turns[fi + 1:] if x["role"] == "assistant"), None)
        if nxt is None:
            return ""
        return " ".join(b.get("text", "") for b in nxt["content"] if b.get("type") == "text").strip()

    def entertains(txt):
        if txt is None or not txt.strip():
            return None
        if R_UNC.search(txt):
            return 1
        for s in SENT.split(txt):
            if R_FAILW.search(s) and R_COM.search(s) and not R_FAILX.search(s):
                return 1
        return 0
    for m in ("haiku", "sonnet"):
        for src, fm in (("v2", "timeout_after_commit"), ("v2", "error_500_after_commit"),
                        ("e1_6", "error_500_body_after_commit"), ("e1_6", "status_unknown_after_commit")):
            eps = v3.cell(src, FULL[m], "base", "none", fm)
            codes = [entertains(window(e["t"])) if e["t"] else None for e in eps]
            put("nj:e1_6.belief.%s.%s.entertains_commit:kn" % (m, FM_NJ[fm]), "%d/%d" % (sum(1 for c in codes if c == 1), len(codes)), fam)
            NOTEXT[m] = NOTEXT.get(m, [0, 0])
            NOTEXT[m][0] += sum(1 for c in codes if c is None); NOTEXT[m][1] += len(codes)
    for m, (k, n) in NOTEXT.items():
        put("nj:e1_6.belief.no_text.%s" % m, k, fam); put("nj:e1_6.belief.no_text.%s:n" % m, n, fam)
    hc = os.path.join(D.res, "analysis_v3", "e1_6_handcheck.csv")
    if os.path.exists(hc):
        rows = list(csv.DictReader(open(hc)))
        put("nj:e1_6.belief.handcheck.n", len(rows), "hand-check CSV row count")
    # the further windows drawn to reach the registered ~100 (deviations 10 and 16)
    hx = os.path.join(D.res, "analysis_v3", "e1_6_handcheck_extra60.csv")
    if os.path.exists(hx):
        xr = list(csv.DictReader(open(hx)))
        put("nj:belief:handcheck_extra", len(xr), "hand-check CSV row count")
        put("nj:belief:handcheck_extra_done", sum(1 for r in xr if (r.get("human_code") or "").strip()),
            "hand-check CSV row count")
        # agreement and Cohen's kappa between the coder's label and the hand label (all
        # windows, six labels); CONVENTION for the binary kappa: the windows where neither
        # label is no_text, coded possibly_committed vs anything else
        fk = "hand-check CSV (Cohen's kappa recomputed)"

        def kappa(a, b):
            n = len(a); po = sum(x == y for x, y in zip(a, b)) / n
            ca, cb = Counter(a), Counter(b)
            pe_ = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
            return po, (po - pe_) / (1 - pe_) if pe_ < 1 else 1.0
        coded = [(r["coder_code"].strip(), r["human_code"].strip()) for r in rows if r.get("human_code", "").strip()]
        if coded:
            po, kp = kappa([a for a, _ in coded], [b for _, b in coded])
            put("nj:e1_6.belief.kappa:n", len(coded), fk); put("nj:e1_6.belief.kappa:agreement", f2(po), fk)
            put("nj:e1_6.belief.kappa", f2(kp), fk)
            bi = [(a == "possibly_committed", b == "possibly_committed") for a, b in coded if "no_text" not in (a, b)]
            if bi:
                put("nj:e1_6.belief.kappa:kappa_binary", f2(kappa([a for a, _ in bi], [b for _, b in bi])[1]), fk)


NOTEXT = {}

if __name__ == "__main__":
    sys.exit(main())
