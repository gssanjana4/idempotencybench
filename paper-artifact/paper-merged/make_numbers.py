#!/usr/bin/env python3
"""Generate paper-merged/numbers.tex: every number in the paper.

Rule: no number in main.tex is typed by hand. main.tex prints values with
\\V{key}. This script reads the released and new result files directly and
writes one \\defnum{key}{value} line per value. A key that is not defined
(results not produced yet) renders as a red \\PENDING box, so a missing
number is visible and never guessed. A cell whose run is still in progress
is written as \\PENDING{key (n/expected)}.

Reads (read-only; nothing outside paper-merged/ is written):
  results/sweep_raw_v2.jsonl, results/sweep2_raw_v2.jsonl  (July, corrected)
  results/<batch>/sweep_raw*.jsonl, results/<batch>/sweep2_raw*.jsonl
      for every batch directory (crossprovider, claude_batch_2026-10, and
      any later Tier-1/Tier-2 directory), skipping smoke/ and *.bak
  results/sweep_transcripts/, results/transcripts2/, results/<batch>/...
  results/summary_by_failure.csv, results/graph_summary.csv  (scripted)
  results/analysis_v2/e0_3_summary.json, e0_4_summary.json,
      e0_5_sampling_protocol.csv  (Tier-0 analyses)
  src/model_agent.py, src2/sweep2.py, src2/graph.py, run_crossprovider.sh
      (design constants only, by regex; nothing is imported)

Writes: paper-merged/numbers.tex, paper-merged/numbers_report.txt

Run from the repo root or anywhere:
  python3 paper-merged/make_numbers.py            # write numbers.tex
  python3 paper-merged/make_numbers.py --check    # also lint main.tex:
      lists \\V keys that are undefined (= PENDING) and literal digits in
      the prose that do not come from a macro.
Deterministic: bootstrap seeds are derived from the cell key.
"""
import csv, glob, hashlib, json, math, os, random, re, subprocess, sys
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
OUT_TEX = os.path.join(HERE, "numbers.tex")
OUT_REP = os.path.join(HERE, "numbers_report.txt")

# ---------------------------------------------------------------- config
MODEL_SHORT = [  # (substring of model id, short key, display name), paper order
    ("haiku", "haiku", "Haiku 4.5"),
    ("sonnet", "sonnet", "Sonnet 5"),
    ("fable", "fable", "Fable 5"),
    ("gpt-5.5", "gpt", "GPT-5.5"),
    ("gemini", "gemini", "Gemini"),
    ("qwen", "qwen", "Qwen3.6-27B"),
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
FM_SHORT = {"none": "none", "timeout_after_commit": "toA",
            "timeout_before_commit": "toB", "error_500_after_commit": "e500",
            # Tier-1 E1-6 signal variants (names from the experiment specification B4)
            "error_500_body_after_commit": "e500body",
            "error_503_retry_after_after_commit": "e503",
            "status_unknown_after_commit": "unknown",
            "timeout_504_after_commit": "to504"}
MIT_SHORT = {"none": "none", "idem_keys": "keys", "runtime_receipts": "rcpt"}
BATCH_NAME = {"": "jul", "crossprovider": "xp", "claude_batch_2026-10": "oct",
              # Tier-1 runs (src/tier1.py): results/tier1/<exp>/raw.jsonl.
              # E1-1 holds only the OpenRouter keys-under-500 cell, which the
              # paper reports with E1-3 (Claude) under one prefix, e13.
              os.path.join("tier1", "e1_0"): "e10", os.path.join("tier1", "e1_1"): "e13",
              os.path.join("tier1", "e1_2"): "e12", os.path.join("tier1", "e1_3"): "e13",
              os.path.join("tier1", "e1_4"): "e14", os.path.join("tier1", "e1_5"): "e15",
              os.path.join("tier1", "e1_6"): "e16"}
EXPECTED_209 = 64          # 32 tasks x 2 samples
# analysis/preregistration.md, "Exclusions": a cell is reported only if it
# reached >= 90% of planned n, and its achieved n is always shown. Episodes
# a provider refused to serve (18 Gemini episodes, one task) are missing rows.
MIN_FRAC = 0.9


def min_n(expected):
    return math.ceil(MIN_FRAC * expected)

EXPECTED_209_BATCH = {"e15": 32}   # E1-5 long chains: 32 tasks x r = 1
CORE = {"jul", "xp", "oct"}        # the core grid; Tier-1 cells are reported
                                   # in their own rows and do not enter the
                                   # cross-cell summary statements
EXPECTED_249 = 32          # 16 graphs x 2 samples
EXPECTED_249_OVERRIDE = {("oct", "plan_only"): 16,   # fan-out-only rerun
                         ("jul", "bare_hint"): 16}   # July ablation: Fable, r = 1
BENCH_TOPOS = {"chain4", "fanout"}
BOOT_N = 10000
CANON = {  # canonical identity fields (checked against src/world.py by regex)
    "send_email": ("to", "subject"),
    "send_payment": ("payee", "amount", "memo"),
    "create_ticket": ("queue", "title"),
    "append_file": ("path", "content_hash"),
    "create_event": ("calendar_id", "title", "start"),
}
DOMAIN = {"send_email": "email", "send_payment": "payments",
          "create_ticket": "tickets", "append_file": "files",
          "create_event": "calendar"}

NUMS = {}            # key -> LaTeX value
FULL209 = ""         # appendix table rows (written as \\fullrows)
DUPROWS = ""         # appendix table rows for the double-run cells (\\duprows)
REPORT = []          # human-readable notes
DUPS = {}            # batch -> {episode_id: [every row, in file order]}
EPI = {}             # 209 qual -> [(task_id, violated 0/1, readback 0/1/None)]
EPI249 = {}          # 249 qual -> {(graph_id, interrupt_node, sample): givr_hit}
PAIRED_MDD = {}      # model -> paired minimum detectable difference (timeout vs 500)
PERM_N = 10000       # paired permutation over tasks: 10,000 permutations,
PERM_SEED = 20261025 # fixed seed (analysis/preregistration.md, "Contrasts")
SNAPSHOT = os.path.join(HERE, "numbers.json")   # pass-1 values for --render-only
_NUM = re.compile(r"^[$€£]?\s*-?[\d,]+(\.\d+)?$")


def put(key, value):
    NUMS[key] = str(value)


def pending(key, note):
    NUMS[key] = r"\PENDING{%s (%s)}" % (key, note)
    NUMS[key + ":isp"] = "1"          # lets main.tex print a short box in tables


def note(msg):
    REPORT.append(msg)


# ---------------------------------------------------------------- helpers
def f2(x):
    """Two decimals, half-up (so 20/32 prints 0.63, as a reader computes it)."""
    from decimal import ROUND_HALF_UP
    return str(Decimal(repr(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(0.0, p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (max(0.0, c - h), min(1.0, c + h))


def ci_str(lo, hi):
    return f"[{f2(lo)}, {f2(hi)}]"


def boot_ci(clusters, key):
    """Percentile cluster bootstrap of a pooled proportion.
    clusters: list of (sum_of_hits, size). Seed derived from key."""
    T = len(clusters)
    if T == 0:
        return (0.0, 1.0)
    rng = random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
    bs = []
    for _ in range(BOOT_N):
        idx = [rng.randrange(T) for _ in range(T)]
        s = sum(clusters[i][0] for i in idx)
        m = sum(clusters[i][1] for i in idx)
        bs.append(s / m)
    bs.sort()
    return bs[int(0.025 * BOOT_N)], bs[int(0.975 * BOOT_N) - 1]


def norm(v):
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


def short_model(model):
    m = model.lower()
    for sub, key, _ in MODEL_SHORT:
        if sub in m:
            return key
    return re.sub(r"[^a-z0-9]+", "-", m)


def batch_of(path):
    """Batch tag: 'jul' for the released July data, 'xp' / 'oct' for the
    October batch folders, 'e12' etc. for a config run named E1-2_*."""
    m = re.match(r"E(\d)-(\d+)_", os.path.basename(path))
    if m:
        return f"e{m.group(1)}{m.group(2)}"
    rel = os.path.relpath(os.path.dirname(path), RES)
    rel = "" if rel == "." else rel
    return BATCH_NAME.get(rel, re.sub(r"[^A-Za-z0-9]+", "-", rel))


def tex_id(s):
    """Verbatim-ish identifier. '#', '%', braces and backslashes cannot sit
    inside \\detokenize in a \\def body, so they are set outside it."""
    s = str(s)
    special = {"#": r"\#", "%": r"\%", "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}"}
    parts, buf = [], ""
    for ch in s:
        if ch in special:
            if buf:
                parts.append(r"\detokenize{%s}" % buf)
                buf = ""
            parts.append(special[ch])
        else:
            buf += ch
    if buf:
        parts.append(r"\detokenize{%s}" % buf)
    return r"\texttt{%s}" % "".join(parts)


def tex_path(s):
    """A file name in typewriter type that LaTeX may break after _ - . or :"""
    rep = {"_": r"\_\allowbreak{}", "-": r"-\allowbreak{}", ".": r".\allowbreak{}",
           ":": r":\allowbreak{}", "#": r"\#", "%": r"\%", "&": r"\&", "~": r"\textasciitilde{}"}
    return r"\texttt{%s}" % "".join(rep.get(c, c) for c in str(s))


def tex_text(s):
    """Escape plain text for LaTeX prose (roman font, line-breakable)."""
    rep = {"\\": r"\textbackslash{}", "$": r"\$", "%": r"\%", "#": r"\#", "&": r"\&",
           "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}"}
    # a straight ' that opens a quote (string start, or after a space or "(")
    # becomes a LaTeX opening quote, so 'Receipt: ...' renders with curly quotes
    s = re.sub(r"(?<![^\s(])'", "`", str(s))
    return "".join(rep.get(c, c) for c in s)


def plain(v):
    """LaTeX value -> plain text for the Markdown templates."""
    v = re.sub(r"\\PENDING\{([^}]*)\}", r"**[PENDING: \1]**", v)
    v = re.sub(r"\\texttt\{(.*)\}$", r"`\1`", v)
    v = re.sub(r"\\detokenize\{([^}]*)\}", r"\1", v)
    v = v.replace(r"\allowbreak{}", "").replace(r"\_", "_").replace(r"\$", "$")
    v = re.sub(r"\$p([<=])([^$]*)\$", r"p \1 \2", v)
    v = v.replace("$-$", "\u2212")
    v = v.replace(r"\%", "%").replace(r"\#", "#").replace(r"\&", "&").replace("--", "–")
    return v


def labels_from_aux():
    """Section/table numbers from main.aux (after a LaTeX run), so the
    checklist and the response letter cite numbers that exist in the PDF."""
    aux = os.path.join(HERE, "main.aux")
    if not os.path.exists(aux):
        note("main.aux not found: compile main.tex before rendering templates")
        return
    for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}\{([^}]*)\}", open(aux).read()):
        lab, num, page = m.groups()
        if lab.startswith(("sec:", "tab:", "fig:", "app:")):
            NUMS["ref:" + lab] = num
            NUMS["page:" + lab] = page


def render_templates():
    """paper-merged/*.template.md -> *.md, replacing {{key}} by its value;
    an unknown key becomes a visible **[PENDING: key]** marker."""
    labels_from_aux()
    out = []
    for tp in sorted(glob.glob(os.path.join(HERE, "*.template.md"))):
        src = open(tp).read()
        missing = []

        def sub(m):
            k = m.group(1).strip()
            if k in NUMS:
                return plain(NUMS[k])
            missing.append(k)
            return f"**[PENDING: {k}]**"
        txt = re.sub(r"\{\{([^}]+)\}\}", sub, src)
        dst = tp.replace(".template.md", ".md")
        with open(dst, "w") as fh:
            if os.path.basename(dst) == "response_to_reviewers.md":
                txt = re.sub(r"[ \t]*<!--(?!\s*OPEN ITEM).*?-->", "", txt, flags=re.S)
            else:
                fh.write("<!-- GENERATED from %s by make_numbers.py; edit the template, "
                         "not this file -->\n" % os.path.basename(tp))
            fh.write(txt)
        out.append((os.path.basename(dst), sorted(set(missing))))
    return out


def read_jsonl(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# episodes re-scored by src/rescore_c1.py (ISO date-time seconds); their
# transcripts keep the pre-fix score, the _v2 raw row holds the new one
_C1 = os.path.join(RES, "analysis_v3", "rescore_c1.csv")
C1_RESCORED = {r["episode_id"] for r in csv.DictReader(open(_C1))} if os.path.exists(_C1) else set()


def prefer_v2(path):
    """x.jsonl -> x_v2.jsonl when it exists (C1 re-score, src/rescore_c1.py;
    the original x.jsonl is never edited)."""
    root, ext = os.path.splitext(path)
    p2 = root + "_v2" + ext
    return p2 if not root.endswith("_v2") and os.path.exists(p2) else path


def raw_files(pattern_v2, pattern_any):
    """Corrected July file at results/ root plus every batch directory, plus
    Tier-1/2 config runs written as results/**/E1-x_raw.jsonl / E2-x_raw.jsonl
    (the registered metric definitions). Paper is told apart by the row fields."""
    files = [os.path.join(RES, pattern_v2)]
    cands = glob.glob(os.path.join(RES, "**", pattern_any), recursive=True) + \
        glob.glob(os.path.join(RES, "**", "E[0-9]-*_raw.jsonl"), recursive=True) + \
        [prefer_v2(p) for p in glob.glob(os.path.join(RES, "tier1", "e1_*", "raw.jsonl"))]
    want_249 = pattern_any.startswith("sweep2")
    for p in sorted(set(cands)):
        rel = os.path.relpath(p, RES)
        if os.path.dirname(rel) == "":          # root: only the v2 file
            continue
        if "smoke" in rel.split(os.sep) or p.endswith(".bak") or "analysis_v2" in rel:
            continue
        if rel.split(os.sep)[0] in ("r2_wording", "r2_wording_or", "r3_wording"):   # R2 / R3 have their own analyses (r2_wording.py, r3_wording.py)
            continue
        if rel.split(os.sep)[0] == "r4_precommit":   # R4: own analysis (r4_precommit.py), merged in emit_analysis_v3
            continue
        if prefer_v2(p) != p:                   # superseded by its C1 re-score
            continue
        if os.path.basename(p).startswith("E") or os.path.basename(p) in ("raw.jsonl", "raw_v2.jsonl"):
            try:
                first = json.loads(open(p).readline())
            except Exception:
                continue
            if ("resume_context" in first) != want_249:
                continue
        files.append(p)
    return [f for f in files if os.path.exists(f)]


# ---------------------------------------------------------------- 209 data
FNAME209 = re.compile(
    r"^(?P<model>.+?)_(?P<prompt>base|careful|informed)_"
    r"(?P<mit>none|idem_keys|runtime_receipts)_(?P<fm>.+?)__s(?P<s>\d+)__(?P<task>.+)\.json$")


def cell209_tags(r):
    tags = []
    for fld, default in (("retry_mode", "agent_retry"), ("task_set", "base"),
                         ("schema", "lean")):
        v = r.get(fld)
        if v and v != default and not (fld == "task_set" and v == "core"):
            tags.append(str(v))
    return tags


def load_209():
    """Return {(batch, model, prompt, mit, fm, tags): {episode_id: row}}.
    Dedupe rule (fixed in src/sweep.py, e78e23d, before the two runs' scores
    were compared): keep the LAST row per episode_id, because its transcript
    is the one on disk. Every row is also kept in DUPS, so that the first-run
    values can be reported next to the reported ones (Appendix)."""
    cells = defaultdict(dict)
    dup_conflicts = 0
    for path in raw_files("sweep_raw_v2.jsonl", "sweep_raw*.jsonl"):
        b = batch_of(path)
        seen = {}
        allrows = defaultdict(list)
        for r in read_jsonl(path):
            eid = r["episode_id"]
            allrows[eid].append(r)
            if eid in seen and seen[eid]["violations"] != r["violations"]:
                dup_conflicts += 1
            seen[eid] = r            # last row wins (= transcript on disk)
        d = {e: rs for e, rs in allrows.items() if len(rs) > 1}
        if d:
            DUPS.setdefault(b, {}).update(d)
            put(f"dup:{b}:rows", f"{sum(len(rs) for rs in allrows.values()):,}")
            put(f"dup:{b}:unique", f"{len(allrows):,}")
            put(f"dup:{b}:ids", len(d))
            put(f"dup:{b}:maxcopies", max(len(rs) for rs in d.values()))
        for eid, r in seen.items():
            k = (b, r["model"], r["prompt"], r["mitigation"], r["failure_mode"],
                 tuple(cell209_tags(r)))
            cells[k][eid] = r
        note(f"209 raw {os.path.relpath(path, ROOT)}: batch={b}, "
             f"{len(seen)} unique episodes")
    put("hygiene:dup_conflicts", dup_conflicts)
    if dup_conflicts:
        note(f"WARNING 209: {dup_conflicts} duplicated episode ids with "
             f"different scores; the last row was kept (see e78e23d)")
    return cells


def transcripts_209():
    """{(batch, model, prompt, mit, fm, task, sample): transcript dict}"""
    out = {}
    dirs = [os.path.join(RES, "sweep_transcripts")] + [
        d for d in glob.glob(os.path.join(RES, "**", "sweep_transcripts"), recursive=True)
        if "smoke" not in d and os.path.dirname(d) != RES] + \
        glob.glob(os.path.join(RES, "**", "E1-*_transcripts"), recursive=True) + \
        glob.glob(os.path.join(RES, "tier1", "e1_*", "transcripts"))
    for d in dirs:
        b = batch_of(d)
        for f in glob.glob(os.path.join(d, "*.json")):
            m = FNAME209.match(os.path.basename(f))
            if not m:
                continue
            t = json.load(open(f))
            t["_file"] = os.path.basename(f)
            out[(b, t.get("model", m["model"]), m["prompt"], m["mit"], m["fm"],
                 m["task"], int(m["s"]))] = t
    return out


KEYTEXT = re.compile(r"idempoten", re.I)


def retry_text(t):
    """Visible assistant text between the first effectful call and its retry
    (the first later effectful call with the same identity). Returns None if
    there is no retry, else (text, sentence naming the key or '')."""
    seq = []
    for turn in t.get("turns") or []:
        if turn.get("role") != "assistant":
            continue
        for b in turn["content"]:
            if b.get("type") == "tool_use":
                seq.append(("call", b))
            elif b.get("type") == "text":
                seq.append(("text", b.get("text", "")))
    eff = [i for i, (k, b) in enumerate(seq) if k == "call" and b["name"] in CANON]
    if len(eff) < 2:
        return None
    b0 = seq[eff[0]][1]
    try:
        s0 = sid(b0["name"], {k: v for k, v in b0["input"].items() if k != "idempotency_key"})
    except Exception:
        return None
    i1 = None
    for i in eff[1:]:
        b = seq[i][1]
        try:
            if sid(b["name"], {k: v for k, v in b["input"].items()
                               if k != "idempotency_key"}) == s0:
                i1 = i
                break
        except Exception:
            pass
    if i1 is None:
        return None
    txt = " ".join(x for k, x in seq[eff[0] + 1:i1] if k == "text").strip()
    sent = ""
    for s in re.split(r"(?<=[.!?:])\s+", txt.replace("\n", " ")):
        if KEYTEXT.search(s):
            sent = s.strip()
            break
    k0 = b0["input"].get("idempotency_key")
    k1 = seq[i1][1]["input"].get("idempotency_key")
    return txt, sent, bool(k0) and k0 == k1


def eff_calls(t):
    """Ordered list of (index, name, args_without_key, key) of all tool calls."""
    turns = t.get("turns")
    if turns is None and "deliveries" in t:
        turns = [x for dlv in t["deliveries"] for x in dlv.get("turns", [])]
    calls = []
    for turn in turns or []:
        if turn.get("role") != "assistant":
            continue
        for b in turn["content"]:
            if b.get("type") == "tool_use":
                a = dict(b.get("input") or {})
                k = a.pop("idempotency_key", "") or ""
                calls.append((b["name"], a, k if isinstance(k, str) else json.dumps(k)))
    return calls


def episode_features(t):
    """read-back after the first effectful call (before the next effectful
    call), key on first call, retry category."""
    calls = eff_calls(t)
    eff = [i for i, c in enumerate(calls) if c[0] in CANON]
    if not eff:
        return None
    texts = [b.get("text", "") for turn in (t.get("turns") or [])
             if turn.get("role") == "assistant" for b in turn["content"]
             if b.get("type") == "text"]
    said_done = bool(texts) and "DONE" in texts[-1]
    i0 = eff[0]
    name, a0, k0 = calls[i0]
    try:
        s0 = sid(name, a0)
    except Exception:
        return None
    later = [i for i in eff if i > i0]
    nxt = later[0] if later else len(calls)
    readback = any(calls[j][0] == "list_effects" for j in range(i0 + 1, nxt))
    retry = None
    rb_before_retry = False
    for i in later:
        try:
            if sid(calls[i][0], calls[i][1]) == s0:
                retry = calls[i]
                rb_before_retry = any(calls[j][0] == "list_effects" for j in range(i0 + 1, i))
                break
        except Exception:
            pass
    cat = ""
    if retry:
        k1 = retry[2]
        cat = ("same_key" if k0 == k1 else "new_key") if (k0 and k1) else \
              "dropped_key" if k0 else "added_key" if k1 else "keyless"
    return dict(readback=readback, key_first=bool(k0), retried=retry is not None,
                cat=cat, rbretry=readback and retry is not None,
                rb_before_retry=rb_before_retry,
                gaveup=retry is None and not readback and not said_done)


def emit_209(cells, trans):
    seen_unq = defaultdict(list)
    haiku_first_keys = [0, 0]
    for (b, model, prompt, mit, fm, tags), eps in sorted(cells.items()):
        ms = short_model(model)
        parts = [ms, prompt, MIT_SHORT.get(mit, mit), FM_SHORT.get(fm, fm)] + list(tags)
        base = ":".join(parts)
        qual = b + ":" + base
        seen_unq[base].append(qual)
        rows = list(eps.values())
        n = len(rows)
        exp = EXPECTED_209_BATCH.get(b, EXPECTED_209)
        if n < min_n(exp):
            for metric in ("ivr", "ci", "tci", "n", "k", "succ"):
                pending(f"{qual}:{metric}", f"{n}/{exp}")
            note(f"PENDING 209 {qual}: {n}/{exp} episodes so far")
            continue
        k = sum(1 for r in rows if r["violations"] > 0)
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_id"]].append(1 if r["violations"] > 0 else 0)
        T = len(by_task)
        p = k / n
        lo, hi = wilson(k, n)
        tlo, thi = wilson(p * T, T)      # design-effect 2 (rho = 1) interval on IVR
        blo, bhi = boot_ci([(sum(v), len(v)) for v in by_task.values()], qual)
        tany = sum(1 for v in by_task.values() if any(v))
        alo, ahi = wilson(tany, T)
        succ = sum(1 for r in rows if r["achieved"]) / n
        mis = sum(1 for r in rows if r.get("misorder"))
        depth = sum(r["violations"] for r in rows) / n
        put(f"{qual}:n", n); put(f"{qual}:T", T); put(f"{qual}:k", k)
        put(f"{qual}:kn", f"{k}/{n}")
        put(f"{qual}:ivr", f2(p)); put(f"{qual}:ci", ci_str(lo, hi))
        put(f"{qual}:tci", ci_str(tlo, thi))
        # a bootstrap over all-zero (or all-one) clusters is degenerate
        put(f"{qual}:boot", "--" if k in (0, n) else ci_str(blo, bhi))
        put(f"{qual}:tvr", f2(tany / T)); put(f"{qual}:tvrci", ci_str(alo, ahi))
        put(f"{qual}:succ", f2(succ)); put(f"{qual}:mis", mis)
        put(f"{qual}:depth", f2(depth))
        put(f"{qual}:depth_eq_ivr", "yes" if abs(depth - p) < 1e-9 else "no")
        put(f"{qual}:trunc", sum(1 for r in rows if r.get("truncated")))
        put(f"{qual}:invalid", sum(int(r.get("invalid_calls") or 0) for r in rows))
        fam = Counter(r["family"] for r in rows if r["violations"] > 0)
        for fm_name in ("atomic", "multi_step", "read_then_act", "ambiguous"):
            put(f"{qual}:fam:{fm_name}", fam.get(fm_name, 0))
        put(f"{qual}:fams_violating", len(fam))
        # transcript features
        feats = []
        mism = 0
        epi = []
        rtxt = [0, 0, 0]      # retries, with visible text before, naming the key
        rkey_ex = None
        gaveup_succ = [0, 0]  # gave-up episodes that still count as task success
        for r in rows:
            t = trans.get((b, model, prompt, mit, fm, r["task_id"], r["sample"]))
            v = 1 if r["violations"] > 0 else 0
            if t is None:
                epi.append((r["task_id"], v, None))
                continue
            if b != "jul" and t.get("score", {}).get("violations") != r["violations"] \
                    and r["episode_id"] not in C1_RESCORED:   # transcript keeps its pre-fix score
                mism += 1
            fe = episode_features(t)
            epi.append((r["task_id"], v, (1 if fe["readback"] else 0) if fe else None))
            if ms == "haiku" and b == "jul":
                haiku_first_keys[1] += 1
                haiku_first_keys[0] += bool(fe and fe["key_first"])
            if fe:
                feats.append(fe)
                if fe["gaveup"]:
                    gaveup_succ[1] += 1
                    gaveup_succ[0] += bool(r["achieved"])
            rt = retry_text(t)
            if rt is not None:
                rtxt[0] += 1
                rtxt[1] += bool(rt[0])
                rtxt[2] += bool(rt[1])
                # example: a duplicating retry that reused the key and whose
                # preceding text names the key (sorted for determinism below)
                if rt[1] and rt[2] and v:
                    cand = (t["_file"], rt[1])
                    if rkey_ex is None or cand[0] < rkey_ex[0]:
                        rkey_ex = cand
        EPI[qual] = epi
        put(f"{qual}:rtext:retries", rtxt[0])
        put(f"{qual}:rtext:any", rtxt[1])
        put(f"{qual}:rtext:key", rtxt[2])
        put(f"{qual}:rtext:none", rtxt[0] - rtxt[1])
        put(f"{qual}:gaveup_succ", f"{gaveup_succ[0]}/{gaveup_succ[1]}")
        if rkey_ex:
            put(f"{qual}:rtext:ex_file", tex_path(rkey_ex[0]))
            put(f"{qual}:rtext:ex", tex_text(rkey_ex[1]))
        if mism:
            note(f"WARNING {qual}: {mism} transcripts disagree with the kept raw row")
        # E1-5 injects the failure on call k > 1, but episode_features assumes
        # the first effectful call failed; its read-back for E1-5 comes from
        # analysis_v3 (readback_after_failure, logged by the runner) instead.
        if len(feats) == n and b != "e15":
            rb = sum(f["readback"] for f in feats)
            put(f"{qual}:rb", f"{rb}/{n}"); put(f"{qual}:rbrate", f2(rb / n))
            kf = sum(f["key_first"] for f in feats)
            put(f"{qual}:keyfirst", f"{kf}/{n}")
            ret = [f for f in feats if f["retried"]]
            put(f"{qual}:retried", len(ret))
            put(f"{qual}:rbretry", sum(f["rbretry"] for f in feats))
            put(f"{qual}:norb_retry", sum(1 for f in ret if not f["readback"]))
            # retried even though list_effects had been called before the retry
            put(f"{qual}:rb_before_retry", sum(1 for f in ret if f["rb_before_retry"]))
            # neither retried nor checked, and did not report DONE (reported failure)
            put(f"{qual}:gaveup", sum(1 for f in feats if f["gaveup"]))
            for c in ("same_key", "new_key", "dropped_key", "added_key", "keyless"):
                put(f"{qual}:{c}", sum(1 for f in ret if f["cat"] == c))
            put(f"{qual}:reuse", f2(sum(1 for f in ret if f["cat"] == "same_key") / len(ret))
                if ret else "--")
        else:
            note(f"{qual}: transcript features for {len(feats)}/{n} episodes; not emitted")
    # unqualified aliases when exactly one batch has the cell
    for base, quals in seen_unq.items():
        if len(quals) == 1:
            q = quals[0]
            for key in [x for x in list(NUMS) if x.startswith(q + ":")]:
                NUMS[base + key[len(q):]] = NUMS[key]
        else:
            note(f"209 cell {base} exists in several batches {quals}: use a "
                 f"batch-qualified key")
    if haiku_first_keys[1]:
        put("haiku:keyfirst:all", f"{haiku_first_keys[0]}/{haiku_first_keys[1]}")


# ---------------------------------------------------------------- 249 data
def load_249():
    cells = defaultdict(dict)
    for path in raw_files("sweep2_raw_v2.jsonl", "sweep2_raw*.jsonl"):
        b = batch_of(path)
        seen = {}
        for r in read_jsonl(path):
            seen[r["episode_id"]] = r
        for eid, r in seen.items():
            cells[(b, r["model"], r["resume_context"], r["mitigation"])][eid] = r
        note(f"249 raw {os.path.relpath(path, ROOT)}: batch={b}, {len(seen)} unique episodes")
    return cells


def transcripts_249():
    out = {}
    dirs = [os.path.join(RES, "transcripts2")] + [
        d for d in glob.glob(os.path.join(RES, "**", "transcripts2"), recursive=True)
        if "smoke" not in d and os.path.dirname(d) != RES] + \
        glob.glob(os.path.join(RES, "**", "E2-*_transcripts"), recursive=True)
    for d in dirs:
        b = batch_of(d)
        for f in glob.glob(os.path.join(d, "*.json")):
            t = json.load(open(f))
            out[(b, t["model"], t["resume_context"], t["mitigation"], t["graph_id"],
                 t["interrupt_node"], t["sample"])] = t
    return out


def lookups(rows, b, trans, with_dup=False):
    """Session-2 effect-log lookups. with_dup: also count episodes that
    looked up the log and still duplicated."""
    resumed = looked = looked_dup = 0
    for r in rows:
        t = trans.get((b, r["model"], r["resume_context"], r["mitigation"],
                       r["graph_id"], r["interrupt_node"], r["sample"]))
        if t is None or len(t["sessions"]) < 2:
            continue
        resumed += 1
        names = [x["name"] for turn in t["sessions"][1]["turns"]
                 if turn.get("role") == "assistant"
                 for x in turn["content"] if x.get("type") == "tool_use"]
        if "list_effects" in names:
            looked += 1
            looked_dup += r["givr_hit"] > 0
    return (looked, resumed, looked_dup) if with_dup else (looked, resumed)


def emit_249_cell(qual, rows, b, trans, expected):
    n = len(rows)
    if n < min_n(expected):
        for metric in ("givr", "ci", "gci", "n", "kn", "chk"):
            pending(f"{qual}:{metric}", f"{n}/{expected}")
        note(f"PENDING 249 {qual}: {n}/{expected} episodes so far")
        return
    k = sum(r["givr_hit"] for r in rows)
    G = len({r["graph_id"] for r in rows})
    p = k / n
    lo, hi = wilson(k, n)
    glo, ghi = wilson(p * G, G)
    EPI249[qual] = {(r["graph_id"], r["interrupt_node"], r["sample"]): r["givr_hit"]
                    for r in rows}
    st = sum(1 for r in rows if r.get("order_violation"))
    comp = sum(r["completion"] for r in rows)
    viol = [r["violations"] for r in rows]
    put(f"{qual}:n", n); put(f"{qual}:G", G); put(f"{qual}:k", k)
    put(f"{qual}:kn", f"{k}/{n}"); put(f"{qual}:givr", f2(p))
    put(f"{qual}:ci", ci_str(lo, hi)); put(f"{qual}:gci", ci_str(glo, ghi))
    put(f"{qual}:stale", f2(st / n)); put(f"{qual}:stalekn", f"{st}/{n}")
    # stale episodes that are not also duplicating episodes
    put(f"{qual}:stale_nodup", sum(1 for r in rows if r.get("order_violation")
                                   and not r["givr_hit"]))
    # stale by interruption point: after node 0 no dependent has committed yet
    pos = [r for r in rows if r["interrupt_node"] > 0]
    put(f"{qual}:stale_k0", sum(1 for r in rows if r["interrupt_node"] == 0
                                and r.get("order_violation")))
    put(f"{qual}:stale_kpos", f"{sum(1 for r in pos if r.get('order_violation'))}/{len(pos)}")
    put(f"{qual}:comp", f2(comp / n)); put(f"{qual}:compkn", f"{comp}/{n}")
    put(f"{qual}:meanviol", f"{sum(viol) / n:.1f}"); put(f"{qual}:maxviol", max(viol))
    put(f"{qual}:trunc", sum(1 for r in rows if r.get("truncated")))
    put(f"{qual}:leaked", sum(1 for r in rows if r.get("notes_leaked")))
    lk, rs, lkd = lookups(rows, b, trans, with_dup=True)
    put(f"{qual}:chk", f"{lk}/{rs}")
    put(f"{qual}:resumed", rs)
    put(f"{qual}:chkdup", f"{lkd}/{k}")    # duplicating episodes that had looked
    put(f"{qual}:notresumed", n - rs)
    put(f"{qual}:dupG", len({r["graph_id"] for r in rows if r["givr_hit"]}))
    # upstream providers that served the duplicating episodes (OpenRouter)
    ups = set()
    for r in rows:
        if not r["givr_hit"]:
            continue
        t = trans.get((b, r["model"], r["resume_context"], r["mitigation"],
                       r["graph_id"], r["interrupt_node"], r["sample"]))
        for ses in (t or {}).get("sessions", []):
            for turn in ses["turns"]:
                up = (turn.get("provider_meta") or {}).get("upstream_provider")
                if up:
                    ups.add(up)
    if ups:
        put(f"{qual}:dup_upstream", ", ".join(sorted(ups)))
        put(f"{qual}:dup_upstream_n", len(ups))


def emit_249(cells, trans):
    seen_unq = defaultdict(list)
    for (b, model, ctx, mit), eps in sorted(cells.items()):
        rows = list(eps.values())
        # chain4 + fanout form the benchmark set; any other topology (diamond,
        # chain8 from Tier 2) is reported as its own cell
        groups = defaultdict(list)
        for r in rows:
            groups["bench" if r["topology"] in BENCH_TOPOS else r["topology"]].append(r)
        for tag, rs in groups.items():
            base = ":".join([short_model(model), ctx, MIT_SHORT.get(mit, mit)]
                            + ([] if tag == "bench" else [tag]))
            qual = b + ":" + base
            seen_unq[base].append(qual)
            expected = EXPECTED_249_OVERRIDE.get((b, ctx), EXPECTED_249)
            if tag != "bench":
                expected = EXPECTED_249
            emit_249_cell(qual, rs, b, trans, expected)
    for base, quals in seen_unq.items():
        if len(quals) == 1:
            q = quals[0]
            for key in [x for x in list(NUMS) if x.startswith(q + ":")]:
                NUMS[base + key[len(q):]] = NUMS[key]
        else:
            note(f"249 cell {base} exists in several batches {quals}: use a "
                 f"batch-qualified key")

    # derived: plan_only corrected = July chain4 (no ack leak possible: the
    # leak needs several calls in one turn at the seam, seen only in fan-out)
    # + October fan-out reruns under the fixed acknowledgement code.
    for (b, model, ctx, mit), eps in cells.items():
        if b != "oct" or ctx != "plan_only" or mit != "none":
            continue
        jul = cells.get(("jul", model, "plan_only", "none"), {})
        ch = [r for r in jul.values() if r["topology"] == "chain4"]
        leaked_ch = sum(1 for r in ch if r.get("notes_leaked"))
        fan = [r for r in eps.values() if r["topology"] == "fanout"]
        ms = short_model(model)
        jf = {(r["graph_id"], r["interrupt_node"], r["sample"]) for r in jul.values()
              if r["topology"] == "fanout"}
        of = {(r["graph_id"], r["interrupt_node"], r["sample"]) for r in fan}
        if jf != of:
            note(f"WARNING {ms}: October fan-out rerun episodes differ from the July set")
        put(f"pocorr:{ms}:same_episodes", "yes" if jf == of else "no")
        if leaked_ch:
            note(f"WARNING {ms}: July chain4 plan_only has {leaked_ch} leaked notes")
        if len(ch) + len(fan) < EXPECTED_249:
            pending(f"pocorr:{ms}:plan_only:none:givr", f"{len(ch) + len(fan)}/32")
            continue
        rows = ch + fan
        k = sum(r["givr_hit"] for r in rows)
        n = len(rows); G = len({r["graph_id"] for r in rows})
        p = k / n
        lo, hi = wilson(k, n); glo, ghi = wilson(p * G, G)
        lk1, rs1, d1 = lookups(ch, "jul", trans, with_dup=True)
        lk2, rs2, d2 = lookups(fan, "oct", trans, with_dup=True)
        q = f"pocorr:{ms}:plan_only:none"
        put(f"{q}:chkdup", f"{d1 + d2}/{k}")
        put(f"{q}:n", n); put(f"{q}:G", G); put(f"{q}:kn", f"{k}/{n}")
        put(f"{q}:givr", f2(p)); put(f"{q}:ci", ci_str(lo, hi)); put(f"{q}:gci", ci_str(glo, ghi))
        put(f"{q}:stale", f2(sum(1 for r in rows if r.get("order_violation")) / n))
        put(f"{q}:comp", f2(sum(r["completion"] for r in rows) / n))
        put(f"{q}:chk", f"{lk1 + lk2}/{rs1 + rs2}")
        put(f"{q}:chain_kn", f"{sum(r['givr_hit'] for r in ch)}/{len(ch)}")
        put(f"{q}:fan_kn", f"{sum(r['givr_hit'] for r in fan)}/{len(fan)}")
        # the July clean subset, for transparency
        clean = [r for r in jul.values() if not r.get("notes_leaked")]
        leak = [r for r in jul.values() if r.get("notes_leaked")]
        cf = [r for r in clean if r["topology"] == "fanout"]
        put(f"pojulfanclean:{ms}:kn", f"{sum(r['givr_hit'] for r in cf)}/{len(cf)}")
        if clean:
            kc = sum(r["givr_hit"] for r in clean)
            put(f"poclean:{ms}:kn", f"{kc}/{len(clean)}")
            put(f"poclean:{ms}:givr", f2(kc / len(clean)))
            put(f"poclean:{ms}:ci", ci_str(*wilson(kc, len(clean))))
            put(f"poleak:{ms}:kn", f"{sum(r['givr_hit'] for r in leak)}/{len(leak)}")
            put(f"pojul:{ms}:kn", f"{sum(r['givr_hit'] for r in jul.values())}/{len(jul)}")

    # what the models actually ran (all core batches), vs what the harness has
    cr = [r for (b, *_), eps in cells.items() if b in CORE for r in eps.values()]
    put("p2:model_topologies", len({r["topology"] for r in cr}))
    put("p2:model_contexts", len({r["resume_context"] for r in cr}))
    # the model graph set (July)
    jr = [r for (b, *_), eps in cells.items() if b == "jul" for r in eps.values()]
    if jr:
        gids = {r["graph_id"] for r in jr}
        put("p2:graphs_used", len(gids))
        for tp in sorted({r["topology"] for r in jr}):
            put(f"p2:graphs:{tp}", len({r["graph_id"] for r in jr if r["topology"] == tp}))
        nn = {r["n_nodes"] for r in jr}
        put("p2:nodes", "--".join(str(x) for x in sorted(nn)))
        put("p2:interrupt_nodes", len({r["interrupt_node"] for r in jr}))
        put("p2:k_max", max(int(r["interrupt_node"]) for r in jr))   # k = 0..k_max
        put("p2:timing_after_only", "yes" if {r.get("interrupt_timing") for r in jr} == {"after_commit"} else "no")
        put("p2:jul_episodes", len(jr))

    # pooled bare_hint over the three Claude models (October batch)
    pool = [r for (b, model, ctx, mit), eps in cells.items()
            if b == "oct" and ctx == "bare_hint" and mit == "none"
            and short_model(model) in ("haiku", "sonnet", "fable")
            for r in eps.values()]
    if len(pool) >= 3 * EXPECTED_249:
        k = sum(r["givr_hit"] for r in pool)
        G = len({(r["model"], r["graph_id"]) for r in pool})
        put("pool:bare_hint:claude:kn", f"{k}/{len(pool)}")
        put("pool:bare_hint:claude:G", G)
        put("pool:bare_hint:claude:gci", ci_str(*wilson(k / len(pool) * G, G)))
    else:
        pending("pool:bare_hint:claude:kn", f"{len(pool)}/{3 * EXPECTED_249}")

    # range of amnesiac gIVR over models with a complete cell
    am = []
    for (b, model, ctx, mit), eps in cells.items():
        if ctx == "amnesiac" and mit == "none" and len(eps) >= min_n(EXPECTED_249):
            rs = [r for r in eps.values() if r["topology"] in BENCH_TOPOS]
            am.append((short_model(model), sum(r["givr_hit"] for r in rs) / len(rs)))
    if am:
        put("range:amnesiac:min", f2(min(x[1] for x in am)))
        put("range:amnesiac:max", f2(max(x[1] for x in am)))
        put("range:amnesiac:nmodels", len(am))
        put("range:amnesiac:n_at_1", sum(1 for x in am if x[1] == 1.0))

    # bare_hint: per model the largest complete cell (October/xp over July)
    best = {}
    for (b, model, ctx, mit), eps in cells.items():
        if ctx != "bare_hint" or mit != "none":
            continue
        rs = [r for r in eps.values() if r["topology"] in BENCH_TOPOS]
        exp = EXPECTED_249_OVERRIDE.get((b, ctx), EXPECTED_249)
        if len(rs) >= exp and len(rs) > len(best.get(short_model(model), [])):
            best[short_model(model)] = rs
    if best:
        vals = {m: sum(r["givr_hit"] for r in rs) / len(rs) for m, rs in best.items()}
        put("range:bare_hint:min", f2(min(vals.values())))
        put("range:bare_hint:max", f2(max(vals.values())))
        put("range:bare_hint:nmodels", len(vals))
        put("range:bare_hint:n_at_0", sum(1 for v in vals.values() if v == 0))
        allr = [r for rs in best.values() for r in rs]
        put("range:bare_hint:episodes", len(allr))

    # completion in cells that produced duplicates (complete cells only)
    dupc = []
    for (b, model, ctx, mit), eps in cells.items():
        rs = list(eps.values())
        exp = EXPECTED_249_OVERRIDE.get((b, ctx), EXPECTED_249)
        if len(rs) >= exp and any(r["givr_hit"] for r in rs):
            dupc.append(sum(r["completion"] for r in rs) / len(rs))
    if dupc:
        put("all:comp_min_duplicating", f2(min(dupc)))

    # stale co-occurrence with duplication, per batch and over all batches
    # unit: episodes with at least one stale re-execution (order_violation);
    # complete cells only, so a partial run cannot change the count
    per = defaultdict(lambda: [0, 0, 0])   # stale episodes, of them duplicating, episodes
    for (b, model, ctx, mit), eps in cells.items():
        if len(eps) < EXPECTED_249_OVERRIDE.get((b, ctx), EXPECTED_249):
            continue
        for r in eps.values():
            per[b][2] += 1
            per["all"][2] += 1
            if r.get("order_violation"):
                for x in (b, "all"):
                    per[x][0] += 1
                    per[x][1] += r["givr_hit"] > 0
    for b, (ev, co, n) in per.items():
        put(f"stale:{b}:events", ev); put(f"stale:{b}:cooccur", co)
        put(f"stale:{b}:episodes", f"{n:,}")
        put(f"stale:{b}:without_dup", ev - co)


# ---------------------------------------------------------------- scripted
def emit_scripted():
    p = os.path.join(RES, "summary_by_failure.csv")
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        k = ":".join(["scr1", r["agent"], r["retry_mode"], MIT_SHORT[r["mitigation"]],
                      FM_SHORT[r["failure_mode"]]])
        put(k + ":ivr", f2(float(r["ivr"])))
        put(k + ":succ", f2(float(r["success"])))
    put("scr1:ntasks", rows[0]["n_tasks"])
    tb = [r for r in rows if r["failure_mode"] == "timeout_before_commit"]
    put("scr1:toB:cells", len(tb))
    put("scr1:toB:allzero", "yes" if all(float(r["ivr"]) == 0 for r in tb) else "no")
    rc = [r for r in rows if r["mitigation"] == "runtime_receipts"]
    put("scr1:rcpt:allzero", "yes" if all(float(r["ivr"]) == 0 for r in rc) else "no")

    g = list(csv.DictReader(open(os.path.join(RES, "graph_summary.csv"))))
    put("scr2:episodes", f"{sum(int(r['n_episodes']) for r in g):,}")
    for r in g:
        k = ":".join(["scr2", r["resume_context"], MIT_SHORT[r["mitigation"]],
                      r["key_posture"], r["interrupt_timing"]])
        put(k + ":givr", f2(float(r["givr"])))
        put(k + ":stale", f2(float(r["order_violation_rate"])))
        put(k + ":meanviol", f"{float(r['mean_violations']):.2f}")
        put(k + ":n", r["n_episodes"])
        put(k + ":comp", f2(float(r["completion"])))
    # pooled over timings, as the old 249 table did (to report the correction)
    for ctx in ("amnesiac", "plan_only"):
        for post in ("naive", "stable"):
            sel = [r for r in g if r["resume_context"] == ctx and r["mitigation"] == "none"
                   and r["key_posture"] == post]
            n = sum(int(r["n_episodes"]) for r in sel)
            gv = sum(float(r["givr"]) * int(r["n_episodes"]) for r in sel) / n
            st = sum(float(r["order_violation_rate"]) * int(r["n_episodes"]) for r in sel) / n
            put(f"scr2pool:{ctx}:{post}:givr", f2(gv))
            put(f"scr2pool:{ctx}:{post}:stale", f2(st))
            put(f"scr2pool:{ctx}:{post}:n", f"{n:,}")


# ---------------------------------------------------------------- tier 0
def emit_tier0():
    a = os.path.join(RES, "analysis_v2")
    s3 = json.load(open(os.path.join(a, "e0_3_summary.json")))
    put("match:pairs", s3["pairs"]); put("match:raw", s3["raw_identical"])
    put("match:normonly", s3["normalization_only"]); put("match:near", s3["near_miss"])
    put("match:distinct", s3["distinct"]); put("match:effcalls", f"{s3['effectful_calls']:,}")
    put("match:noncanon", s3["calls_with_non_canonical_fields"])
    put("match:normtasks", s3["normalization_only_distinct_tasks"])
    ch = s3["strict_matcher_ivr_changes"].get("claude-haiku-4-5-20251001|base/none/error_500_after_commit")
    if ch:
        put("match:strict:haiku500:ivr", f2(ch["ivr_strict"]))
        put("match:strict:haiku500:kn", f"{ch['viol_strict']}/{ch['n']}")
        put("match:paper:haiku500:kn", f"{ch['viol_paper']}/{ch['n']}")
    put("match:strict:cells_changed", len(s3["strict_matcher_ivr_changes"]))
    hc = os.path.join(a, "e0_3_handcheck.csv")
    verdicts = [r for r in csv.DictReader(open(hc))
                if (r.get("your_verdict_same_action_Y_N") or "").strip()]
    if verdicts:
        fp = sum(1 for r in verdicts if r["your_verdict_same_action_Y_N"].strip().upper().startswith("N"))
        put("match:hand:n", len(verdicts)); put("match:hand:fp", fp)
        same = lambda rs: sum(1 for r in rs if r["your_verdict_same_action_Y_N"].strip().upper().startswith("Y"))
        nrows = [r for r in verdicts if r["auto_class"] == "normalization_only"]
        ctrl = [r for r in verdicts if r["auto_class"] != "normalization_only"]
        put("match:hand:norm:n", len(nrows)); put("match:hand:norm:same", same(nrows))
        # "these pairs" in the paper refers to the normalization-only pairs; the controls get their own clause
        put("match:hand:clause", f"our hand check judged {same(nrows)} of these {len(nrows)} pairs "
            "to be the same action")
        put("match:hand:controls:clause", f", of which it judged {same(ctrl)} the same action")
    else:
        note("matcher hand-check: e0_3_handcheck.csv has no author verdicts (rule-based result reported)")
        put("match:hand:clause", "the pairs are listed in the artifact "
            r"(\texttt{e0\_3\_\allowbreak handcheck.csv}) for audit")
        put("match:hand:controls:clause", "")
    # the worked example of the normalization rule: first normalization-only pair
    for r in csv.DictReader(open(hc)):
        m = re.search(r'"(\w+)":\s*\["([^"]*)",\s*"([^"]*)"\]', r["reason"])
        if r["auto_class"] == "normalization_only" and m:
            put("match:example:field", tex_id(f"{r['tool']}.{m.group(1)}"))
            put("match:example:a", tex_id(m.group(2)))
            put("match:example:b", tex_id(m.group(3)))
            put("match:example:norm", tex_id(norm(m.group(2))))
            tp = os.path.join(RES, "sweep_transcripts", r["file"])
            if os.path.exists(tp):
                t = json.load(open(tp))
                calls = [c for c in eff_calls(t) if c[0] == r["tool"]]
                if calls:
                    for fld in ("payee", "memo"):
                        if fld in calls[0][1]:
                            put(f"match:example:{fld}", tex_id(calls[0][1][fld]))
                errs = [x.get("content", "") for turn in t["turns"]
                        if turn["role"] == "tool_results" for x in turn["content"]
                        if x.get("is_error")]
                if errs:
                    put("match:example:error", tex_id(errs[0]))
            break

    s4 = json.load(open(os.path.join(a, "e0_4_summary.json")))
    for r in s4["retry_mode"]:
        ms = short_model(r["model"])
        put(f"{ms}:wholerun", f"{r['whole_run_would_trigger']}/{r['n_episodes']}")
        put(f"{ms}:unprompted_rb", f"{r['base_readback_episodes']}/{r['base_episodes']}")
        put(f"{ms}:maxturns_used", r["max_assistant_turns"])
        put(f"{ms}:turncap_hits", r["episodes_at_turn_cap"])

    # counterfactual replays (labelled as such in the paper)
    for r in csv.DictReader(open(os.path.join(a, "e0_4_keys500_counterfactual.csv"))):
        ms = short_model(r["model"])
        lo, hi = int(r["ivr_idem_keys_replay_valid"]), int(r["ivr_idem_keys_upper"])
        put(f"cf:keys500:{ms}", f"{lo}/{r['n']}" if lo == hi else f"{lo}--{hi}/{r['n']}")
        put(f"cf:keys500:{ms}:none", f"{r['ivr_none_replay']}/{r['n']}")
    for r in csv.DictReader(open(os.path.join(a, "e0_4_alo_counterfactual.csv"))):
        ms = short_model(r["model"])
        put(f"cf:alo:{ms}:{MIT_SHORT[r['mitigation']]}", r["bounds"])
        put(f"cf:alo:{ms}:valid", r["valid_pairs"])

    for r in csv.DictReader(open(os.path.join(a, "e0_5_sampling_protocol.csv"))):
        ms = short_model(r["model"])
        q = f"samp:{r['paper']}:{ms}"
        put(q + ":ident", f"{r['identical_pairs_text']}/{r['sample_pairs']}")
        put(q + ":dates", r["run_dates_utc_from_mtime"].replace("2026-", "").replace(" .. ", "--"))
        put(q + ":id", tex_id(r["model"]))
        put(q + ":dated", "yes" if re.search(r"-\d{8}$", r["model"]) else "no")
        te = r["temperature_effective"]
        put(q + ":temp", "0" if te.startswith("0.0") else "default")
        put(q + ":temp_logged", "logged" if "(logged)" in te else "inferred")


def emit_misorder_turns(cells209, trans):
    """For each misordered episode: were both effectful calls issued in the
    same assistant turn?"""
    same = tot = 0
    for (b, model, prompt, mit, fm, tags), v in cells209.items():
        if b not in CORE or len(v) < min_n(EXPECTED_209):
            continue
        for r in v.values():
            if not r.get("misorder"):
                continue
            t = trans.get((b, model, prompt, mit, fm, r["task_id"], r["sample"]))
            if t is None:
                continue
            tot += 1
            for turn in t.get("turns", []):
                if turn.get("role") != "assistant":
                    continue
                names = [x["name"] for x in turn["content"] if x.get("type") == "tool_use"
                         and x["name"] in CANON]
                if len(set(names)) >= 2:
                    same += 1
                    break
    put("misorder:sameturn", f"{same} of {tot}")
    put("misorder:sameturn:all", "yes" if same == tot and tot else "no")
    # how often each model sends both steps of a two-step task in its first
    # effectful turn (complete jul/xp base-task cells, all conditions)
    par = defaultdict(lambda: [0, 0])
    for (b, model, prompt, mit, fm, tags), v in cells209.items():
        if b not in ("jul", "xp") or tags or len(v) < min_n(EXPECTED_209):
            continue
        for r in v.values():
            if r["family"] != "multi_step":
                continue
            t = trans.get((b, model, prompt, mit, fm, r["task_id"], r["sample"]))
            if t is None:
                continue
            for turn in t.get("turns", []):
                if turn.get("role") != "assistant":
                    continue
                names = {x["name"] for x in turn["content"]
                         if x.get("type") == "tool_use" and x["name"] in CANON}
                if names:
                    par[short_model(model)][1] += 1
                    par[short_model(model)][0] += len(names) >= 2
                    break
    for ms, (k, n) in par.items():
        put(f"par:{ms}:multistep", f"{k}/{n}")


def temp_label(values):
    """Effective temperature label from the transcript field values."""
    vs = set(values)
    if vs == {"0.0"} or vs == {"0"}:
        return "0"
    if vs == {"None"}:
        return "default"
    if vs == {"unverified"}:
        return "0 sent, unverified"
    if vs == {"absent"}:
        return None
    return "mixed"


def emit_ident_calls(trans209):
    """July single-session sample pairs whose tool calls (name, arguments and
    idempotency key, in order) are identical. The text-identical count
    (samp:209:<m>:ident, analysis_v2/e0_5) also requires the same assistant
    text and tool results, ignoring thinking blocks and random ids."""
    by = defaultdict(dict)
    for (b, model, prompt, mit, fm, task, s), t in trans209.items():
        if b != "jul":
            continue
        calls = [(c[0], json.dumps(c[1], sort_keys=True), c[2]) for c in eff_calls(t)]
        by[(short_model(model), prompt, mit, fm, task)][s] = calls
    n, k = Counter(), Counter()
    for key, d in by.items():
        if 0 in d and 1 in d:
            n[key[0]] += 1
            k[key[0]] += d[0] == d[1]
    for m in n:
        put(f"samp:209:{m}:ident_calls", f"{k[m]}/{n[m]}")


def emit_temps(trans209, trans249):
    """Effective temperature per (run, model) read from the transcripts.
    Overrides the Tier-0 CSV values for the July runs; where July session
    transcripts lack the field, the Tier-0 inferred value is kept and marked."""
    runs = defaultdict(list)
    for (b, model, *_), t in trans209.items():
        runs[("209" if b == "jul" else b, short_model(model))].append(str(t.get("temperature", "absent")))
    for (b, model, *_), t in trans249.items():
        runs[("249" if b == "jul" else b, short_model(model))].append(
            str(t["temperature"]) if "temperature" in t else "absent")
    for (run, ms), vals in runs.items():
        q = f"samp:{run}:{ms}"
        lab = temp_label(vals)
        put(q + ":temp_n", len(vals))
        if lab is None:          # field missing in every transcript
            put(q + ":temp_logged", "inferred")
            note(f"{q}: temperature not in transcripts; Tier-0 inferred value kept")
        else:
            put(q + ":temp", lab)
            put(q + ":temp_logged", "logged")
    # Haiku across runs: is it the same everywhere?
    hk = {run: NUMS.get(f"samp:{run}:haiku:temp") for run in ("209", "249", "oct")}
    put("haiku:temp:jul209", hk["209"] or "?"); put("haiku:temp:jul249", hk["249"] or "?")
    put("haiku:temp:oct", hk["oct"] or "?")
    put("haiku:temp:changed", "yes" if hk["oct"] and hk["209"] and hk["oct"] != hk["209"] else "no")
    # corrected plan_only cell mixes July chain and October fan-out episodes
    put("pocorr:haiku:temp_chain", hk["249"] or "?")
    put("pocorr:haiku:temp_fan", hk["oct"] or "?")


def emit_v1():
    """Values printed in the July submissions (v1, before the fixes), for the
    response letter only. Read from the untouched v1 CSVs."""
    p = os.path.join(RES, "sweep_summary.csv")
    for r in csv.DictReader(open(p)):
        q = ":".join(["v1", short_model(r["model"]), r["prompt"],
                      MIT_SHORT[r["mitigation"]], FM_SHORT[r["failure_mode"]]])
        put(q + ":ivr", f2(float(r["ivr"])))
        put(q + ":succ", f2(float(r["success"])))
    p = os.path.join(RES, "sweep2_summary.csv")
    for r in csv.DictReader(open(p)):
        q = ":".join(["v1", short_model(r["model"]), r["resume_context"],
                      MIT_SHORT[r["mitigation"]]])
        put(q + ":givr", f2(float(r["givr"]))); put(q + ":n", r["n"])
        put(q + ":ci", ci_str(float(r["givr_lo"]), float(r["givr_hi"])))


def emit_batch_sampling(trans209, trans249):
    """Per (batch, model) for the new batches: temperature actually logged,
    access dates, served ids, upstream providers, identical sample pairs."""
    info = defaultdict(lambda: dict(temps=Counter(), dates=set(), served=Counter(),
                                    upstream=Counter(), sigs=defaultdict(dict),
                                    refusals=0))
    def refused(turn_lists):
        return any(turn.get("stop_reason") == "refusal"
                   for turns in turn_lists for turn in turns)
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
    for (b, model, prompt, mit, fm, task, s), t in trans209.items():
        if b == "jul":
            continue
        d = info[(b, model)]
        d["temps"][str(t.get("temperature"))] += 1
        if t.get("started_utc"):
            d["dates"].add(t["started_utc"][:10])
        d["served"][str(t.get("model_served"))] += 1
        for turn in t.get("turns", []):
            up = (turn.get("provider_meta") or {}).get("upstream_provider")
            if up:
                d["upstream"][up] += 1
        d["sigs"][("p1", prompt, mit, fm, task)][s] = sig([t.get("turns", [])])
        d["refusals"] += refused([t.get("turns", [])])
    for (b, model, ctx, mit, gid, k, s), t in trans249.items():
        if b == "jul":
            continue
        d = info[(b, model)]
        d["temps"][str(t.get("temperature"))] += 1
        if t.get("started_utc"):
            d["dates"].add(t["started_utc"][:10])
        d["served"][str(t.get("model_served"))] += 1
        for ses in t["sessions"]:
            for turn in ses["turns"]:
                up = (turn.get("provider_meta") or {}).get("upstream_provider")
                if up:
                    d["upstream"][up] += 1
        d["sigs"][("p2", ctx, mit, gid, k)][s] = sig([x["turns"] for x in t["sessions"]])
        d["refusals"] += refused([x["turns"] for x in t["sessions"]])
    for (b, model), d in sorted(info.items()):
        ms = short_model(model)
        q = f"samp:{b}:{ms}"
        put(q + ":id", tex_id(model.replace("openrouter:", "")))
        put(q + ":dated", "yes" if re.search(r"-\d{8}$", model) else "no")
        temps = d["temps"]
        if set(temps) == {"0.0"}:
            put(q + ":temp", "0")
        elif set(temps) == {"None"}:
            put(q + ":temp", "default")
        elif set(temps) == {"unverified"}:
            put(q + ":temp", "0 sent, unverified")
        else:
            put(q + ":temp", "mixed")
        put(q + ":temp_logged", "logged")
        dates = sorted(d["dates"])
        if dates:
            put(q + ":dates", dates[0][5:] if dates[0] == dates[-1]
                else f"{dates[0][5:]}--{dates[-1][5:]}")
        pairs = [v for v in d["sigs"].values() if 0 in v and 1 in v]
        put(q + ":ident", f"{sum(1 for v in pairs if v[0] == v[1])}/{len(pairs)}")
        put(q + ":upstream_n", len(d["upstream"]))
        # served model id as recorded by the API (provider suffix removed)
        sv = Counter()
        for s, c in d["served"].items():
            sv[s.split(" @")[0] if s != "None" else "not recorded"] += c
        put(q + ":served", tex_id(", ".join(sorted(sv))))
        put(q + ":served_dated", "yes" if all(re.search(r"-\d{8}$", s) for s in sv) else "no")
        up_ep = Counter()
        for s, c in d["served"].items():
            if " @" in s:
                up_ep[s.split(" @", 1)[1]] += c
        if up_ep:
            put(q + ":served_counts", ", ".join(f"{u} {c}" for u, c in
                                                sorted(up_ep.items(), key=lambda x: (-x[1], x[0]))))
        put(q + ":refusals", d["refusals"])
        put(q + ":episodes", sum(d["temps"].values()))
        if d["upstream"]:
            put(q + ":upstream", ", ".join(sorted(d["upstream"])))
        note(f"sampling {q}: temps={dict(temps)} served={dict(d['served'])} "
             f"upstream={dict(d['upstream'])}")


# ---------------------------------------------------------------- summaries
def write_full209(cells209):
    """Appendix table rows: every complete single-session cell, paper order."""
    order = {k: i for i, (_, k, _) in enumerate(MODEL_SHORT)}
    disp = {k: d for _, k, d in MODEL_SHORT}
    lines = []
    keys = []
    for (b, model, prompt, mit, fm, tags), v in cells209.items():
        if len(v) < min_n(EXPECTED_209) or b not in CORE:
            continue
        ms = short_model(model)
        q = ":".join([b, ms, prompt, MIT_SHORT.get(mit, mit), FM_SHORT.get(fm, fm)] + list(tags))
        keys.append((order.get(ms, 99), b, prompt, mit, fm, q, ms, tags))
    for _, b, prompt, mit, fm, q, ms, tags in sorted(keys):
        cond = "/".join([prompt, mit, fm] + list(tags)).replace("_", r"\_")
        # the July Haiku key cell after a timeout was superseded by the
        # registered E1-0 rerun (Table tab:within reports the rerun)
        if q == "jul:haiku:base:keys:toA" and "e10:haiku:base:keys:toA:ivr" in NUMS:
            cond += r" (v2, July; superseded by \mbox{E1-0}, %s)" % NUMS["e10:haiku:base:keys:toA:ivr"]
        lines.append(" & ".join([
            disp.get(ms, ms) + (f" ({b})" if b not in ("jul", "xp") else ""), cond,
            NUMS[q + ":ivr"], NUMS[q + ":kn"], NUMS[q + ":ci"], NUMS[q + ":tci"],
            NUMS[q + ":boot"], f"{NUMS[q + ':tvr']} {NUMS[q + ':tvrci']}",
            NUMS[q + ":succ"]]) + r" \\")
    global FULL209
    FULL209 = "\n".join(lines)
    put("full209:rows", len(lines))


def emit_summaries(cells209):
    """Derived statements over the July Claude data (fixed data)."""
    jul = {k: v for k, v in cells209.items() if k[0] == "jul"}
    n_cells = len(jul)
    succ1 = sum(1 for v in jul.values() if all(r["achieved"] for r in v.values()))
    put("jul:cells", n_cells); put("jul:cells_succ1", succ1)
    put("jul:episodes", f"{sum(len(v) for v in jul.values()):,}")
    mis = sum(1 for v in jul.values() for r in v.values() if r.get("misorder"))
    put("jul:misorder_total", mis)
    tasks = {r["task_id"] for v in jul.values() for r in v.values()}
    fams = {r["family"] for v in jul.values() for r in v.values()}
    templ = {r["task_id"].rsplit("-", 1)[0] for v in jul.values() for r in v.values()}
    put("p1:tasks_used", len(tasks)); put("p1:families", len(fams))
    put("p1:templates", len(templ)); put("p1:per_template", len(tasks) // max(1, len(templ)))
    depth_eq = sum(1 for v in jul.values()
                   if abs(sum(r["violations"] for r in v.values()) / len(v)
                          - sum(1 for r in v.values() if r["violations"] > 0) / len(v)) < 1e-9)
    put("jul:depth_eq_cells", depth_eq)
    # over every complete core-grid 209 cell (Tier-1 cells have their own rows)
    comp = {k: v for k, v in cells209.items() if len(v) >= min_n(EXPECTED_209) and k[0] in CORE}
    viol = [v for v in comp.values() if any(r["violations"] > 0 for r in v.values())]
    if viol:
        put("all:succ_min_violating", f2(min(
            sum(1 for r in v.values() if r["achieved"]) / len(v) for v in viol)))
    put("all:cells_violating", len(viol)); put("all:cells_complete", len(comp))
    put("all:omission_max", f2(max(sum(1 for r in v.values() if r.get("omission")) / len(v)
                                   for v in comp.values())))
    put("all:depth_eq_cells", sum(
        1 for v in comp.values()
        if abs(sum(r["violations"] for r in v.values())
               - sum(1 for r in v.values() if r["violations"] > 0)) == 0))
    put("all:episodes_complete", f"{sum(len(v) for v in comp.values()):,}")
    mis_cells = defaultdict(int)
    for k, v in comp.items():
        for r in v.values():
            if r.get("misorder"):
                mis_cells[(short_model(k[1]), k[4])] += 1
    put("misorder:all", sum(mis_cells.values()))
    disp = {k: d for _, k, d in MODEL_SHORT}
    per_m = Counter()
    for (m, _), c in mis_cells.items():
        per_m[m] += c
    order = [k for _, k, _ in MODEL_SHORT]
    parts = [f"{disp.get(m, m)}: {per_m[m]}" for m in order if per_m.get(m)]
    put("misorder:bymodel", ", ".join(parts))
    put("misorder:models", len({m for m, _ in mis_cells}))
    put("misorder:only_toB", "yes" if all(f == "timeout_before_commit" for _, f in mis_cells) else "no")
    mis_fam = {r["family"] for v in comp.values() for r in v.values() if r.get("misorder")}
    put("misorder:only_multistep", "yes" if mis_fam <= {"multi_step"} else "no")
    put("misorder:with_violation", sum(1 for v in comp.values() for r in v.values()
                                       if r.get("misorder") and r["violations"] > 0))
    rc = [v for k, v in comp.items() if k[3] == "runtime_receipts"]
    put("rcpt:cells", len(rc))
    put("rcpt:allzero", "yes" if all(r["violations"] == 0 for v in rc for r in v.values()) else "no")
    # check-first prompt: completed cells and the ones above zero
    car = {k: v for k, v in comp.items() if k[2] == "careful" and k[3] == "none"}
    nz = [k for k, v in car.items() if any(r["violations"] > 0 for r in v.values())]
    put("careful:cells", len(car)); put("careful:cells_nonzero", len(nz))
    put("careful:cells_zero", len(car) - len(nz))
    put("careful:nonzero", "; ".join(f"{short_model(k[1])}/{FM_SHORT[k[4]]}" for k in nz) or "none")
    # range of IVR across models with complete cells, per failure signal
    for fm in ("timeout_after_commit", "error_500_after_commit"):
        vals = []
        for (b, model, prompt, mit, f, tags), v in cells209.items():
            if prompt == "base" and mit == "none" and f == fm and not tags \
                    and len(v) >= min_n(EXPECTED_209) and b in CORE:
                vals.append((short_model(model),
                             sum(1 for r in v.values() if r["violations"] > 0) / len(v)))
        if vals:
            fs = FM_SHORT[fm]
            put(f"range:{fs}:min", f2(min(x[1] for x in vals)))
            put(f"range:{fs}:max", f2(max(x[1] for x in vals)))
            put(f"range:{fs}:nmodels", len(vals))
            names = [d for _, k, d in MODEL_SHORT if k in {x[0] for x in vals}]
            put(f"range:{fs}:models", ", ".join(names[:-1]) + " and " + names[-1]
                if len(names) > 1 else names[0])


FIG1 = {  # Figure 1: one task, one model, one committed world state, two signals
    "to": "claude-sonnet-5_base_none_timeout_after_commit__s0__payments-18aa8e1466.json",
    "e5": "claude-sonnet-5_base_none_error_500_after_commit__s0__payments-18aa8e1466.json",
}


def emit_fig1():
    for tag, fn in FIG1.items():
        p = os.path.join(RES, "sweep_transcripts", fn)
        t = json.load(open(p))
        put(f"fig1:{tag}:instruction", tex_id(t["instruction"]))
        steps = []
        for turn in t["turns"]:
            for b in turn["content"]:
                if turn["role"] == "assistant" and b.get("type") == "tool_use":
                    a = dict(b["input"]); a.pop("idempotency_key", None)
                    steps.append(("call", b["name"] + "(" + ", ".join(
                        f"{k}={v}" for k, v in a.items()) + ")"))
                elif turn["role"] == "tool_results":
                    steps.append(("obs", b["content"]))
                elif turn["role"] == "assistant" and b.get("type") == "text":
                    steps.append(("say", b["text"].replace("\n", " ").strip()))
        for i, (kind, txt) in enumerate(steps):
            put(f"fig1:{tag}:{i}:kind", kind)
            put(f"fig1:{tag}:{i}", tex_id(txt if len(txt) < 120 else txt[:117] + "..."))
        put(f"fig1:{tag}:steps", len(steps))
        put(f"fig1:{tag}:dups", t["score"]["violations"])
        put(f"fig1:{tag}:file", tex_path(fn))
        put(f"fig1:{tag}:task", tex_id(t["task_id"]))
        says = [x for k, x in steps if k == "say"]
        if says:
            first = re.split(r"(?<=[.!?])\s", says[-1].replace("DONE", "").strip())[0]
            put(f"fig1:{tag}:say", tex_text(first))
        put(f"fig1:{tag}:instr_text", tex_text(t["instruction"]))
        put(f"fig1:{tag}:sample", t["sample"])
        rt = retry_text(t)
        put(f"fig1:{tag}:samekey", "yes" if rt and rt[2] else "no")
        put(f"fig1:{tag}:retry_text", "yes" if rt and rt[0] else "no")
        # does the closing message mention the error or the retry?
        last = says[-1].lower() if says else ""
        put(f"fig1:{tag}:say_mentions_error",
            "yes" if re.search(r"error|retr|again|timed out|time out", last) else "no")


FIGSEAM = {  # Figure 1: one graph, one model, one interruption, two resume notes
    "am": "openrouter:qwen_qwen3.6-27b-20260422__amnesiac__none__chain4-27d504eba1__k2__s0.json",
    "bh": "openrouter:qwen_qwen3.6-27b-20260422__bare_hint__none__chain4-27d504eba1__k2__s0.json",
}


def emit_figseam():
    for tag, fn in FIGSEAM.items():
        t = json.load(open(os.path.join(RES, "crossprovider", "transcripts2", fn)))
        put(f"figs:{tag}:file", tex_path(fn))
        put(f"figs:{tag}:dups", t["score"]["violations"])
        put(f"figs:{tag}:k", t["interrupt_node"])
        put(f"figs:{tag}:committed", t["interrupt_node"] + 1)
        put(f"figs:{tag}:graph", tex_id(t["graph_id"]))
        put(f"figs:{tag}:instr", tex_text(t["instruction"]))
        for i, ses in enumerate(t["sessions"]):
            names = [x["name"] for turn in ses["turns"] if turn.get("role") == "assistant"
                     for x in turn["content"] if x.get("type") == "tool_use"]
            put(f"figs:{tag}:s{i + 1}calls", tex_text(", ".join(names)))
            put(f"figs:{tag}:s{i + 1}n", len(names))
            looked = "list_effects" in names
            put(f"figs:{tag}:s{i + 1}looked", "yes" if looked else "no")
        put(f"figs:{tag}:served", tex_id(t.get("model_served", "")))
        put(f"figs:{tag}:sample", t.get("sample", 0))


def emit_finish_reasons():
    """Assistant turns that hit the length limit or an in-band provider error
    but still carried a tool call (so they were executed, not flagged as
    truncated): OpenRouter native_finish_reason "length", stop_reason "error"."""
    for tag, dirs in (("xp", [os.path.join(RES, "crossprovider", "sweep_transcripts"),
                              os.path.join(RES, "crossprovider", "transcripts2")]),
                      ("t1", glob.glob(os.path.join(RES, "tier1", "e1_*", "transcripts")))):
        c = Counter()
        for d in dirs:
            for f in glob.glob(os.path.join(d, "*.json")):
                t = json.load(open(f))
                turns = [x for s in t["sessions"] for x in s["turns"]] if "sessions" in t else \
                    [x for dl in t["deliveries"] for x in dl["turns"]] if "deliveries" in t else t["turns"]
                for x in turns:
                    if x.get("role") != "assistant" or not any(b.get("type") == "tool_use" for b in x["content"]):
                        continue
                    m = short_model(t.get("model", ""))
                    if (x.get("provider_meta") or {}).get("native_finish_reason") == "length":
                        c[(m, "length")] += 1
                    if x.get("stop_reason") == "error":
                        c[(m, "error")] += 1
        for m in ("qwen", "gemini", "gpt"):
            for r in ("length", "error"):
                put(f"fin:{tag}:{m}:{r}", c[(m, r)])


def emit_launches():
    """Start times of the overlapping cross-provider launches (run.log)."""
    p = os.path.join(RES, "crossprovider", "run.log")
    if not os.path.exists(p):
        return
    starts = re.findall(r"=== start \w+ (\w+) (\d+) (\d+:\d+):\d+ (\w+) (\d{4})", open(p).read())
    if len(starts) >= 2:
        for i, (mon, day, hm, tz, yr) in enumerate(starts[:2]):
            put(f"dup:xp:launch{i + 1}", f"{hm} {tz}")
        months = dict(Jan="January", Feb="February", Mar="March", Apr="April", May="May",
                      Jun="June", Jul="July", Aug="August", Sep="September", Oct="October",
                      Nov="November", Dec="December")
        put("dup:xp:date", f"{int(starts[0][1])} {months[starts[0][0]]} {starts[0][4]}")


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
         7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def emit_totals(cells209, cells249):
    """Totals over every complete or partial cell actually on disk."""
    n209 = sum(len(v) for v in cells209.values())
    n249 = sum(len(v) for v in cells249.values())
    put("total:p1", f"{n209:,}"); put("total:p2", f"{n249:,}")
    put("total:all", f"{n209 + n249:,}")
    put("total:core", f"{sum(len(v) for k, v in cells209.items() if k[0] in CORE) + n249:,}")
    models = sorted({short_model(k[1]) for k, v in cells209.items()
                     if len(v) >= min_n(EXPECTED_209) and k[0] in CORE} |
                    {short_model(k[1]) for k, v in cells249.items()
                     if len(v) >= EXPECTED_249_OVERRIDE.get((k[0], k[2]), EXPECTED_249)})
    put("models:n", len(models)); put("models:nwords", WORDS.get(len(models), len(models)))
    prov = {"haiku": "Anthropic", "sonnet": "Anthropic", "fable": "Anthropic",
            "gpt": "OpenAI", "gemini": "Google", "qwen": "Alibaba"}
    ps = {prov.get(m, m) for m in models}
    put("providers:n", len(ps)); put("providers:nwords", WORDS.get(len(ps), len(ps)))
    cl = [m for m in models if m in ("haiku", "sonnet", "fable")]
    put("models:claude:nwords", WORDS.get(len(cl), len(cl)))
    other = [m for m in models if m not in ("haiku", "sonnet", "fable")]
    put("models:other:nwords", WORDS.get(len(other), len(other)))
    for b in sorted({k[0] for k in cells209} | {k[0] for k in cells249}):
        r1 = [r for k, v in cells209.items() if k[0] == b for r in v.values()]
        r2 = [r for k, v in cells249.items() if k[0] == b for r in v.values()]
        put(f"total:{b}:episodes", f"{len(r1) + len(r2):,}")
        put(f"total:{b}:trunc", sum(1 for r in r1 + r2 if r.get("truncated")))
        put(f"total:{b}:invalid", sum(int(r.get("invalid_calls") or 0) for r in r1 + r2))
    # minimum detectable difference, two proportions, T tasks per arm,
    # alpha = .05 two-sided, power .80, at p = 0.5
    for T in (16, 32, 96):
        put(f"power:mdd{T}", f2((1.959964 + 0.841621) * math.sqrt(2 * 0.25 / T)))


def emit_dups(cells209):
    """Episodes that were run more than once (two concurrent launches before
    the per-model run lock). For every affected cell: the value under the
    reported rule (last row per episode_id) and under the first run (first
    row per episode_id); all rows stay in the released raw file."""
    global DUPROWS
    disp = {k: d for _, k, d in MODEL_SHORT}
    lines = []
    tot_conf = 0
    for b, d in sorted(DUPS.items()):
        affected = defaultdict(list)
        for eid, rs in d.items():
            r = rs[-1]
            affected[(b, r["model"], r["prompt"], r["mitigation"], r["failure_mode"],
                      tuple(cell209_tags(r)))].append(eid)
        put(f"dup:{b}:cells", len(affected))
        models = {short_model(k[1]) for k in affected}
        put(f"dup:{b}:models", ", ".join(disp.get(m, m) for m in sorted(models)))
        for key, eids in sorted(affected.items(), key=lambda kv: kv[0][2:]):
            eps = cells209[key]
            last = {e: r["violations"] > 0 for e, r in eps.items()}
            first = dict(last)
            for e in eids:
                first[e] = d[e][0]["violations"] > 0
            n = len(eps)
            kl, kf = sum(last.values()), sum(first.values())
            conf = sum(1 for e in eids if (d[e][0]["violations"] > 0) != (d[e][-1]["violations"] > 0))
            tot_conf += conf
            ms = short_model(key[1])
            c = ":".join([ms, key[2], MIT_SHORT.get(key[3], key[3]), FM_SHORT.get(key[4], key[4])])
            q = f"dup:{b}:{c}"
            put(q + ":rerun", len(eids)); put(q + ":n", n)
            put(q + ":last", f"{kl}/{n}"); put(q + ":first", f"{kf}/{n}")
            put(q + ":last_ivr", f2(kl / n)); put(q + ":first_ivr", f2(kf / n))
            put(q + ":disagree", conf)
            cond = "/".join([key[2], MIT_SHORT.get(key[3], key[3]),
                             {"error_500_after_commit": "500", "timeout_before_commit": "timeout, before"}
                             .get(key[4], FM_SHORT.get(key[4], key[4]))])
            lines.append(" & ".join([disp.get(ms, ms).replace("Qwen3.6-27B", "Qwen"), cond, str(len(eids)),
                                     f"{kf}/{n} ({f2(kf / n)})", f"{kl}/{n} ({f2(kl / n)})",
                                     str(conf)]) + r" \\")
            # the timeout-vs-server-error test, recomputed with first-run rows
            if key[2] == "base" and key[3] == "none" and key[4] == "error_500_after_commit":
                qa = f"{b}:{ms}:base:none:toA"
                if qa in EPI:
                    ra = task_rates(EPI[qa], 1)
                    rf = defaultdict(list)
                    for e, r in eps.items():
                        rf[r["task_id"]].append(1 if first[e] else 0)
                    tasks = sorted(set(ra) & set(rf))
                    dm, p = perm_test([(ra[t], sum(rf[t]) / len(rf[t])) for t in tasks], q)
                    put(q + ":first_test_p", pfmt(p))
                    put(q + ":first_test_d", f2(dm))
    put("dup:disagree:total", tot_conf)
    DUPROWS = "\n".join(lines)


def perm_test(pairs, key):
    """Two-sided paired permutation test over tasks (sign flip of the per-task
    difference), PERM_N permutations, fixed seed. pairs: [(x_task, y_task)]
    with per-task rates. Returns (mean difference y - x, p)."""
    diffs = [y - x for x, y in pairs]
    obs = abs(sum(diffs))
    rng = random.Random(PERM_SEED)
    hit = 0
    for _ in range(PERM_N):
        s = sum(dv if rng.random() < 0.5 else -dv for dv in diffs)
        if abs(s) >= obs - 1e-12:
            hit += 1
    return sum(diffs) / len(diffs), (hit + 1) / (PERM_N + 1)


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def holm(ps):
    """Holm step-down adjusted p-values, same order as input."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adj = [0.0] * m
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * ps[i]))
        adj[i] = run
    return adj


def task_rates(epi, idx):
    """{task: mean of field idx over its samples} from EPI entries."""
    d = defaultdict(list)
    for e in epi:
        if e[idx] is not None:
            d[e[0]].append(e[idx])
    return {t: sum(v) / len(v) for t, v in d.items()}


def emit_tests():
    """Paired task-level permutation tests for every contrast stated in the
    text. Single-session: timeout vs server error, per model, on IVR and on
    read-back. Sessions: amnesiac vs bare_hint per model (paired by graph,
    interrupt node and sample; same-month cells where they exist)."""
    family = []      # (key, p) for the timeout-vs-500 family, Holm within it
    for _, m, _ in MODEL_SHORT:
        for b in ("jul", "xp"):
            qa, qe = f"{b}:{m}:base:none:toA", f"{b}:{m}:base:none:e500"
            if qa in EPI and qe in EPI:
                break
        else:
            continue
        for metric, idx in (("ivr", 1), ("rb", 2)):
            ra, re_ = task_rates(EPI[qa], idx), task_rates(EPI[qe], idx)
            tasks = sorted(set(ra) & set(re_))
            if not tasks:
                continue
            dmean, p = perm_test([(ra[t], re_[t]) for t in tasks], f"{m}:{metric}")
            k = f"test:{m}:toA_e500:{metric}"
            put(k + ":d", ("+" if dmean > 0 else "") + f2(dmean) if dmean else "0.00")
            put(k + ":p", pfmt(p)); put(k + ":T", len(tasks))
            put(k + ":plus", sum(1 for t in tasks if re_[t] > ra[t]))
            put(k + ":minus", sum(1 for t in tasks if re_[t] < ra[t]))
            family.append((k, p))
            if metric == "ivr" and len(tasks) > 1:
                # smallest paired difference detectable at 80% power, alpha .05
                # two-sided (normal approximation, observed SD of d)
                dd = [re_[t] - ra[t] for t in tasks]
                mu = sum(dd) / len(dd)
                sd = math.sqrt(sum((x - mu) ** 2 for x in dd) / (len(dd) - 1))
                PAIRED_MDD[m] = (1.959964 + 0.841621) * sd / math.sqrt(len(dd))
    if PAIRED_MDD:
        disp = {k: d for _, k, d in MODEL_SHORT}
        lo = min(PAIRED_MDD, key=PAIRED_MDD.get); hi = max(PAIRED_MDD, key=PAIRED_MDD.get)
        put("power:paired:min", f2(PAIRED_MDD[lo])); put("power:paired:minname", disp[lo])
        put("power:paired:max", f2(PAIRED_MDD[hi])); put("power:paired:maxname", disp[hi])
    if family:
        for (k, _), a in zip(family, holm([p for _, p in family])):
            put(k + ":holm", pfmt(a))
        put("test:toA_e500:family", len(family))
        put("test:perm_n", f"{PERM_N:,}")
    # sessions: hint vs no hint, paired by graph (EPI249 holds complete cells
    # only). Primary = same month, same harness and settings (Claude: the
    # October batch; the OpenRouter models: their one batch). Secondary,
    # Claude only = July amnesiac vs October bare_hint.
    def graph_test(qa, qh, k):
        ga, gh = defaultdict(list), defaultdict(list)
        for (g, kk, s), v in EPI249[qa].items():
            ga[g].append(v)
        for (g, kk, s), v in EPI249[qh].items():
            gh[g].append(v)
        gs = sorted(set(ga) & set(gh))
        if not gs:
            return
        dmean, p = perm_test([(sum(ga[g]) / len(ga[g]), sum(gh[g]) / len(gh[g]))
                              for g in gs], k)
        put(k + ":d", f2(dmean)); put(k + ":p", pfmt(p)); put(k + ":G", len(gs))
    sm = {}
    for _, m, _ in MODEL_SHORT:
        b = "oct" if m in ("haiku", "sonnet", "fable") else "xp"
        qa, qh = f"{b}:{m}:amnesiac:none", f"{b}:{m}:bare_hint:none"
        if qh not in EPI249:
            continue
        if qa in EPI249:
            graph_test(qa, qh, f"test:{m}:hint")
            sm[m] = float(NUMS[qa + ":givr"])
        else:
            pending(f"test:{m}:hint:p", "same-month amnesiac cell not complete")
            sm[m] = None
        qj = f"jul:{m}:amnesiac:none"
        if b == "oct" and qj in EPI249:
            graph_test(qj, qh, f"test:{m}:hintjul")
    # same-month ranges per group: Claude (October batch) and the OpenRouter
    # models (one batch each). amn = no hint, hint = bare_hint.
    disp = {k: d for _, k, d in MODEL_SHORT}
    for g, ms, b in (("cl", ("haiku", "sonnet", "fable"), "oct"),
                     ("or", ("gpt", "gemini", "qwen"), "xp")):
        have = [m for m in ms if f"{b}:{m}:bare_hint:none" in EPI249]
        if not have:
            continue
        put(f"sm:{g}:names", " and ".join([", ".join(disp[m] for m in have[:-1]), disp[have[-1]]])
            if len(have) > 1 else disp[have[0]])
        put(f"sm:{g}:n", len(have))
        hv = [float(NUMS[f"{b}:{m}:bare_hint:none:givr"]) for m in have]
        put(f"sm:{g}:hint:min", f2(min(hv))); put(f"sm:{g}:hint:max", f2(max(hv)))
        hk = [NUMS[f"{b}:{m}:bare_hint:none:kn"] for m in have]
        put(f"sm:{g}:hint:kn", f"{sum(int(x.split('/')[0]) for x in hk)}/"
                               f"{sum(int(x.split('/')[1]) for x in hk)}")
        if all(sm.get(m) is not None for m in have):
            av = [sm[m] for m in have]
            put(f"sm:{g}:amn:min", f2(min(av))); put(f"sm:{g}:amn:max", f2(max(av)))
            ak = [NUMS[f"{b}:{m}:amnesiac:none:kn"] for m in have]
            put(f"sm:{g}:amn:kn", f"{sum(int(x.split('/')[0]) for x in ak)}/"
                                  f"{sum(int(x.split('/')[1]) for x in ak)}")
            ps = [NUMS.get(f"test:{m}:hint:p", "") for m in have]
            put(f"sm:{g}:pmax", max(ps, key=lambda s: 0 if s.startswith("<") else float(s)))
        else:
            done = sum(1 for m in have if sm.get(m) is not None)
            for k in ("amn:min", "amn:max", "amn:kn", "pmax"):
                pending(f"sm:{g}:{k}", f"{done}/{len(have)} same-month no-hint cells")


def emit_baselines(cells249):
    """Same-month (October 2026) Claude baselines for the bare_hint contrast:
    amnesiac and ledger_readback run by run_claude_baselines.sh into
    results/claude_batch_2026-10/ with the same harness and settings as the
    October bare_hint cells. A cell counts only at full n (32)."""
    claude = ("haiku", "sonnet", "fable")
    done = []
    for ctx in ("amnesiac", "ledger_readback"):
        for m in claude:
            q = f"oct:{m}:{ctx}:none"
            ok = q + ":givr" in NUMS and not NUMS[q + ":givr"].startswith(r"\PENDING")
            if not ok and q + ":givr" not in NUMS:
                pending(q + ":givr", f"0/{EXPECTED_249}")
                for met in ("gci", "kn", "chk"):
                    pending(f"{q}:{met}", f"0/{EXPECTED_249}")
            done.append(ok)
    put("base:oct:complete", "yes" if all(done) else "no")
    # pooled same-month amnesiac over the three Claude models, if complete
    pool = [r for (b, model, ctx, mit), eps in cells249.items()
            if b == "oct" and ctx == "amnesiac" and mit == "none"
            and short_model(model) in claude for r in eps.values()]
    if len(pool) >= 3 * EXPECTED_249 and all(done[:3]):
        k = sum(r["givr_hit"] for r in pool)
        G = len({(r["model"], r["graph_id"]) for r in pool})
        put("pool:amnesiac:oct:kn", f"{k}/{len(pool)}")
        put("pool:amnesiac:oct:gci", ci_str(*wilson(k / len(pool) * G, G)))
    else:
        pending("pool:amnesiac:oct:kn", f"{len(pool)}/{3 * EXPECTED_249}")


def emit_derived():
    """Yes/no facts that sentences in main.tex are asserted against."""
    disp_to_key = {d: k for _, k, d in MODEL_SHORT}
    def names_to_keys(v):
        return [disp_to_key[x.strip()] for x in re.split(r",| and ", v or "")
                if x.strip() in disp_to_key]
    def sig(k, alpha=0.05):
        v = NUMS.get(k)
        if v is None or v.startswith(r"\PENDING"):
            return None
        return v.startswith("<") or float(v) < alpha
    chk = names_to_keys(NUMS.get("checkers:names"))
    non = names_to_keys(NUMS.get("noncheckers:names"))
    if chk:
        put("test:checkers:ivr_holm_sig", "yes" if all(
            sig(f"test:{m}:toA_e500:ivr:holm") for m in chk) else "no")
        put("test:checkers:rb_holm_sig", "yes" if all(
            sig(f"test:{m}:toA_e500:rb:holm") for m in chk) else "no")
    if non:
        put("test:noncheckers:ivr_raw_sig", "yes" if all(
            sig(f"test:{m}:toA_e500:ivr:p") for m in non) else "no")
        put("test:noncheckers:ivr_holm_ns", "yes" if all(
            sig(f"test:{m}:toA_e500:ivr:holm") is False for m in non) else "no")
        put("test:noncheckers:rb_raw_ns", "yes" if all(
            sig(f"test:{m}:toA_e500:rb:p") is False for m in non) else "no")
    for grp, ms, b in (("cl", ("haiku", "sonnet", "fable"), "oct"),
                       ("or", ("gpt", "gemini", "qwen"), "xp")):
        ok = []
        for m in ms:
            a, h = NUMS.get(f"{b}:{m}:amnesiac:none:givr"), NUMS.get(f"{b}:{m}:bare_hint:none:givr")
            if not a or not h or a.startswith(r"\PENDING") or h.startswith(r"\PENDING"):
                ok = None
                break
            ok.append(float(h) < float(a) and sig(f"test:{m}:hint:p"))
        if ok is not None and ok:
            put(f"sm:{grp}:hint_lower_sig", "yes" if all(ok) else "no")
    # all models whose bare_hint and no-hint cells ran in the same month
    allm = []
    for _, m, _ in MODEL_SHORT:
        b = "oct" if m in ("haiku", "sonnet", "fable") else "xp"
        a, h = NUMS.get(f"{b}:{m}:amnesiac:none:kn"), NUMS.get(f"{b}:{m}:bare_hint:none:kn")
        p = NUMS.get(f"test:{m}:hint:p")
        if a and h and p and not any(x.startswith(r"\PENDING") for x in (a, h, p)):
            allm.append((m, a, h, p))
    if allm:
        put("sm:all:n", len(allm))
        put("sm:all:amn:kn", f"{sum(int(x[1].split('/')[0]) for x in allm)}/"
                             f"{sum(int(x[1].split('/')[1]) for x in allm)}")
        put("sm:all:hint:kn", f"{sum(int(x[2].split('/')[0]) for x in allm)}/"
                              f"{sum(int(x[2].split('/')[1]) for x in allm)}")
        worst = max((x[3] for x in allm), key=lambda v: 0 if v.startswith("<") else float(v))
        put("sm:all:pmaxe", "$p<0.001$" if worst.startswith("<") else f"$p={worst}$")
        put("sm:all:sig", "yes" if all(sig(f"test:{x[0]}:hint:p") for x in allm) else "no")
    # double-run cell: does the first run give the same direction?
    fd = NUMS.get("dup:xp:qwen:base:none:e500:first_test_d")
    ld = NUMS.get("test:qwen:toA_e500:ivr:d")
    if fd and ld:
        put("dup:xp:qwen:same_direction",
            "yes" if (float(fd) < 0) == (float(ld.lstrip("+")) < 0) else "no")
    g = NUMS.get("jul:haiku:base:none:e500:gaveup_succ", "0/0").split("/")
    put("jul:haiku:base:none:e500:gaveup_allsucc", "yes" if g[0] == g[1] else "no")
    # p-values as math for prose: key + "e" -> "$p=0.034$" / "$p<0.001$"
    for k, v in list(NUMS.items()):
        if re.search(r":(p|holm|pmax|first_test_p)$", k) and not v.startswith(r"\PENDING"):
            NUMS[k + "e"] = "$p<0.001$" if v.startswith("<") else f"$p={v}$"


def emit_contrasts():
    """Per-model timeout vs 500 contrast at the same committed state, from the
    values already computed. A model counts only when both cells are complete."""
    disp = {k: d for _, k, d in MODEL_SHORT}
    order = [k for _, k, _ in MODEL_SHORT]
    rows = []
    for m in order:
        def get(cell, met):
            for b in ("jul", "xp", ""):
                key = (f"{b}:" if b else "") + f"{m}:base:none:{cell}:{met}"
                v = NUMS.get(key)
                if v and not v.startswith(r"\PENDING"):
                    return v
            return None
        ta, ea = get("toA", "ivr"), get("e500", "ivr")
        tr, er = get("toA", "rbrate"), get("e500", "rbrate")
        if None in (ta, ea, tr, er):
            continue
        rows.append((m, float(ta), float(ea), float(tr), float(er)))
    if not rows:
        return
    put("contrast:nmodels", len(rows))
    hi = [r for r in rows if r[2] > r[1]]
    lo = [r for r in rows if r[2] < r[1]]
    def names(rs):
        ns = [disp[r[0]] for r in rs]
        return "none" if not ns else ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]
    put("contrast:e500_higher:n", len(hi)); put("contrast:e500_higher:names", names(hi))
    put("contrast:e500_lower:n", len(lo)); put("contrast:e500_lower:names", names(lo))
    chk = [r for r in rows if r[3] >= 0.5]       # models that check after a timeout
    put("checkers:n", len(chk)); put("checkers:names", names(chk))
    non = [r for r in rows if r[3] < 0.5]
    put("noncheckers:n", len(non)); put("noncheckers:names", names(non))
    if chk:
        put("checkers:rb_toA:min", f2(min(r[3] for r in chk)))
        put("checkers:rb_e500:max", f2(max(r[4] for r in chk)))
        put("checkers:all_e500_higher", "yes" if all(r[2] > r[1] for r in chk) else "no")
        put("checkers:all_rb_lower", "yes" if all(r[4] < r[3] for r in chk) else "no")
        sm = min(chk, key=lambda r: r[2] - r[1])
        put("checkers:smallest_rise", disp[sm[0]])
    if non:
        put("noncheckers:rb_max", f2(max(max(r[3], r[4]) for r in non)))
        put("noncheckers:ivr:min", f2(min(min(r[1], r[2]) for r in non)))
        put("noncheckers:ivr:max", f2(max(max(r[1], r[2]) for r in non)))
    for k in ("toA", "e500"):
        put(f"range5:{k}:min", f2(min(r[1 if k == "toA" else 2] for r in rows)))
        put(f"range5:{k}:max", f2(max(r[1 if k == "toA" else 2] for r in rows)))
    for w in ("contrast:nmodels", "checkers:n", "noncheckers:n",
              "contrast:e500_higher:n", "contrast:e500_lower:n"):
        NUMS[w + ":w"] = {0: "zero"}.get(int(NUMS[w]), WORDS.get(int(NUMS[w])))


def emit_lists(cells249):
    """Per model, the primary complete 249 cell for each resume context, and
    name lists derived from them, so prose that names models follows the data.
    Primary cell: corrected plan_only (pocorr) > October batch > xp > July."""
    disp = {k: d for _, k, d in MODEL_SHORT}
    order = [k for _, k, _ in MODEL_SHORT]
    def names(ms):
        ns = [disp[m] for m in order if m in ms]
        return "none" if not ns else ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]
    for ctx in ("amnesiac", "bare_hint", "plan_only", "ledger_readback"):
        vals = {}
        # amnesiac / ledger_readback: the Table 2 cells (July for Claude, the
        # October same-month baselines are reported in their own row)
        # same-month cells first: the October batch for the Claude models
        # (amnesiac and ledger_readback were re-run there as baselines), the
        # one batch for the OpenRouter models; July only as a fallback
        batches = ("oct", "xp", "jul")
        for m in order:
            for q in ((f"pocorr:{m}:plan_only:none",) if ctx == "plan_only" else ()) + \
                     tuple(f"{b}:{m}:{ctx}:none" for b in batches):
                v = NUMS.get(q + ":givr")
                if v and not v.startswith(r"\PENDING") and NUMS.get(q + ":n") and \
                        (int(NUMS[q + ":n"]) >= min_n(EXPECTED_249)):
                    vals[m] = (float(v), q)
                    break
        if not vals:
            continue
        put(f"list:{ctx}:n", len(vals))
        put(f"list:{ctx}:names", names(vals))
        put(f"list:{ctx}:zero", names([m for m, x in vals.items() if x[0] == 0]))
        put(f"list:{ctx}:zero:n", sum(1 for x in vals.values() if x[0] == 0))
        put(f"list:{ctx}:nonzero", names([m for m, x in vals.items() if x[0] > 0]))
        put(f"list:{ctx}:nonzero:n", sum(1 for x in vals.values() if x[0] > 0))
        put(f"list:{ctx}:one", names([m for m, x in vals.items() if x[0] == 1]))
        put(f"list:{ctx}:one:n", sum(1 for x in vals.values() if x[0] == 1))
        put(f"list:{ctx}:below1", names([m for m, x in vals.items() if x[0] < 1]))
        put(f"list:{ctx}:below1:n", sum(1 for x in vals.values() if x[0] < 1))
        put(f"list:{ctx}:max", f2(max(x[0] for x in vals.values())))
        put(f"list:{ctx}:min", f2(min(x[0] for x in vals.values())))
        for m, (v, q) in vals.items():
            put(f"best:{m}:{ctx}", q)
        allchk = all(NUMS.get(q + ":chk", "").split("/")[0] == NUMS.get(q + ":resumed")
                     for _, q in vals.values())
        try:
            lk = sum(int(NUMS[q + ":chk"].split("/")[0]) for _, q in vals.values())
            rs = sum(int(NUMS[q + ":chk"].split("/")[1]) for _, q in vals.values())
            put(f"list:{ctx}:chk_total", f"{lk}/{rs}")
        except (KeyError, ValueError):
            pass
        put(f"list:{ctx}:allchk", "yes" if allchk else "no")
        put(f"list:{ctx}:dupchk_none", "yes" if all(
            NUMS.get(q + ":chkdup", "0/0").split("/")[0] == "0" for _, q in vals.values()) else "no")
    for k in [k for k in list(NUMS) if k.startswith("list:") and k.endswith(":n")]:
        NUMS[k + ":w"] = {0: "zero"}.get(int(NUMS[k]), WORDS.get(int(NUMS[k])))


def emit_design():
    def grab(path, pat, cast=str):
        try:
            m = re.search(pat, open(os.path.join(ROOT, path)).read(), re.M)
            return cast(m.group(1)) if m else None
        except OSError:
            return None
    s1 = json.load(open(os.path.join(RES, "analysis_v2", "e0_1_summary.json")))
    put("jul:trunc_suspect", len(s1["truncated_suspect_files"]))
    put("jul:turncap_hits", s1["hit_turn_cap"])
    # Haiku verdict agreement between its two samples, worst cell
    rows = list(csv.DictReader(open(os.path.join(RES, "analysis_v2", "e0_1_corrected_table.csv"))))
    hk = [r for r in rows if "haiku" in r["model"]]
    if hk:
        worst = min(hk, key=lambda r: int(r["verdict_agree_tasks"]))
        put("haiku:agree_min", f"{worst['verdict_agree_tasks']} of {worst['n_tasks']}")
    hc = os.path.join(RES, "analysis_v2", "e0_3_handcheck.csv")
    ctl = [r for r in csv.DictReader(open(hc)) if r["auto_class"] != "normalization_only"]
    put("match:hand:controls", len(ctl))
    v = grab("src/model_agent.py", r"^MAX_TURNS\s*=\s*(\d+)")
    v and put("design:maxturns1", v)
    v = grab("src2/sweep2.py", r"^MAX_TURNS_SESSION\s*=\s*(\d+)")
    v and put("design:maxturns2", v)
    v = grab("src/model_agent.py", r'BENCH_MAX_TOKENS",\s*"(\d+)"')
    v and put("design:maxtokens", v)
    v = grab("run_crossprovider.sh", r"BENCH_MAX_TOKENS=(\d+)")
    v and put("design:maxtokens_or", v)
    v = grab("src/agents.py", r"max_attempts\s*=\s*(\d+)")
    v and put("design:agent_retry_attempts", v)
    # tool schema text the model sees (Appendix "Prompts"), read from the source
    for k, pat in (("design:schema:keydesc", r'"idempotency_key"\]\s*=\s*\{[^}]*"description":\s*"([^"]+)"'),
                   ("design:schema:tooldesc", r'"description":\s*f"(Execute \{tool\}\.[^"]*)"'),
                   ("design:schema:listdesc", r'"name":\s*"list_effects",\s*"description":\s*"([^"]+)"')):
        v = grab("src/model_agent.py", pat)
        if v:
            put(k, tex_text(v.replace("{tool}", "<tool>")))
    v = grab("src/model_agent.py", r"^TEMPERATURE\s*=\s*([\d.]+)")
    v and put("design:temp_req", v.rstrip("0").rstrip(".") if "." in v else v)
    mix = grab("src2/graph.py", r"^MIX\s*=\s*(\{[^}]*\})")
    if mix:
        d = json.loads(mix.replace("'", '"'))
        put("design:graphs", sum(d.values()))
        for k, n in d.items():
            put(f"design:graphs:{k}", n)
    # canonical fields drift check
    w = open(os.path.join(ROOT, "src", "world.py")).read()
    for tool, fields in CANON.items():
        m = re.search(r'"%s":\s*\("(\w+)",\s*\(([^)]*)\)\)' % tool, w)
        got = tuple(re.findall(r'"(\w+)"', m.group(2))) if m else None
        if got != fields:
            note(f"WARNING canonical fields for {tool} differ from src/world.py: {got}")
    put("design:domains", len(CANON))
    # anthropic SDK version used by the run environments (dist-info folder)
    vs = sorted({re.search(r"anthropic-([\d.]+)\.dist-info", p).group(1) for p in
                 glob.glob(os.path.join(ROOT, ".venv*", "lib", "*", "site-packages",
                                        "anthropic-*.dist-info"))})
    if not vs and os.path.exists(os.path.join(HERE, 'provenance.json')):
        vs = json.load(open(os.path.join(HERE, 'provenance.json'))).get('anthropic_sdk_versions') or []
    if vs:
        put("design:sdk_version", ", ".join(vs))
    # Stripe API version pinned on every R1 request (run config of the slice)
    try:
        put("design:stripe_version", json.load(open(os.path.join(ROOT, "configs",
            "real_stripe.json")))["stripe_api_version"])
    except (OSError, KeyError, ValueError):
        pass
    rm = grab("src/agents.py", r"^RETRY_MODES\s*=\s*\(([^)]*)\)")
    rm and put("design:retry_modes", len(re.findall(r'"\w+"', rm)))
    fmd = grab("src/world.py", r"^FAILURE_MODES\s*=\s*\(([^)]*)\)")
    fmd and put("design:failure_modes", len(re.findall(r'"\w+"', fmd)))
    fam = grab("src/tasks.py", r"^FAMILIES\s*=\s*\(([^)]*)\)")
    fam and put("design:families", len(re.findall(r'"\w+"', fam)))
    tp = grab("src2/graph.py", r"^TOPOLOGIES\s*=\s*\(([^)]*)\)")
    tp and put("design:topologies", len(re.findall(r'"\w+"', tp)))
    rc = grab("src2/resume.py", r"^RESUME_CONTEXTS\s*=\s*\(([^)]*)\)")
    rc and put("design:resume_contexts", len(re.findall(r'"\w+"', rc)))
    tc = grab("src/tasks.py", r"^TEMPLATE_COUNT\s*=.*#\s*(\d+)")
    tc and put("design:templates", tc)
    put("design:ntasks_generated", list(csv.DictReader(open(os.path.join(
        RES, "summary_by_failure.csv"))))[0]["n_tasks"])


def check_samples(cells209):
    ss = {r["sample"] for v in cells209.values() for r in v.values()}
    if ss != {0, 1}:
        note(f"WARNING samples per task are {sorted(ss)}, not {{0,1}}")
    put("design:samples", len(ss))



# ---------------------------------------------------------------- analysis v3
# results/analysis_v3/numbers.json is written by analysis/run_all_v3.sh (the
# pre-registered E1-0 ... E1-6 tests, the same-month bare_hint contrast, the
# refusal bounds and the provisional E1-6 belief coding). Every entry becomes
# macros under the prefix "nj:" + its key, so main.tex prints the test results
# exactly as the analysis computed them:
#   \V{nj:<key>}            the value (rate 0.69, p "<0.001", d "+0.63", text)
#   \V{nj:<key>:kn}         k/n, when the entry has k and n
#   \V{nj:<key>:ci}         [lo, hi] (the entry's own CI: task-, graph- or
#                           episode-level, as its ci_method says)
#   \V{nj:<key>:pe}         "$p=0.034$" for p-values, for prose
#   \V{nj:<key>:<field>}    any other scalar field (n, T, G, plus, minus,
#                           holm_p, verdict, status, ...), lists as [a, b]
# A null value (the E1-6 hand-check kappa, not coded yet) is written as the
# word "provisional", never as a number.
NJ_PATH = os.path.join(RES, "analysis_v3", "numbers.json")
# Post hoc extras requested by the iteration-1 writing review (written by
# analysis/v3_iter1_extra.py, run from analysis/run_all_v3.sh). They are kept
# out of numbers.json so the registered outputs are unchanged; here they are
# merged under their own "x." keys, so their macros are "nj:x.<key>...".
X_PATH = os.path.join(RES, "analysis_v3", "iter1_extra.json")
# Exploratory extras for the real-API slice (analysis/v3_real_extra.py, added
# after the R1 results; payment-only simulator subsets). Same treatment as
# iter1_extra.json: not merged into numbers.json, macros "nj:xr.<key>...".
XR_PATH = os.path.join(RES, "analysis_v3", "real_extra.json")
# R2, wording robustness (analysis/r2_wording.py, registered 2026-10-31 as
# section R2 of analysis/preregistration.md; results/r2_wording/raw.jsonl).
# Its keys r2.A.*, r2.B.*, r2.C.* and r2.total:* are merged like real_extra.json,
# so the macros are "nj:r2.<key>...". Cells below the registered 90% rule
# carry a "status" that starts with "incomplete"; emit_r2() never prints them.
R2_PATH = os.path.join(RES, "analysis_v3", "r2_wording.json")
NJ = {}


def _is_p(key, field):
    return field in ("holm_p", "p_bound", "holm_sensitivity") or \
        (field == "value" and re.search(r":(p|holm_sensitivity)$", key))


def _fmt(key, field, v):
    if v is None:
        return "provisional"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}" if abs(v) >= 10000 else str(v)
    if isinstance(v, float):
        if _is_p(key, field):
            return pfmt(v)
        if field in ("value", "d") and re.search(r":d$|diff", key):
            return ("+" + f2(v) if v > 0 else "$-$" + f2(-v)) if v else "0.00"
        return f2(v)
    if isinstance(v, (list, tuple)):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v) and len(v) == 2:
            if _is_p(key, field):
                return f"[{pfmt(v[0])}, {pfmt(v[1])}]"
            return ci_str(*v)
        return tex_text(", ".join(str(x) for x in v))
    if isinstance(v, dict):
        return tex_text(", ".join(f"{a} {b}" for a, b in v.items()))
    return tex_text(v)


# October sensitivity (deviation 8, added after the results, not confirmatory).
# analysis/v3_sens_oct.py once keyed the July v2 cells without the prompt, so
# its sens.oct.<m>.drift.* keys compared October base with July base+careful
# pooled. It now keys by prompt and keeps prompt == "base" (fixed 2026-10-26);
# analysis/v3_sens_oct_fix.py is kept as the regression check and writes
# sens.oct_fix.* (merged by analysis/v3_numbers.py via OPTIONAL_PARTS). Here
# every sens.oct.<m>.drift key must equal its sens.oct_fix twin, or it is
# dropped so no macro can print it; every .check. key must match.
SENS_FIX_PATH = os.path.join(RES, "analysis_v3", "sens_oct_fix.json")
OCT_RAW = prefer_v2(os.path.join(RES, "tier1", "e1_6_oct", "raw.jsonl"))


def load_sens_oct_fix(nums):
    if not any(k.startswith("sens.oct.") for k in nums):
        return
    if not os.path.exists(SENS_FIX_PATH):
        note("WARNING sens_oct_fix.json missing: run bash analysis/run_all_v3.sh")
    else:
        if os.path.exists(OCT_RAW) and os.path.getmtime(SENS_FIX_PATH) < os.path.getmtime(OCT_RAW):
            note("WARNING sens_oct_fix.json is older than results/tier1/e1_6_oct/raw.jsonl: re-run it")
        fix = json.load(open(SENS_FIX_PATH))
        for k, e in fix.items():
            if ".check." in k and isinstance(e, dict) and e.get("matches_sens_oct") is not True:
                note(f"WARNING {k}: fix script does not reproduce the sens.oct value")
            if k not in nums:
                nums[k] = e
    for k in [k for k in nums if re.match(r"sens\.oct\.\w+\.drift\.", k)]:
        twin = nums.get(k.replace("sens.oct.", "sens.oct_fix.", 1), {})
        if not isinstance(twin, dict) or twin.get("value") != nums[k].get("value"):
            note(f"WARNING {k} disagrees with its sens.oct_fix twin: dropped")
            del nums[k]
    rows = [json.loads(l) for l in open(OCT_RAW) if l.strip()] if os.path.exists(OCT_RAW) else []
    put("sens:oct:episodes", len({r["episode_id"] for r in rows}))
    days = sorted({(r.get("started_utc") or "")[:10] for r in rows} - {""})
    months = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]
    if days:
        y, mo, dd = days[0].split("-")
        put("sens:oct:date", f"{int(dd)} {months[int(mo) - 1]} {y}")
        put("sens:oct:days", len(days))


def emit_analysis_v3():
    if not os.path.exists(NJ_PATH):
        note("WARNING analysis_v3/numbers.json not found: run analysis/run_all_v3.sh")
        return
    d = json.load(open(NJ_PATH))
    meta, nums = d.get("meta", {}), d.get("numbers", {})
    load_sens_oct_fix(nums)
    if os.path.exists(X_PATH):
        x = json.load(open(X_PATH))
        assert not set(x) & set(nums), set(x) & set(nums)
        nums.update(x)
        if os.path.getmtime(X_PATH) < os.path.getmtime(NJ_PATH) - 3600:
            note("WARNING iter1_extra.json is older than numbers.json: re-run analysis/run_all_v3.sh")
    else:
        note("WARNING analysis_v3/iter1_extra.json not found: run analysis/run_all_v3.sh")
    if os.path.exists(XR_PATH):
        xr = json.load(open(XR_PATH))
        assert not set(xr) & set(nums), set(xr) & set(nums)
        nums.update(xr)
        rs = os.path.join(RES, "analysis_v3", "real_stripe.json")
        if os.path.exists(rs) and os.path.getmtime(XR_PATH) < os.path.getmtime(rs):
            note("WARNING real_extra.json is older than real_stripe.json: re-run analysis/run_all_v3.sh")
    elif os.path.exists(os.path.join(RES, "real_stripe", "raw.jsonl")):
        note("WARNING analysis_v3/real_extra.json not found: run analysis/run_all_v3.sh")
    r2_path = os.path.join(RES, "analysis_v3", "r2_wording.json")   # RES at call time (tests)
    if os.path.exists(r2_path):
        r2 = json.load(open(r2_path)).get("numbers", {})
        # the registered 90% rule: an incomplete cell is dropped here, so no
        # macro can print it and \V{nj:r2...} of such a cell renders as PENDING
        r2 = {k: e for k, e in r2.items()
              if not (isinstance(e, dict) and str(e.get("status", "")).startswith("incomplete"))}
        assert not set(r2) & set(nums), set(r2) & set(nums)
        nums.update(r2)
        r2raw = os.path.join(RES, "r2_wording", "raw.jsonl")
        if os.path.exists(r2raw) and os.path.getmtime(r2_path) < os.path.getmtime(r2raw):
            note("WARNING r2_wording.json is older than results/r2_wording/raw.jsonl: re-run analysis/r2_wording.py")
    elif os.path.exists(os.path.join(RES, "r2_wording", "raw.jsonl")):
        note("WARNING analysis_v3/r2_wording.json not found: run analysis/r2_wording.py")
    # R3 (cross-provider wording), merged exactly like R2: keys r3.A.*, r3.B.*,
    # r3.total:*; cells and verdicts below the registered 90% rule are dropped,
    # so \V{nj:r3...} of such a cell renders as PENDING and emit_r3() reads a
    # missing verdict as "not evaluable"
    r3_path = os.path.join(RES, "analysis_v3", "r3_wording.json")
    if os.path.exists(r3_path):
        r3 = json.load(open(r3_path)).get("numbers", {})
        r3 = {k: e for k, e in r3.items()
              if not (isinstance(e, dict) and str(e.get("status", "")).startswith("incomplete"))}
        assert not set(r3) & set(nums), set(r3) & set(nums)
        nums.update(r3)
        r3raw = os.path.join(RES, "r3_wording", "raw.jsonl")
        if os.path.exists(r3raw) and os.path.getmtime(r3_path) < os.path.getmtime(r3raw):
            note("WARNING r3_wording.json is older than results/r3_wording/raw.jsonl: re-run analysis/r3_wording.py")
    elif os.path.exists(os.path.join(RES, "r3_wording", "raw.jsonl")):
        note("WARNING analysis_v3/r3_wording.json not found: run analysis/r3_wording.py")
    # R4 (outcome unknown before commit), merged like R3: keys r4.<model>.*,
    # r4.total:*; a cell below the registered 90% rule is dropped, so its
    # \V{nj:r4...} renders as PENDING and emit_r4() reads its verdict as "not evaluable"
    r4_path = os.path.join(RES, "r4_precommit", "analysis", "r4_precommit.json")
    if os.path.exists(r4_path):
        r4 = json.load(open(r4_path)).get("numbers", {})
        r4 = {k: e for k, e in r4.items()
              if not (isinstance(e, dict) and str(e.get("status", "")).startswith("incomplete"))}
        assert not set(r4) & set(nums), set(r4) & set(nums)
        nums.update(r4)
        r4raw = os.path.join(RES, "r4_precommit", "raw.jsonl")
        if os.path.exists(r4raw) and os.path.getmtime(r4_path) < os.path.getmtime(r4raw):
            note("WARNING r4_precommit.json is older than results/r4_precommit/raw.jsonl: re-run analysis/r4_precommit.py")
    elif os.path.exists(os.path.join(RES, "r4_precommit", "raw.jsonl")):
        note("WARNING r4_precommit/analysis/r4_precommit.json not found: run analysis/r4_precommit.py")
    NJ.update(nums)
    # the snapshot must have been computed from the result files on disk now
    stale = []
    for rel, h in (meta.get("input_sha256_16") or {}).items():
        p = os.path.join(ROOT, rel)
        if os.path.exists(p) and hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] != h:
            stale.append(rel)
    put("nj:meta:stale_inputs", len(stale))
    for rel in stale:
        note(f"WARNING analysis_v3/numbers.json is older than {rel}: re-run analysis/run_all_v3.sh")
    put("nj:meta:n", len(nums))
    put("nj:meta:perm_n", f"{meta.get('perm_n', 0):,}")
    put("nj:meta:perm_seed", meta.get("perm_seed", ""))
    put("nj:meta:handcheck_seed", meta.get("handcheck_seed", ""))
    skip = {"source", "ci_method", "stat", "note", "criterion", "reading", "unit"}
    for key, e in nums.items():
        base = "nj:" + key
        if not isinstance(e, dict):
            put(base, _fmt(key, "value", e))
            continue
        put(base, _fmt(key, "value", e.get("value")))
        if isinstance(e.get("value"), float) and re.search(r":d$|diff", key):
            put(base + ":abs", f2(abs(e["value"])))
        if key.endswith(":p_bound") and isinstance(e.get("value"), list) and len(e["value"]) == 2:
            put(base + ":lo", pfmt(e["value"][0])); put(base + ":hi", pfmt(e["value"][1]))
        if "k" in e and "n" in e:
            put(base + ":kn", f"{e['k']}/{e['n']}")
        elif isinstance(e.get("value"), float) and isinstance(e.get("n"), int) and \
                0 <= e["value"] <= 1 and not _is_p(key, "value") and \
                not re.search(r":d$|diff", key) and \
                abs(e["value"] * e["n"] - round(e["value"] * e["n"])) < 0.01:
            put(base + ":kn", f"{round(e['value'] * e['n'])}/{e['n']}")
        if isinstance(e.get("verdict"), str):
            vt = e["verdict"]
            short = ("reversed" if "reversed" in vt else
                     "not supported" if vt.startswith("not supported") else
                     "supported" if vt.startswith("supported") else
                     "null" if vt.startswith("null") else
                     "reduced" if "reduced" in vt else
                     "no change" if "no change" in vt else vt.split(" (")[0])
            put(base + ":v", short)
        for fld, v in e.items():
            if fld in skip or fld == "value":
                continue
            if isinstance(v, (dict, list)) and fld not in ("ci", "p_bound", "d_bound", "codes",
                                                           "retry_key_categories"):
                continue
            put(f"{base}:{fld}", _fmt(key, fld, v))
        if _is_p(key, "value") and isinstance(e.get("value"), float):
            v = e["value"]
            put(base + ":pe", "$p<0.001$" if v < 0.001 else f"$p={pfmt(v)}$")
        if isinstance(e.get("holm_p"), float):
            v = e["holm_p"]
            put(base + ":holm_pe", "$p<0.001$" if v < 0.001 else f"$p={pfmt(v)}$")
        for bf in ("p_bound", "d_bound"):
            if isinstance(e.get(bf), list):
                put(f"{base}:{bf}", _fmt(key, bf, e[bf]))
                if len(e[bf]) == 2 and all(isinstance(x, (int, float)) for x in e[bf]):
                    fm = pfmt if bf == "p_bound" else f2
                    put(f"{base}:{bf}:lo", fm(e[bf][0])); put(f"{base}:{bf}:hi", fm(e[bf][1]))
        # one macro per entry of a count dictionary (key categories, codes,
        # per-template or per-k counts): nj:<key>:<field>:<entry>
        for fld, v in e.items():
            if isinstance(v, dict):
                pre = base if fld == "value" else f"{base}:{fld}"
                for sub, sv in v.items():
                    if isinstance(sv, (int, float, str)) and not isinstance(sv, bool):
                        put(f"{pre}:{sub}", sv if isinstance(sv, (int, str)) else f2(sv))
    disp = {k: dd for _, k, dd in MODEL_SHORT}
    order = [k for _, k, _ in MODEL_SHORT]
    def names(ms):
        ns = [disp[m] for m in order if m in ms]
        return "none" if not ns else ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]
    # E1-1 P (per model): which models met the registered >= 0.20 gap
    cat = defaultdict(list)
    for m in order:
        e = nums.get(f"e1_1.P.H2.{m}:p")
        if isinstance(e, dict) and "verdict" in e:
            v = e["verdict"]
            c = ("reversed" if "reversed" in v else "supported" if v.startswith("supported")
                 else "null" if v.startswith("null") else "margin")
            cat[c].append(m)
    for c in ("supported", "reversed", "null", "margin"):
        put(f"e11:{c}:names", names(cat[c])); put(f"e11:{c}:n", len(cat[c]))
    put("e11:n", sum(len(v) for v in cat.values()))
    # bare_hint vs same-month amnesiac, pooled, from the analysis entries
    # (Claude: October batch; OpenRouter models: their one batch). The one
    # Sonnet episode refused before the seam is excluded, as the analysis does.
    ak = an = hk = hn = 0
    for m in order:
        b = "oct" if m in ("haiku", "sonnet", "fable") else "xp"
        a, h = nums.get(f"bh.{m}.{b}.amnesiac:givr"), nums.get(f"bh.{m}.{b}.bare_hint:givr")
        if a and h:
            ak += a["k"]; an += a["n"]; hk += h["k"]; hn += h["n"]
    if an:
        put("smnj:all:amn:kn", f"{ak}/{an}"); put("smnj:all:hint:kn", f"{hk}/{hn}")
    # per-model significance of the same-month hint contrast
    sig, ns = [], []
    for m in order:
        v = NUMS.get(f"test:{m}:hint:p")
        if v is None or v.startswith("\\PENDING"):
            continue
        (sig if (v.startswith("<") or float(v) < 0.05) else ns).append(m)
    put("hint:sig:names", names(sig)); put("hint:sig:n", len(sig))
    put("hint:ns:names", names(ns)); put("hint:ns:n", len(ns))
    ps = [NUMS[f"test:{m}:hint:p"] for m in sig]
    if ps:
        w = max(ps, key=lambda x: 0 if x.startswith("<") else float(x))
        put("hint:sig:pmaxe", "$p<0.001$" if w.startswith("<") else f"$p={w}$")
    # E1-4 / E1-5 registered P: how many of the four cells met it
    for ex, pat in (("e14", r"e1_4\.(\w+)\.(toA|e500)\.P\.new_within_released_ci"),
                    ("e15", r"e1_5\.(\w+)\.(toA|e500)\.P\.within_e1_4_pooled96_ci")):
        cells = [(k, v) for k, v in nums.items() if re.fullmatch(pat, k)]
        put(f"{ex}:P:cells", len(cells))
        put(f"{ex}:P:supported", sum(1 for _, v in cells if v.get("value") is True))
        fails = [k for k, v in cells if v.get("value") is not True]
        put(f"{ex}:P:failed", ", ".join(re.fullmatch(pat, k).group(1) + "/" +
                                         re.fullmatch(pat, k).group(2) for k in fails) or "none")
    # do registered verdicts survive Holm within the experiment?
    def keeps(prefix):
        ok = True
        for k, v in nums.items():
            if k.startswith(prefix) and k.endswith(":holm_sensitivity"):
                raw = nums.get(k.replace(":holm_sensitivity", ":p"), {}).get("value")
                if raw is None:
                    continue
                if (raw < 0.05) != (v["value"] < 0.05):
                    ok = False
        return "yes" if ok else "no"
    put("e16:holm_keeps", keeps("e1_6.P"))
    put("e11:holm_keeps", keeps("e1_1.P"))
    # refusal bounds: does any test's significance change between the
    # primary coding and either bound?
    flips = 0
    for k, e in nums.items():
        pb = e.get("p_bound") if isinstance(e, dict) else None
        if isinstance(pb, list) and isinstance(e.get("value"), float):
            sig = [x < 0.05 for x in [e["value"]] + pb]
            flips += len(set(sig)) > 1
    put("refusal:verdict_flips", flips)
    # Tier-1 episodes on disk, from the analysis hygiene counts
    t1 = sum(int(nums.get(f"hygiene.e1_{i}.rows", {}).get("value", 0)) for i in range(7))
    put("total:tier1", f"{t1:,}")
    # E1-6 belief coding: provisional until the hand check exists
    kap = nums.get("e1_6.belief.kappa", {})
    put("nj:belief:provisional", "yes" if kap.get("value") is None else "no")
    if kap.get("value") is None:
        put("belief:audit", "rule-based and unaudited")
        put("belief:audit:short", "rule-based, unaudited")
    else:
        put("belief:audit", "rule-based, checked against a hand coding of %s windows "
            "($\\kappa=%s$)" % (kap.get("n", "?"), f2(kap["value"])))
        put("belief:audit:short", "$\\kappa=%s$" % f2(kap["value"]))
    hc = os.path.join(RES, "analysis_v3", "e1_6_handcheck.csv")
    if os.path.exists(hc):
        rows = list(csv.DictReader(open(hc)))
        put("nj:belief:handcheck_rows", len(rows))
        put("nj:belief:handcheck_done", sum(1 for r in rows if (r.get("human_code") or "").strip()))
    # the further 60 windows prepared to reach the registered ~100 (deviations 10 and 16)
    hx = os.path.join(RES, "analysis_v3", "e1_6_handcheck_extra60.csv")
    if os.path.exists(hx):
        rows = list(csv.DictReader(open(hx)))
        put("nj:belief:handcheck_extra", len(rows))
        put("nj:belief:handcheck_extra_done", sum(1 for r in rows if (r.get("human_code") or "").strip()))
    # refusals: count of provider blocks over every file the analysis read
    put("nj:refusal:total", int(nums.get("refusal.xp.episodes", {}).get("value", 0)) +
        int(nums.get("refusal.tier1.episodes", {}).get("value", 0)))
    put("nj:refusal:tasks_n", len(nums.get("refusal.all.tasks", {}).get("value") or []))
    # receipts never cost task success (min success over the receipt cells)
    rs = [float(v) for k, v in NUMS.items() if re.fullmatch(r"(jul|xp):\w+:\w+:rcpt:\w+:succ", k)]
    if rs:
        put("rcpt:succ_min", f2(min(rs))); put("rcpt:cells", len(rs))
    # E1-2: cells with a spontaneous duplicate inside delivery 1
    d1 = [k for k, e in nums.items() if re.fullmatch(r"e1_2\.\w+\.\w+\.\w+\.none:d1_spontaneous_dup", k)]
    put("e12:d1:cells", len(d1))
    put("e12:d1:nonzero_n", sum(1 for k in d1 if (nums[k].get("value") or 0) > 0))
    # Real-API validation slice (preregistration R1, Stripe test mode). The
    # paper prints its sentences only once every planned episode has a row
    # (\IfReal in main.tex reads real:ready:yes). Until then the keys the
    # sentences use are defined as "--" so the lint does not count them, and
    # the response letter says the run is in progress.
    ne = nums.get("real.n_episodes", {})
    ready = isinstance(ne, dict) and bool(ne.get("planned")) and ne.get("value", 0) >= ne["planned"]
    put("real:ready", "yes" if ready else "no")
    if ready:
        put("real:ready:yes", "1")
    for k in ("nj:real.n_episodes", "nj:real.H1.within", "nj:real.H1.within:n",
              "nj:real.agreement.ledger_vs_stripe:kn"):
        if k not in NUMS:
            put(k, "--")
    # R1-H1 in words: did fewer than half of the registered cells contain the real rate?
    h1 = nums.get("real.H1.within", {})
    if isinstance(h1, dict) and isinstance(h1.get("value"), int) and h1.get("n"):
        put("nj:real.H1.within:lt_half", "yes" if 2 * h1["value"] < h1["n"] else "no")
    put("r1:status", "has finished as well" if ready else "is still running")
    # R1-H1 asks for d = real - simulator per registered comparison (Table
    # tab:real); the raw field prints unsigned, so a signed copy goes to :dsg.
    for key, e in nums.items():
        if key.startswith("real.") and isinstance(e, dict) and "sim_key" in e:
            v = e.get("d")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                put("nj:" + key + ":dsg", "0.00" if f2(abs(v)) == "0.00" else
                    ("+" if v > 0 else "$-$") + f2(abs(v)))
            else:
                put("nj:" + key + ":dsg", "--")
    if ready:
        put("r1:resp", "Pre-registered R1 (registered before any real-API data) ran %s episodes "
            "on Stripe test mode with Haiku 4.5, Sonnet 5 and Qwen3.6-27B: the real-API IVR fell "
            "inside the simulator's task-level interval in %s of %s cells, and the simulator's verdict "
            "on the same calls matched Stripe's record in %s episodes (Limitations)."
            % (NUMS["nj:real.n_episodes"], NUMS["nj:real.H1.within"], NUMS["nj:real.H1.within:n"],
               NUMS["nj:real.agreement.ledger_vs_stripe:kn"]))
    else:
        put("r1:resp", "A pre-registered real-API check on Stripe test mode (R1, "
            "registered before any of its data; Haiku 4.5, Sonnet 5 and Qwen3.6-27B, payments only) "
            "is still running; its results will be added to the paper and to this answer when it "
            "finishes.")


def emit_r2():
    """Derived R2 macros (wording robustness): verdict words, the paraphrase
    ranges, the same-batch comparisons the main text states in words, episode
    and spend totals, and the data window. Registration timing is not read from
    git here (the anonymized artifact has no history); the registration date
    comes from the section heading of analysis/preregistration.md, and git is
    only consulted to WARN if the commit does not precede the first R2 episode."""
    nums = NJ
    if not any(k.startswith("r2.") for k in nums):
        return
    def e(k):
        return nums.get(k, {}) if isinstance(nums.get(k), dict) else {}
    def kn(k):
        x = e(k)
        return f"{x['k']}/{x['n']}" if "k" in x and "n" in x else None
    paras = ("p1", "p2", "p3")
    # Part A (restart wording): verdict per model, and whether every hinted
    # wording of a model gave zero duplicates and a session-2 log lookup
    for m in ("haiku", "fable"):
        v = e(f"r2.A.{m}.H1:supported").get("value")
        put(f"r2:A:{m}:verdict", "supported" if v is True else "not supported" if v is False
            else "not evaluable")
        cells = [f"r2.A.{m}.bare_hint:givr"] + [f"r2.A.{m}.hint_{p}:givr" for p in paras]
        have = [c for c in cells if "k" in e(c)]
        put(f"r2:A:{m}:hinted_cells", len(have))
        put(f"r2:A:{m}:hinted_all_zero", "yes" if have and all(e(c)["k"] == 0 for c in have) else "no")
        looks = [f"r2.A.{m}.bare_hint:session2_lookup"] + \
                [f"r2.A.{m}.hint_{p}:session2_lookup" for p in paras]
        looks = [c for c in looks if "k" in e(c)]
        put(f"r2:A:{m}:hinted_all_looked", "yes" if looks and
            all(e(c)["k"] == e(c)["n"] for c in looks) else "no")
        sig = e(f"r2.A.{m}.H1:supported").get("n_significant")
        if isinstance(sig, int):
            put(f"r2:A:{m}:nsig", sig)
    # Part B (error wording): verdict per model, paraphrase range, and the
    # registered secondary contrasts against the same-batch timeout
    for m in ("haiku", "sonnet"):
        v = e(f"r2.B.{m}.H2:supported").get("value")
        put(f"r2:B:{m}:verdict", "supported" if v is True else "not supported" if v is False
            else "not evaluable")
        sig = e(f"r2.B.{m}.H2:supported").get("n_significant")
        if isinstance(sig, int):
            put(f"r2:B:{m}:nsig", sig)
        cells = {p: e(f"r2.B.{m}.unknown_{p}:ivr") for p in paras}
        if all("k" in c for c in cells.values()):
            lo = min(cells.values(), key=lambda c: c["k"]); hi = max(cells.values(), key=lambda c: c["k"])
            put(f"r2:B:{m}:par:min:kn", f"{lo['k']}/{lo['n']}"); put(f"r2:B:{m}:par:max:kn", f"{hi['k']}/{hi['n']}")
            put(f"r2:B:{m}:par:min:ivr", f2(lo["value"])); put(f"r2:B:{m}:par:max:ivr", f2(hi["value"]))
            put(f"r2:B:{m}:par:all_zero", "yes" if hi["k"] == 0 else "no")
            put(f"r2:B:{m}:par:max_k", hi["k"])
        # every unknown wording (W3 and the paraphrases) below the same-batch
        # timeout (d < 0, two-sided p < 0.05), or equal to it (d == 0 for all)
        # Holm-adjusted within the registered secondary block where the analysis
        # provides it (r2_wording.json from commit b655b89 on), else the raw p
        ds = [e(f"r2.B.{m}.{w}_vs_toA:d").get("value") for w in ("w3",) + tuple("unknown_" + p for p in paras)]
        ps = [e(f"r2.B.{m}.{w}_vs_toA:p").get("holm_p", e(f"r2.B.{m}.{w}_vs_toA:p").get("value"))
              for w in ("w3",) + tuple("unknown_" + p for p in paras)]
        if all(isinstance(d, float) for d in ds) and all(isinstance(p, float) for p in ps):
            put(f"r2:B:{m}:below_timeout:all", "yes" if all(d < 0 and p < 0.05 for d, p in zip(ds, ps)) else "no")
            put(f"r2:B:{m}:equal_timeout:all", "yes" if all(d == 0 for d in ds) else "no")
            put(f"r2:B:{m}:below_timeout:n", sum(1 for d, p in zip(ds, ps) if d < 0 and p < 0.05))
        # every unknown wording below the same-batch 500 (the registered direction)
        ds = [e(f"r2.B.{m}.unknown_{p}_vs_e500:d").get("value") for p in paras]
        if all(isinstance(d, float) for d in ds):
            put(f"r2:B:{m}:below_500:all", "yes" if all(d < 0 for d in ds) else "no")
        # read-back under the four unknown wordings: min and max
        rbs = [e(f"r2.B.{m}.{w}:readback_after_failure") for w in ("w3",) + tuple("unknown_" + p for p in paras)]
        if all("k" in r for r in rbs):
            put(f"r2:B:{m}:rb:min:kn", "%d/%d" % (min(r["k"] for r in rbs), rbs[0]["n"]))
            put(f"r2:B:{m}:rb:max:kn", "%d/%d" % (max(r["k"] for r in rbs), rbs[0]["n"]))
    # the largest Holm-adjusted p over a model's three paraphrases (primary tests)
    for part, m, base in (("A", "haiku", "hint_{}_vs_amnesiac"), ("A", "fable", "hint_{}_vs_amnesiac"),
                          ("B", "haiku", "unknown_{}_vs_e500"), ("B", "sonnet", "unknown_{}_vs_e500")):
        hp = [e(f"r2.{part}.{m}." + base.format(p) + ":p").get("holm_p") for p in paras]
        if all(isinstance(x, float) for x in hp):
            v = max(hp)
            put(f"r2:{part}:{m}:holm_pmax_e", "$p<0.001$" if v < 0.001 else f"$p={pfmt(v)}$")
    # the registered wordings, read from the harness source by regex (as the
    # schema texts are in emit_design); printed in Appendix "Prompts"
    def grab_src(rel, pat):
        try:
            m = re.search(pat, open(os.path.join(ROOT, rel)).read(), re.S)
        except OSError:
            return None
        return re.sub(r"\s+", " ", "".join(re.findall(r'"([^"]*)"', m.group(1)))) if m else None
    for p in paras:
        t = grab_src("src2/sweep2.py", r'"hint_%s":\s*((?:"[^"]*"\s*)+),' % p)
        if t:
            put(f"design:r2:hint_{p}", tex_text(t))
        t = grab_src("src/world.py", r'"status_unknown_%s_after_commit":\s*\(\s*"timeout",\s*((?:"[^"]*"\s*)+)\)' % p)
        if t:
            put(f"design:r2:unknown_{p}", tex_text(t))
    # how many model-level verdicts of R2-H1 and R2-H2 are supported / evaluable
    vs = [NUMS.get(k) for k in ("r2:A:haiku:verdict", "r2:A:fable:verdict",
                                 "r2:B:haiku:verdict", "r2:B:sonnet:verdict")]
    put("r2:verdicts:n", len(vs))
    put("r2:verdicts:supported", sum(1 for v in vs if v == "supported"))
    put("r2:verdicts:not_evaluable", sum(1 for v in vs if v == "not evaluable"))
    put("r2:verdicts:evaluable", sum(1 for v in vs if v in ("supported", "not supported")))
    put("r2:paraphrases:n", len(paras))
    # totals: episodes on disk, planned, spend (from the analysis output; the
    # rows are summed again in emit_cost)
    t = e("r2.total:episodes")
    if "value" in t:
        put("r2:episodes", f"{int(t['value']):,}")
        if "total:all" in NUMS:   # simulated episodes including R2 (abstract, section 1)
            put("total:all_r2", f"{int(str(NUMS['total:all']).replace(',', '')) + int(t['value']):,}")
        if t.get("planned"):
            put("r2:planned", f"{int(t['planned']):,}")
    # Part C did not run (credit ran out): every Part-C cell is incomplete, so
    # none is in nums; say so with a macro rather than a hand-typed claim
    c_any = any(k.startswith("r2.C.") and k.endswith("plan_only:givr") for k in nums)
    put("r2:C:ran", "yes" if c_any else "no")
    # data window and registration date (heading of the R2 section)
    raw = os.path.join(RES, "r2_wording", "raw.jsonl")
    if os.path.exists(raw):
        st = [json.loads(l).get("started_utc") for l in open(raw) if l.strip()]
        st = [x for x in st if x]
        if st:
            put("r2:first_data_utc", min(st).replace("T", " ")[:16] + " UTC")
            put("r2:last_data_utc", max(st).replace("T", " ")[:16] + " UTC")
    pr = os.path.join(ROOT, "analysis", "preregistration.md")
    if os.path.exists(pr):
        m = re.search(r"^## R2\..*registered (\d{4})-(\d{2})-(\d{2}), before any R2 data", open(pr).read(), re.M)
        if m:
            months = ["January", "February", "March", "April", "May", "June", "July", "August",
                      "September", "October", "November", "December"]
            put("r2:reg_date", f"{int(m.group(3))} {months[int(m.group(2)) - 1]} {m.group(1)}")
            put("r2:reg_dm", f"{int(m.group(3))} {months[int(m.group(2)) - 1]}")
            put("r2:reg_iso", "-".join(m.groups()))
        try:
            reg = subprocess.run(["git", "-C", ROOT, "log", "--diff-filter=A", "--format=%cI",
                                  "--", "configs/r2_wording.json"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:
            reg = []
        if not reg and os.path.exists(os.path.join(HERE, "provenance.json")):
            # anonymized artifact: no git history; the build froze the commit time
            reg = json.load(open(os.path.join(HERE, "provenance.json"))).get("r2_registered_utc") or []
        if reg and "r2:first_data_utc" in NUMS:
            from datetime import datetime, timezone
            a = datetime.fromisoformat(reg[-1]).astimezone(timezone.utc)
            z = datetime.fromisoformat(min(st).replace("Z", "+00:00")).astimezone(timezone.utc)
            if not a < z:
                note("WARNING R2 runner config was committed after the first R2 episode")
            # registration commit (the commit that added the runner config and the R2
            # section) against the first episode, to the minute (Section sec:metrics)
            put("r2:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC")
            put("r2:commit_hm", a.strftime("%H:%M") + " UTC")
            put("r2:first_data_hm", z.strftime("%H:%M") + " UTC")
            put("r2:before_data", "yes" if a < z else "no")


def emit_r3():
    """Derived R3 macros (cross-provider wording, registered 2026-11-01 as
    section R3 of analysis/preregistration.md): verdict words per model and
    hypothesis, the registered floor reading (a "not supported" R3-H2 for a
    model whose same-batch bare-500 IVR is <= 0.10 is read as "no duplication
    to remove", not as a failure of the wording), the same-batch comparisons
    the text states in words, counts of supported / evaluable verdicts,
    episode and spend totals, and the registration-versus-data timing, as
    for R2. A verdict the analysis dropped under the 90% rule is "not
    evaluable" here, so every sentence built on these macros stays true
    while the last R3 cells are still being completed."""
    nums = NJ
    if not any(k.startswith("r3.") for k in nums):
        return
    def e(k):
        return nums.get(k, {}) if isinstance(nums.get(k), dict) else {}
    disp = {k: d for _, k, d in MODEL_SHORT}
    order = [k for _, k, _ in MODEL_SHORT]
    def names(ms):
        ns = [disp[m] for m in order if m in ms]
        return "none" if not ns else ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]
    def verdict(v):
        return "supported" if v is True else "not supported" if v is False else "not evaluable"
    FLOOR = 0.10                     # registered: same-batch baseline IVR / gIVR <= 0.10
    put("r3:floor", f2(FLOOR))
    A_MODELS, B_MODELS = ("gpt", "gemini", "qwen"), ("gpt", "gemini", "qwen", "fable")
    notes = ("bare_hint", "hint_p1", "hint_p3")
    unknown = ("w3", "unknown_p1", "unknown_p3")
    put("r3:A:notes:n", len(notes)); put("r3:B:wordings:n", len(unknown))
    # ---- R3-A (restart note vs same-batch amnesiac), per model
    cat = defaultdict(list)
    for m in A_MODELS:
        v = e(f"r3.A.{m}.H1:supported").get("value")
        put(f"r3:A:{m}:verdict", verdict(v))
        sig = e(f"r3.A.{m}.H1:supported").get("n_significant")
        if isinstance(sig, int):
            put(f"r3:A:{m}:nsig", sig)
        base = e(f"r3.A.{m}.amnesiac:givr")
        if "value" in base:
            put(f"r3:A:{m}:floor", "yes" if base["value"] <= FLOOR else "no")
        cells = [e(f"r3.A.{m}.{c}:givr") for c in notes]
        have = [c for c in cells if "k" in c]
        # planned minus achieved episodes over the model's four R3-A cells (deviation 15)
        allc = [base] + cells
        if all("planned" in c and "n" in c for c in allc):
            put(f"r3:A:{m}:missing", sum(int(c["planned"]) - int(c["n"]) for c in allc))
        put(f"r3:A:{m}:note_cells", len(have))
        put(f"r3:A:{m}:notes_complete", "yes" if len(have) == len(notes) else "no")
        if have:
            hi = max(have, key=lambda c: c["k"])
            put(f"r3:A:{m}:hinted_all_zero", "yes" if hi["k"] == 0 else "no")
            put(f"r3:A:{m}:hinted_max:kn", f"{hi['k']}/{hi['n']}")
            put(f"r3:A:{m}:hinted_max_k", hi["k"])
            put(f"r3:A:{m}:hinted_max:givr", f2(hi["value"]))
        looks = [e(f"r3.A.{m}.{c}:session2_lookup") for c in notes]
        looks = [c for c in looks if "k" in c]
        if looks:
            put(f"r3:A:{m}:hinted_all_looked", "yes" if all(c["k"] == c["n"] for c in looks) else "no")
            lo = min(looks, key=lambda c: c["value"])
            put(f"r3:A:{m}:looked_min:kn", f"{lo['k']}/{lo['n']}")
        hp = [e(f"r3.A.{m}.{c}_vs_amnesiac:p").get("holm_p") for c in notes]
        if all(isinstance(x, float) for x in hp):
            put(f"r3:A:{m}:holm_pmax_e", "$p<0.001$" if max(hp) < 0.001 else f"$p={pfmt(max(hp))}$")
        cat[NUMS[f"r3:A:{m}:verdict"]].append(m)
    for c, lab in (("supported", "supported"), ("not supported", "notsupported"),
                   ("not evaluable", "notevaluable")):
        put(f"r3:A:{lab}:n", len(cat[c])); put(f"r3:A:{lab}:names", names(cat[c]))
    put("r3:A:evaluable:n", len(cat["supported"]) + len(cat["not supported"]))
    put("r3:A:models:n", len(A_MODELS))
    # same-batch R2 500 vs timeout (registered S2), detectable after Holm (Sections 7.1, 8)
    for m in ("haiku", "sonnet"):
        hp = e(f"r2.B.{m}.e500_vs_toA:p").get("holm_p")
        if isinstance(hp, float):
            put(f"r2:B:{m}:e500_vs_toA:sig", "yes" if hp < 0.05 else "no")
    # ---- R3-B (unknown wordings vs same-batch bare 500), per model
    cat = defaultdict(list); floor_models = []; at_timeout = []
    for m in B_MODELS:
        v = e(f"r3.B.{m}.H2:supported").get("value")
        put(f"r3:B:{m}:verdict", verdict(v))
        sig = e(f"r3.B.{m}.H2:supported").get("n_significant")
        if isinstance(sig, int):
            put(f"r3:B:{m}:nsig", sig)
        base = e(f"r3.B.{m}.e500:ivr")
        if "value" in base:
            fl = base["value"] <= FLOOR
            put(f"r3:B:{m}:floor", "yes" if fl else "no")
            if fl:
                floor_models.append(m)
        # the registered reading of the verdict: "no duplication to remove"
        # when the model is at the floor and the hypothesis is not supported
        put(f"r3:B:{m}:reading", "floor" if (v is False and NUMS.get(f"r3:B:{m}:floor") == "yes")
            else verdict(v))
        cells = [e(f"r3.B.{m}.{c}:ivr") for c in unknown]
        have = [c for c in cells if "k" in c]
        put(f"r3:B:{m}:unknown_cells", len(have))
        put(f"r3:B:{m}:unknown_complete", "yes" if len(have) == len(unknown) else "no")
        if have:
            lo = min(have, key=lambda c: c["k"]); hi = max(have, key=lambda c: c["k"])
            put(f"r3:B:{m}:par:min:kn", f"{lo['k']}/{lo['n']}"); put(f"r3:B:{m}:par:max:kn", f"{hi['k']}/{hi['n']}")
            put(f"r3:B:{m}:par:min:ivr", f2(lo["value"])); put(f"r3:B:{m}:par:max:ivr", f2(hi["value"]))
            put(f"r3:B:{m}:par:max_k", hi["k"])
            put(f"r3:B:{m}:par:all_zero", "yes" if hi["k"] == 0 else "no")
        # each complete unknown wording against the same-batch timeout (registered S1,
        # Holm within the block): all below it, all equal to it, or none detectably different
        ds = [e(f"r3.B.{m}.{c}_vs_toA:d").get("value") for c in unknown if "k" in e(f"r3.B.{m}.{c}:ivr")]
        ps = [e(f"r3.B.{m}.{c}_vs_toA:p").get("holm_p", e(f"r3.B.{m}.{c}_vs_toA:p").get("value"))
              for c in unknown if "k" in e(f"r3.B.{m}.{c}:ivr")]
        if ds and all(isinstance(d, float) for d in ds) and all(isinstance(p, float) for p in ps):
            below = all(d < 0 and p < 0.05 for d, p in zip(ds, ps))
            put(f"r3:B:{m}:below_timeout:all", "yes" if below else "no")
            put(f"r3:B:{m}:below_timeout:n", sum(1 for d, p in zip(ds, ps) if d < 0 and p < 0.05))
            put(f"r3:B:{m}:equal_timeout:all", "yes" if all(d == 0 for d in ds) else "no")
            put(f"r3:B:{m}:timeout_level", "below" if below else
                "equal" if all(d == 0 for d in ds) else
                "at" if all(p >= 0.05 for p in ps) else "mixed")
            if NUMS[f"r3:B:{m}:timeout_level"] in ("at", "equal"):
                at_timeout.append(m)
        ds = [e(f"r3.B.{m}.{c}_vs_e500:d").get("value") for c in unknown]
        if all(isinstance(d, float) for d in ds):
            put(f"r3:B:{m}:below_500:all", "yes" if all(d < 0 for d in ds) else "no")
        # same-batch 500 vs timeout (registered S2): direction word for the text
        d = e(f"r3.B.{m}.e500_vs_toA:d").get("value")
        if isinstance(d, float):
            put(f"r3:B:{m}:e500_vs_toA:dir", "above" if d > 0 else "below" if d < 0 else "equal to")
        hp = e(f"r3.B.{m}.e500_vs_toA:p").get("holm_p")
        if isinstance(hp, float):   # detectable after Holm within the block (Section 7.1)
            put(f"r3:B:{m}:e500_vs_toA:sig", "yes" if hp < 0.05 else "no")
        if m == "fable":
            # deviation 17: were W3 and unknown_p1 significant under Holm over three
            # whatever the third (unknown_p3) cell gave? Their Holm-adjusted p is at
            # most 3 x the larger raw p for any third p, so 3 * max < 0.05 suffices
            # and the third cell could only move "not evaluable" to "supported"
            pr = [e(f"r3.B.fable.{c}_vs_e500:p").get("value") for c in ("w3", "unknown_p1")]
            if all(isinstance(x, float) for x in pr):
                put("r3:B:fable:third_cell_moot", "yes" if 3 * max(pr) < 0.05 else "no")
        if have and "value" in base:   # some unknown wording above the same-batch bare 500
            put(f"r3:B:{m}:par:max_above_500", "yes" if hi["value"] > base["value"] else "no")
        rbs = [e(f"r3.B.{m}.{c}:readback_after_failure") for c in unknown]
        rbs = [r for r in rbs if "k" in r]
        if rbs:   # the cells' n can differ (a refused or missing episode), so print each cell's own k/n
            lo = min(rbs, key=lambda r: r["value"]); hi = max(rbs, key=lambda r: r["value"])
            put(f"r3:B:{m}:rb:min:kn", f"{lo['k']}/{lo['n']}"); put(f"r3:B:{m}:rb:max:kn", f"{hi['k']}/{hi['n']}")
        hp = [e(f"r3.B.{m}.{c}_vs_e500:p").get("holm_p") for c in unknown]
        if all(isinstance(x, float) for x in hp):
            put(f"r3:B:{m}:holm_pmax_e", "$p<0.001$" if max(hp) < 0.001 else f"$p={pfmt(max(hp))}$")
        cat[NUMS[f"r3:B:{m}:verdict"]].append(m)
    for c, lab in (("supported", "supported"), ("not supported", "notsupported"),
                   ("not evaluable", "notevaluable")):
        put(f"r3:B:{lab}:n", len(cat[c])); put(f"r3:B:{lab}:names", names(cat[c]))
    put("r3:B:evaluable:n", len(cat["supported"]) + len(cat["not supported"]))
    put("r3:B:models:n", len(B_MODELS))
    put("r3:B:floor:n", len(floor_models)); put("r3:B:floor:names", names(floor_models))
    # "not supported" verdicts that are NOT explained by the floor (a failure of the wording)
    fail = [m for m in cat["not supported"] if m not in floor_models]
    put("r3:B:fail:n", len(fail)); put("r3:B:fail:names", names(fail))
    put("r3:B:abovefloor:evaluable:n", len(cat["supported"]) + len(fail))
    put("r3:B:at_timeout:names", names(at_timeout)); put("r3:B:at_timeout:n", len(at_timeout))
    # how many of the seven model-level verdicts are supported / not evaluable
    vs = [NUMS.get(f"r3:A:{m}:verdict") for m in A_MODELS] + [NUMS.get(f"r3:B:{m}:verdict") for m in B_MODELS]
    put("r3:verdicts:n", len(vs))
    put("r3:verdicts:supported", sum(1 for v in vs if v == "supported"))
    put("r3:verdicts:not_supported", sum(1 for v in vs if v == "not supported"))
    put("r3:verdicts:not_evaluable", sum(1 for v in vs if v == "not evaluable"))
    put("r3:verdicts:evaluable", sum(1 for v in vs if v in ("supported", "not supported")))
    # the non-Claude models for which each wording effect is shown (R3), by name
    # the models for which each wording effect is shown with paraphrases, over R2
    # and R3 together (abstract, sections 1, 5 and 8), and the providers they span
    a_ok = [m for m in ("haiku", "fable") if NUMS.get(f"r2:A:{m}:verdict") == "supported"] + \
           [m for m in A_MODELS if NUMS.get(f"r3:A:{m}:verdict") == "supported"]
    b_ok = [m for m in ("haiku", "sonnet") if NUMS.get(f"r2:B:{m}:verdict") == "supported"] + \
           [m for m in B_MODELS if NUMS.get(f"r3:B:{m}:verdict") == "supported"]
    put("wording:A:supported:names", names(a_ok)); put("wording:A:supported:n", len(set(a_ok)))
    put("wording:B:supported:names", names(b_ok)); put("wording:B:supported:n", len(set(b_ok)))
    nonclaude = [m for m in set(a_ok) | set(b_ok) if m in ("gpt", "gemini", "qwen")]
    put("wording:nonclaude:names", names(nonclaude)); put("wording:nonclaude:n", len(nonclaude))
    provider = {"haiku": "Anthropic", "sonnet": "Anthropic", "fable": "Anthropic",
                "gpt": "OpenAI", "gemini": "Google", "qwen": "Alibaba"}
    put("wording:providers:n", len({provider[m] for m in set(a_ok) | set(b_ok)}))
    # providers spanned by each effect on its own (Conclusion)
    put("wording:A:providers:n", len({provider[m] for m in a_ok}))
    put("wording:B:providers:n", len({provider[m] for m in b_ok}))
    # totals: episodes on disk, planned, spend; simulated-episode total with R2 and R3
    t = e("r3.total:episodes")
    if "value" in t:
        put("r3:episodes", f"{int(t['value']):,}")
        if t.get("planned"):
            put("r3:planned", f"{int(t['planned']):,}")
            put("r3:complete", "yes" if int(t["value"]) >= int(t["planned"]) else "no")
            put("r3:missing", f"{int(t['planned']) - int(t['value']):,}")
            # "all 1,664 planned episodes" or "1,580 of 1,664 planned episodes" (Limitations)
            put("r3:coverage", f"all {int(t['planned']):,} planned episodes" if NUMS["r3:complete"] == "yes"
                else f"{int(t['value']):,} of {int(t['planned']):,} planned episodes")
        if "total:all_r2" in NUMS:
            put("total:all_r23", f"{int(str(NUMS['total:all_r2']).replace(',', '')) + int(t['value']):,}")
        if "r2:episodes" in NUMS:   # R2 + R3 episodes (section 1)
            put("r23:episodes", f"{int(str(NUMS['r2:episodes']).replace(',', '')) + int(t['value']):,}")
    r = e("r3.total:refusals")
    if "value" in r:
        put("r3:refusals", int(r["value"]))
        by = r.get("by_cell") or {}
        put("r3:refusals:cells", len(by))
        if by:
            put("r3:refusals:per_cell:min", min(by.values())); put("r3:refusals:per_cell:max", max(by.values()))
        # were the refused R3 episodes all on the task Google blocked in October?
        csvp = os.path.join(RES, "analysis_v3", "r3_wording_refusals.csv")
        oct_tasks = set((nums.get("refusal.all.tasks") or {}).get("value") or [])
        if os.path.exists(csvp) and oct_tasks:
            with open(csvp) as fh:
                units = {row["unit"] for row in csv.DictReader(fh)}
            put("r3:refusals:same_task", "yes" if units and units <= oct_tasks else "no")
    # data window and registration date (heading of the R3 section), as for R2
    raw = os.path.join(RES, "r3_wording", "raw.jsonl")
    st = []
    if os.path.exists(raw):
        st = [json.loads(l).get("started_utc") for l in open(raw) if l.strip()]
        st = [x for x in st if x]
        if st:
            put("r3:first_data_utc", min(st).replace("T", " ")[:16] + " UTC")
            put("r3:last_data_utc", max(st).replace("T", " ")[:16] + " UTC")
    pr = os.path.join(ROOT, "analysis", "preregistration.md")
    if os.path.exists(pr):
        m = re.search(r"^## R3\..*registered (\d{4})-(\d{2})-(\d{2}), before any R3 data", open(pr).read(), re.M)
        if m:
            months = ["January", "February", "March", "April", "May", "June", "July", "August",
                      "September", "October", "November", "December"]
            put("r3:reg_date", f"{int(m.group(3))} {months[int(m.group(2)) - 1]} {m.group(1)}")
            put("r3:reg_dm", f"{int(m.group(3))} {months[int(m.group(2)) - 1]}")
        # the registered OpenRouter spending stop of R3, enforced in code (deviation 15)
        mc = re.search(r"OpenRouter hard stop at \$(\d+)", open(pr).read())
        if mc:
            put("r3:or_cap", f"\\${mc.group(1)}")
            put("r3:reg_iso", "-".join(m.groups()))
        try:
            reg = subprocess.run(["git", "-C", ROOT, "log", "--diff-filter=A", "--format=%cI",
                                  "--", "configs/r3_wording.json"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:
            reg = []
        if not reg and os.path.exists(os.path.join(HERE, "provenance.json")):
            reg = json.load(open(os.path.join(HERE, "provenance.json"))).get("r3_registered_utc") or []
        if reg and st:
            from datetime import datetime, timezone
            a = datetime.fromisoformat(reg[-1]).astimezone(timezone.utc)
            z = datetime.fromisoformat(min(st).replace("Z", "+00:00")).astimezone(timezone.utc)
            if not a < z:
                note("WARNING R3 runner config was committed after the first R3 episode")
            put("r3:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC")
            put("r3:commit_hm", a.strftime("%H:%M") + " UTC")
            put("r3:first_data_hm", z.strftime("%H:%M") + " UTC")
            put("r3:before_data", "yes" if a < z else "no")


def emit_r4():
    """Derived R4 macros (outcome unknown before commit, registered 2026-11-01
    as section R4 of analysis/preregistration.md): the R4-H1 verdict word per
    model (non-inferiority of omission, W3 versus timeout, both before commit),
    supported / not supported names and counts, which models omitted no more
    under W3 than under the timeout, which never duplicated, the episode
    total (and the simulated total with R2, R3 and R4), and the
    registration-versus-data timing, as for R3. A verdict the analysis
    dropped under the 90% rule is "not evaluable" here."""
    nums = NJ
    if not any(k.startswith("r4.") for k in nums):
        return
    def e(k):
        return nums.get(k, {}) if isinstance(nums.get(k), dict) else {}
    disp = {k: d for _, k, d in MODEL_SHORT}
    order = [k for _, k, _ in MODEL_SHORT]
    def names(ms):
        ns = [disp[m] for m in order if m in ms]
        return "none" if not ns else ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]
    MODELS = [m for m in order if any(k.startswith(f"r4.{m}.") for k in nums)]
    CELLS = ("toB", "e500B", "w3B")
    cat = defaultdict(list)
    no_dup, w3_eq_to, e500_more = [], [], []
    for m in MODELS:
        v = e(f"r4.{m}.H1:supported").get("value")
        word = v if v in ("supported", "not supported") else "not evaluable"
        put(f"r4:{m}:verdict", word)
        cat[word].append(m)
        om = {c: e(f"r4.{m}.{c}:omission") for c in CELLS}
        if all("k" in om[c] for c in CELLS):
            put(f"r4:{m}:w3_eq_timeout", "yes" if om["w3B"]["k"] == om["toB"]["k"] else "no")
            if om["w3B"]["k"] == om["toB"]["k"]:
                w3_eq_to.append(m)
            # the bare 500 before commit against the timeout: direction and Holm
            hp = e(f"r4.{m}.omission.e500B_vs_toB:p").get("holm_p")
            if hp is not None:
                put(f"r4:{m}:e500_omits_more", "yes" if om["e500B"]["k"] > om["toB"]["k"] and hp < 0.05 else "no")
                if om["e500B"]["k"] > om["toB"]["k"] and hp < 0.05:
                    e500_more.append(m)
            hp = e(f"r4.{m}.omission.w3B_vs_e500B:p").get("holm_p")
            if hp is not None:   # W3 against the bare 500, both before commit: detectable after Holm?
                put(f"r4:{m}:w3_vs_e500:sig", "yes" if hp < 0.05 else "no")
        iv = [e(f"r4.{m}.{c}:ivr") for c in CELLS]
        if all("k" in c for c in iv):
            put(f"r4:{m}:ivr:all_zero", "yes" if all(c["k"] == 0 for c in iv) else "no")
            put(f"r4:{m}:ivr:max_k", max(c["k"] for c in iv))
            if all(c["k"] == 0 for c in iv):
                no_dup.append(m)
    for word, lab in (("supported", "supported"), ("not supported", "not_supported"),
                      ("not evaluable", "not_evaluable")):
        put(f"r4:{lab}:n", len(cat[word])); put(f"r4:{lab}:names", names(cat[word]))
    put("r4:models:n", len(MODELS)); put("r4:models:names", names(MODELS))
    put("r4:evaluable:n", len(cat["supported"]) + len(cat["not supported"]))
    put("r4:w3_eq_timeout:names", names(w3_eq_to)); put("r4:w3_eq_timeout:n", len(w3_eq_to))
    put("r4:no_dup:names", names(no_dup)); put("r4:no_dup:n", len(no_dup))
    # registered S item: models whose omission under the bare 500 before commit
    # is above the timeout's, detectable after Holm within model (Discussion)
    put("r4:e500_omits_more:names", names(e500_more)); put("r4:e500_omits_more:n", len(e500_more))
    # POST HOC (not registered): per cell, omissions in which the model neither
    # read the log after the failure nor retried, and completion among episodes
    # that did / did not read back (Section 7.1); last row per episode id
    raw4 = os.path.join(RES, "r4_precommit", "raw.jsonl")
    if os.path.exists(raw4):
        FM4 = {"timeout_before_commit": "toB", "error_500_before_commit": "e500B",
               "status_unknown_before_commit": "w3B"}
        last = {}
        for line in open(raw4):
            if line.strip():
                r = json.loads(line)
                last[r["episode_id"]] = r
        acc = defaultdict(lambda: [0, 0, 0, 0, 0, 0])   # om, om_blind, rb, rb_compl, norb, norb_compl
        for r in last.values():
            c = FM4.get(r.get("failure_mode"))
            if c is None or r.get("refusal"):
                continue
            a = acc[(short_model(r["model"]), c)]
            rb = bool(r.get("readback_after_failure"))
            if r.get("omission"):
                a[0] += 1
                a[1] += (not rb) and (not r.get("retried"))
            if rb:
                a[2] += 1; a[3] += bool(r.get("achieved"))
            else:
                a[4] += 1; a[5] += bool(r.get("achieved"))
        for (m, c), a in acc.items():
            put(f"r4:{m}:{c}:omit_blind:k", a[1]); put(f"r4:{m}:{c}:omit:k", a[0])
            put(f"r4:{m}:{c}:rb_compl:kn", f"{a[3]}/{a[2]}")
            put(f"r4:{m}:{c}:norb_compl:kn", f"{a[5]}/{a[4]}")
    meta = {}
    r4_path = os.path.join(RES, "r4_precommit", "analysis", "r4_precommit.json")
    if os.path.exists(r4_path):
        meta = json.load(open(r4_path)).get("meta", {})
    if "margin" in meta:
        put("r4:margin", f2(float(meta["margin"])))
    t = e("r4.total:episodes")
    if "value" in t:
        put("r4:episodes", f"{int(t['value']):,}")
        pl = meta.get("planned_episodes")
        if pl:
            put("r4:planned", f"{int(pl):,}")
            put("r4:complete", "yes" if int(t["value"]) >= int(pl) else "no")
        if "total:all_r23" in NUMS:   # every simulated episode, R4 included (abstract, section 1)
            put("total:all_r234", f"{int(str(NUMS['total:all_r23']).replace(',', '')) + int(t['value']):,}")
    r = e("r4.total:refusals")
    if "value" in r:
        put("r4:refusals", int(r["value"]))
    raw = os.path.join(RES, "r4_precommit", "raw.jsonl")
    st = []
    if os.path.exists(raw):
        st = [json.loads(l).get("started_utc") for l in open(raw) if l.strip()]
        st = [x for x in st if x]
        if st:
            put("r4:first_data_utc", min(st).replace("T", " ")[:16] + " UTC")
            put("r4:last_data_utc", max(st).replace("T", " ")[:16] + " UTC")
    pr = os.path.join(ROOT, "analysis", "preregistration.md")
    if os.path.exists(pr):
        m = re.search(r"^## R4\..*registered (\d{4})-(\d{2})-(\d{2}), before any R4 data", open(pr).read(), re.M)
        if m:
            put("r4:reg_date", f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]} {m.group(1)}")
            put("r4:reg_dm", f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]}")
        try:
            reg = subprocess.run(["git", "-C", ROOT, "log", "--diff-filter=A", "--format=%cI",
                                  "--", "configs/r4_precommit.json"],
                                 capture_output=True, text=True, timeout=30).stdout.split()
        except Exception:
            reg = []
        if not reg and os.path.exists(os.path.join(HERE, "provenance.json")):
            reg = json.load(open(os.path.join(HERE, "provenance.json"))).get("r4_registered_utc") or []
        if reg and st:
            from datetime import datetime, timezone
            a = datetime.fromisoformat(reg[-1]).astimezone(timezone.utc)
            z = datetime.fromisoformat(min(st).replace("Z", "+00:00")).astimezone(timezone.utc)
            if not a < z:
                note("WARNING R4 runner config was committed after the first R4 episode")
            put("r4:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC")
            put("r4:commit_hm", a.strftime("%H:%M") + " UTC")
            put("r4:first_data_hm", z.strftime("%H:%M") + " UTC")
            put("r4:before_data", "yes" if a < z else "no")


def emit_completion():
    """Task completion next to IVR in the same-run wording cells of R2 and R3
    (an agent that does nothing scores IVR 0). Completion is the
    raw row's `achieved` (every goal step committed with the instructed values,
    in order); refusals are excluded as for IVR. compl:incomplete:allran says
    whether every incomplete episode under a note or an unknown wording still
    committed one distinct call per step (a wrong field value, not a skipped step)."""
    A_CTX = ("amnesiac", "bare_hint", "hint_p1", "hint_p2", "hint_p3")
    B_FM = {"error_500_after_commit": "e500", "timeout_after_commit": "toA",
            "status_unknown_after_commit": "w3", "status_unknown_p1_after_commit": "unknown_p1",
            "status_unknown_p2_after_commit": "unknown_p2", "status_unknown_p3_after_commit": "unknown_p3"}
    BASE = ("amnesiac", "e500", "toA")
    EFF = {"send_payment", "send_email", "create_ticket", "append_file", "create_event"}
    cells, incomplete = {}, []
    for run in ("r2", "r3"):
        raw = os.path.join(RES, f"{run}_wording", "raw.jsonl")
        if not os.path.exists(raw):
            continue
        last = {}
        for r in read_jsonl(raw):
            last[r["episode_id"]] = r
        for r in last.values():
            if r.get("refusal"):
                continue
            if r.get("kind") == "restart":
                part, cond = "A", r.get("resume_context")
                if cond not in A_CTX:
                    continue
            else:
                part, cond = "B", B_FM.get(r.get("failure_mode"))
                if cond is None:
                    continue
            c = cells.setdefault((run, part, short_model(r["model"]), cond), [0, 0])
            c[0] += bool(r["achieved"]); c[1] += 1
            if not r["achieved"] and cond not in BASE:
                incomplete.append((run, r))
    if not cells:
        return
    for (run, part, m, cond), (k, n) in cells.items():
        put(f"compl:{run}.{part}.{m}.{cond}:kn", f"{k}/{n}")

    def lowest(vs):
        k, n = min(vs, key=lambda v: (v[0] / v[1], -v[1]))
        return f"{k}/{n}"
    note_c = [v for (_, p, _, c), v in cells.items() if p == "A" and c not in BASE]
    unk_c = [v for (_, p, _, c), v in cells.items() if p == "B" and c not in BASE]
    base_c = [v for (_, _, _, c), v in cells.items() if c in BASE]
    if note_c:
        put("compl:A:note:min:kn", lowest(note_c))
    if unk_c:
        put("compl:B:unk:min:kn", lowest(unk_c))
    if note_c and unk_c:
        put("compl:treat:min:kn", lowest(note_c + unk_c))   # under any note or unknown wording
        k_, n_ = (int(x) for x in lowest(note_c + unk_c).split("/"))
        put("compl:treat:min:pctfloor", f"{(100 * k_) // n_}\\%")   # rounded down, so "at least" stays true
    put("compl:base:all", "yes" if base_c and all(k == n for k, n in base_c) else "no")
    put("compl:incomplete:n", len(incomplete))
    # every incomplete episode: one distinct committed effectful call per step?
    ok = True
    for run, r in incomplete:
        tdir = os.path.join(RES, f"{run}_wording", "transcripts")
        if r.get("kind") == "restart":
            f = os.path.join(tdir, r["episode_id"] + ".json")
            t = json.load(open(f)) if os.path.exists(f) else None
            if t is None:
                ok = False; continue
            calls = set()
            for si, sess in enumerate(t["sessions"]):
                turns = sess["turns"]
                res = {x["tool_use_id"]: x for turn in turns if turn.get("role") == "tool_results"
                       for x in turn["content"]}
                for turn in turns:
                    if turn.get("role") != "assistant":
                        continue
                    for x in turn["content"]:
                        if x.get("type") != "tool_use" or x["name"] not in EFF:
                            continue
                        rr = res.get(x["id"])
                        # session 1's interrupted call committed but got no result
                        committed = (rr is None and si == 0) or (rr is not None and not rr.get("is_error")
                                                                 and str(rr.get("content", "")).startswith("ok"))
                        if committed:
                            calls.add(sid(x["name"], x["input"]))
            ok = ok and len(calls) >= int(r["n_nodes"])
        else:
            hits = glob.glob(os.path.join(tdir, "*%s*__s%d__%s.json" % (
                r["failure_mode"], int(r["sample"]), r["task_id"])))
            t = None
            for f in hits:
                tt = json.load(open(f))
                if tt.get("episode_id") == r["episode_id"]:
                    t = tt
            if t is None:
                ok = False; continue
            goals = {g["tool"] for g in t["score"]["goals"]}
            ok = ok and goals <= {e["tool"] for e in t["ledger"]}
    put("compl:incomplete:allran", "yes" if ok else "no")


def check_analysis_v3():
    """Cross-check: every cell value the analysis wrote must equal the value
    this script computes from the same raw rows (k/n). Mismatches are
    reported, and \\ASSERT{nj:check:mismatch}{0} in main.tex fails visibly."""
    exp_b = {"e1_0": "e10", "e1_1": "e13", "e1_3": "e13", "e1_4": "e14",
             "e1_5": "e15", "e1_6": "e16", "v2": "jul", "xp": "xp", "e1_2": "e12"}
    fm = {"w1": "e500body", "w3": "unknown"}
    checked = bad = 0
    for key, e in NJ.items():
        m = re.fullmatch(r"(e1_[0-6]|v2|xp)\.(\w+)\.(base|careful)\.(none|keys|rcpt)\.(\w+):ivr", key)
        if not m or not isinstance(e, dict) or "k" not in e:
            continue
        b, model, prompt, mit, f = m.groups()
        q = f"{exp_b[b]}:{model}:{prompt}:{mit}:{fm.get(f, f)}"
        if b == "e1_2":
            q += ":at_least_once"
        mine = NUMS.get(q + ":kn")
        if mine is None or mine.startswith("\\PENDING"):
            continue
        checked += 1
        if mine != f"{e['k']}/{e['n']}":
            bad += 1
            note(f"WARNING analysis_v3 {key} = {e['k']}/{e['n']} but make_numbers {q} = {mine}")
    put("nj:check:cells", checked)
    put("nj:check:mismatch", bad)


def emit_prereg_timing():
    """When the pre-registration was committed (git, read-only) and when the
    first and last Tier-1 episode started (transcripts), so the paper can
    state that the registration precedes the data."""
    import subprocess
    try:
        out = subprocess.run(["git", "-C", ROOT, "log", "--diff-filter=A", "--format=%cI",
                              "--", "analysis/preregistration.md"],
                             capture_output=True, text=True, timeout=30).stdout.split()
        dev = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%cI %h",
                              "--", "analysis/preregistration.md"],
                             capture_output=True, text=True, timeout=30).stdout.split()
    except Exception:
        out, dev = [], []
    if not out:   # anonymized artifact: no git history; use frozen provenance
        _pv = os.path.join(HERE, 'provenance.json')
        if os.path.exists(_pv):
            _pv = json.load(open(_pv))
            out = _pv.get('preregistration_added_utc') or []
            dev = _pv.get('preregistration_last_change_utc') or []
    starts = []
    for f in glob.glob(os.path.join(RES, "tier1", "e1_*", "transcripts", "*.json")):
        try:
            t = json.load(open(f))
        except Exception:
            continue
        s = t.get("started_utc")
        if s:
            starts.append(s)
    if out:
        reg = out[-1]
        put("prereg:commit_utc", reg.replace("T", " ")[:16].replace("+00:00", "") + " UTC")
        put("prereg:date", reg[:10])
    if dev:
        put("prereg:last_change_date", dev[0][:10])
    if starts:
        put("prereg:first_data_utc", min(starts).replace("T", " ")[:16] + " UTC")
        put("prereg:last_data_utc", max(starts).replace("T", " ")[:16] + " UTC")
    if out and starts:
        from datetime import datetime
        a = datetime.fromisoformat(out[-1])
        z = datetime.fromisoformat(min(starts).replace("Z", "+00:00"))
        put("prereg:before_data", "yes" if a < z else "no")
        h = (z - a).total_seconds() / 3600
        put("prereg:lead_hours", f"{h:.1f}")
        mins = round(h * 60)
        put("prereg:lead_hm", f"{mins // 60} h {mins % 60} min")
    # R1 (real-API slice): registered in the commit that added its runner config
    try:
        r1 = subprocess.run(["git", "-C", ROOT, "log", "--diff-filter=A", "--format=%cI",
                             "--", "configs/real_stripe.json"],
                            capture_output=True, text=True, timeout=30).stdout.split()
    except Exception:
        r1 = []
    if not r1 and os.path.exists(os.path.join(HERE, 'provenance.json')):
        r1 = json.load(open(os.path.join(HERE, 'provenance.json'))).get('r1_registered_utc') or []
    rraw = os.path.join(RES, "real_stripe", "raw.jsonl")
    rst = [json.loads(l).get("started_utc") for l in open(rraw) if l.strip()] if os.path.exists(rraw) else []
    rst = [x for x in rst if x]
    if r1 and rst:
        from datetime import datetime, timezone
        a = datetime.fromisoformat(r1[-1]).astimezone(timezone.utc)
        z = datetime.fromisoformat(min(rst).replace("Z", "+00:00"))
        put("r1:commit_utc", a.strftime("%Y-%m-%d %H:%M") + " UTC")
        put("r1:first_data_utc", min(rst).replace("T", " ")[:16] + " UTC")
        put("r1:last_data_utc", max(rst).replace("T", " ")[:16] + " UTC")
        put("r1:before_data", "yes" if a < z else "no")
        # the R1 heading of the registration carries the committer's local date (some UTC
        # offset from -12 to +14 h); the paper states times in UTC and names that date once
        from datetime import timedelta as _td, date as _date
        _h = re.search(r"^## R1:.*registered (\d{4})-(\d{2})-(\d{2})", open(os.path.join(ROOT, "analysis", "preregistration.md")).read(), re.M)
        _d = _date(int(_h.group(1)), int(_h.group(2)), int(_h.group(3))) if _h else None
        pac = _d if _d in {(a + _td(hours=h)).date() for h in range(-12, 15)} else a.date()
        put("r1:commit_dm", f"{a.day} {MONTHS[a.month - 1]}")
        put("r1:commit_hm", a.strftime("%H:%M") + " UTC")
        put("r1:commit_local_dm", f"{pac.day} {MONTHS[pac.month - 1]}")
    # deviations: numbered entries in the dated deviations log
    pr = os.path.join(ROOT, "analysis", "preregistration.md")
    if os.path.exists(pr):
        m = re.search(r"^## R1:.*registered (\d{4})-(\d{2})-(\d{2})", open(pr).read(), re.M)
        if m:
            put("r1:reg_dm", f"{int(m.group(3))} {MONTHS[int(m.group(2)) - 1]}")
        src = open(pr).read().split("## Deviations log", 1)[-1]
        put("prereg:deviations", len(re.findall(r"^\d+\.\s", src, re.M)))
        put("prereg:experiments", len(re.findall(r"^## E1-\d", open(pr).read(), re.M)))
        full = open(pr).read()
        def sec(name):
            m = re.search(r"^## " + re.escape(name) + r".*?(?=^## )", full, re.M | re.S)
            return m.group(0) if m else ""
        m = re.search(r"within ±([\d.]+)", sec("E1-0"))
        if m: put("prereg:e10_tol", m.group(1))
        m = re.search(r"≥ ([\d.]+), tested", sec("E1-1"))
        if m: put("prereg:e11_margin", f2(float(m.group(1))))
        m = re.search(r"is rare \(≤ ([\d.]+)\)", sec("E1-5"))
        if m: put("prereg:e15_stale_max", f2(float(m.group(1))))
        m = re.search(r"H1: controls 1–2 are ≈ 0 \(task-CI upper bound ≤ ([\d.]+)\)", sec("E1-1"))
        if m: put("prereg:e11_ctrl_max", f2(float(m.group(1))))
        try:
            put("design:maxturns_long", json.load(open(os.path.join(ROOT, "configs",
                "e1_5_long_chains.json")))["max_turns"])
        except (OSError, KeyError, ValueError):
            pass
        m = re.search(r"≥ ([\d.]+)% of planned n", full)
        if m: put("prereg:min_pct", m.group(1) + "\\%")
    # E1-1 cells 1-7 are imported from results/crossprovider/ (registration,
    # "E1-1"). Count the imported episodes of the registered P test (timeout
    # vs server error, base/none) that STARTED before the registration commit.
    if out:
        reg = out[-1]
        pre, tot = Counter(), Counter()
        for f in glob.glob(os.path.join(RES, "crossprovider", "sweep_transcripts", "*.json")):
            mm = re.search(r"_base_none_(timeout_after_commit|error_500_after_commit)__s", f)
            if not mm:
                continue
            t = json.load(open(f))
            m = short_model(t.get("model", ""))
            tot[m] += 1
            st = t.get("started_utc") or ""
            if st and st.replace("Z", "+00:00") < reg:
                pre[m] += 1
        put("prereg:e1_1_pre", f"{sum(pre.values())}/{sum(tot.values())}")
        if "prereg:experiments" in NUMS:
            put("prereg:clean", int(NUMS["prereg:experiments"]) - (1 if sum(pre.values()) else 0))
            # E1-0..E1-6 plus R1: how many were NOT registered before any of their data
            if "r1:before_data" in NUMS:
                n_all = int(NUMS["prereg:experiments"]) + 1
                n_clean = int(NUMS["prereg:clean"]) + (1 if NUMS["r1:before_data"] == "yes" else 0)
                # R2 (wording robustness) is registered in the same file; its
                # heading states that it precedes its data (emit_r2 warns if git disagrees)
                if "r2:reg_date" in NUMS:
                    n_all += 1; n_clean += 1
                # R3 (cross-provider wording): the same, registered 2026-11-01
                if "r3:reg_date" in NUMS:
                    n_all += 1; n_clean += 1
                # R4 (outcome unknown before commit): the same, registered 2026-11-01
                if "r4:reg_date" in NUMS:
                    n_all += 1; n_clean += 1
                put("prereg:all:n", n_all)
                put("prereg:all:notclean", n_all - n_clean)
                put("prereg:followups:n", n_all - 1)   # E1-0..E1-6, R2, R3 and R4 (all but R1): experiments after the reviews
        for m in tot:
            put(f"prereg:e1_1_pre:{m}", f"{pre[m]}/{tot[m]}")


def emit_cost():
    """API spend, from logged data only. Claude: token usage in every October
    transcript x the list prices hard-coded in the runners (read by regex);
    Tier-1: the spend lines the launcher logged; OpenRouter core runs: no
    per-call cost was logged, so only the key limit bounds them."""
    src = open(os.path.join(ROOT, "src2", "sweep2.py")).read()
    prices = {m: (float(a), float(b)) for m, a, b in
              re.findall(r'"(claude-[\w-]+)": \(([\d.]+), ([\d.]+)\)', src)}
    def price(model):
        for p, v in prices.items():
            if model.startswith(p):
                return v
        return None
    claude_oct = 0.0
    for f in glob.glob(os.path.join(RES, "claude_batch_2026-10", "transcripts2", "*.json")):
        t = json.load(open(f))
        pr = price(t.get("model", ""))
        if not pr:
            continue
        for s in t.get("sessions", []):
            u = (s.get("meta") or {}).get("usage") or {}
            claude_oct += ((u.get("input") or 0) * pr[0] + (u.get("output") or 0) * pr[1]) / 1e6
    put("cost:claude_oct", f"\\${claude_oct:.2f}")
    # the launcher logs the cumulative spend under results/tier1; the October
    # sensitivity run (e1_6_oct) has its own log, so take the largest total
    a = o = None
    for log in (os.path.join(RES, "tier1", "run.log"), os.path.join(RES, "tier1", "e1_6_oct", "run.log")):
        if not os.path.exists(log):
            continue
        for line in open(log):
            m = re.search(r"logged spend under .*: Anthropic \$([\d.]+).*OpenRouter \$([\d.]+)", line)
            if m and (a is None or float(m.group(1)) + float(m.group(2)) >= a + o):
                a, o = float(m.group(1)), float(m.group(2))
    if a is not None:
        put("cost:tier1_anthropic", f"\\${a:.2f}")
        put("cost:tier1_openrouter", f"\\${o:.2f}")
    m = re.search(r"\$(\d+) key limit", open(os.path.join(ROOT, "run_crossprovider.sh")).read())
    if m:
        put("cost:or_key_limit", f"\\${m.group(1)}")
    # real-API slice (preregistration R1): cost_usd of every scored episode
    # (Anthropic at list prices, OpenRouter usage.cost); Stripe test mode is free
    ra = ro = 0.0
    rraw = os.path.join(RES, "real_stripe", "raw.jsonl")
    if os.path.exists(rraw):
        rr = {}
        for line in open(rraw):
            if line.strip():
                r = json.loads(line)
                rr[r["episode_id"]] = r
        for r in rr.values():
            if r["model"].startswith("openrouter:"):
                ro += r.get("cost_usd") or 0
            else:
                ra += r.get("cost_usd") or 0
        put("cost:r1", f"\\${ra + ro:.2f}")
        put("cost:r1_anthropic", f"\\${ra:.2f}")
        put("cost:r1_openrouter", f"\\${ro:.2f}")
    # R2 (wording robustness, Anthropic only): cost_usd of every scored episode,
    # last row per episode id, as for R1; cached tokens are priced in the rows
    r2c = 0.0
    r2raw = os.path.join(RES, "r2_wording", "raw.jsonl")
    if os.path.exists(r2raw):
        rr = {}
        for line in open(r2raw):
            if line.strip():
                r = json.loads(line)
                rr[r["episode_id"]] = r
        r2c = sum(r.get("cost_usd") or 0 for r in rr.values())
        put("cost:r2", f"\\${r2c:.2f}")
        # the R2 completion (deviation 14) was launched by run_r3.sh, which logs
        # what it spent as "Part 1 spent $X"; the value before it is deviation 13's
        log3 = os.path.join(RES, "r3_wording", "run.log")
        if os.path.exists(log3):
            spent = [float(x) for x in re.findall(r"Part 1 spent \$([\d.]+)", open(log3).read())]
            if spent:
                put("cost:r2_completion", f"\\${max(spent):.2f}")
                put("cost:r2_before_completion", f"\\${r2c - max(spent):.2f}")
    # R3 (cross-provider wording): cost_usd of every scored episode, by provider
    # (Anthropic at list prices with cache tokens priced; OpenRouter usage.cost)
    r3a = r3o = 0.0
    r3raw = os.path.join(RES, "r3_wording", "raw.jsonl")
    if os.path.exists(r3raw):
        rr = {}
        for line in open(r3raw):
            if line.strip():
                r = json.loads(line)
                rr[r["episode_id"]] = r
        for r in rr.values():
            if r["model"].startswith("openrouter:"):
                r3o += r.get("cost_usd") or 0
            else:
                r3a += r.get("cost_usd") or 0
        put("cost:r3", f"\\${round(r3a + r3o, 2):.2f}")
        put("cost:r3_anthropic", f"\\${r3a:.2f}")
        put("cost:r3_openrouter", f"\\${r3o:.2f}")
    # R4 (outcome unknown before commit): cost_usd of every scored episode, by provider, as for R3
    r4a = r4o = 0.0
    r4raw = os.path.join(RES, "r4_precommit", "raw.jsonl")
    if os.path.exists(r4raw):
        rr = {}
        for line in open(r4raw):
            if line.strip():
                r = json.loads(line)
                rr[r["episode_id"]] = r
        for r in rr.values():
            if r["model"].startswith("openrouter:"):
                r4o += r.get("cost_usd") or 0
            else:
                r4a += r.get("cost_usd") or 0
        put("cost:r4", f"\\${round(r4a + r4o, 2):.2f}")
        put("cost:r4_anthropic", f"\\${r4a:.2f}")
        put("cost:r4_openrouter", f"\\${r4o:.2f}")
    if a is not None:
        before_r2 = round(claude_oct + a + o + ra + ro, 2)
        put("cost:logged_total", f"\\${before_r2:.2f}")
        # Running totals. Each experiment's total is its unrounded sum, rounded
        # once (cost:r3 = round(r3a + r3o, 2)); a running total adds those
        # ROUNDED per-experiment totals, so the printed total equals the sum of
        # the printed per-experiment values (44.62 + 18.39 + 22.57 + 3.19 =
        # 88.77). independent/check.py recomputes these keys the same way.
        t_r2, t_r3, t_r4 = round(r2c, 2), round(r3a + r3o, 2), round(r4a + r4o, 2)
        put("cost:logged_total_r2", f"\\${before_r2 + t_r2:.2f}")
        if os.path.exists(r3raw):
            put("cost:logged_total_r23", f"\\${before_r2 + t_r2 + t_r3:.2f}")
            put("cost:r23", f"\\${t_r2 + t_r3:.2f}")
            if os.path.exists(r4raw):
                put("cost:logged_total_r234", f"\\${before_r2 + t_r2 + t_r3 + t_r4:.2f}")
                # the printed parts (Limitations) add up to the printed total, in cents
                cents = [round(float(NUMS[k].strip("\\$")) * 100) for k in
                         ("cost:logged_total", "cost:r2", "cost:r3", "cost:r4")]
                put("cost:r234:parts_add", "yes" if sum(cents) ==
                    round(float(NUMS["cost:logged_total_r234"].strip("\\$")) * 100) else "no")
        # cost:total is the pre-R2 total with its composition; the sentence in
        # main.tex (Limitations) states which runs were not logged
        put("cost:total", f"\\${before_r2:.2f} (Claude \\${claude_oct + a + ra:.2f} "
            f"at list prices; OpenRouter \\${o + ro:.2f}"
            + (f"; the real-API check \\${ra + ro:.2f} of the total, with no Stripe charge"
               if ra + ro else "")
            + ")")



# ---------------------------------------------------------------- registered tests table
PREREG = ""   # appendix rows (\preregrows)


def emit_prereg_table():
    """One row per pre-registered test (analysis/preregistration.md), values
    and verdicts read from analysis_v3/numbers.json. Kind: P = primary
    (uncorrected, as registered), S = secondary (Holm within experiment)."""
    global PREREG
    nums = NJ
    disp = {k: d for _, k, d in MODEL_SHORT}
    rows, tally = [], Counter()
    def short(v):
        return ("reversed" if "reversed" in v else "not supported" if v.startswith("not supported")
                else "supported" if v.startswith("supported") else "null" if v.startswith("null")
                else "reduced" if "reduced" in v else "no change" if "no change" in v
                else "consistent" if v.startswith("consistent") else v.split(" (")[0])
    def pv(k):
        e = nums.get(k, {})
        v = e.get("value")
        return "--" if not isinstance(v, float) else pfmt(v)
    def add(exp, kind, hyp, who, result, p, verdict):
        rows.append(" & ".join([exp, kind, hyp, who, result, p, verdict]) + r" \\")
        if kind == "P":
            tally[verdict] += 1
    def dval(k):
        v = nums.get(k, {}).get("value")
        return "--" if v is None else (("+" if v > 0 else "$-$") + f2(abs(v)) if v else "0.00")
    def cell(k):
        e = nums.get(k, {})
        return f"{f2(e['value'])} ({e['k']}/{e['n']})" if "k" in e else f2(e.get("value", 0))
    v = nums["e1_0.P.diff_rerun_minus_v2"]
    add("E1-0", "P", r"$|$rerun$-$v2$|\le$ " + NUMS.get("prereg:e10_tol", "?"), "Haiku 4.5",
        cell("e1_0.haiku.base.keys.toA:ivr") + ", d " + dval("e1_0.P.diff_rerun_minus_v2"), "--",
        short(v["verdict"]))
    for k, h in (("e1_0.S1.key_attach_first", "key on first call 0/64"),
                 ("e1_0.S2.released_calls_identical", "$>$half calls identical to v1")):
        add("E1-0", "S", h, "Haiku 4.5", str(nums[k]["value"]), "--", short(nums[k]["verdict"]))
    for m in ("gpt", "gemini", "qwen"):
        k = f"e1_1.P.H2.{m}"
        add("E1-1", "P", r"500$-$timeout $\ge$ " + NUMS.get("prereg:e11_margin", "?"), disp[m],
            "d " + dval(k + ":d"), pv(k + ":p"), short(nums[k + ":p"]["verdict"]))
    add("E1-1", "S", r"controls: CI upper $\le$ 0.10", "3 models", "0 duplicates in 6 cells", "--",
        "not evaluable")
    ok = sum(1 for k, e in nums.items() if k.startswith("e1_1.S.H3.") and e["verdict"] == "supported")
    tot = sum(1 for k in nums if k.startswith("e1_1.S.H3."))
    add("E1-1", "S", r"check-first $\le$ 0.10", "3 models", f"{ok} of {tot} cells", "--",
        "supported" if ok == tot else "partly")
    # S-H4 (descriptive in the registration): under keys, IVR tracks key
    # attach x reuse; a model that keys its first call and reuses the key
    # gets IVR ~ 0. Supported if every keyed-and-reused episode is clean.
    h4 = {k: e for k, e in nums.items() if re.fullmatch(r"e1_1\.S\.H4\.\w+\.\w+", k)}
    kr = sum(e.get("n_keyed_and_reused", 0) for e in h4.values())
    kr_dup = sum(round((e.get("ivr_keyed_and_reused") or 0) * e.get("n_keyed_and_reused", 0))
                 for e in h4.values())
    if h4:
        put("e11:h4:keyed_reused", kr); put("e11:h4:keyed_reused_dup", kr_dup)
        add("E1-1", "S", r"keyed and reused $\Rightarrow$ IVR $\approx$ 0", "3 models",
            f"{kr_dup} of {kr} keyed, reused episodes duplicated", "--",
            "supported" if kr and kr_dup == 0 else "not supported")
    k = "e1_2.P.H2.sonnet_minus_haiku"
    add("E1-2", "P", "Sonnet $<$ Haiku (redelivery)", "Sonnet 5, Haiku 4.5",
        cell("e1_2.sonnet.base.none.none:ivr") + " vs " + f2(nums["e1_2.haiku.base.none.none:ivr"]["value"]),
        pv(k + ":p"), short(nums[k + ":p"]["verdict"]))
    add("E1-2", "S", r"Haiku $\ge$ 0.85", "Haiku 4.5", cell("e1_2.S.H1.haiku_base_none"), "--",
        short(nums["e1_2.S.H1.haiku_base_none"]["verdict"]))
    for m in ("haiku", "sonnet", "gpt", "gemini", "qwen"):
        k = f"e1_2.S.H3.{m}"
        # S-H3 predicts Haiku unchanged and the others reduced
        add("E1-2", "S", "keys do not change Haiku" if m == "haiku" else "keys reduce redelivery dups",
            disp[m], "d " + dval(k + ":d"),
            pfmt(nums[k + ":p"]["holm_p"]) + " (Holm)", short(nums[k + ":p"]["verdict"]))
    for m in ("haiku", "sonnet", "gpt", "gemini", "qwen"):
        k = f"e1_2.S.H4.{m}"
        d = nums[k + ":d"]["value"]
        verdict = short(nums[k + ":p"]["verdict"])
        if nums[k + ":p"]["holm_p"] < 0.05 and d > 0:
            verdict = "supported; increase"
        add("E1-2", "S", "check-first does not reduce", disp[m], "d " + dval(k + ":d"),
            pfmt(nums[k + ":p"]["holm_p"]) + " (Holm)", verdict)
    k = "e1_3.P.sonnet_keys_minus_none"
    add("E1-3", "P", "keys lower 500 IVR", "Sonnet 5", "d " + dval(k + ":d"), pv(k + ":p"),
        short(nums[k + ":p"]["verdict"]))
    k = "e1_3.S.haiku_keys_minus_none"
    add("E1-3", "S", "Haiku unchanged", "Haiku 4.5", "d " + dval(k + ":d"), pv(k + ":p"),
        short(nums[k + ":p"]["verdict"]))
    fmn = {"toA": "timeout", "e500": "500"}
    for m in ("haiku", "sonnet"):
        for f in ("toA", "e500"):
            k = f"e1_4.{m}.{f}.P.new_within_released_ci"
            s0 = nums[f"e1_4.{m}.{f}.released_s0.ivr"]
            add("E1-4", "P", "new tasks in old CI", f"{disp[m]}, {fmn[f]}",
                cell(f"e1_4.{m}.{f}.new.ivr") + " vs " + f2(s0["value"]) + " " + ci_str(*s0["ci"]),
                "--", short(nums[k]["verdict"]))
    for m in ("haiku", "sonnet"):
        for f in ("toA", "e500"):
            k = f"e1_5.{m}.{f}.P.within_e1_4_pooled96_ci"
            # two decimals, as everywhere else; 30/32 = 0.9375 prints as 0.94
            # against an upper bound 0.942 (also 0.94): k/n shows it is inside
            e5 = nums[f"e1_5.{m}.{f}:ivr"]
            add("E1-5", "P", "long chain in 96-task CI", f"{disp[m]}, {fmn[f]}",
                f"{f2(e5['value'])} ({e5['k']}/{e5['n']}) vs {ci_str(*nums[k]['ci'])}",
                "--", short(nums[k]["verdict"]))
    for m in ("haiku", "sonnet"):
        for f in ("toA", "e500"):
            k = f"e1_5.{m}.{f}.S.stale_reexec"
            add("E1-5", "S", r"stale re-execution $\le$ " + NUMS.get("prereg:e15_stale_max", "?"),
                f"{disp[m]}, {fmn[f]}", cell(k), "--", short(nums[k]["verdict"]))
    mis = sum(1 for k, e in nums.items() if re.fullmatch(r"e1_5\.\w+\.\w+\.S\.misorder", k) and e["value"] == 0)
    add("E1-5", "S", "misorder only with parallel calls", "both", f"0 misorders ({mis} cells)", "--",
        "not testable")
    for m in ("haiku", "sonnet"):
        k = f"e1_6.P1.{m}"
        add("E1-6", "P", "Retry-After body $\\ge$ 500", disp[m], "d " + dval(k + ":d"), pv(k + ":p"),
            short(nums[k + ":p"]["verdict"]))
    # P2 is one registered hypothesis per model: IVR falls, mediated by a
    # higher read-back rate. Both parts share one row and count as one P test;
    # the row is "supported" only if both parts are.
    for m in ("haiku", "sonnet"):
        k, kr = f"e1_6.P2.{m}", f"e1_6.P2.{m}.readback"
        v1, v2 = short(nums[k + ":p"]["verdict"]), short(nums[kr + ":p"]["verdict"])
        add("E1-6", "P", "``outcome unknown'' $<$ 500, via read-back", disp[m],
            "d " + dval(k + ":d") + "; read-back d " + dval(kr + ":d"),
            pv(k + ":p") + "; " + pv(kr + ":p"),
            v1 if v1 == v2 else f"{v1}; read-back {v2}")
    bel = NUMS.get("belief:audit:short", "rule-based, unaudited")
    # the counted label (does the window entertain a possible commit) reached
    # only fair agreement with the hand coding, so the row says the counts are
    # unreliable and the verdict column carries no verdict
    kb = NUMS.get("nj:e1_6.belief.kappa:kappa_binary")
    # the counts themselves are not reported anywhere: the row
    # records that the hypothesis was registered and why it has no verdict
    add("E1-6", "S", "stated commit belief", "Haiku 4.5",
        ("no counts ($\\kappa$=%s)" % kb) if kb else "no counts",
        "--", "not evaluable" if kb or bel.startswith("rule") else bel)
    put("prereg:rows", len(rows))
    # added sensitivity (deviation 8): E1-6 contrasts against October bare
    # baselines, plus the October 500 - timeout split and the corrected
    # October - July drift. Kind "sens.": not counted among the P tests.
    sens = []
    def reading(k):
        e = nums.get(k + ":p", {})
        p, d = e.get("value"), nums.get(k + ":d", {}).get("value")
        if not isinstance(p, float) or d is None:
            return "--"
        return ("null" if p >= 0.05 else "lower" if d < 0 else "higher") + " (no verdict)"
    def okn(k):
        e = nums.get(k, {})
        return f"{e['k']}/{e['n']}" if "k" in e else "?"
    if any(k.startswith("sens.oct.") for k in nums):
        for m in ("haiku", "sonnet"):
            oc = f"sens.oct.{m}.base.none."
            for exp, k, hyp, res in (
                    ("E1-6", f"sens.oct.{m}.w3_minus_oct_e500", "``outcome unknown'' vs Oct.\\ 500",
                     okn(f"e1_6.{m}.base.none.w3:ivr") + " vs " + okn(oc + "e500:ivr")),
                    ("E1-6", f"sens.oct.{m}.w1_minus_oct_e500", "Retry-After body vs Oct.\\ 500",
                     okn(f"e1_6.{m}.base.none.w1:ivr") + " vs " + okn(oc + "e500:ivr")),
                    ("E1-1", f"sens.oct.{m}.oct_e500_minus_oct_toA", "Oct.\\ 500 vs Oct.\\ timeout",
                     okn(oc + "e500:ivr") + " vs " + okn(oc + "toA:ivr")),
                    ("--", f"sens.oct_fix.{m}.drift.e500", "Oct.\\ vs July bare 500",
                     okn(oc + "e500:ivr") + " vs " + okn(f"sens.oct_fix.{m}.july.base.none.e500:ivr")),
                    ("--", f"sens.oct_fix.{m}.drift.toA", "Oct.\\ vs July timeout",
                     okn(oc + "toA:ivr") + " vs " + okn(f"sens.oct_fix.{m}.july.base.none.toA:ivr"))):
                if k + ":p" not in nums:
                    continue
                sens.append(" & ".join([exp, "sens.", hyp, disp[m], res + ", d " + dval(k + ":d"),
                                        pv(k + ":p"), reading(k)]) + r" \\")
    if sens:
        rows += [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Added sensitivity analysis, deviation (8): "
                 r"October bare baselines, run after the E1-6 results; not confirmatory}} \\"] + sens
    put("prereg:sens_rows", len(sens))
    PREREG = "\n".join(rows)
    put("prereg:P:n", sum(tally.values()))
    for vname in ("supported", "not supported", "reversed", "null"):
        put("prereg:P:" + vname.replace(" ", "_"), tally.get(vname, 0))
    put("prereg:P:against", sum(c for v, c in tally.items() if v != "supported"))


# ---------------------------------------------------------------- lint
def lint(main_tex):
    src = open(main_tex).read()
    body = src.split(r"\begin{document}", 1)[-1]
    body_nc = re.sub(r"(?<!\\)%.*", "", body)
    keys = re.findall(r"\\[VT]\{([^}]*)\}", body_nc) + \
        re.findall(r"\\ASSERT\{([^}]*)\}", body_nc) + \
        [k + ":ivr" for k in re.findall(r"\\C\{([^}]*)\}", body_nc)] + \
        [k + ":givr" for k in re.findall(r"\\G\{([^}]*)\}", body_nc)]
    lit = os.path.join(HERE, "lit_numbers.tex")
    litkeys = set(re.findall(r"\\defnum\{([^}]*)\}", open(lit).read())) if os.path.exists(lit) else set()
    undefined = sorted(set(k for k in keys if k not in NUMS and k not in litkeys))
    pend = sorted(set(k for k in keys if k in NUMS and NUMS[k].startswith(r"\PENDING")))
    # literal digits in prose: strip comments, macros args that are allowed
    s = body_nc
    s = re.sub(r"\\(V|T|C|G)\{[^}]*\}", "", s)
    s = re.sub(r"\\ASSERT\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\(ref|label|cite[pt]?|citep|citet|cref|Cref|eqref|input|PENDING|"
               r"includegraphics|begin|end|hspace|vspace|setlength|resizebox|"
               r"multicolumn|cmidrule|small|footnotesize|arraystretch|url|href)"
               r"(\[[^\]]*\])?\{[^}]*\}", "", s)
    s = re.sub(r"\{[lcrp|@{}\d.\\ ]*\}", "", s)
    allowed = re.compile(r"(Haiku[~ ]4\.5|Sonnet[~ ]5|Fable[~ ]5|GPT-5\.5|Qwen3\.6-27B|"
                         r"[0-9.]+\\(columnwidth|textwidth)|red!\d+|\d+\\%|"
                         r"p=0\.5|80\\% power|"
                         r"Gemini|500|503|504|409|202|[12][0-9]{3}[a-z]?\b|"
                         r"\\S\d|Table~?\d|Figure~?\d|E[012]-\d|T\d\d|"
                         r"[A-Za-z]+\d|\d+(pt|em|ex|cm|mm|in)\b)")
    s2 = allowed.sub("", s)
    lits = [m.group(0) for m in re.finditer(r"(?<![A-Za-z\\])\d[\d.,]*", s2)]
    return undefined, pend, Counter(lits)


# ---------------------------------------------------------------- PII sweep
# Checklist B4 / Ethics Statement: a regex sweep of every released result file
# (transcripts, raw rows, logs; smoke runs and *.bak are not released) for
# e-mail addresses outside example.com, phone-number-shaped strings and URLs.
PII_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
PII_PHONE = re.compile(r"(?<![\w.-])(?:\+?\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]\d{3}[ .-]\d{4}(?![\w-])")
PII_URL = re.compile(r"\b(?:https?://|www\.)[^\s\"'<>\\)]+")
PII_URL_ALLOWED = re.compile(r"https://api\.stripe\.com/?$")
PII_SKIP = {"sweep_run2.log", "sweep_run.log", "swap3_pairs.csv", "swap3_summary.csv"}   # not shipped (make_anon_release_merged.sh)


def untracked_results():
    """Files under results/ that git does not track. make_anon_release_merged.sh does not
    ship them (e.g. the blind model-coder hand-check CSVs), so the sweep skips them too and
    the count is the same in the repository and in the artifact (which has no .git)."""
    try:
        out = subprocess.run(["git", "-C", RES, "ls-files", "--others", "--exclude-standard",
                              "--full-name", "-z", "."], capture_output=True, text=True, timeout=30)
        top = subprocess.run(["git", "-C", RES, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return set()
    if out.returncode != 0 or not top or os.path.realpath(top) != os.path.realpath(os.path.dirname(RES)):
        return set()     # results/ is not the top of its own repository (e.g. an unzipped artifact)
    return {os.path.realpath(os.path.join(top, f)) for f in out.stdout.split("\0") if f}


def emit_errors():
    """Error analysis (Section sec:errors): two cases read from the transcripts.
    (i) Qwen3.6-27B resumed sessions that called list_effects: did the first
    lookup precede every effectful call, and what happened after it;
    (ii) Sonnet 5 under at-least-once redelivery (E1-2): duplication on the
    second delivery split by whether that delivery looked at the log first."""
    EFF = {"send_payment", "send_email", "create_ticket", "append_file", "create_event"}
    def calls_of(turns):
        return [x["name"] for turn in turns if turn.get("role") == "assistant"
                for x in turn["content"] if x.get("type") == "tool_use"]
    # (i) Qwen3.6-27B restarts, cross-provider batch (the released transcripts)
    after = Counter(); after_dup = Counter(); after_done = Counter(); total_after = 0
    k_last = Counter()
    for f in sorted(glob.glob(os.path.join(RES, "crossprovider", "transcripts2", "*qwen*.json"))):
        t = json.load(open(f))
        if len(t.get("sessions", [])) < 2:
            continue
        rc = t["resume_context"]
        turns = t["sessions"][1]["turns"]
        names = calls_of(turns)
        if "list_effects" not in names:
            continue
        i = names.index("list_effects")
        dup = t["score"]["violations"] > 0
        if any(n in EFF for n in names[:i]):
            after[rc] += 1; total_after += 1
            after_dup[rc] += dup
            # nothing after the lookup but the closing text
            after_done[rc] += not any(n in EFF for n in names[i + 1:])
        if rc == "bare_hint" and t["interrupt_node"] == 3:
            # resumed after the last node committed: nothing is left to do
            k_last["n"] += 1
            k_last["stopped"] += not any(n in EFF for n in names)
            k_last["redid"] += dup
            if dup:
                # the duplicating sessions re-issued every step in one turn, each with a key
                one_turn = max(sum(1 for x in turn["content"] if x.get("type") == "tool_use")
                               for turn in turns if turn.get("role") == "assistant")
                k_last["redid_one_turn"] += one_turn == 4
                # every re-issued call carried a key, and no session-1 effectful call had one
                # (the listing the model read shows identity fields only, never a key)
                s2_keyed = all("idempotency_key" in x["input"] for turn in turns
                               if turn.get("role") == "assistant"
                               for x in turn["content"] if x.get("type") == "tool_use"
                               and x["name"] in EFF)
                s1_keyed = any((x.get("input") or {}).get("idempotency_key")
                               for turn in t["sessions"][0]["turns"] if turn.get("role") == "assistant"
                               for x in turn["content"] if x.get("type") == "tool_use"
                               and x["name"] in EFF)
                k_last["redid_keyed_s1none"] += s2_keyed and not s1_keyed
    for rc in ("amnesiac", "plan_only", "bare_hint"):
        put(f"err:qwen:{rc}:look_after", after[rc])
        put(f"err:qwen:{rc}:look_after:dup", f"{after_dup[rc]}/{after[rc]}")
        put(f"err:qwen:{rc}:look_after:done", f"{after_done[rc]}/{after[rc]}")
    put("err:qwen:look_after:total", total_after)
    put("err:qwen:k3:n", k_last["n"])
    put("err:qwen:k3:stopped:kn", f"{k_last['stopped']}/{k_last['n']}")
    put("err:qwen:k3:redid:kn", f"{k_last['redid']}/{k_last['n']}")
    put("err:qwen:k3:redid_one_turn:kn", f"{k_last['redid_one_turn']}/{k_last['redid']}")
    put("err:qwen:k3:redid_keyed_s1none:kn", f"{k_last['redid_keyed_s1none']}/{k_last['redid']}")
    # (ii) Sonnet 5 under redelivery (E1-2), base and check-first prompts, no mitigation
    for prompt in ("base", "careful"):
        c = Counter()
        for f in sorted(glob.glob(os.path.join(RES, "tier1", "e1_2", "transcripts",
                                               f"claude-sonnet-5_{prompt}_none_none__*.json"))):
            t = json.load(open(f))
            dup = t["score"]["violations"] > 0
            for i, d in enumerate(t["deliveries"]):
                names = calls_of(d["turns"])
                first = bool(names) and names[0] == "list_effects"
                c[f"d{i + 1}_n"] += 1
                c[f"d{i + 1}_read"] += first
                if i == 1:
                    c["d2_read_dup"] += first and dup
                    c["d2_noread"] += not first
                    c["d2_noread_dup"] += (not first) and dup
        q = f"err:e12:sonnet:{prompt}"
        put(f"{q}:d1_read:kn", f"{c['d1_read']}/{c['d1_n']}")
        put(f"{q}:d2_read:kn", f"{c['d2_read']}/{c['d2_n']}")
        put(f"{q}:d2_read_dup:kn", f"{c['d2_read_dup']}/{c['d2_read']}")
        put(f"{q}:d2_noread_dup:kn", f"{c['d2_noread_dup']}/{c['d2_noread']}")


def emit_pii():
    files = trans = 0
    hits = Counter()
    stripe_other = set()   # non-transcript files that record the Stripe base address
    skip = untracked_results()
    for d, dirs, fs in os.walk(RES):
        dirs[:] = [x for x in dirs if x != "smoke"]   # R4 (results/r4_precommit/) is swept like R3
        for f in fs:
            if "smoke" in f or f in PII_SKIP or f.endswith(".bak") or \
                    not f.endswith((".json", ".jsonl", ".csv", ".log", ".txt", ".md")) or \
                    os.path.realpath(os.path.join(d, f)) in skip:
                continue
            files += 1
            trans += "transcript" in os.path.basename(d)
            txt = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()
            hits["email"] += sum(1 for m in PII_EMAIL.finditer(txt)
                                 if m.group(1).lower() != "example.com"
                                 and not m.group(1).lower().endswith(".example.com"))
            hits["phone"] += len(PII_PHONE.findall(txt))
            for u in PII_URL.findall(txt):
                # the Stripe test-mode transcripts (preregistration R1) record the
                # API base address; it is not personal data and is counted apart
                if PII_URL_ALLOWED.match(u):
                    hits["url_stripe_api"] += 1
                    if "transcript" not in os.path.basename(d):
                        stripe_other.add(f)
                else:
                    hits["url"] += 1
    put("pii:files", f"{files:,}")
    put("pii:transcripts", f"{trans:,}")
    for k in ("email", "phone", "url", "url_stripe_api"):
        put(f"pii:{k}", hits[k])
    put("pii:total", hits["email"] + hits["phone"] + hits["url"])
    put("pii:url_clause", "no URL" if not hits["url_stripe_api"] else
        "no URL other than the Stripe API base address that the real-API-check transcripts"
        + (" and its key-check record" if stripe_other == {"stripe_check.json"} else
           f" and {len(stripe_other)} other real-API-check files" if stripe_other else "")
        + " contain")
    if hits["email"] + hits["phone"] + hits["url"]:
        note(f"WARNING PII sweep: {dict(hits)} hits in released results; check before release")


# ---------------------------------------------------------------- main
def main():
    if "--render-only" in sys.argv:
        return render_only()
    cells209 = load_209()
    check_samples(cells209)
    trans209 = transcripts_209()
    emit_209(cells209, trans209)
    emit_dups(cells209)
    emit_summaries(cells209)
    emit_misorder_turns(cells209, trans209)
    cells249 = load_249()
    trans249 = transcripts_249()
    emit_249(cells249, trans249)
    emit_baselines(cells249)
    emit_tests()
    emit_scripted()
    emit_tier0()
    emit_v1()
    emit_batch_sampling(trans209, trans249)
    emit_temps(trans209, trans249)
    emit_ident_calls(trans209)
    emit_design()
    emit_totals(cells209, cells249)
    emit_fig1()
    emit_figseam()
    emit_launches()
    emit_finish_reasons()
    emit_contrasts()
    emit_lists(cells249)
    emit_derived()
    emit_analysis_v3()
    emit_r2()
    emit_r3()
    emit_r4()
    emit_completion()
    check_analysis_v3()
    emit_prereg_timing()
    emit_cost()
    emit_errors()
    emit_pii()
    emit_prereg_table()
    write_full209(cells209)
    for _, k, disp in MODEL_SHORT:
        put(f"name:{k}", disp)
    # word forms for small counts ("five domains"): key + ":w"; capitalized
    # for a sentence start ("Nine follow-up experiments"): key + ":W"
    for k, v in list(NUMS.items()):
        if re.fullmatch(r"\d+", v) and int(v) <= 10:
            NUMS[k + ":w"] = {0: "zero"}.get(int(v), WORDS.get(int(v)))
            NUMS[k + ":W"] = NUMS[k + ":w"].capitalize()
    # denominators of the text-identical pair counts (Table tab:sampling caption)
    for k, v in list(NUMS.items()):
        m = re.fullmatch(r"(\d+)/(\d+)", v)
        if m and re.fullmatch(r"samp:\w+:\w+:ident", k):
            NUMS[k + ":n"] = m.group(2)
    # percentages for k/n values: key + ":pct"
    for k, v in list(NUMS.items()):
        m = re.fullmatch(r"(\d+)/(\d+)", v)
        if m and int(m.group(2)) > 0:
            NUMS[k + ":pct"] = f"{100 * int(m.group(1)) / int(m.group(2)):.0f}\\%"

    # list of every value main.tex uses that is not available yet
    mt = os.path.join(HERE, "main.tex")
    und, pend = [], []
    if os.path.exists(mt):
        und, pend, _ = lint(mt)

    with open(OUT_TEX, "w") as fh:
        fh.write("% GENERATED by make_numbers.py -- do not edit by hand.\n")
        fh.write("% Re-run: python3 paper-merged/make_numbers.py\n")
        for k in sorted(NUMS):
            fh.write("\\defnum{%s}{%s}\n" % (k, NUMS[k]))
        items = [r"\item \PENDINGKEY{%s} (no data yet)" % k for k in und] + \
                [r"\item \PENDINGKEY{%s} (run in progress: %s)" %
                 (k, re.sub(r".*\(([^)]*)\)\}$", r"\1", NUMS[k])) for k in pend]
        fh.write("\\def\\fullrows{%%\n%s}\n" % FULL209.replace("%", "\\%"))
        fh.write("\\def\\duprows{%%\n%s}\n" % DUPROWS.replace("%", "\\%"))
        fh.write("\\def\\preregrows{%%\n%s}\n" % PREREG)
        fh.write("\\def\\pendingcount{%d}\n" % len(items))
        fh.write("\\def\\pendinglist{%s}\n" % (
            "\\begin{itemize}\\setlength\\itemsep{0pt}\\footnotesize " + " ".join(items)
            + "\\end{itemize}" if items else "None."))
    # snapshot, so the post-LaTeX pass renders the .md files from exactly the
    # values the PDF was built with (results/ may grow while runs continue)
    with open(SNAPSHOT, "w") as fh:
        json.dump(NUMS, fh, sort_keys=True)
    print(f"main.tex values pending: {len(und)} without data, {len(pend)} in progress")
    for name, missing in render_templates():
        print(f"rendered {name}: {len(missing)} pending keys")
        for k in missing:
            print("    ", k)
    with open(OUT_REP, "w") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print(f"wrote {OUT_TEX} ({len(NUMS)} values)")
    for line in REPORT:
        if line.startswith(("PENDING", "WARNING")):
            print(" ", line)
    if "--check" in sys.argv:
        report_check()


def render_only():
    """Second build pass: re-render the .md templates with section numbers
    from main.aux, using the values numbers.tex was written with. Does not
    read results/ and does not rewrite numbers.tex."""
    NUMS.update(json.load(open(SNAPSHOT)))
    for k in [k for k in NUMS if k.startswith(("ref:", "page:"))]:
        del NUMS[k]
    for name, missing in render_templates():
        print(f"rendered {name}: {len(missing)} pending keys")
        for k in missing:
            print("    ", k)
    if "--check" in sys.argv:
        report_check()


def report_check():
    if True:
        mt = os.path.join(HERE, "main.tex")
        und, pend, lits = lint(mt)
        print(f"\n\\V keys undefined in numbers.tex (render as PENDING): {len(und)}")
        for k in und:
            print("   ", k)
        print(f"\\V keys that are in-progress PENDING: {len(pend)}")
        for k in pend:
            print("   ", k)
        print(f"literal numbers in main.tex prose (check each is not a result): "
              f"{sum(lits.values())}")
        for k, c in lits.most_common():
            print(f"    {k!r} x{c}")
        # numbers in the generated .md files must equal the snapshot the PDF used
        stale = []
        for md in ("checklist.md", "response_to_reviewers.md"):
            p = os.path.join(HERE, md)
            if os.path.exists(p) and os.path.getmtime(p) < os.path.getmtime(OUT_TEX):
                stale.append(md)
        print(f"generated .md files older than numbers.tex: {len(stale)} {stale}")


if __name__ == "__main__":
    main()
