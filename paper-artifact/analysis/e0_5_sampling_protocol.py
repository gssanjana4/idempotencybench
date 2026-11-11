"""E0-5: sampling-protocol table for the v1 data of BOTH papers.

Per model (and paper): requested vs effective temperature, dated id vs
alias, served id, run dates, identical-sample-pair rate, verdict agreement
and effective n.

Sources and what each can and cannot tell us:
  - temperature: 209 transcripts carry a `temperature` field (effective, as
    recorded by the runner). 249 transcripts carry NO temperature field; the
    value is inferred from the code at the run commit (src2/sweep2.py sent
    T=0.0 with the same reject-and-fall-back logic as 209) and labelled
    'inferred (not logged)'.
  - requested temperature: TEMPERATURE constant in src/model_agent.py at the
    run commits (read from git history, c49a67e for 209, 009d234 for 249).
  - served model id: not logged in v1 (no `model_served` field) -> 'not logged'.
  - run dates: v1 transcripts have no timestamps. Reported: file mtimes (UTC,
    first..last; valid only if the files were not rewritten since, which
    their July mtimes indicate) and the git commits that added them.
  - identical pair: s0 and s1 transcripts identical after stripping
    tool_use ids and the random ack ids (ok-xxxxxxxx). Two variants:
    'text' ignores thinking blocks (stored content-free as {'type':'thinking'}),
    'full' also requires the same number of thinking blocks.
  - verdict agreement: s0 and s1 give the same violation verdict (209: v2
    rescored violations > 0 from E0-1; 249: givr_hit).
  - n_eff (209): sum over cells of n/(1+ICC) from E0-1 (ICC clipped to [0,1],
    a cell with undefined ICC contributes n_tasks if its verdicts all agree,
    else n). Reported alongside the conservative bound n_tasks.

Reads : results/sweep_transcripts, results/transcripts2, results/sweep2_raw.jsonl,
        results/analysis_v2/e0_1_corrected_table.csv (run e0_1 first), git history
Writes: results/analysis_v2/e0_5_sampling_protocol.csv, e0_5_summary.json
Run   : python3 analysis/e0_5_sampling_protocol.py
No sampling; deterministic.
"""
import csv, datetime, glob, json, os, re, subprocess
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "analysis_v2")
F209 = re.compile(r"^(?P<cell>.+)__s(?P<s>\d)__(?P<task>.+)\.json$")
ACK = re.compile(r"ok-[0-9a-f]{8}")
DATED = re.compile(r"-\d{8}$")


def canon_turns(turns, keep_thinking):
    out = []
    for tu in turns:
        if tu["role"] not in ("assistant", "tool_results"):
            continue
        c = []
        for b in tu["content"]:
            b = dict(b); b.pop("id", None); b.pop("tool_use_id", None)
            if "content" in b:
                b["content"] = ACK.sub("ok-X", str(b["content"]))
            if not keep_thinking and b.get("type") == "thinking":
                continue
            c.append(b)
        out.append((tu["role"], c))
    return json.dumps(out, sort_keys=True)


def git(*a):
    try:
        return subprocess.run(["git", "-C", ROOT, *a], capture_output=True, text=True, check=True).stdout
    except Exception as e:
        return f"ERROR: {e}"


def requested_temperature(commit, path):
    src = git("show", f"{commit}:{path}")
    m = re.search(r"^TEMPERATURE\s*=\s*([0-9.]+)", src, re.M)
    return float(m.group(1)) if m else "not found"


def utc(ts):
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")


def commits_adding(path):
    return [l for l in git("log", "--diff-filter=A", "--format=%h %ad", "--date=short", "--", path).splitlines() if l]


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []

    # ---------------- single-session study ----------------
    req209 = requested_temperature("c49a67e", "src/model_agent.py")
    by = defaultdict(dict); temps = defaultdict(lambda: defaultdict(int)); mt = defaultdict(list)
    served = defaultdict(int)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "sweep_transcripts", "*.json"))):
        m = F209.match(os.path.basename(f)); t = json.load(open(f))
        mo = t["model"]
        temps[mo][repr(t.get("temperature", "absent"))] += 1
        served[mo] += "model_served" in t
        mt[mo].append(os.stat(f).st_mtime)
        by[(mo, m["cell"], m["task"])][m["s"]] = (canon_turns(t["turns"], False), canon_turns(t["turns"], True))
    verdict = {}
    e01 = os.path.join(OUT, "e0_1_episodes.csv")
    if not os.path.exists(e01):
        raise SystemExit("run python3 analysis/e0_1_corrected_table.py first")
    for r in csv.DictReader(open(e01)):
        verdict[(r["model"], r["prompt"], r["mitigation"], r["failure_mode"], r["task_id"], r["sample"])] = int(r["violations"]) > 0
    neff = defaultdict(float); ntask = defaultdict(int); nep = defaultdict(int)
    for r in csv.DictReader(open(os.path.join(OUT, "e0_1_corrected_table.csv"))):
        mo = r["model"]; n = int(r["n_episodes"]); T = int(r["n_tasks"])
        nep[mo] += n; ntask[mo] += T
        if r["n_eff_icc"]:
            neff[mo] += float(r["n_eff_icc"])
        else:
            neff[mo] += T if int(r["verdict_agree_tasks"]) == T else n
    agree = defaultdict(int); npairs = defaultdict(int)
    for (mo, p, mi, fm, tk, s), v in verdict.items():
        if s == "0" and (mo, p, mi, fm, tk, "1") in verdict:
            npairs[mo] += 1; agree[mo] += v == verdict[(mo, p, mi, fm, tk, "1")]
    added209 = commits_adding("results/sweep_transcripts")
    for mo in sorted(temps):
        pairs = [d for (m2, _, _), d in by.items() if m2 == mo and "0" in d and "1" in d]
        ident_text = sum(d["0"][0] == d["1"][0] for d in pairs)
        ident_full = sum(d["0"][1] == d["1"][1] for d in pairs)
        eff = dict(temps[mo])
        rows.append(dict(
            paper="209", model=mo, id_type="dated" if DATED.search(mo) else "alias (undated)",
            model_served="not logged" if served[mo] == 0 else f"logged in {served[mo]}",
            temperature_requested=req209,
            temperature_effective="; ".join(f"{k} x{v}" for k, v in eff.items()) + " (logged)",
            episodes=sum(eff.values()),
            run_dates_utc_from_mtime=f"{utc(min(mt[mo]))} .. {utc(max(mt[mo]))}",
            git_commits_adding_transcripts=" | ".join(added209) or "none found",
            sample_pairs=len(pairs), identical_pairs_text=ident_text, identical_pairs_full=ident_full,
            identical_pair_rate=round(ident_text / len(pairs), 4),
            verdict_agree_pairs=agree[mo], verdict_agree_rate=round(agree[mo] / npairs[mo], 4),
            n_episodes_total=nep[mo], n_tasks_conservative=f"{ntask[mo]} task-cell clusters (32 x 9 cells)", n_eff_icc=round(neff[mo], 1)))

    # ---------------- session-resume study ----------------
    req249 = requested_temperature("009d234", "src/model_agent.py")
    sw2 = git("show", "009d234:src2/sweep2.py")
    sends_temp_249 = "kwargs[\"temperature\"] = TEMPERATURE" in sw2
    by2 = defaultdict(dict); temp_field = defaultdict(int); n2 = defaultdict(int); mt2 = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "transcripts2", "*.json"))):
        t = json.load(open(f)); mo = t["model"]
        n2[mo] += 1; temp_field[mo] += "temperature" in t; mt2[mo].append(os.stat(f).st_mtime)
        k = (mo, t["resume_context"], t["mitigation"], t["graph_id"], t["interrupt_node"])
        by2[k][t["sample"]] = (json.dumps([canon_turns(s["turns"], False) for s in t["sessions"]]),
                               json.dumps([canon_turns(s["turns"], True) for s in t["sessions"]]))
    hit = {}
    for line in open(os.path.join(ROOT, "results", "sweep2_raw.jsonl")):
        r = json.loads(line)
        hit[(r["model"], r["resume_context"], r["mitigation"], r["graph_id"], r["interrupt_node"], r["sample"])] = r["givr_hit"]
    added249 = commits_adding("results/transcripts2")
    t209 = {r["model"]: r["temperature_effective"] for r in rows}
    for mo in sorted(n2):
        pairs = [(k, d) for k, d in by2.items() if k[0] == mo and 0 in d and 1 in d]
        it = sum(d[0][0] == d[1][0] for _, d in pairs); fu = sum(d[0][1] == d[1][1] for _, d in pairs)
        ag = sum(hit.get(k + (0,)) == hit.get(k + (1,)) for k, _ in pairs)
        graphs = len({k[3] for k in by2 if k[0] == mo})
        inferred = ("0.0" if "0.0 x" in t209.get(mo, "") else "None (provider default)" if "None x" in t209.get(mo, "") else "unknown")
        rows.append(dict(
            paper="249", model=mo, id_type="dated" if DATED.search(mo) else "alias (undated)",
            model_served="not logged",
            temperature_requested=req249 if sends_temp_249 else "not sent",
            temperature_effective=(f"{inferred} - inferred (not logged; {temp_field[mo]}/{n2[mo]} transcripts have a "
                                   f"temperature field); same reject-and-fall-back code as 209"),
            episodes=n2[mo],
            run_dates_utc_from_mtime=f"{utc(min(mt2[mo]))} .. {utc(max(mt2[mo]))}",
            git_commits_adding_transcripts=" | ".join(added249) or "none found",
            sample_pairs=len(pairs), identical_pairs_text=it, identical_pairs_full=fu,
            identical_pair_rate=round(it / len(pairs), 4) if pairs else None,
            verdict_agree_pairs=ag, verdict_agree_rate=round(ag / len(pairs), 4) if pairs else None,
            n_episodes_total=n2[mo], n_tasks_conservative=f"{graphs} distinct graphs",
            n_eff_icc="not computed (use graph-level n)"))

    with open(os.path.join(OUT, "e0_5_sampling_protocol.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = dict(rows=rows, notes=[
        "Served model ids were not logged in v1 for either paper: report alias + access window as a limitation.",
        "v1 transcripts have no timestamps; run dates are file mtimes (UTC) cross-checked against git commit dates.",
        "249 effective temperature is inferred from code + the 209 behaviour of the same model ids, not logged.",
        "identical_pairs_text ignores thinking blocks (they are stored content-free); identical_pairs_full also "
        "requires the same number of thinking blocks. REVISION-NOTES' 4/288 for Sonnet matches the text variant."])
    json.dump(summary, open(os.path.join(OUT, "e0_5_summary.json"), "w"), indent=1)
    for r in rows:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
