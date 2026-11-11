#!/usr/bin/env python3
"""Robustness probe for independent/check.py.

Applies seeded random, number-changing edits (mutants) to the paper's own
number pipeline in a SCRATCH clone of the repository, regenerates the
pipeline output there (bash analysis/run_all_v3.sh && python3
paper-merged/make_numbers.py) and runs independent/check.py against it.
A mutant counts as "number-changing" when at least one macro that main.tex
prints differs from the unmutated baseline; the probe keeps drawing until
TARGET such mutants have been scored. A mutant is "caught" when the check
reports at least one unexplained mismatch.

    python3 independent/probe.py init  <repo> <scratch>   # clone + baseline
    python3 independent/probe.py step  <scratch> [seconds]  # score mutants until the budget is used
    python3 independent/probe.py report <scratch>

Never touches the real repository: everything happens under <scratch>.
"""
import ast, json, os, random, re, shutil, subprocess, sys, time

SEED = 20261030
TARGET = 20
FILES = ["paper-merged/make_numbers.py", "analysis/v3_tier1_prereg.py", "analysis/v3_bare_hint.py",
         "analysis/v3_iter1_extra.py", "analysis/v3_real_extra.py", "analysis/real_stripe.py",
         "analysis/v3_common.py", "analysis/v3_load.py", "analysis/v3_sens_oct.py", "analysis/v3_e1_6_belief.py",
         "analysis/v3_matcher_audit_all.py", "analysis/v3_numbers.py"]
CMP = {ast.Lt: ("<", "<="), ast.LtE: ("<=", "<"), ast.Gt: (">", ">="), ast.GtE: (">=", ">"),
       ast.Eq: ("==", "!="), ast.NotEq: ("!=", "==")}
ARI = {ast.Add: ("+", "-"), ast.Sub: ("-", "+"), ast.Mult: ("*", "/"), ast.Div: ("/", "*")}


def sites(path):
    """Every single-token mutation site: (start, end, old, new, description)."""
    src = open(path).read()
    tree = ast.parse(src)
    starts = [0]
    for line in src.splitlines(True):
        starts.append(starts[-1] + len(line))
    off = lambda ln, col: starts[ln - 1] + col
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in CMP:
            a = off(node.left.end_lineno, node.left.end_col_offset); b = off(node.comparators[0].lineno, node.comparators[0].col_offset)
            mid = src[a:b]; o, n = CMP[type(node.ops[0])]
            toks = re.findall(r"<=|>=|==|!=|<|>", mid)
            if toks == [o]:
                i = mid.index(o); out.append((a + i, a + i + len(o), o, n, "L%d cmp %s->%s" % (node.lineno, o, n)))
        elif isinstance(node, ast.BinOp) and type(node.op) in ARI:
            a = off(node.left.end_lineno, node.left.end_col_offset); b = off(node.right.lineno, node.right.col_offset)
            mid = src[a:b]; o, n = ARI[type(node.op)]
            if mid.count(o) == 1 and "**" not in mid and "//" not in mid:
                i = mid.index(o); out.append((a + i, a + i + 1, o, n, "L%d arith %s->%s" % (node.lineno, o, n)))
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float) and not isinstance(node.value, bool):
            a = off(node.lineno, node.col_offset); b = off(node.end_lineno, node.end_col_offset)
            txt = src[a:b]
            if isinstance(node.value, int) and txt.isdigit():
                new = str(node.value + 1)
            elif isinstance(node.value, float) and re.fullmatch(r"\d+\.\d+", txt):
                new = repr(node.value / 2 if node.value > 0 else 1.0)
            else:
                continue
            out.append((a, b, txt, new, "L%d const %s->%s" % (node.lineno, txt, new)))
    return out


def run(cmd, cwd, timeout=170):
    return subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True, timeout=timeout)


def defnums(path):
    out = {}
    for line in open(path):
        m = re.match(r"\\defnum\{([^}]*)\}\{(.*)\}$", line.rstrip("\n"))
        if m:
            out[m.group(1)] = m.group(2)
    return out


def printed_keys(main_tex):
    body = open(main_tex).read().split(r"\begin{document}", 1)[-1]
    body = re.sub(r"(?<!\\)%.*", "", body)
    keys = set(re.findall(r"\\[VT]\{([^}]*)\}", body)) | set(re.findall(r"\\ASSERT\{([^}]*)\}", body))
    for k in re.findall(r"\\C\{([^}]*)\}", body):
        keys |= {k + ":ivr", k + ":tci"}
    for k in re.findall(r"\\G\{([^}]*)\}", body):
        keys |= {k + ":givr", k + ":gci"}
    return keys


def pipeline(repo, analysis=True):
    """Regenerate the pipeline output; the analysis layer is re-run only when
    a file under analysis/ was edited (make_numbers.py reads its JSON output)."""
    t0 = time.time()
    cmd = ("bash analysis/run_all_v3.sh > /dev/null 2> .probe_v3.err && " if analysis else "") + \
        "python3 paper-merged/make_numbers.py > .probe_mk.out 2> .probe_mk.err"
    r = run(cmd, repo)
    return r.returncode, time.time() - t0


def init(repo, scratch):
    os.makedirs(scratch, exist_ok=True)
    clone = os.path.join(scratch, "repo")
    if not os.path.exists(clone):
        subprocess.run(["git", "clone", "-q", "--no-hardlinks", repo, clone], check=True)
    # the check itself (the version under test) travels with the clone
    os.makedirs(os.path.join(clone, "independent"), exist_ok=True)
    shutil.copy(os.path.join(repo, "independent", "check.py"), os.path.join(clone, "independent", "check.py"))
    rc, dt = pipeline(clone)
    base = os.path.join(scratch, "base_numbers.tex")
    shutil.copy(os.path.join(clone, "paper-merged", "numbers.tex"), base)
    same = defnums(base) == defnums(os.path.join(repo, "paper-merged", "numbers.tex"))
    state = {"seed": SEED, "repo": repo, "clone": clone, "base": base, "pipeline_rc": rc, "pipeline_s": round(dt, 1),
             "baseline_equals_repo_numbers": same, "draws": 0, "mutants": [], "discarded": []}
    json.dump(state, open(os.path.join(scratch, "probe_state.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in state.items() if k not in ("mutants", "discarded")}, indent=1))


def step(scratch, budget):
    t_start = time.time()
    sp = os.path.join(scratch, "probe_state.json")
    state = json.load(open(sp))
    clone = state["clone"]
    base = defnums(state["base"])
    keys = printed_keys(os.path.join(clone, "paper-merged", "main.tex"))
    allsites = []
    for f in FILES:
        p = os.path.join(clone, f)
        if os.path.exists(p):
            allsites += [(f,) + s for s in sites(p)]
    rng = random.Random(state["seed"])
    for _ in range(state["draws"]):
        rng.randrange(len(allsites))
    while len(state["mutants"]) < TARGET and time.time() - t_start < budget:
        idx = rng.randrange(len(allsites)); state["draws"] += 1
        f, a, b, old, new, desc = allsites[idx]
        path = os.path.join(clone, f)
        orig = open(path).read()
        if orig[a:b] != old:
            state["discarded"].append({"draw": state["draws"], "file": f, "desc": desc, "why": "site drifted"})
            continue
        try:
            open(path, "w").write(orig[:a] + new + orig[b:])
            rc, dt = pipeline(clone, analysis=f.startswith("analysis/"))
            if rc != 0:
                state["discarded"].append({"draw": state["draws"], "file": f, "desc": desc, "why": "pipeline failed (rc %d)" % rc})
                continue
            mut = defnums(os.path.join(clone, "paper-merged", "numbers.tex"))
            changed = sorted(k for k in keys if base.get(k) != mut.get(k))
            if not changed:
                state["discarded"].append({"draw": state["draws"], "file": f, "desc": desc, "why": "no printed macro changed"})
                continue
            rep = os.path.join(scratch, "report_%02d.json" % (len(state["mutants"]) + 1))
            r = run("python3 -I independent/check.py --root . --report %s --quiet" % rep, clone)
            report = json.load(open(rep)) if os.path.exists(rep) else {}
            mism = [m["key"] for m in report.get("mismatches", []) if not m.get("explained")]
            covered = sorted(k for k in changed if k in report.get("recomputed_values", {}))
            state["mutants"].append({"id": len(state["mutants"]) + 1, "draw": state["draws"], "file": f, "mutation": desc,
                                     "changed_printed_macros": len(changed), "changed_examples": changed[:8],
                                     "changed_covered_by_check": len(covered), "check_rc": r.returncode,
                                     "unexplained_mismatches": len(mism), "caught": bool(mism) or r.returncode != 0,
                                     "pipeline_s": round(dt, 1)})
            print(json.dumps(state["mutants"][-1]))
        finally:
            open(path, "w").write(orig)
            subprocess.run(["git", "-C", clone, "checkout", "-q", "--", "results", "paper-merged"], check=False)
            if os.path.exists(os.path.join(clone, "paper-merged", "numbers.tex")) is False:
                pass
        json.dump(state, open(sp, "w"), indent=1)
    json.dump(state, open(sp, "w"), indent=1)
    print("mutants scored: %d / %d, draws %d, discarded %d" % (len(state["mutants"]), TARGET, state["draws"], len(state["discarded"])))


def report(scratch):
    state = json.load(open(os.path.join(scratch, "probe_state.json")))
    ms = state["mutants"]
    caught = sum(1 for m in ms if m["caught"])
    cov = [m for m in ms if m["changed_covered_by_check"]]
    print("seed %d: %d number-changing mutants from %d draws (%d discarded: %s)" % (
        state["seed"], len(ms), state["draws"], len(state["discarded"]),
        dict(__import__("collections").Counter(d["why"] for d in state["discarded"]))))
    print("caught %d / %d; with at least one changed macro covered by the check: %d, of which caught %d" % (
        caught, len(ms), len(cov), sum(1 for m in cov if m["caught"])))
    for m in ms:
        print("  #%02d %-32s %-28s changed %3d (covered %3d) -> %s" % (m["id"], m["file"].split("/")[-1], m["mutation"],
              m["changed_printed_macros"], m["changed_covered_by_check"], "CAUGHT" if m["caught"] else "missed"))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "init":
        init(os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3]))
    elif cmd == "step":
        step(os.path.abspath(sys.argv[2]), float(sys.argv[3]) if len(sys.argv) > 3 else 120)
    else:
        report(os.path.abspath(sys.argv[2]))
