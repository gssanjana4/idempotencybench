"""Cost estimate for R3 (configs/r3_wording.json) and for the R2 completion,
from the token usage logged in the October cross-provider runs and in R2,
plus the registered cut order when the OpenRouter estimate exceeds its cap.

  python3 src/r3_estimate.py [--config configs/r3_wording.json] [--json]
  python3 src/r3_estimate.py --cuts     # apply the cut order to the config (prints what would be cut)

Basis (read-only):
  OpenRouter prices   fitted per model from results/tier1/e1_1 and e1_2
      transcripts, where OpenRouter's usage.cost is logged per turn
      (least squares of episode cost on input / output tokens; the fit is
      exact to < $0.001 per episode).
  R3-B (OpenRouter)   results/crossprovider/sweep_transcripts/, base prompt,
      no mitigation: timeout_after_commit and error_500_after_commit as
      logged; the three unknown wordings (W3, p1, p3) priced like the bare
      500 (no OpenRouter W3 episode exists); worst case = one more model
      turn per episode (a read-back before the retry).
  R3-A (OpenRouter)   results/crossprovider/transcripts2/: amnesiac and
      bare_hint as logged; hint_p1 / hint_p3 priced like bare_hint; worst
      case like amnesiac (the note did nothing).
  R3-B (Fable 5)      no Fable retry episode has logged tokens (the July
      transcripts predate usage logging), so the Sonnet 5 R2 Part-B cells
      (results/r2_wording/transcripts/, same cells, prompt caching on) are
      re-priced at Fable list prices (cache tokens as logged); worst case
      x1.5 (Fable's October restart sessions used ~1.4x Haiku's output).
  R2 completion       the remaining episodes of configs/r2_wording.json:
      Fable hint cells at the R2 logged Fable bare_hint / hint_p1 mean
      (worst: amnesiac mean); Part C plan_only at the October plan_only
      usage under prompt caching (src/r2_estimate.cached_input_equiv).
Cut order (registered), applied until the OpenRouter estimate fits the cap:
  1. drop status_unknown_p3 from R3-B; 2. drop hint_p3 from R3-A;
  3. drop Gemini from R3-A.
"""
import argparse, copy, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import sweep                       # noqa: E402
import r2_estimate                 # noqa: E402

RES = os.path.join(HERE, "results")
XP = os.path.join(RES, "crossprovider")
R2 = os.path.join(RES, "r2_wording", "transcripts")
PRICE_SRC = glob.glob(os.path.join(RES, "tier1", "e1_[12]", "transcripts"))
GEMINI = "openrouter:google/gemini-4-argon-20261013"
P3_B = "status_unknown_p3_after_commit"
P3_A = "hint_p3"
PROXY_B = {"status_unknown_after_commit": "error_500_after_commit",
           "status_unknown_p1_after_commit": "error_500_after_commit",
           "status_unknown_p2_after_commit": "error_500_after_commit",
           P3_B: "error_500_after_commit"}
PROXY_A = {"hint_p1": "bare_hint", "hint_p2": "bare_hint", P3_A: "bare_hint"}
FABLE_WORST = 1.5


def _load(f):
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception:
        return None


def _usage(t):
    """(input, output, cache_read, cache_write, assistant turns) summed over sessions."""
    u = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    turns = 0
    if "sessions" in t:
        for s in t["sessions"]:
            for k, v in ((s.get("meta") or {}).get("usage") or {}).items():
                u[k] = u.get(k, 0) + (v or 0)
            turns += sum(1 for x in s["turns"] if x["role"] == "assistant")
    else:
        for k, v in (t.get("usage") or {}).items():
            u[k] = u.get(k, 0) + (v or 0)
        turns = sum(1 for x in t.get("turns", []) if x["role"] == "assistant")
    return u, turns


def fit_prices(dirs=None) -> dict:
    """{model: (usd per input MTok, usd per output MTok)} from logged usage.cost."""
    pts = defaultdict(list)
    for d in dirs or PRICE_SRC:
        for f in glob.glob(os.path.join(d, "*.json")):
            t = _load(f)
            if not t or not str(t.get("model", "")).startswith("openrouter:"):
                continue
            u, _ = _usage(t)
            c = sweep.episode_cost(t)
            if c > 0 and u["input"]:
                pts[t["model"]].append((u["input"], u["output"], c))
    out = {}
    for m, xs in pts.items():
        sxx = sum(x * x for x, _, _ in xs); syy = sum(y * y for _, y, _ in xs)
        sxy = sum(x * y for x, y, _ in xs); sxc = sum(x * c for x, _, c in xs)
        syc = sum(y * c for _, y, c in xs)
        det = sxx * syy - sxy * sxy
        if det <= 0:
            continue
        a = (sxc * syy - syc * sxy) / det; b = (sxx * syc - sxy * sxc) / det
        out[m] = (a * 1e6, b * 1e6, len(xs))
    return out


def _or_cost(u, pr):
    return (u["input"] * pr[0] + u["output"] * pr[1]) / 1e6


def logged_openrouter(prices) -> dict:
    """(model, condition) -> list of (cost, worst_cost) per logged episode."""
    out = defaultdict(list)
    for f in glob.glob(os.path.join(XP, "sweep_transcripts", "*.json")):
        t = _load(f)
        m = t and t.get("model")
        if not t or m not in prices:
            continue
        rest = os.path.basename(f).split("__")[0][len(m.replace("/", "_")) + 1:]
        prompt, mit, fm = rest.split("_", 2)
        if (prompt, mit) != ("base", "none"):
            continue
        u, turns = _usage(t)
        c = _or_cost(u, prices[m])
        out[(m, fm)].append((c, c * (turns + 1) / max(turns, 1)))
    for f in glob.glob(os.path.join(XP, "transcripts2", "*.json")):
        t = _load(f)
        m = t and t.get("model")
        if not t or m not in prices or t.get("mitigation") != "none":
            continue
        u, _ = _usage(t)
        c = _or_cost(u, prices[m])
        out[(m, t["resume_context"])].append((c, c))
    return out


def logged_fable_b() -> dict:
    """condition -> list of (cost, worst) with Sonnet's R2 Part-B tokens at Fable prices."""
    pr = sweep.price_for("claude-fable-5")
    out = defaultdict(list)
    for f in glob.glob(os.path.join(R2, "claude-sonnet-5_*.json")):
        t = _load(f)
        if not t or "sessions" in t:
            continue
        u, _ = _usage(t)
        c = (u["input"] * pr[0] + u["output"] * pr[1] + u["cache_write"] * pr[0] * sweep.CACHE_WRITE_MULT
             + u["cache_read"] * pr[0] * sweep.CACHE_READ_MULT) / 1e6
        out[t["failure_mode"]].append((c, c * FABLE_WORST))
    return out


def logged_r2_restart() -> dict:
    """(model, context) -> list of logged R2 restart costs (prompt caching as logged)."""
    out = defaultdict(list)
    for f in glob.glob(os.path.join(R2, "*.json")):
        t = _load(f)
        if not t or "sessions" not in t:
            continue
        out[(t["model"], t["resume_context"])].append(sweep.episode_cost(t))
    return out


def mean(xs, i=None):
    if not xs:
        return None
    return sum(x if i is None else x[i] for x in xs) / len(xs)


def _cells(cfg):
    """[(group index, part, model, condition, n)] of a config."""
    out = []
    for gi, g in enumerate(cfg["groups"]):
        for m in g["models"]:
            for c in g.get("cells", []):
                out.append((gi, "B", m, c[2], 32 * cfg["samples"]))
            for c in g.get("restart_cells", []):
                out.append((gi, "A", m, c[0], (16 if c[2] == "all" else 8) * cfg.get("restart_samples", cfg["samples"])))
    return out


def estimate_r3(cfg, prices=None, logs=None, fable=None) -> dict:
    prices = prices if prices is not None else fit_prices()
    logs = logs if logs is not None else logged_openrouter(prices)
    fable = fable if fable is not None else logged_fable_b()
    rows = []
    for gi, part, m, cond, n in _cells(cfg):
        if m.startswith("openrouter:"):
            key = (PROXY_B if part == "B" else PROXY_A).get(cond, cond)
            xs = logs.get((m, key))
            worst = logs.get((m, "amnesiac")) if part == "A" else xs
            prov = "openrouter"
            basis = f"{len(xs)} October {key} episodes" if xs else "none"
            c, w = (mean(xs, 0), mean(worst or xs, 1)) if xs else (None, None)
        else:
            xs = fable.get(cond)
            prov = "anthropic"
            basis = f"{len(xs)} Sonnet R2 {cond} episodes at Fable prices" if xs else "none"
            c, w = (mean(xs, 0), mean(xs, 1)) if xs else (None, None)
        rows.append({"group": gi, "part": part, "model": m, "condition": cond, "n": n, "provider": prov,
                     "basis": basis, "per_ep": c, "cost": n * c if c is not None else None,
                     "worst": n * w if w is not None else None})
    tot = {}
    for prov in ("openrouter", "anthropic"):
        rs = [r for r in rows if r["provider"] == prov and r["cost"] is not None]
        tot[prov] = {"episodes": sum(r["n"] for r in rows if r["provider"] == prov),
                     "cost": sum(r["cost"] for r in rs), "worst": sum(r["worst"] for r in rs),
                     "unpriced": sum(r["n"] for r in rows if r["provider"] == prov and r["cost"] is None)}
    return {"rows": rows, "totals": tot, "prices": {m: list(p) for m, p in prices.items()}}


def estimate_r2_completion(cfg_path=None) -> dict:
    """Remaining R2 episodes (per model / cell) and their estimated Anthropic cost."""
    import tier1
    cfg = tier1.load_config(cfg_path or os.path.join(HERE, "configs", "r2_wording.json"))
    done = tier1.done_ids(cfg)
    rem = [e for e in tier1.plan(cfg) if tier1.eid_of(e) not in done]
    r2 = logged_r2_restart()
    oct_a = r2_estimate.logged("A")        # (model, ctx) -> (plain, cached)
    rows = defaultdict(lambda: {"n": 0, "cost": 0.0, "worst": 0.0, "basis": ""})
    for e in rem:
        m = e[0]
        if tier1.is_restart(e):
            ctx = e[1]
            if ctx == "plan_only":
                xs = oct_a.get((m, "plan_only"))
                c = w = mean(xs, 1) if xs else None
                basis = f"{len(xs)} October plan_only episodes (cached pricing)" if xs else "none"
            else:
                xs = r2.get((m, "bare_hint"), []) + r2.get((m, "hint_p1"), [])
                ws = r2.get((m, "amnesiac"))
                c = mean(xs); w = mean(ws) if ws else c
                basis = f"{len(xs)} R2 {m} bare_hint/hint_p1 episodes" if xs else "none"
            key = (m, "restart", ctx)
        else:
            xs = r2.get((m, e[3]))
            c = w = mean(xs) if xs else None
            basis = f"{len(xs)} R2 episodes" if xs else "none"
            key = (m, "retry", e[3])
        r = rows[key]
        r["n"] += 1; r["basis"] = basis
        if c is not None:
            r["cost"] += c; r["worst"] += w
        else:
            r["unpriced"] = r.get("unpriced", 0) + 1
    out = [{"model": k[0], "kind": k[1], "condition": k[2], **v} for k, v in sorted(rows.items())]
    return {"rows": out, "remaining": len(rem), "planned": len(rem) + len(done),
            "cost": sum(r["cost"] for r in out), "worst": sum(r["worst"] for r in out)}


def apply_cuts(cfg, cap, est_fn=None) -> tuple:
    """Apply the registered cut order to a copy of cfg until the OpenRouter
    estimate (est_fn(cfg) -> usd) is <= cap. Returns (cfg, [cut names], est)."""
    est_fn = est_fn or (lambda c: estimate_r3(c)["totals"]["openrouter"]["cost"])
    cfg = copy.deepcopy(cfg)
    cuts = []
    steps = [
        ("drop status_unknown_p3 from R3-B",
         lambda c: [g.__setitem__("cells", [x for x in g.get("cells", []) if x[2] != P3_B]) for g in c["groups"]]),
        ("drop hint_p3 from R3-A",
         lambda c: [g.__setitem__("restart_cells", [x for x in g.get("restart_cells", []) if x[0] != P3_A]) for g in c["groups"]]),
        ("drop Gemini from R3-A",
         lambda c: [g.__setitem__("models", [m for m in g["models"] if m != GEMINI]) for g in c["groups"] if g.get("restart_cells")]),
    ]
    est = est_fn(cfg)
    for name, fn in steps:
        if est <= cap:
            break
        fn(cfg)
        cfg["groups"] = [g for g in cfg["groups"] if g["models"] and (g.get("cells") or g.get("restart_cells"))]
        cuts.append(name)
        est = est_fn(cfg)
    return cfg, cuts, est


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "r3_wording.json"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cuts", action="store_true", help="apply the registered cut order to the config's OpenRouter cap")
    ap.add_argument("--no-r2", action="store_true", help="skip the R2 completion estimate")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    est = estimate_r3(cfg)
    r2c = None if a.no_r2 else estimate_r2_completion()
    cap_or = float(cfg.get("exp_or_budget_usd") or 0)
    cuts = []
    if a.cuts and cap_or:
        _, cuts, _ = apply_cuts(cfg, cap_or)
    if a.json:
        print(json.dumps({"r3": est, "r2_completion": r2c, "cuts": cuts}, indent=1))
        return 0
    print("R3 cost estimate (October logged tokens x fitted OpenRouter prices; Fable from Sonnet's R2 Part-B tokens at Fable list prices)")
    for m, p in sorted(est["prices"].items()):
        print(f"  OpenRouter price fit {m}: ${p[0]:.2f} in / ${p[1]:.2f} out per MTok ({int(p[2])} episodes)")
    by = defaultdict(lambda: {"n": 0, "cost": 0.0, "worst": 0.0})
    for r in est["rows"]:
        k = (r["part"], r["model"]); by[k]["n"] += r["n"]
        if r["cost"] is not None:
            by[k]["cost"] += r["cost"]; by[k]["worst"] += r["worst"]
    print(f"  {'part':4s} {'model':44s} {'episodes':>8s} {'estimate':>9s} {'worst':>9s}")
    for (part, m), v in sorted(by.items()):
        print(f"  {part:4s} {m:44s} {v['n']:8d} ${v['cost']:8.2f} ${v['worst']:8.2f}")
    t = est["totals"]
    print(f"  OpenRouter: {t['openrouter']['episodes']} episodes, ${t['openrouter']['cost']:.2f} "
          f"(worst ${t['openrouter']['worst']:.2f}); cap exp_or_budget_usd ${cap_or:.2f}")
    print(f"  Anthropic (Fable R3-B): {t['anthropic']['episodes']} episodes, ${t['anthropic']['cost']:.2f} "
          f"(worst ${t['anthropic']['worst']:.2f})")
    if r2c:
        print(f"R2 completion: {r2c['remaining']} of {r2c['planned']} episodes remaining, "
              f"est. ${r2c['cost']:.2f} (worst ${r2c['worst']:.2f}) Anthropic")
        for r in r2c["rows"]:
            print(f"  {r['model']:28s} {r['kind']:8s} {r['condition']:14s} {r['n']:4d}  ${r['cost']:.2f}  ({r['basis']})")
        print(f"Anthropic total (R2 completion + Fable R3-B): ${r2c['cost'] + t['anthropic']['cost']:.2f} "
              f"(worst ${r2c['worst'] + t['anthropic']['worst']:.2f}); shared cap exp_budget_usd ${float(cfg.get('exp_budget_usd') or 0):.2f}")
    if a.cuts and cap_or:
        print("cuts needed for the OpenRouter cap: " + (", ".join(cuts) if cuts else "none"))
    missing = [r for r in est["rows"] if r["cost"] is None]
    if missing:
        print("no logged basis for: " + ", ".join(f"{r['model']}/{r['condition']}" for r in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
