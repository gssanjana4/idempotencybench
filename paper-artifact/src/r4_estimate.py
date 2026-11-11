"""Cost and runtime estimate for R4 (configs/r4_precommit.json), from the
token usage logged in October, plus the registered rule for the optional
OpenRouter models (GPT-5.5, Gemini).

  python3 src/r4_estimate.py [--config configs/r4_precommit.json] [--json]
  python3 src/r4_estimate.py --optional   # also price GPT-5.5 and Gemini and apply the rule

Basis (read-only; no network):
  OpenRouter  results/crossprovider/sweep_transcripts/ (2026-10-25/26), prompt
      base, mitigation none, `timeout_before_commit` AS LOGGED (the same
      pre-commit failure on the same call); `error_500_before_commit` and
      `status_unknown_before_commit` priced like it. Prices per model fitted
      from the October usage.cost logs (r3_estimate.fit_prices). Worst case =
      one more model turn per episode (a read-back before the retry).
  Anthropic   no Claude pre-commit episode has logged tokens (the July
      timeout_before_commit cells predate usage logging), so the R2 Part-B
      cells of the same model (results/r2_wording/transcripts/, 2026-10-31,
      prompt caching ON, cache tokens as logged and priced at 1.25x / 0.1x)
      are used: timeout_after_commit -> timeout_before_commit,
      error_500_after_commit -> error_500_before_commit, status_unknown_after_commit
      (W3) -> status_unknown_before_commit, each scaled by (turns + 1) / turns
      (one more model turn: the retry of the action that did not commit;
      October OpenRouter pre-commit episodes ran ~1 turn longer than
      post-commit ones); worst case (turns + 2) / turns.
  Runtime     mean logged wall time per episode (started_utc -> finished_utc)
      of the same model in R2 (Claude) / R3 (OpenRouter, the newest logged
      OpenRouter wall times; the October sweep logged no finish time), scaled
      like the cost, divided by the config's workers; phases run one after
      the other.
Optional models (registered rule): GPT-5.5 and Gemini are added only if the
estimate WITH both stays <= $8 in total AND the OpenRouter part stays <= the
$4 OpenRouter cap (exp_or_budget_usd); otherwise both are left out.
"""
import argparse, glob, json, os, sys
from collections import defaultdict
from datetime import datetime

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import sweep                       # noqa: E402
import r3_estimate                 # noqa: E402

RES = os.path.join(HERE, "results")
R2T = os.path.join(RES, "r2_wording", "transcripts")
R3T = os.path.join(RES, "r3_wording", "transcripts")
GPT = "openrouter:openai/gpt-5.5-20260423"
GEMINI = "openrouter:google/gemini-4-argon-20261013"
OPTIONAL = [GPT, GEMINI]
TOTAL_MAX = 8.0                    # registered: optional models only if the total stays <= $8
TOB = "timeout_before_commit"
CELLS = [TOB, "error_500_before_commit", "status_unknown_before_commit"]
# Claude basis: the post-commit cell with the same error text and class
PROXY_CLAUDE = {TOB: "timeout_after_commit", "error_500_before_commit": "error_500_after_commit",
                "status_unknown_before_commit": "status_unknown_after_commit"}


def _turns(t):
    return max(1, sum(1 for x in t.get("turns", []) if x["role"] == "assistant"))


def _dur(t):
    try:
        a = datetime.fromisoformat(t["started_utc"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(t["finished_utc"].replace("Z", "+00:00"))
        return (b - a).total_seconds()
    except Exception:
        return None


def logged_claude(tdir=None) -> dict:
    """(model, R4 cell) -> [(cost, worst, seconds)] from R2 Part-B transcripts."""
    inv = {v: k for k, v in PROXY_CLAUDE.items()}
    out = defaultdict(list)
    for f in glob.glob(os.path.join(tdir or R2T, "*.json")):
        t = r3_estimate._load(f)
        if not t or "sessions" in t or (t.get("prompt"), t.get("mitigation")) != ("base", "none"):
            continue
        cell = inv.get(t.get("failure_mode"))
        if cell is None:
            continue
        n = _turns(t); c = sweep.episode_cost(t); d = _dur(t)
        out[(t["model"], cell)].append((c * (n + 1) / n, c * (n + 2) / n,
                                        d * (n + 1) / n if d is not None else None))
    return out


def or_seconds(tdir=None) -> dict:
    """model -> mean logged wall seconds per retry episode (R3, OpenRouter)."""
    xs = defaultdict(list)
    for f in glob.glob(os.path.join(tdir or R3T, "*.json")):
        t = r3_estimate._load(f)
        if not t or "sessions" in t or not str(t.get("model", "")).startswith("openrouter:"):
            continue
        d = _dur(t)
        if d is not None:
            xs[t["model"]].append(d)
    return {m: sum(v) / len(v) for m, v in xs.items()}


def mean(xs, i):
    xs = [x[i] for x in xs if x[i] is not None]
    return sum(xs) / len(xs) if xs else None


def estimate_r4(cfg, models=None, prices=None, logs=None, claude=None, secs=None) -> dict:
    prices = prices if prices is not None else r3_estimate.fit_prices()
    logs = logs if logs is not None else r3_estimate.logged_openrouter(prices)
    claude = claude if claude is not None else logged_claude()
    secs = secs if secs is not None else or_seconds()
    rows = []
    for g in cfg["groups"]:
        for m in (models or g["models"]):
            for c in g["cells"]:
                cond = c[2]; n = 32 * cfg["samples"]
                if m.startswith("openrouter:"):
                    xs = logs.get((m, TOB)); prov = "openrouter"
                    basis = f"{len(xs)} October {TOB} episodes" if xs else "none"
                    per = r3_estimate.mean(xs, 0) if xs else None
                    worst = r3_estimate.mean(xs, 1) if xs else None
                    sec = secs.get(m)
                else:
                    xs = claude.get((m, cond)); prov = "anthropic"
                    basis = f"{len(xs)} R2 {PROXY_CLAUDE[cond]} episodes x (turns+1)/turns" if xs else "none"
                    per, worst, sec = (mean(xs, 0), mean(xs, 1), mean(xs, 2)) if xs else (None, None, None)
                rows.append({"model": m, "condition": cond, "n": n, "provider": prov, "basis": basis,
                             "per_ep": per, "cost": n * per if per is not None else None,
                             "worst": n * worst if worst is not None else None,
                             "seconds": n * sec if sec is not None else None})
    tot = {}
    for prov in ("anthropic", "openrouter"):
        rs = [r for r in rows if r["provider"] == prov]
        tot[prov] = {"episodes": sum(r["n"] for r in rs),
                     "cost": sum(r["cost"] or 0 for r in rs), "worst": sum(r["worst"] or 0 for r in rs),
                     "unpriced": sum(r["n"] for r in rs if r["cost"] is None)}
    w = max(1, int(cfg.get("workers") or 1))
    tot["minutes"] = sum(r["seconds"] or 0 for r in rows) / w / 60.0
    tot["cost"] = tot["anthropic"]["cost"] + tot["openrouter"]["cost"]
    tot["worst"] = tot["anthropic"]["worst"] + tot["openrouter"]["worst"]
    return {"rows": rows, "totals": tot, "prices": {m: list(p) for m, p in prices.items()}}


def optional_rule(cfg, est_fn=None) -> dict:
    """The registered rule: add GPT-5.5 and Gemini only if total <= $8 and
    OpenRouter <= its cap with both added. Returns the decision and numbers."""
    est_fn = est_fn or (lambda c, ms: estimate_r4(c, ms)["totals"])
    base = list(cfg["groups"][0]["models"])
    withopt = base + [m for m in OPTIONAL if m not in base]
    t = est_fn(cfg, withopt)
    cap = float(cfg.get("exp_or_budget_usd") or 0)
    ok = t["cost"] <= TOTAL_MAX and (not cap or t["openrouter"]["cost"] <= cap)
    return {"add_optional": ok, "total_with": t["cost"], "or_with": t["openrouter"]["cost"],
            "total_max": TOTAL_MAX, "or_cap": cap}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "r4_precommit.json"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--optional", action="store_true", help="price GPT-5.5 and Gemini and apply the registered rule")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    est = estimate_r4(cfg)
    opt = optional_rule(cfg) if a.optional else None
    if a.json:
        print(json.dumps({"r4": est, "optional": opt}, indent=1))
        return 0
    t = est["totals"]
    print("R4 estimate (October logged tokens; OpenRouter prices fitted from October usage.cost; "
          "Claude from R2 Part-B tokens with prompt caching, +1 turn)")
    print(f"  {'model':40s} {'condition':30s} {'n':>4s} {'estimate':>9s} {'worst':>8s}  basis")
    for r in est["rows"]:
        c = "-" if r["cost"] is None else f"${r['cost']:.2f}"
        w = "-" if r["worst"] is None else f"${r['worst']:.2f}"
        print(f"  {r['model']:40s} {r['condition']:30s} {r['n']:4d} {c:>9s} {w:>8s}  {r['basis']}")
    print(f"  Anthropic: {t['anthropic']['episodes']} episodes, ${t['anthropic']['cost']:.2f} "
          f"(worst ${t['anthropic']['worst']:.2f}); cap exp_budget_usd ${float(cfg.get('exp_budget_usd') or 0):.2f}")
    print(f"  OpenRouter: {t['openrouter']['episodes']} episodes, ${t['openrouter']['cost']:.2f} "
          f"(worst ${t['openrouter']['worst']:.2f}); cap exp_or_budget_usd ${float(cfg.get('exp_or_budget_usd') or 0):.2f}")
    print(f"  total ${t['cost']:.2f} (worst ${t['worst']:.2f}); runtime about {t['minutes']:.0f} min "
          f"at {cfg.get('workers', 1)} episodes in flight")
    if opt:
        print(f"  optional GPT-5.5 + Gemini: total with them ${opt['total_with']:.2f} (max ${opt['total_max']:.0f}), "
              f"OpenRouter with them ${opt['or_with']:.2f} (cap ${opt['or_cap']:.0f}) -> "
              + ("added" if opt["add_optional"] else "left out"))
    missing = [r for r in est["rows"] if r["cost"] is None]
    if missing:
        print("no logged basis for: " + ", ".join(f"{r['model']}/{r['condition']}" for r in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
