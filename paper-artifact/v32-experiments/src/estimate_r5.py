"""Cost and runtime estimate for R5 from LOGGED per-episode spend (no network).

Basis: the same model's single-session after-commit cells, prompt base,
mitigation none, run through the same runner (src/tier1.py) with the same
settings R5 uses:
  Haiku 4.5, Sonnet 5  results/r2_wording/transcripts/ (2026-10-31, prompt caching ON)
  Fable 5, Qwen3.6-27B, GPT-5.5, Gemini  results/r3_wording/transcripts/ (2026-11-01; Fable caching ON)
  cells: error_500_after_commit, timeout_after_commit, status_unknown_after_commit
Per-episode cost = src/sweep.episode_cost of the logged transcript (Anthropic:
list price x logged tokens, cache writes 1.25x / reads 0.1x; OpenRouter: the
usage.cost OpenRouter returned).

Expected cost of a cell = mean logged cost + text-length extra, where the
extra is the injected text's length beyond the paper's 'internal server
error' (characters / 4 tokens) x (mean turns - 1) later requests x the
model's full input price (as if never cached).
Worst case = (mean cost x (turns + 1) / turns  [one more turn, e.g. a read-back]
              + extra) x 1.10  [drift].
Runtime = mean logged wall time per episode (started_utc -> finished_utc) x
episodes / workers.

Registered rule: the planned models run only if the worst-case total is
<= estimate_cap_usd ($45); otherwise the cut order applies (drop Fable 5,
then Sonnet 5). GPT-5.5 and Gemini are priced and shown, not run (floors).

  python3 v32-experiments/src/estimate_r5.py [--config ...] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.dirname(HERE)
REPO = os.path.dirname(V32)
for d in (os.path.join(REPO, "src"), HERE):
    if d not in sys.path:
        sys.path.insert(0, d)
import sweep                                     # noqa: E402
import r5_texts                                  # noqa: E402

RES = os.path.join(REPO, "results")
BASIS_CELLS = ("error_500_after_commit", "timeout_after_commit", "status_unknown_after_commit")
BASIS_DIR = {"claude-haiku-4-5-20251001": "r2_wording", "claude-sonnet-5": "r2_wording",
             "claude-fable-5": "r3_wording", "openrouter:qwen/qwen3.6-27b-20260422": "r3_wording",
             "openrouter:openai/gpt-5.5-20260423": "r3_wording",
             "openrouter:google/gemini-4-argon-20261013": "r3_wording"}
ANCHOR_TEXT = {"error_500_after_commit": "internal server error",
               "status_unknown_after_commit": "The request may have been processed; its outcome is unknown."}
WORST_DRIFT = 1.10


def _ts(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError):
        return None


def logged_basis(model, res_dir=RES) -> dict:
    d = os.path.join(res_dir, BASIS_DIR[model], "transcripts")
    costs, turns, walls = [], [], []
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(d, fn)) as f:
            t = json.load(f)
        if t.get("model") != model or "sessions" in t or t.get("failure_mode") not in BASIS_CELLS:
            continue
        if (t.get("prompt"), t.get("mitigation")) != ("base", "none"):
            continue
        costs.append(sweep.episode_cost(t))
        turns.append(max(1, sum(1 for x in t["turns"] if x["role"] == "assistant")))
        a, b = _ts(t.get("started_utc")), _ts(t.get("finished_utc"))
        if a and b:
            walls.append((b - a).total_seconds())
    assert costs, f"no logged basis episodes for {model} in {d}"
    n = len(costs)
    return {"n": n, "cost": sum(costs) / n, "turns": sum(turns) / n,
            "wall_s": sum(walls) / len(walls) if walls else None, "source": os.path.relpath(d, REPO)}


def estimate(cfg, models=None, res_dir=RES) -> dict:
    r5 = cfg["r5"]
    cat = r5_texts.load(r5["catalogue"] if os.path.isabs(r5["catalogue"]) else os.path.join(REPO, r5["catalogue"]))
    ids = r5_texts.by_id(cat)
    ntasks = 8 * int(cfg.get("per_template", 4))
    per_cell = ntasks * int(cfg["samples"])
    models = models or cfg["models"]
    rows, tot = [], {"expected": 0.0, "worst": 0.0}
    for m in models:
        b = logged_basis(m, res_dir)
        pr = sweep.price_for(m) or tuple(r5["or_prices_per_mtok"][m])
        exp_m = worst_m = 0.0
        for fm in [r5_texts.fm_of(i) for i in r5["injected"]] + list(r5["anchors"]):
            tid = r5_texts.id_of(fm)
            text = ids[tid]["text"] if tid else ANCHOR_TEXT[fm]
            dtok = max(0, len(text) - len(ANCHOR_TEXT["error_500_after_commit"])) / 4.0
            extra = dtok * max(0.0, b["turns"] - 1) * pr[0] / 1e6
            e = b["cost"] + extra
            w = (b["cost"] * (b["turns"] + 1) / b["turns"] + extra) * WORST_DRIFT
            exp_m += e * per_cell
            worst_m += w * per_cell
        wall = (b["wall_s"] or 0) * per_cell * (len(r5["injected"]) + len(r5["anchors"])) / max(1, int(cfg.get("workers", 1)))
        rows.append({"model": m, "episodes": per_cell * (len(r5["injected"]) + len(r5["anchors"])),
                     "basis_n": b["n"], "basis": b["source"], "logged_cost_per_episode": round(b["cost"], 6),
                     "mean_turns": round(b["turns"], 2), "expected_usd": round(exp_m, 2),
                     "worst_usd": round(worst_m, 2), "runtime_min": round(wall / 60, 1),
                     "provider": "openrouter" if m.startswith("openrouter:") else "anthropic"})
        tot["expected"] += exp_m
        tot["worst"] += worst_m
    return {"rows": rows, "expected_usd": round(tot["expected"], 2), "worst_usd": round(tot["worst"], 2),
            "by_provider": {p: {"expected": round(sum(r["expected_usd"] for r in rows if r["provider"] == p), 2),
                                "worst": round(sum(r["worst_usd"] for r in rows if r["provider"] == p), 2)}
                            for p in ("anthropic", "openrouter")},
            "runtime_min": round(sum(r["runtime_min"] for r in rows), 1)}


def apply_rule(cfg, res_dir=RES) -> dict:
    """The registered cut rule: run the planned models if worst <= estimate_cap_usd,
    else drop Fable 5, then Sonnet 5. Also prices the not-run models."""
    cap = float(cfg["r5"]["estimate_cap_usd"])
    models = list(cfg["models"])
    cuts = []
    est = estimate(cfg, models, res_dir)
    for step, m in (("drop claude-fable-5", "claude-fable-5"), ("drop claude-sonnet-5", "claude-sonnet-5")):
        if est["worst_usd"] <= cap:
            break
        models.remove(m)
        cuts.append(step)
        est = estimate(cfg, models, res_dir)
    with_not_run = estimate(cfg, list(cfg["models"]) + list(cfg["r5"]["not_run"]), res_dir)
    return {"estimate": est, "cuts_applied": cuts, "models_run": models, "estimate_cap_usd": cap,
            "fits": est["worst_usd"] <= cap,
            "if_gpt_gemini_added": {"expected_usd": with_not_run["expected_usd"],
                                    "worst_usd": with_not_run["worst_usd"]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(V32, "configs", "r5_real_errors.json"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write", default="", help="also write the result to this JSON file")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(REPO, "src"))
    import tier1
    cfg = tier1.load_config(a.config)
    out = apply_rule(cfg)
    if a.write:
        with open(a.write, "w") as f:
            json.dump(out, f, indent=1)
    if a.json:
        print(json.dumps(out, indent=1))
        return 0
    e = out["estimate"]
    print(f"{'model':42s} {'episodes':>8s} {'logged $/ep':>11s} {'turns':>5s} {'expected':>9s} {'worst':>8s} {'runtime':>8s}")
    for r in e["rows"]:
        print(f"{r['model']:42s} {r['episodes']:8d} {r['logged_cost_per_episode']:11.5f} {r['mean_turns']:5.2f} "
              f"${r['expected_usd']:8.2f} ${r['worst_usd']:7.2f} {r['runtime_min']:6.0f}min")
    bp = e["by_provider"]
    print(f"TOTAL expected ${e['expected_usd']:.2f}  worst ${e['worst_usd']:.2f}  "
          f"(Anthropic ${bp['anthropic']['expected']:.2f}/${bp['anthropic']['worst']:.2f}, "
          f"OpenRouter ${bp['openrouter']['expected']:.2f}/${bp['openrouter']['worst']:.2f})  "
          f"runtime ~{e['runtime_min'] / 60:.1f} h at {cfg.get('workers')} in flight")
    print(f"registered rule: worst <= ${out['estimate_cap_usd']:.0f} -> {'fits, nothing cut' if out['fits'] and not out['cuts_applied'] else out['cuts_applied']}; "
          f"hard stop at ${cfg['r5']['total_cap_usd']} cumulative (spend guard)")
    print(f"not run (floors): GPT-5.5, Gemini; with them the total would be expected "
          f"${out['if_gpt_gemini_added']['expected_usd']:.2f}, worst ${out['if_gpt_gemini_added']['worst_usd']:.2f}")
    return 0 if out["fits"] else 1


if __name__ == "__main__":
    sys.exit(main())
