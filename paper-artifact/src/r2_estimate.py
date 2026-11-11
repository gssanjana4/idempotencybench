"""Cost estimate for R2 (configs/r2_wording.json) from the token usage logged
in the October runs of the same models and kinds of episode.

  python3 src/r2_estimate.py [--config configs/r2_wording.json]

Basis (read-only):
  Part A (restart wording)   results/claude_batch_2026-10/transcripts2/
      amnesiac cells -> amnesiac; bare_hint cells -> bare_hint and the three
      paraphrases (assumed to behave like bare_hint; the "worst case" column
      prices them like amnesiac, i.e. as if the hint did nothing).
  Part B (error wording)     results/tier1/e1_6_oct/transcripts/ (bare 500,
      timeout) and results/tier1/e1_6/transcripts/ (W3 -> W3 and paraphrases).
  Part C (chain plan_only)   October plan_only (fan-out) transcripts.
Cost = Anthropic list prices (src/sweep.PRICES) x logged tokens. "cached"
re-prices the same token counts under prompt caching (cache write 1.25x,
read 0.1x), assuming the prompt grows linearly over the turns of a session
from ~600 tokens (tools + system + instruction) and that a prefix is cached
only once it reaches the model's minimum cacheable length (Haiku 4.5: 4096,
Sonnet: 1024, Fable: 1024 assumed). Haiku prompts rarely reach 4096, so
caching saves little there. This is an estimate, not the spend log: the run
logs the real cache tokens per call.
"""
import argparse, glob, json, os, sys
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
import sweep  # noqa: E402

RES = os.path.join(HERE, "results")
MIN_CACHE = {"claude-haiku-4-5": 4096, "claude-sonnet-5": 1024, "claude-fable-5": 1024}
L1 = 600            # tokens of tools + system + instruction (first prompt)
SRC = {
    "A": [(os.path.join(RES, "claude_batch_2026-10", "transcripts2"), None)],
    "B": [(os.path.join(RES, "tier1", "e1_6_oct", "transcripts"), None),
          (os.path.join(RES, "tier1", "e1_6", "transcripts"), None)],
    "C": [(os.path.join(RES, "claude_batch_2026-10", "transcripts2"), None)],
}
# condition in the plan -> condition whose logged episodes stand in for it
PROXY = {"hint_p1": "bare_hint", "hint_p2": "bare_hint", "hint_p3": "bare_hint",
         "status_unknown_p1_after_commit": "status_unknown_after_commit",
         "status_unknown_p2_after_commit": "status_unknown_after_commit",
         "status_unknown_p3_after_commit": "status_unknown_after_commit"}
WORST = {"hint_p1": "amnesiac", "hint_p2": "amnesiac", "hint_p3": "amnesiac", "bare_hint": "amnesiac"}


def min_cache(model):
    for k, v in MIN_CACHE.items():
        if model.startswith(k):
            return v
    return 1024


def cached_input_equiv(total_in, turns, model):
    """Input-token equivalents billed under caching for one session with
    `turns` assistant turns and `total_in` uncached input tokens."""
    if turns <= 1 or total_in <= L1 * turns:
        return float(total_in)
    thr = min_cache(model)
    delta = (total_in - L1 * turns) * 2.0 / (turns * (turns - 1))
    L = [L1 + t * delta for t in range(turns)]
    cost = 0.0
    prev_cached = False
    for t, Lt in enumerate(L):
        if t > 0 and prev_cached:
            cost += 0.1 * L[t - 1] + 1.25 * (Lt - L[t - 1])
        else:
            cost += 1.25 * Lt if Lt >= thr else Lt
        prev_cached = Lt >= thr
    return cost


def sessions_of(t):
    if "sessions" in t:
        for s in t["sessions"]:
            u = (s.get("meta") or {}).get("usage") or {}
            yield u.get("input") or 0, u.get("output") or 0, \
                sum(1 for x in s["turns"] if x["role"] == "assistant")
    else:
        u = t.get("usage") or {}
        yield u.get("input") or 0, u.get("output") or 0, \
            sum(1 for x in t["turns"] if x["role"] == "assistant")


def logged(part):
    """(model, condition) -> list of (plain_cost, cached_cost)."""
    out = defaultdict(list)
    for d, _ in SRC[part]:
        for f in glob.glob(os.path.join(d, "*.json")):
            try:
                with open(f) as fh:
                    t = json.load(fh)
            except Exception:
                continue
            m = t.get("model", "")
            pr = sweep.price_for(m)
            if pr is None:
                continue
            cond = t.get("resume_context") or t.get("failure_mode")
            if "sessions" in t and t.get("mitigation") != "none":
                continue
            plain = cached = 0.0
            for tin, tout, turns in sessions_of(t):
                plain += (tin * pr[0] + tout * pr[1]) / 1e6
                cached += (cached_input_equiv(tin, turns, m) * pr[0] + tout * pr[1]) / 1e6
            out[(m, cond)].append((plain, cached))
    return out


def mean(xs, i):
    return sum(x[i] for x in xs) / len(xs) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "configs", "r2_wording.json"))
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args(argv)
    with open(a.config) as fh:
        cfg = json.load(fh)
    logs = {p: logged(p) for p in ("A", "B", "C")}
    rows, totals = [], {"plain": 0.0, "cached": 0.0, "worst_plain": 0.0, "worst_cached": 0.0}
    unpriced = set()
    for g in cfg["groups"]:
        part = g["part"]
        for m in g["models"]:
            if sweep.price_for(m) is None:
                unpriced.add(m)
                continue
            cells = [(c[2], 32 * cfg["samples"]) for c in g.get("cells", [])] + \
                    [(c[0], (16 if c[2] == "all" else 8) * cfg.get("restart_samples", cfg["samples"]))
                     for c in g.get("restart_cells", [])]
            for cond, n in cells:
                key = PROXY.get(cond, cond)
                xs = logs[part].get((m, key)) or (logs["A"].get((m, "plan_only")) if part == "C" else None)
                if part == "C" and not xs:
                    xs = logs["A"].get((m, "plan_only"))
                worst = logs[part].get((m, WORST.get(cond, cond))) or xs
                if not xs:
                    rows.append({"part": part, "model": m, "condition": cond, "n": n, "basis": "none"})
                    continue
                r = {"part": part, "model": m, "condition": cond, "n": n,
                     "basis": f"{len(xs)} logged {key} episodes",
                     "per_ep_plain": mean(xs, 0), "per_ep_cached": mean(xs, 1),
                     "plain": n * mean(xs, 0), "cached": n * mean(xs, 1),
                     "worst_plain": n * mean(worst, 0), "worst_cached": n * mean(worst, 1)}
                rows.append(r)
                for k in totals:
                    totals[k] += r[k]
    if a.json:
        print(json.dumps({"rows": rows, "totals": totals}, indent=1))
        return 0
    by_part = defaultdict(lambda: defaultdict(float))
    for r in rows:
        if "plain" in r:
            for k in ("plain", "cached", "worst_plain", "worst_cached"):
                by_part[(r["part"], r["model"])][k] += r[k]
            by_part[(r["part"], r["model"])]["n"] += r["n"]
    print("R2 cost estimate (Anthropic list prices x October logged tokens; 'cached' = same tokens under prompt caching)")
    print(f"{'part':4s} {'model':28s} {'episodes':>8s} {'plain':>9s} {'cached':>9s}   worst case (hints behave like amnesiac): plain / cached")
    last = None
    for (part, m), v in sorted(by_part.items()):
        if last and last != part:
            print()
        last = part
        print(f"{part:4s} {m:28s} {int(v['n']):8d} ${v['plain']:8.2f} ${v['cached']:8.2f}   ${v['worst_plain']:.2f} / ${v['worst_cached']:.2f}")
    for part in ("A", "B", "C"):
        v = {k: sum(x[k] for (p, _), x in by_part.items() if p == part) for k in ("plain", "cached", "n")}
        print(f"Part {part} total: {int(v['n'])} episodes, ${v['plain']:.2f} plain, ${v['cached']:.2f} with caching")
    print(f"TOTAL: ${totals['plain']:.2f} plain, ${totals['cached']:.2f} with caching "
          f"(worst case ${totals['worst_plain']:.2f} / ${totals['worst_cached']:.2f}); "
          f"cap exp_budget_usd = ${cfg.get('exp_budget_usd', 0):.2f}")
    if unpriced:
        print(f"no list price for: {', '.join(sorted(unpriced))} (OpenRouter: the key limit / exp_or_budget_usd is the cap)")
    missing = [r for r in rows if r["basis"] == "none"]
    if missing:
        print(f"no logged basis for {len(missing)} cells: " + ", ".join(f"{r['model']}/{r['condition']}" for r in missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
