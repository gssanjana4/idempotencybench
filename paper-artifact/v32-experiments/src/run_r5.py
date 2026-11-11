"""R5 runner: real error texts in the single-session after-commit cell.

Everything goes through the existing Tier-1 runner (src/tier1.py: plan, episode
ids, World seeding, model loop, scoring, raw.jsonl / summary.csv, resume,
refusal rows, per-provider caps, prompt caching). This module adds only:

  1. the eight injected texts, registered as World failure modes
     (r5_texts.register -> world.WORDING_VARIANTS, at run time);
  2. the cumulative spend guard: every provider is wrapped in
     spend_guard.GuardedProvider, which reserves each call's worst case in the
     shared ledger before the call and stops the run before the cap ($50).

Calls are synchronous, as in R2/R3/R4 (Anthropic prompt caching on).

  python3 v32-experiments/src/run_r5.py --dry-run         plan, injected texts as the model sees them; no API call
  python3 v32-experiments/src/run_r5.py --verify-texts    checklist for LAUNCH.md step 0 (no network)
  python3 v32-experiments/src/run_r5.py --estimate        cost estimate from logged R2/R3 spend (no API call)
  python3 v32-experiments/src/run_r5.py --check-keys      one 8-token Anthropic request + OpenRouter GET /key
  python3 v32-experiments/src/run_r5.py --launch [--limit N] [--models a,b]
  python3 v32-experiments/src/run_r5.py --status          episodes done / remaining, ledger spend
  python3 v32-experiments/src/run_r5.py --aggregate       summary.csv + registered analysis

Exit codes of --launch are tier1's: 0 all done; 1 some failed (rerun);
2 Anthropic stopped (guard, cap or credit); 3 OpenRouter stopped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.dirname(HERE)
REPO = os.path.dirname(V32)
for d in (os.path.join(REPO, "src"), os.path.join(REPO, "src2"), HERE):
    if d not in sys.path:
        sys.path.insert(0, d)

import tier1                                     # noqa: E402
import sweep                                     # noqa: E402
import r5_texts                                  # noqa: E402
from spend_guard import (SpendGuard, GuardedProvider, anthropic_call_cost,  # noqa: E402
                         SpendCapReached)

DEFAULT_CFG = os.path.join(V32, "configs", "r5_real_errors.json")
CHECK_KEYS_CHARGE_USD = 0.001     # the 8-token Haiku request of tier1.check_keys, rounded up


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)


def load(cfg_path=DEFAULT_CFG) -> dict:
    cfg = tier1.load_config(cfg_path)
    cfg["cfg_path"] = cfg_path
    r5 = cfg["r5"]
    cat = r5_texts.load(_abs(r5["catalogue"]))
    assert r5["injected"] == r5_texts.select_subset(cat), \
        "config 'injected' differs from the catalogue's registered selection rule"
    want = [r5_texts.fm_of(i) for i in r5["injected"]] + list(r5["anchors"])
    for g in cfg["groups"]:
        assert [c[2] for c in g["cells"]] == want, f"group {g.get('label')}: cells differ from the registered list"
        assert all(c[0] == "base" and c[1] == "none" for c in g["cells"])
    return cfg


def guard_for(cfg) -> SpendGuard:
    r5 = cfg["r5"]
    return SpendGuard(_abs(r5["ledger"]), r5["total_cap_usd"])


def price_table(cfg) -> dict:
    out = {}
    for m in cfg["models"]:
        pr = sweep.price_for(m) or cfg["r5"]["or_prices_per_mtok"].get(m)
        assert pr is not None, f"no price for {m}"
        out[m] = tuple(pr)
    return out


def install(cfg, guard):
    """Register the texts and route every provider through the guard.
    Returns the restore function (tests call it)."""
    r5_texts.register(cfg["r5"])
    prices = price_table(cfg)
    saved = (tier1._client_for, tier1.BUDGET_USD, tier1.OR_BUDGET_USD)
    cache = {}
    inner_for = saved[0]

    def client_for(model, anthropic_client):
        if model not in cache:
            inner = inner_for(model, anthropic_client)
            if tier1.provider_of(model) == "anthropic":
                cache[model] = GuardedProvider(inner, guard, prices, cost_fn=anthropic_call_cost,
                                               worst_in_mult=1.25 if cfg.get("prompt_cache") else 1.0)
            else:
                cache[model] = GuardedProvider(inner, guard, prices)
        return cache[model]

    tier1._client_for = client_for
    tier1.BUDGET_USD = tier1.OR_BUDGET_USD = 0.0       # the config caps and the guard apply
    for k in ("BENCH_EXP_BUDGET_USD", "BENCH_EXP_OR_BUDGET_USD"):
        os.environ.pop(k, None)                         # only the registered caps

    def restore():
        tier1._client_for, tier1.BUDGET_USD, tier1.OR_BUDGET_USD = saved
    return restore


def ledger_line(guard) -> str:
    s = guard.summarize(guard.entries())
    prov = ", ".join(f"{k} ${v:.2f}" for k, v in sorted(s["by_provider"].items())) or "-"
    return (f"cumulative v32 spend (ledger {os.path.relpath(guard.path, REPO)}): ${s['actual']:.4f} actual"
            f" + ${s['held']:.4f} held by {len(s['open'])} open reservations"
            f" = ${s['committed']:.4f} of the ${guard.cap:.2f} cap   [{prov}]")


def launch(cfg, limit=0, models=None) -> int:
    guard = guard_for(cfg)
    print(ledger_line(guard))
    if guard.remaining() < 0.01:          # each call is reserved at its worst case below
        print(f"!! {SpendCapReached(guard.committed(), 0.01, guard.cap)}")
        return 2
    restore = install(cfg, guard)
    tier1.LIMIT = int(limit or 0)
    try:
        rc = tier1.launch(cfg, models)
    finally:
        restore()
    print(ledger_line(guard))
    return rc


def show_texts(cfg):
    from world import World, ToolTimeout, ToolServerError
    r5_texts.register(cfg["r5"])
    cat = r5_texts.load(_abs(cfg["r5"]["catalogue"]))
    ids = r5_texts.by_id(cat)
    for fm in [r5_texts.fm_of(i) for i in cfg["r5"]["injected"]] + list(cfg["r5"]["anchors"]):
        w = World(failure_mode=fm, seed=1)
        try:
            w.call("send_email", {"to": "a@example.org", "subject": "Q3 update"}, "", "t")
            seen = "(no error!)"
        except (ToolTimeout, ToolServerError) as e:
            seen = f"ERROR: {e}"
        tid = r5_texts.id_of(fm)
        cls = ids[tid]["class"] if tid else r5_texts.ANCHORS[fm]
        print(f"  [{cls:11s}] {fm}\n      committed={len(w.ledger.effects)}  model sees: {seen!r}")


def verify_texts(cfg) -> int:
    cat = r5_texts.load(_abs(cfg["r5"]["catalogue"]))
    ids = r5_texts.by_id(cat)
    print("LAUNCH.md step 0: open each URL and compare the text character for character "
          "(punctuation, apostrophes, final period). Fix the catalogue BEFORE the registration commit.")
    for i in cfg["r5"]["injected"]:
        t = ids[i]
        h = hashlib.sha256(t["text"].encode()).hexdigest()[:12]
        print(f"\n[{t['class']}] {i}   ({t['provider']}; {t['status']}; retrieval={t['retrieval']}; sha256 {h})")
        print(f"  url: {t['url']}")
        for u in t.get("also_at", []):
            print(f"  also: {u}")
        print(f"  text: {t['text']!r}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R5 real error texts (v32)")
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-texts", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--check-keys", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N new episodes")
    ap.add_argument("--models", default="", help="comma subset of the config's models")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args(argv)
    cfg = load(a.config)
    models = [m for m in a.models.split(",") if m] or None
    if a.verify_texts:
        return verify_texts(cfg)
    if a.estimate:
        import estimate_r5
        return estimate_r5.main(["--config", a.config])
    if a.check_keys:
        guard = guard_for(cfg)
        rc = tier1.check_keys([a.config])
        if not os.environ.get("BENCH_CHECK_SKIP_ANTHROPIC"):
            guard.charge(CHECK_KEYS_CHARGE_USD, provider="anthropic", kind="check_keys")
        return rc
    if a.launch:
        return launch(cfg, a.limit, models)
    if a.status:
        rc = tier1.main(["--status", "--config", a.config])
        print("  " + ledger_line(guard_for(cfg)))
        return rc
    if a.aggregate:
        tier1.aggregate(cfg)
        import analysis_r5
        return analysis_r5.main(["--config", a.config])
    # default / --dry-run: the plan and what the model sees; no API call
    eps = tier1.plan(cfg, models)
    done = tier1.done_ids(cfg)
    by = {}
    for e in eps:
        by.setdefault(e[0], [0, 0])
        by[e[0]][0] += 1
        by[e[0]][1] += tier1.eid_of(e) in done
    for g in cfg["groups"]:
        print(f"{g['label']}: {len(g['models'])} models x {len(g['cells'])} cells x "
              f"{len(tier1.tasks_for(cfg))} tasks x r={cfg['samples']}   [{', '.join(g['models'])}]")
    for m, (n, d) in by.items():
        print(f"  {m:45s} {d:4d}/{n:<4d} done")
    print(f"R5: {len(eps)} episodes planned, {sum(v[1] for v in by.values())} done")
    print("injected texts and anchors (the failure fires on the first valid effectful call, AFTER commit):")
    show_texts(cfg)
    print(f"out_dir: {os.path.relpath(cfg['out_dir'], REPO)}  ({'dry run: ' if a.dry_run else ''}no API calls made)")
    print("  " + ledger_line(guard_for(cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
