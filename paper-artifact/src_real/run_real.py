"""Real-API validation slice: the payment tool on Stripe test mode.

  python3 src_real/run_real.py --config configs/real_stripe.json --estimate
  python3 src_real/run_real.py --config configs/real_stripe.json --stripe-check
  python3 src_real/run_real.py --config configs/real_stripe.json --launch [--models M]
  python3 src_real/run_real.py --config configs/real_stripe.json --verify-search
  python3 src_real/run_real.py --config configs/real_stripe.json --status
  python3 src_real/run_real.py --config configs/real_stripe.json --aggregate

Answers "results come from simulated tools" (209-dXix W2, 249-AC, 249-GFU6):
the effect is a real PaymentIntent, the read-back is Stripe's list API, the
dedup under idem_keys is Stripe's server-side Idempotency-Key, and the score
is Stripe's own record. The simulator's verdict for the same call sequence is
kept next to it (shadow World), and their agreement is reported.

Reused unchanged: src/providers.py (models), model_agent.run_model_episode
(tool loop, retry slice), src2/sweep2.run_model_session / resume_user_msg
(restart slice, same wording), src/runlock.py, tier1.episode_row (metrics),
sweep.episode_cost (spend). Only the tool list is narrowed to the two tools
that exist here (send_payment, list_effects; their schemas are the
simulator's own entries), by a scoped patch, not an edit.

Writes ONLY under out_dir (results/real_stripe/): raw.jsonl (one row per
episode; the "done" marker), transcripts/, stripe_ground_truth.jsonl,
summary.csv, stripe_check.json, search_check.jsonl.

Resumable: an episode is done when its row is in raw.jsonl. An episode that
crashes writes no row; the rerun uses a NEW attempt (new Stripe Customer, new
key namespace), so the crashed attempt's charges are never counted.
Spend guards as src/tier1.py: BENCH_BUDGET_USD caps Claude spend (list
prices x logged tokens) over this out_dir; BENCH_OR_BUDGET_USD optional for
OpenRouter (usage.cost); HTTP 402 stops OpenRouter only.
Exit: 0 done; 1 some episodes failed (rerun); 2 Claude cap; 3 OpenRouter
stopped; 5 unsafe/invalid Stripe key; 6 Stripe unreachable; 7 check failed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ("src", "src2", "src_real"):
    sys.path.insert(0, os.path.join(HERE, p))

import sweep  # noqa: E402
import tier1  # noqa: E402
from stripe_client import (StripeClient, UnsafeKeyError, StripeInfraError,  # noqa: E402
                           StripeAPIError, assert_test_key)
from stripe_world import (StripeWorld, stripe_ground_truth, TOOLS,  # noqa: E402
                          COMMITTED, REAL_FAILURE_MODES)
import tasks_real  # noqa: E402

OR_PREFIX = "openrouter:"
LIMIT = int(os.environ.get("BENCH_LIMIT", "0"))
BUDGET_USD = float(os.environ.get("BENCH_BUDGET_USD", "0"))
OR_BUDGET_USD = float(os.environ.get("BENCH_OR_BUDGET_USD", "0"))
SKIP_PROVIDERS = {p for p in os.environ.get("BENCH_REAL_SKIP_PROVIDERS", "").split(",") if p}
SKIP_MODELS = {m for m in os.environ.get("BENCH_REAL_SKIP_MODELS", "").split(",") if m}
BROKEN_AFTER = 3
STRIPE_FAIL_STOP = 3       # consecutive Stripe infra failures -> stop (exit 6)
RESCAN_EVERY = 20          # re-read spend from disk (parallel per-model runs)

DEFAULTS = {"samples": 2, "n_tasks": 16, "max_turns": 8, "max_turns_restart": 10,
            "max_tokens": {"default": 1024, OR_PREFIX: 4096},
            "restart_cells": ["amnesiac", "bare_hint"],
            "stripe_api_version": "2024-06-20", "est_tokens_by_model": {},
            "est_prices_per_mtok": {}}


def provider_of(m):
    return "openrouter" if m.startswith(OR_PREFIX) else "anthropic"


# ---------------------------------------------------------------------------
# config / plan
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    for k in ("exp", "models", "cells", "out_dir"):
        if k not in cfg:
            raise ValueError(f"{path}: missing {k}")
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    cfg["cells"] = [tuple(c) for c in cfg["cells"]]
    for p, mit, fm in cfg["cells"]:
        assert p == "base" and mit in ("none", "idem_keys") and fm in REAL_FAILURE_MODES, (p, mit, fm)
    for c in cfg["restart_cells"]:
        assert c in ("amnesiac", "bare_hint"), c
    out = cfg["out_dir"]
    cfg["out_dir"] = out if os.path.isabs(out) else os.path.join(HERE, out)
    if os.path.realpath(cfg["out_dir"]) == os.path.realpath(os.path.join(HERE, "results")):
        raise ValueError("out_dir must not be the released results/ dir")
    d = cfg["out_dir"]
    cfg.update(raw_path=os.path.join(d, "raw.jsonl"),
               gt_path=os.path.join(d, "stripe_ground_truth.jsonl"),
               summary_path=os.path.join(d, "summary.csv"),
               transcript_dir=os.path.join(d, "transcripts"),
               check_path=os.path.join(d, "stripe_check.json"),
               search_path=os.path.join(d, "search_check.jsonl"))
    return cfg


def plan(cfg, models=None):
    """[(kind, model, a, b, c, i, s)]: retry (prompt, mit, fm) or restart
    (ctx, "none", "none"). Model-major; a model's first episode is its smoke."""
    out = []
    for m in (models or cfg["models"]):
        for (p, mit, fm) in cfg["cells"]:
            for i in range(cfg["n_tasks"]):
                for s in range(cfg["samples"]):
                    out.append(("retry", m, p, mit, fm, i, s))
        for ctx in cfg["restart_cells"]:
            for i in range(cfg["n_tasks"]):
                for s in range(cfg["samples"]):
                    out.append(("restart", m, ctx, "none", "none", i, s))
    return out


def eid_of(e) -> str:
    kind, m, a, b, c, i, s = e
    if kind == "retry":
        return sweep.episode_id(m, a, b, c, tasks_real.task_id(i), s)
    return f"{m}__restart__{a}__none__{tasks_real.plan_id(i)}__k0__s{s}"


def done_ids(cfg) -> set:
    return {r["episode_id"] for r in tier1.read_rows(cfg["raw_path"])}


def new_attempt(eid: str) -> str:
    return hashlib.sha256(eid.encode()).hexdigest()[:10] + "-" + uuid.uuid4().hex[:10]


# ---------------------------------------------------------------------------
# providers / Stripe (replaceable in tests)
# ---------------------------------------------------------------------------

def _anthropic_client(models):
    return tier1._anthropic_client(models)


def _client_for(model, anthropic_client):
    return tier1._client_for(model, anthropic_client)


def _stripe_client(cfg):
    return StripeClient(api_version=cfg["stripe_api_version"])


@contextlib.contextmanager
def real_tools(max_tokens: int, max_turns: int, max_turns_restart: int):
    """Scoped: the model loops offer only send_payment + list_effects (the
    simulator's own schema entries for them), with this run's caps."""
    import model_agent
    import sweep2
    full = model_agent._tool_schemas()
    schemas = [s for s in full if s["name"] in TOOLS]
    saved = (model_agent._tool_schemas, model_agent.MAX_TURNS, model_agent.MAX_TOKENS,
             sweep2._tool_schemas, sweep2.MAX_TOKENS, sweep2.MAX_TURNS_SESSION)
    model_agent._tool_schemas = sweep2._tool_schemas = lambda: [dict(s) for s in schemas]
    model_agent.MAX_TURNS, model_agent.MAX_TOKENS = max_turns, max_tokens
    sweep2.MAX_TOKENS, sweep2.MAX_TURNS_SESSION = max_tokens, max_turns_restart
    try:
        yield
    finally:
        (model_agent._tool_schemas, model_agent.MAX_TURNS, model_agent.MAX_TOKENS,
         sweep2._tool_schemas, sweep2.MAX_TOKENS, sweep2.MAX_TURNS_SESSION) = saved


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _refusal(turns) -> bool:
    return any(t.get("role") == "assistant" and t.get("stop_reason") == "refusal" for t in turns)


def _stripe_block(world, stripe, gt, cfg):
    return {"api_version": cfg["stripe_api_version"], "base": stripe.base,
            "customer": world.customer, "attempt_id": world.attempt_id,
            "events": world.events, "ground_truth": gt}


def _gt_line(cfg, eid, kind, gt):
    with open(cfg["gt_path"], "a") as f:
        f.write(json.dumps({"episode_id": eid, "kind": kind, "recorded_utc": _now(), **gt}) + "\n")


def _start(stripe, eid, kind, model):
    att = new_attempt(eid)
    cus = stripe.create_customer(att, {"ibench": "benchmark-real-v1",
                                       "episode_id": eid[:500], "attempt_id": att,
                                       "kind": kind, "model": model[:500]})
    return att, cus["id"]


def run_retry(cfg, prov, stripe, e):
    import model_agent
    kind, model, prompt, mit, fm, i, s = e
    eid = eid_of(e)
    task = tasks_real.payment_task(i, eid)
    att, cus = _start(stripe, eid, kind, model)
    world = StripeWorld(stripe, episode_id=eid, attempt_id=att, customer=cus,
                        mitigation=mit, failure_mode=fm)
    tr = model_agent.run_model_episode(prov, world, task, model_agent.BASE_SYSTEM, model=model)
    gt = stripe_ground_truth(stripe, cus, att, task.goals)
    tr.update({"retry_mode": "agent_retry", "sample": s, "exp": cfg["exp"], "episode_id": eid,
               "kind": kind, "prompt": prompt, "mitigation": mit, "failure_mode": fm,
               "fail_on_call": world.fail_on_call,
               "failure_fired": fm != "none" and not world._failure_armed,
               "max_turns": cfg["max_turns"], "max_tokens": model_agent.MAX_TOKENS,
               "finished_utc": _now(), "backend": "stripe_test_mode",
               "stripe": _stripe_block(world, stripe, gt, cfg)})
    model_agent._score_and_log(world, task, tr, cfg["transcript_dir"],
                               f"{model}_{prompt}_{mit}_{fm}__s{s}")
    row = tier1.episode_row(tr, task)
    sent = [ev for ev in world.events if ev.get("sent")]
    row.update({"kind": kind, "attempt_id": att, "customer": cus,
                "violations_ledger": row["violations"], "ivr_ledger": row["violations"] > 0,
                "violations_stripe": gt["violations"], "ivr_stripe": gt["ivr_hit"],
                "achieved_stripe": gt["achieved"], "n_committed_stripe": gt["n_committed"],
                "agree": (gt["violations"] > 0) == (row["violations"] > 0),
                "agree_exact": gt["violations"] == row["violations"],
                "n_sent": len(sent), "forwarded_keys": sum(1 for ev in sent if ev.get("forwarded_key")),
                "replayed": sum(1 for ev in sent if ev.get("replayed")),
                "stripe_errors": sum(1 for ev in sent if ev.get("stripe_error")),
                "refusal": _refusal(tr["turns"])})
    return eid, row, gt


def run_restart(cfg, prov, stripe, e):
    import sweep2
    from resume import score_graph
    kind, model, ctx, _, _, i, s = e
    eid = eid_of(e)
    graph = tasks_real.payment_plan(i, eid)
    att, cus = _start(stripe, eid, kind, model)
    world = StripeWorld(stripe, episode_id=eid, attempt_id=att, customer=cus,
                        mitigation="none", failure_mode="none")
    started = _now()
    t1, interrupted, acked, meta1 = sweep2.run_model_session(
        prov, model, world, graph, sweep2.BASE_SYSTEM, graph.instruction, interrupt_node=0)
    sessions = [{"n": 1, "turns": t1, "interrupted": interrupted,
                 "acknowledged": sorted(acked), "meta": meta1}]
    if interrupted:
        acked.discard(0)
        msg2 = sweep2.resume_user_msg(graph, ctx, acked)
        t2, _, _, meta2 = sweep2.run_model_session(prov, model, world, graph,
                                                   sweep2.BASE_SYSTEM, msg2)
        sessions.append({"n": 2, "turns": t2, "meta": meta2, "user_msg": msg2})
    sim = score_graph(world, graph)
    gt = stripe_ground_truth(stripe, cus, att, graph.goals(), ordered=True)
    usage = {k: sum((x["meta"].get("usage") or {}).get(k) or 0 for x in sessions)
             for k in ("input", "output")}
    tr = {"model": model, "provider": meta1["provider"], "model_served": meta1["model_served"],
          "temperature_requested": meta1["temperature_requested"],
          "temperature": meta1["temperature"], "usage": usage,
          "exp": cfg["exp"], "episode_id": eid, "kind": kind, "resume_context": ctx,
          "mitigation": "none", "graph_id": graph.graph_id, "topology": graph.topology,
          "interrupt_node": 0, "interrupt_timing": "after_commit", "sample": s,
          "instruction": graph.instruction, "system": sweep2.BASE_SYSTEM,
          "sessions": sessions, "started_utc": started, "finished_utc": _now(),
          "backend": "stripe_test_mode",
          "ledger": [{"tool": x.tool, "args": x.args, "key": x.key, "seq": x.seq}
                     for x in world.ledger.for_task(graph.graph_id)],
          "score": {k: sim[k] for k in ("violations", "givr_hit", "completion", "order_violation")},
          "stripe": _stripe_block(world, stripe, gt, cfg)}
    os.makedirs(cfg["transcript_dir"], exist_ok=True)
    with open(os.path.join(cfg["transcript_dir"], f"{eid}.json".replace("/", "_")), "w") as f:
        json.dump(tr, f, indent=1)
    s2 = sessions[1]["turns"] if len(sessions) > 1 else []
    s2_calls = [b for t in s2 if t.get("role") == "assistant" for b in t["content"]
                if b.get("type") == "tool_use"]
    all_turns = [t for x in sessions for t in x["turns"]]
    row = {"episode_id": eid, "exp": cfg["exp"], "kind": kind, "model": model,
           "prompt": "restart", "resume_context": ctx, "mitigation": "none",
           "failure_mode": "restart_after_commit", "task_id": graph.graph_id,
           "family": "pay2", "sample": s, "interrupted": interrupted,
           "violations_ledger": sim["violations"], "ivr_ledger": sim["givr_hit"] > 0,
           "completion_ledger": bool(sim["completion"]),
           "order_violation_ledger": bool(sim["order_violation"]),
           "violations_stripe": gt["violations"], "ivr_stripe": gt["ivr_hit"],
           "achieved_stripe": gt["achieved"], "order_violation_stripe": gt["order_violation"],
           "n_committed_stripe": gt["n_committed"],
           "agree": (gt["violations"] > 0) == (sim["givr_hit"] > 0),
           "agree_exact": gt["violations"] == sim["violations"],
           "s2_lookup": (any(b["name"] == "list_effects" for b in s2_calls)
                         if len(sessions) > 1 else None),
           "s2_lookup_first": bool(s2_calls) and s2_calls[0]["name"] == "list_effects",
           "refusal": _refusal(all_turns),
           "truncated": any(x["meta"].get("truncated") for x in sessions),
           "invalid_calls": sum(len(x["meta"].get("invalid_calls", [])) for x in sessions),
           "model_served": meta1["model_served"], "temperature": meta1["temperature"],
           "usage_input": usage["input"], "usage_output": usage["output"],
           "cost_usd": round(sweep.episode_cost(tr), 6), "attempt_id": att, "customer": cus,
           "started_utc": started, "finished_utc": tr["finished_utc"]}
    return eid, row, gt


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------

def spent(cfg) -> dict:
    return tier1._spent_in(cfg["transcript_dir"], {"anthropic": 0.0, "openrouter": 0.0})


def launch(cfg, models=None) -> int:
    models = [m for m in (models or cfg["models"])
              if provider_of(m) not in SKIP_PROVIDERS and m not in SKIP_MODELS]
    if not models:
        print("real_stripe: no models to run")
        return 0
    from runlock import acquire
    if not acquire(cfg["out_dir"], "real", models):
        return 0
    import providers
    providers.RECORD_COST = True
    try:
        stripe = _stripe_client(cfg)
    except UnsafeKeyError as ex:
        print(f"!! {ex}")
        return 5
    eps = plan(cfg, models)
    done = done_ids(cfg)
    remaining = [e for e in eps if eid_of(e) not in done]
    print(f"{cfg['exp']}: {len(eps)} episodes planned for {', '.join(models)}, "
          f"{len(eps) - len(remaining)} done, {len(remaining)} remaining")
    if not remaining:
        aggregate(cfg)
        return 0
    os.makedirs(cfg["out_dir"], exist_ok=True)
    client = _anthropic_client({e[1] for e in remaining})
    sp = spent(cfg)
    caps = {"anthropic": BUDGET_USD, "openrouter": OR_BUDGET_USD}
    for p, cap in caps.items():
        if cap:
            print(f"{p} spend so far in {os.path.relpath(cfg['out_dir'], HERE)}: "
                  f"${sp[p]:.2f} of ${cap:.2f}")
    max_ep = {"anthropic": 0.0, "openrouter": 0.0}
    stopped, broken, tries, fails = {}, set(), {}, {}
    ran = failed = stripe_fail = 0
    for n, e in enumerate(remaining, 1):
        kind, model = e[0], e[1]
        prov = provider_of(model)
        if prov in stopped or model in broken:
            continue
        if ran and ran % RESCAN_EVERY == 0:
            disk = spent(cfg)
            sp = {p: max(sp[p], disk[p]) for p in sp}
        cap = caps[prov]
        if cap and sp[prov] + max(max_ep[prov], 0.5) > cap:
            print(f"!! {prov} budget ${cap:.2f} would be exceeded (spent ${sp[prov]:.2f}); "
                  f"stopping {prov} episodes")
            stopped[prov] = "budget"
            continue
        try:
            mt = tier1.max_tokens_for(cfg, model)
            with real_tools(mt, cfg["max_turns"], cfg["max_turns_restart"]):
                fn = run_retry if kind == "retry" else run_restart
                eid, row, gt = fn(cfg, _client_for(model, client), stripe, e)
        except UnsafeKeyError as ex:
            print(f"!! {ex} - stopping everything")
            return 5
        except StripeInfraError as ex:
            stripe_fail += 1
            failed += 1
            print(f"!! [{n}/{len(remaining)}] {eid_of(e)}: Stripe failure, episode not "
                  f"recorded (rerun retries it with a fresh attempt): {stripe.redact(ex):.300}")
            if stripe_fail >= STRIPE_FAIL_STOP:
                print(f"!! {STRIPE_FAIL_STOP} Stripe failures in a row - stopping (exit 6)")
                aggregate(cfg)
                return 6
            time.sleep(float(os.environ.get("BENCH_REAL_STRIPE_SLEEP", "10")))
            continue
        except Exception as ex:
            msg = stripe.redact(ex)
            if prov == "anthropic" and "credit balance" in msg.lower():
                print(f"!! Anthropic credit balance too low: {msg:.300}")
                stopped[prov] = "credits"
                continue
            if prov == "openrouter" and "HTTP 402" in msg:
                print(f"!! OpenRouter HTTP 402 (key limit / credits): {msg:.300}")
                stopped[prov] = "402"
                continue
            failed += 1
            tries[model] = tries.get(model, 0) + 1
            fails[model] = fails.get(model, 0) + 1
            print(f"!! [{n}/{len(remaining)}] {eid_of(e)} failed, will retry on next run: "
                  f"{type(ex).__name__}: {msg:.300}")
            if fails[model] >= BROKEN_AFTER and fails[model] == tries[model]:
                print(f"!! {model}: first {BROKEN_AFTER} episodes all failed; skipping it this run")
                broken.add(model)
            continue
        stripe_fail = 0
        tries[model] = tries.get(model, 0) + 1
        _gt_line(cfg, eid, kind, gt)
        with open(cfg["raw_path"], "a") as f:          # the "done" marker, last
            f.write(json.dumps(row) + "\n")
        c = row["cost_usd"]
        sp[prov] += c
        max_ep[prov] = max(max_ep[prov], c)
        ran += 1
        print(f"[{n}/{len(remaining)}] {eid} stripe_v={row['violations_stripe']} "
              f"ledger_v={row['violations_ledger']} agree={row['agree']} ${c:.4f}")
        if LIMIT and ran >= LIMIT:
            print(f"BENCH_LIMIT={LIMIT} reached")
            break
    print(f"{cfg['exp']}: {ran} new episodes -> {cfg['raw_path']}"
          + (f"; {failed} failed (rerun to retry)" if failed else ""))
    aggregate(cfg)
    if stopped:
        print(f"stopped: {stopped}")
    if "anthropic" in stopped:
        return 2
    if "openrouter" in stopped:
        return 3
    return 1 if (failed or broken) else 0


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def _rate(rows, k):
    rows = [r for r in rows if r.get(k) is not None]
    return round(sum(1 for r in rows if r.get(k)) / len(rows), 4) if rows else None


def cell_key(r):
    return (r["kind"], r["model"], r["prompt"] if r["kind"] == "retry" else r["resume_context"],
            r["mitigation"], r["failure_mode"])


def aggregate(cfg) -> list:
    rows = tier1.read_rows(cfg["raw_path"])
    if not rows:
        print(f"no raw data at {cfg['raw_path']}")
        return []
    cells = {}
    for r in rows:
        cells.setdefault(cell_key(r), []).append(r)
    out = []
    for key, eps in sorted(cells.items()):
        sc = [r for r in eps if not r.get("refusal")]
        n = len(sc)
        k = sum(1 for r in sc if r["ivr_stripe"])
        T = len({r["task_id"] for r in sc})
        p = k / n if n else 0.0
        lo, hi = sweep.wilson_ci(p * T, T) if n else (0.0, 1.0)
        d = dict(zip(("kind", "model", "condition", "mitigation", "failure_mode"), key))
        d.update({"n_episodes": n, "n_tasks": T, "refusals": len(eps) - n,
                  "ivr_stripe": round(p, 4), "ivr_stripe_ci_tlo": round(lo, 4),
                  "ivr_stripe_ci_thi": round(hi, 4),
                  "ivr_ledger": _rate(sc, "ivr_ledger"),
                  "agree": sum(1 for r in sc if r["agree"]),
                  "agree_exact": sum(1 for r in sc if r["agree_exact"]),
                  "success_stripe": _rate(sc, "achieved_stripe"),
                  "readback_after_failure": _rate([r for r in sc if r.get("failure_fired")],
                                                  "readback_after_failure"),
                  "retry_rate": _rate(sc, "retried"), "key_attach_rate": _rate(sc, "key_attached_first"),
                  "forwarded_keys": sum(r.get("forwarded_keys") or 0 for r in sc),
                  "stripe_replayed": sum(r.get("replayed") or 0 for r in sc),
                  "stripe_errors": sum(r.get("stripe_errors") or 0 for r in sc),
                  "interrupted": sum(1 for r in sc if r.get("interrupted")),
                  "s2_lookup": _rate(sc, "s2_lookup"),
                  "truncated": _rate(sc, "truncated"),
                  "cost_usd": round(sum(r.get("cost_usd") or 0 for r in eps), 4)})
        out.append(d)
    fields = []
    for d in out:
        fields += [x for x in d if x not in fields]
    with open(cfg["summary_path"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)
    print(f"{len(out)} cells -> {cfg['summary_path']}")
    return out


# ---------------------------------------------------------------------------
# Stripe round-trip check, delayed search cross-check, status, estimate
# ---------------------------------------------------------------------------

def stripe_check(cfg) -> int:
    """One real round trip that also checks the idempotency semantics the
    slice relies on: create+confirm a $0.50 PaymentIntent, replay the same
    Idempotency-Key (same id, Idempotent-Replayed: true, still one charge),
    list it back by customer, and reuse the key with a different amount
    (must be refused)."""
    try:
        stripe = _stripe_client(cfg)
    except UnsafeKeyError as ex:
        print(f"!! {ex}")
        return 5
    res = {"utc": _now(), "base": stripe.base, "api_version": stripe.api_version}
    try:
        att = "check-" + uuid.uuid4().hex[:12]
        cus = stripe.create_customer(att, {"ibench": "benchmark-real-v1",
                                           "attempt_id": att, "kind": "stripe_check"})
        params = {"amount": 50, "currency": "usd", "confirm": True,
                  "payment_method": "pm_card_visa", "customer": cus["id"],
                  "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
                  "metadata": {"ibench": "benchmark-real-v1", "attempt_id": att}}
        k = f"ib-{att}-k1"
        pi1, r1 = stripe.create_payment_intent(params, k)
        pi2, r2 = stripe.create_payment_intent(params, k)
        listed = stripe.list_payment_intents(cus["id"])
        mismatch = None
        try:
            stripe.create_payment_intent(dict(params, amount=51), k)
            mismatch = "accepted"
        except StripeAPIError as ex:
            mismatch = f"refused: HTTP {ex.status} {ex.type}"
        res.update({"customer": cus["id"], "pi": pi1.get("id"), "status": pi1.get("status"),
                    "livemode": pi1.get("livemode"), "replay_same_id": pi1.get("id") == pi2.get("id"),
                    "first_replayed": r1, "second_replayed": r2,
                    "listed_committed": sum(1 for p in listed if p.get("status") in COMMITTED),
                    "key_reuse_other_params": mismatch})
    except UnsafeKeyError as ex:
        print(f"!! {ex}")
        return 5
    except (StripeInfraError, StripeAPIError) as ex:
        print(f"!! Stripe check failed: {stripe.redact(ex):.300}")
        return 6
    ok = (res["status"] == "succeeded" and res["livemode"] is False and res["replay_same_id"]
          and res["second_replayed"] and not res["first_replayed"]
          and res["listed_committed"] == 1 and res["key_reuse_other_params"].startswith("refused"))
    res["ok"] = ok
    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(cfg["check_path"], "w") as f:
        json.dump(res, f, indent=1)
    print(("Stripe check OK: " if ok else "!! Stripe check FAILED: ")
          + json.dumps({k: v for k, v in res.items() if k not in ("base",)}))
    return 0 if ok else 7


def _read_jsonl(path) -> list:
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def verify_search(cfg, min_age_s: float = 70.0) -> int:
    """Delayed cross-check of the ground truth with Stripe's Search API
    (eventually consistent, so never used during an episode): for every
    recorded episode, search metadata["attempt_id"] and compare the set of
    PaymentIntent ids with the list-API ground truth."""
    try:
        stripe = _stripe_client(cfg)
    except UnsafeKeyError as ex:
        print(f"!! {ex}")
        return 5
    rows = {r["episode_id"]: r for r in tier1.read_rows(cfg["raw_path"])}
    gts = {}
    for g in _read_jsonl(cfg["gt_path"]):
        r = rows.get(g.get("episode_id"))
        if r and r.get("attempt_id") == g.get("attempt_id"):
            gts[g["episode_id"]] = g
    checked = {c["episode_id"]: c for c in _read_jsonl(cfg["search_path"])}
    todo = [g for e, g in gts.items() if not (checked.get(e) or {}).get("agree")
            or checked[e].get("attempt_id") != g["attempt_id"]]
    newest = max((g["recorded_utc"] for g in todo), default=None)
    if newest:
        import calendar
        age = time.time() - calendar.timegm(time.strptime(newest, "%Y-%m-%dT%H:%M:%SZ"))
        if age < min_age_s:
            time.sleep(min_age_s - age)
    agree = 0
    with open(cfg["search_path"], "a") as f:
        for g in todo:
            try:
                found = stripe.search_payment_intents(f'metadata["attempt_id"]:"{g["attempt_id"]}"')
            except (StripeInfraError, StripeAPIError) as ex:
                print(f"!! search failed for {g['episode_id']}: {stripe.redact(ex):.200}")
                return 6
            a = sorted(p["id"] for p in found)
            b = sorted(p["id"] for p in g["payment_intents"])
            c = {"episode_id": g["episode_id"], "attempt_id": g["attempt_id"], "utc": _now(),
                 "search_ids": a, "list_ids": b, "agree": a == b}
            agree += c["agree"]
            f.write(json.dumps(c) + "\n")
    print(f"search cross-check: {agree}/{len(todo)} newly checked episodes agree with the "
          f"list-API ground truth ({len(gts) - len(todo)} agreed earlier)")
    return 0 if agree == len(todo) else 1


def status(cfg) -> int:
    eps = plan(cfg)
    done = done_ids(cfg)
    rem = [e for e in eps if eid_of(e) not in done]
    by = {}
    for e in rem:
        by[e[1]] = by.get(e[1], 0) + 1
    sp = spent(cfg)
    print(f"  {cfg['exp']} {len(eps) - len(rem)}/{len(eps)} done"
          + (f"   remaining: {by}" if by else ""))
    print(f"  logged spend: Anthropic ${sp['anthropic']:.2f} (list prices), "
          f"OpenRouter ${sp['openrouter']:.2f} (usage.cost)")
    return 1 if rem else 0


def estimate(cfg, models=None) -> dict:
    done = done_ids(cfg)
    rem = [e for e in plan(cfg, models) if eid_of(e) not in done]
    out = {"anthropic": 0.0, "openrouter": 0.0, "episodes": len(rem)}
    for e in rem:
        m = e[1]
        per = (cfg["est_tokens_by_model"].get(m) or {}).get(e[0], [4000, 400])
        pr = sweep.price_for(m) or cfg["est_prices_per_mtok"].get(m)
        if pr:
            out[provider_of(m)] += (per[0] * pr[0] + per[1] * pr[1]) / 1e6
    print(f"{cfg['exp']}: {len(rem)} episodes remaining; est. cost Anthropic "
          f"${out['anthropic']:.2f}, OpenRouter ${out['openrouter']:.2f}; Stripe test mode $0")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    for a in ("launch", "aggregate", "status", "estimate", "stripe_check",
              "verify_search", "check_stripe_key"):
        ap.add_argument("--" + a.replace("_", "-"), action="store_true")
    ap.add_argument("--models", default="")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    models = [m for m in a.models.split(",") if m] or None
    if a.launch or a.stripe_check or a.verify_search or a.check_stripe_key:
        try:
            assert_test_key(os.environ.get("STRIPE_SECRET_KEY", ""))
        except UnsafeKeyError as ex:
            print(f"!! {ex}")
            return 5
        if a.check_stripe_key:
            print("STRIPE_SECRET_KEY is a test-mode key (sk_test_...)")
            return 0
    if a.stripe_check:
        return stripe_check(cfg)
    if a.verify_search:
        return verify_search(cfg)
    if a.launch:
        return launch(cfg, models)
    if a.status:
        return status(cfg)
    if a.aggregate:
        aggregate(cfg)
        return 0
    estimate(cfg, models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
