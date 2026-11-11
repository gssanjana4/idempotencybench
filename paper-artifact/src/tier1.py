"""Tier-1 runner for the follow-up experiments (E1-0 ... E1-6).

One command + one config file per experiment:

  python3 src/tier1.py --config configs/e1_2_at_least_once.json --dry-run
  python3 src/tier1.py --config configs/e1_2_at_least_once.json --estimate
  python3 src/tier1.py --config configs/e1_2_at_least_once.json --launch
  python3 src/tier1.py --config configs/e1_2_at_least_once.json --aggregate

A config fixes models, cells, task set, samples, retry mode, turn cap,
max_tokens and out_dir. Every run writes ONLY under its out_dir
(results/tier1/<exp>/): raw.jsonl, summary.csv, transcripts/. Released files
are never read for "done" and never written. Resumable: completed episode
ids in raw.jsonl are skipped. Deterministic: tasks are generated, and ack ids
/ request ids are seeded from the episode id (World(seed=...)).

Retry modes:
  agent_retry    one delivery; the model decides whether to retry (as v1).
  at_least_once  two deliveries of the same task, each a fresh conversation,
                 against the SAME World with failure_mode "none"; scored once
                 on the shared ledger (mirrors agents.run_episode).

Spend guards (checked BEFORE each episode, as in src2/sweep2.py):
  BENCH_BUDGET_USD      cap on Anthropic spend (list prices x logged tokens),
                         summed over every experiment dir under the tier-1
                         root (results/tier1/), not just this one.
  BENCH_OR_BUDGET_USD   optional cap on OpenRouter spend (usage.cost as
                         returned by OpenRouter); the key limit is the backstop.
  On a cap, episodes of that provider stop; the other provider continues.
  OpenRouter HTTP 402 (key limit / credits) stops OpenRouter episodes only.
  exp_budget_usd (config key, optional) caps Anthropic spend logged under
                         this experiment's own out_dir (on top of the global cap).
  exp_or_budget_usd (config key, optional) the same for OpenRouter spend.
Exit code: 0 all planned episodes done; 1 some failed (rerun to retry);
2 stopped by a cap or 402 (rerun later; completed episodes are kept).

R2 (wording robustness, configs/r2_wording.json) additions, all opt-in:
  restart_cells   [[resume_context, mitigation, topology|"all"], ...]: restart
                  episodes run through src2/sweep2.run_episode (session 1 cut
                  after the interrupt node commits; session 2 resumed with
                  resume_user_msg), the same 16 graphs / interrupt points as
                  the October cells, r = restart_samples (default samples).
                  Rows carry kind = "restart" and go to summary_restart.csv.
  groups          [{"part": label, "models": [...], "cells": [...],
                  "restart_cells": [...]}, ...]: per-part model sets in one
                  config (R2 parts A/B/C). "models", "cells" and
                  "restart_cells" at the top level are then their unions.
  prompt_cache    true: Anthropic prompt caching (providers.PROMPT_CACHE).
  workers         N > 1 (or BENCH_WORKERS): N episodes in flight at once.
  BENCH_DEADLINE_MARGIN_S  with BENCH_DEADLINE_S: stop starting episodes
                  when elapsed + max(margin, longest episode so far) would
                  pass the deadline (short resumable chunks).

R3 (cross-provider wording, configs/r3_wording.json) additions, all opt-in:
  plan_order      "groups": episodes are ordered group by group (retry then
                  restart within a group), so that a budget stop loses the
                  LAST groups first (the registered cut order). Default
                  "kind": all retry episodes, then all restart episodes.
  max_tokens phases  models with different max_tokens (OpenRouter 4096,
                  Claude 1024) run in successive phases of the same launch,
                  each phase with the config's workers, instead of one worker;
                  restart sessions get the same max_tokens as retry episodes.
  refusal_rows    true: an OpenRouter error saying the upstream provider
                  blocked the response ("blocked the response", as Gemini
                  did in October on one task) writes a row with refusal =
                  true and no score instead of failing the episode, so the
                  run completes; the analysis excludes and counts it.
  BENCH_EXP_BUDGET_USD / BENCH_EXP_OR_BUDGET_USD  override the config's
                  exp_budget_usd / exp_or_budget_usd (launcher: a cap shared
                  between two runs; "0" = no cap).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait as _wait

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sweep  # noqa: E402  (wilson_ci, episode_id, episode_cost, prices)
from world import World, EFFECTFUL_TOOLS, normalize_value  # noqa: E402
from tasks import generate_tasks, generate_long_tasks  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OR_PREFIX = "openrouter:"
LIMIT = int(os.environ.get("BENCH_LIMIT", "0"))          # smoke: new episodes
DEADLINE_S = float(os.environ.get("BENCH_DEADLINE_S", "0"))
BUDGET_USD = float(os.environ.get("BENCH_BUDGET_USD", "0"))        # Anthropic
OR_BUDGET_USD = float(os.environ.get("BENCH_OR_BUDGET_USD", "0"))  # OpenRouter
# comma list of providers to skip entirely (e.g. "openrouter" if no key)
SKIP_PROVIDERS = {p for p in os.environ.get(
    "BENCH_TIER1_SKIP_PROVIDERS", "").split(",") if p}
# comma list of model ids to skip (launcher: models that failed their smoke)
SKIP_MODELS = {m for m in os.environ.get("BENCH_TIER1_SKIP_MODELS", "").split(",") if m}
BROKEN_AFTER = 3   # a model whose first N episodes in a run all fail is skipped
WORKERS = int(os.environ.get("BENCH_WORKERS", "0"))          # 0 = config / 1
DEADLINE_MARGIN_S = float(os.environ.get("BENCH_DEADLINE_MARGIN_S", "0"))
RESTART = "restart"   # marker in the 7th slot of a restart episode tuple

REQUIRED = ("exp", "models", "cells", "out_dir")
DEFAULTS = {"task_set": "core", "per_template": 4, "template_start": 0,
            "samples": 2, "retry_mode": "agent_retry", "max_turns": 8,
            "max_tokens": {"default": 1024, OR_PREFIX: 4096},
            "seed_acks": True, "tier1_root": "results/tier1",
            "est_tokens_per_episode": [3000, 300], "est_multiplier": 1.0,
            "est_prices_per_mtok": {},
            "restart_cells": [], "restart_samples": None, "prompt_cache": False,
            "workers": 1, "groups": [], "plan_order": "kind", "refusal_rows": False}
# an OpenRouter error text meaning the upstream provider refused to answer
# (a content block), recorded as a refusal row when refusal_rows is on
REFUSAL_MARKERS = ("blocked the response",)


def provider_of(model: str) -> str:
    return "openrouter" if model.startswith(OR_PREFIX) else "anthropic"


# ---------------------------------------------------------------------------
# Config and plan
# ---------------------------------------------------------------------------

def _dedup(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    if cfg.get("groups"):   # per-part model sets: top-level lists = unions
        for g in cfg["groups"]:
            g.setdefault("cells", []); g.setdefault("restart_cells", [])
            g["cells"] = [tuple(c) for c in g["cells"]]
            g["restart_cells"] = [tuple(c) for c in g["restart_cells"]]
        cfg.setdefault("models", _dedup(m for g in cfg["groups"] for m in g["models"]))
        cfg.setdefault("cells", _dedup(c for g in cfg["groups"] for c in g["cells"]))
        cfg.setdefault("restart_cells",
                       _dedup(c for g in cfg["groups"] for c in g["restart_cells"]))
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    for k, v in DEFAULTS.items():
        cfg.setdefault(k, v)
    cfg["cells"] = [tuple(c) for c in cfg["cells"]]
    cfg["restart_cells"] = [tuple(c) for c in cfg["restart_cells"]]
    for c in cfg["restart_cells"]:
        assert len(c) == 3 and c[2] in ("all", "chain4", "fanout"), c
    if cfg["restart_samples"] is None:
        cfg["restart_samples"] = cfg["samples"]
    assert cfg["retry_mode"] in ("agent_retry", "at_least_once"), cfg["retry_mode"]
    assert cfg["plan_order"] in ("kind", "groups"), cfg["plan_order"]
    assert cfg["task_set"] in ("core", "long"), cfg["task_set"]
    out = cfg["out_dir"]
    root = cfg["tier1_root"]
    cfg["out_dir"] = out if os.path.isabs(out) else os.path.join(HERE, out)
    cfg["tier1_root"] = root if os.path.isabs(root) else os.path.join(HERE, root)
    released = os.path.realpath(os.path.join(HERE, "results"))
    if os.path.realpath(cfg["out_dir"]) == released:
        raise ValueError("out_dir must not be the released results/ dir")
    cfg["raw_path"] = os.path.join(cfg["out_dir"], "raw.jsonl")
    cfg["summary_path"] = os.path.join(cfg["out_dir"], "summary.csv")
    cfg["transcript_dir"] = os.path.join(cfg["out_dir"], "transcripts")
    return cfg


def tasks_for(cfg) -> list:
    """core: generate_tasks(per_template), keeping within-template index in
    [template_start, per_template). generate_tasks is prefix-stable per
    template, so per_template=12/template_start=4 = tasks 33-96 (E1-4).
    long: generate_long_tasks(per_template) (E1-5)."""
    n = cfg["per_template"]
    if cfg["task_set"] == "long":
        return generate_long_tasks(per_template=n)
    tasks = generate_tasks(per_domain=n)
    return [t for i, t in enumerate(tasks) if i % n >= cfg["template_start"]]


def _sweep2():
    """src2/sweep2 (restart protocol), imported on demand."""
    d = os.path.join(HERE, "src2")
    if d not in sys.path:
        sys.path.insert(0, d)
    import sweep2
    return sweep2


def restart_graphs(topology="all"):
    """The 16 October benchmark graphs with their interrupt points
    (sweep2.benchmark_graphs: 8 chain4 + 8 fanout, k balanced, after_commit)."""
    sweep2 = _sweep2()
    return [(g, k) for g, k in sweep2.benchmark_graphs()
            if topology == "all" or g.topology == topology]


def plan(cfg, models=None):
    """Retry episodes (m, prompt, mit, fm, task, s) first, then restart
    episodes (m, ctx, mit, graph, k, s, RESTART)."""
    tasks = tasks_for(cfg)
    groups = cfg["groups"] or [{"models": cfg["models"], "cells": cfg["cells"],
                                "restart_cells": cfg["restart_cells"]}]
    eps = []
    order = [(kind, grp) for kind in ("retry", RESTART) for grp in groups]
    if cfg["plan_order"] == "groups":        # group-major (R3: cut order = run order)
        order = [(kind, grp) for grp in groups for kind in ("retry", RESTART)]
    for kind, grp in order:
        ms = [m for m in grp["models"] if models is None or m in models]
        if kind == "retry":
            eps += [(m, p, mit, fm, t, s)
                    for m in ms
                    for (p, mit, fm) in grp["cells"]
                    for t in tasks
                    for s in range(cfg["samples"])]
        else:
            eps += [(m, ctx, mit, g, k, s, RESTART)
                    for m in ms
                    for (ctx, mit, topo) in grp["restart_cells"]
                    for (g, k) in restart_graphs(topo)
                    for s in range(cfg["restart_samples"])]
    return _dedup(eps) if cfg["groups"] else eps


def is_restart(e) -> bool:
    return len(e) == 7 and e[6] == RESTART


def eid_of(e) -> str:
    if is_restart(e):
        sweep2 = _sweep2()
        m, ctx, mit, g, k, s, _ = e
        return sweep2.episode_id(m, ctx, mit, g.graph_id, k, s)
    m, p, mit, fm, t, s = e
    return sweep.episode_id(m, p, mit, fm, t.task_id, s)


def done_ids(cfg) -> set:
    done = set()
    if os.path.exists(cfg["raw_path"]):
        with open(cfg["raw_path"]) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["episode_id"])
                except Exception:
                    pass
    return done


def ack_seed(eid: str) -> int:
    return int(hashlib.sha256(eid.encode()).hexdigest()[:8], 16)


def max_tokens_for(cfg, model) -> int:
    mt = cfg["max_tokens"]
    if model in mt:
        return int(mt[model])
    for prefix, v in mt.items():
        if prefix != "default" and model.startswith(prefix):
            return int(v)
    return int(mt.get("default", 1024))


# ---------------------------------------------------------------------------
# Spend
# ---------------------------------------------------------------------------

def _spent_in(tdir: str, tot: dict) -> dict:
    if os.path.isdir(tdir):
        for fn in os.listdir(tdir):
            try:
                with open(os.path.join(tdir, fn)) as f:
                    t = json.load(f)
            except Exception:
                continue
            tot[provider_of(t.get("model", ""))] += sweep.episode_cost(t)
    return tot


def spent_by_provider(root: str) -> dict:
    """Spend over every tier-1 experiment dir under root, split by provider:
    anthropic = list price x logged tokens; openrouter = logged usage.cost."""
    tot = {"anthropic": 0.0, "openrouter": 0.0}
    if not os.path.isdir(root):
        return tot
    for exp in sorted(os.listdir(root)):
        _spent_in(os.path.join(root, exp, "transcripts"), tot)
    _spent_in(os.path.join(root, "transcripts"), tot)   # root is itself an out_dir
    return tot


def spent_in_exp(cfg) -> dict:
    """Spend logged under this experiment's own out_dir only."""
    return _spent_in(cfg["transcript_dir"], {"anthropic": 0.0, "openrouter": 0.0})


def estimate(cfg, remaining) -> dict:
    """Pre-run cost estimate per provider (printed before each experiment).
    Anthropic: list prices; OpenRouter: est_prices_per_mtok from the config
    (OpenRouter list prices, re-verify) -- the real spend is usage.cost."""
    tin, tout = cfg["est_tokens_per_episode"]
    mult = float(cfg["est_multiplier"])
    out = {"anthropic": 0.0, "openrouter": 0.0, "episodes": len(remaining),
           "unpriced_models": []}
    for e in remaining:
        m = e[0]
        pr = sweep.price_for(m) or cfg["est_prices_per_mtok"].get(m)
        per = cfg.get("est_tokens_by_model", {}).get(m, [tin, tout])
        if pr is None:
            if m not in out["unpriced_models"]:
                out["unpriced_models"].append(m)
            continue
        out[provider_of(m)] += mult * (per[0] * pr[0] + per[1] * pr[1]) / 1e6
    return out


# ---------------------------------------------------------------------------
# Episodes
# ---------------------------------------------------------------------------

def _system_for(prompt):
    from model_agent import BASE_SYSTEM, CAREFUL_SYSTEM
    return {"base": BASE_SYSTEM, "careful": CAREFUL_SYSTEM}[prompt]


def run_restart(cfg, client, e) -> dict:
    """One restart episode through src2/sweep2.run_episode (unchanged
    protocol: unseeded World, failure_mode none, session 1 cut after node k
    commits, session 2 resumed with resume_user_msg(ctx)); the transcript is
    written under this experiment's transcript_dir. Returns the raw row."""
    sweep2 = _sweep2()
    model, ctx, mit, g, k, s, _ = e
    sweep2.MAX_TOKENS = max_tokens_for(cfg, model)   # sessions use sweep2's copy
    row = sweep2.run_episode(client, model, ctx, mit, g, k, s,
                             transcript_dir=cfg["transcript_dir"])
    path = os.path.join(cfg["transcript_dir"],
                        f"{row['episode_id']}.json".replace("/", "_"))
    with open(path) as f:
        tr = json.load(f)
    tr.update({"exp": cfg["exp"], "kind": RESTART, "episode_id": row["episode_id"],
               "max_tokens": sweep2.MAX_TOKENS,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    with open(path, "w") as f:
        json.dump(tr, f, indent=1)
    return restart_row(cfg, row, tr)


def _s2_lookup(tr):
    """Did session 2 call list_effects? None if there was no session 2."""
    if len(tr.get("sessions", [])) < 2:
        return None
    return any(b.get("name") == "list_effects"
               for turn in tr["sessions"][1]["turns"] if turn.get("role") == "assistant"
               for b in turn["content"] if b.get("type") == "tool_use")


def restart_row(cfg, row, tr) -> dict:
    """raw.jsonl row of a restart episode: the sweep2 row (graph_id,
    topology, violations, givr_hit, completion, order_violation,
    resume_context, mitigation, interrupt_node, ...) plus the bookkeeping the
    retry rows carry (exp, cost, usage, served model, UTC window) and the
    fields the paper's 209 loader indexes (prompt, failure_mode, task_id,
    family, achieved), so one raw file can hold both kinds."""
    turns = [t for s_ in tr["sessions"] for t in s_["turns"]]
    cap = _sweep2().MAX_TURNS_SESSION
    u = {}
    for s_ in tr["sessions"]:
        for kk, v in ((s_.get("meta") or {}).get("usage") or {}).items():
            u[kk] = u.get(kk, 0) + (v or 0)
    r = dict(row)
    r.update({"exp": cfg["exp"], "kind": RESTART, "prompt": "base",
              "failure_mode": "none", "retry_mode": None,
              "task_id": row["graph_id"], "family": row["topology"],
              "achieved": bool(row["completion"]),
              "interrupted": bool(tr["sessions"][0].get("interrupted")),
              "n_sessions": len(tr["sessions"]),
              "session2_lookup": _s2_lookup(tr),
              "refusal": any(t.get("stop_reason") == "refusal" for t in turns
                             if t.get("role") == "assistant"),
              "turn_cap_hit": any(
                  sum(1 for t in s_["turns"] if t["role"] == "assistant") >= cap
                  and s_["turns"][-1]["role"] == "tool_results" for s_ in tr["sessions"]),
              "provider": tr.get("provider"), "model_served": tr.get("model_served"),
              "temperature": tr.get("temperature"),
              "usage_input": u.get("input"), "usage_output": u.get("output"),
              "usage_cache_read": u.get("cache_read"), "usage_cache_write": u.get("cache_write"),
              "max_tokens": tr.get("max_tokens"),
              "cost_usd": round(sweep.episode_cost(tr), 6),
              "started_utc": tr.get("started_utc"), "finished_utc": tr.get("finished_utc")})
    return r


def refusal_row(cfg, e, msg) -> dict:
    """Row of an episode the upstream provider refused (refusal_rows): no
    score, counted as done, excluded and reported by the analysis."""
    eid = eid_of(e)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if is_restart(e):
        model, ctx, mit, g, k, s, _ = e
        return {"episode_id": eid, "exp": cfg["exp"], "kind": RESTART, "model": model,
                "resume_context": ctx, "mitigation": mit, "graph_id": g.graph_id,
                "topology": g.topology, "interrupt_node": k, "sample": s, "prompt": "base",
                "failure_mode": "none", "task_id": g.graph_id, "family": g.topology,
                "refusal": True, "refusal_error": msg[:300], "violations": 0, "givr_hit": 0,
                "completion": False, "order_violation": False, "achieved": False,
                "truncated": False, "cost_usd": 0.0, "started_utc": now, "finished_utc": now}
    model, prompt, mit, fm, task, s = e
    return {"episode_id": eid, "exp": cfg["exp"], "model": model, "prompt": prompt,
            "mitigation": mit, "failure_mode": fm, "retry_mode": cfg["retry_mode"],
            "task_id": task.task_id, "family": task.family, "sample": s,
            "refusal": True, "refusal_error": msg[:300], "violations": 0, "achieved": False,
            "truncated": False, "failure_fired": False, "retried": False,
            "readback_after_failure": False, "cost_usd": 0.0,
            "started_utc": now, "finished_utc": now}


def run_one(cfg, client, e) -> dict:
    """Run one planned episode; returns the transcript (scored + written)."""
    import model_agent
    if is_restart(e):
        return run_restart(cfg, client, e)
    model, prompt, mit, fm, task, s = e
    eid = eid_of(e)
    model_agent.MAX_TURNS = int(cfg["max_turns"])
    model_agent.MAX_TOKENS = max_tokens_for(cfg, model)
    world = World(mitigation=mit, failure_mode=fm,
                  fail_on_call=getattr(task, "fail_on_call", 1),
                  seed=ack_seed(eid) if cfg["seed_acks"] else None)
    system = _system_for(prompt)
    if cfg["retry_mode"] == "at_least_once":
        d1 = model_agent.run_model_episode(client, world, task, system, model=model)
        d1_ledger_len = len(world.ledger.for_task(task.task_id))
        d2 = model_agent.run_model_episode(client, world, task, system, model=model)
        tr = {k: d1.get(k) for k in ("model", "provider", "model_served",
                                     "temperature_requested", "temperature",
                                     "started_utc", "task_id", "family",
                                     "instruction", "system")}
        tr["retry_mode"] = "at_least_once"
        tr["d1_ledger_len"] = d1_ledger_len
        tr["deliveries"] = [d1, d2]
        tr["usage"] = {k: (d1["usage"].get(k) or 0) + (d2["usage"].get(k) or 0)
                       for k in ("input", "output")}
        tr["truncated"] = bool(d1.get("truncated") or d2.get("truncated"))
        tr["invalid_calls"] = d1.get("invalid_calls", []) + d2.get("invalid_calls", [])
    else:
        tr = model_agent.run_model_episode(client, world, task, system, model=model)
        tr["retry_mode"] = "agent_retry"
    tr.update({"sample": s, "exp": cfg["exp"], "episode_id": eid,
               "prompt": prompt, "mitigation": mit, "failure_mode": fm,
               "fail_on_call": world.fail_on_call,
               "failure_fired": fm != "none" and not world._failure_armed,
               "max_turns": cfg["max_turns"],
               "max_tokens": model_agent.MAX_TOKENS,
               "ack_seed": ack_seed(eid) if cfg["seed_acks"] else None,
               "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    model_agent._score_and_log(world, task, tr, cfg["transcript_dir"],
                               f"{model}_{prompt}_{mit}_{fm}__s{s}")
    return tr


# ---------------------------------------------------------------------------
# Per-episode metrics (rows)
# ---------------------------------------------------------------------------

def _deliveries(t):
    return t["deliveries"] if "deliveries" in t else [t]


def _calls(turns):
    """[(tool_use block, result dict)] in issue order."""
    results = {}
    for turn in turns:
        if turn["role"] == "tool_results":
            for r in turn["content"]:
                results[r["tool_use_id"]] = r
    out = []
    for turn in turns:
        if turn["role"] == "assistant":
            for b in turn["content"]:
                if b.get("type") == "tool_use":
                    out.append((b, results.get(b["id"], {})))
    return out


def _sid(tool, args):
    _, canon = EFFECTFUL_TOOLS[tool]
    return (tool,) + tuple(normalize_value(args.get(f)) for f in canon)


def _key(b):
    k = (b.get("input") or {}).get("idempotency_key", "") or ""
    return k if isinstance(k, str) else json.dumps(k, sort_keys=True)


def _key_cat(k0, k1):
    if k0 and k1:
        return "same_key" if k0 == k1 else "new_key"
    if k0:
        return "dropped_key"
    return "added_key" if k1 else "keyless"


def transcript_sig(turns, calls_only=False) -> str:
    """Hash of what the model did (text + tool names/args/keys), ignoring
    tool ids, ack ids and thinking blocks: equal sigs = identical pair.
    calls_only: tool calls only ("tool-call-identical", E1-0)."""
    seq = []
    for turn in turns:
        if turn["role"] != "assistant":
            continue
        for b in turn["content"]:
            if b.get("type") == "text":
                if not calls_only:
                    seq.append(["text", b.get("text", "")])
            elif b.get("type") == "tool_use":
                seq.append(["tool", b["name"], b.get("input")])
    return hashlib.sha256(json.dumps(seq, sort_keys=True).encode()).hexdigest()[:16]


def _turns_flat(t):
    return [turn for d in _deliveries(t) for turn in d["turns"]]


def _calls_all(t):
    """Calls of every delivery, paired with results within their delivery."""
    return [c for d in _deliveries(t) for c in _calls(d["turns"])]


def episode_row(t, task) -> dict:
    """Per-episode metrics from the scored transcript alone (no World), so
    imported transcripts (E1-1 cells run by run_crossprovider.sh) get the
    same rows as runs made here."""
    sc = t["score"]
    invalid = {c["tool_use_id"] for c in t.get("invalid_calls", [])}
    allc = _calls_all(t)
    eff = [(b, r) for b, r in allc
           if b["name"] in EFFECTFUL_TOOLS and b["id"] not in invalid]
    # failed call = first valid effectful call whose result was an error
    fi = next((i for i, (b, r) in enumerate(eff) if r.get("is_error")), None)
    row = {"episode_id": t["episode_id"], "exp": t["exp"], "model": t["model"],
           "prompt": t["prompt"], "mitigation": t["mitigation"],
           "failure_mode": t["failure_mode"], "retry_mode": t["retry_mode"],
           "task_id": task.task_id, "family": task.family,
           "sample": t["sample"], "violations": sc["violations"],
           "achieved": sc["achieved"], "omission": sc["omission"],
           "misorder": sc["misorder"],
           "truncated": bool(t.get("truncated")),
           "invalid_calls": len(t.get("invalid_calls", [])),
           "turn_cap_hit": any(
               sum(1 for x in d["turns"] if x["role"] == "assistant") >= t["max_turns"]
               and d["turns"][-1]["role"] != "assistant" for d in _deliveries(t)),
           "domain": task.domain,
           # template = diversity unit (8 core templates; long3/long4)
           "template": task.domain if task.family in ("atomic", "long_chain")
           else task.family,
           "fail_on_call": t["fail_on_call"],
           # the World arms the failure on the k-th VALID effectful call
           "failure_fired": t["failure_fired"] if "failure_fired" in t
           else t["failure_mode"] != "none" and len(eff) >= t["fail_on_call"],
           "n_effectful": len(eff),
           "key_attached_first": bool(eff and _key(eff[0][0])),
           "retried": False, "retry_key_category": "",
           "readback_after_failure": False,
           "model_served": t.get("model_served"),
           "temperature": t.get("temperature"),
           "usage_input": (t.get("usage") or {}).get("input"),
           "usage_output": (t.get("usage") or {}).get("output"),
           "cost_usd": round(sweep.episode_cost(t), 6),
           "sig": transcript_sig(_turns_flat(t)),
           "sig_calls": transcript_sig(_turns_flat(t), calls_only=True),
           "started_utc": t.get("started_utc"), "finished_utc": t.get("finished_utc")}
    u = t.get("usage") or {}
    if "cache_read" in u or "cache_write" in u:   # prompt-caching runs only (R2)
        row["usage_cache_read"] = u.get("cache_read")
        row["usage_cache_write"] = u.get("cache_write")
    if fi is not None:
        b0 = eff[fi][0]
        s0 = _sid(b0["name"], b0["input"])
        retry = next((b for b, _ in eff[fi + 1:] if _sid(b["name"], b["input"]) == s0), None)
        if retry is not None:
            row["retried"] = True
            row["retry_key_category"] = _key_cat(_key(b0), _key(retry))
        # read-back: list_effects after the failed call, before the next
        # effectful call (or before the end of the episode)
        pos = next(i for i, (b, _) in enumerate(allc) if b is b0)
        for b, _ in allc[pos + 1:]:
            if b["name"] == "list_effects":
                row["readback_after_failure"] = True
                break
            if b["name"] in EFFECTFUL_TOOLS and b["id"] not in invalid:
                break
    # ordered tasks: stale re-execution = an earlier step RE-committed after a
    # later step committed; order_violation = the 249 rule (any commit of a
    # dependency after any commit of its dependent, incl. misorder).
    led = t["ledger"]   # this task's committed effects, in commit order
    # strict matcher (E0-3): raw canonical strings, no normalization
    from collections import Counter
    raw = Counter((e["tool"],) + tuple(str(e["args"].get(f)) for f in EFFECTFUL_TOOLS[e["tool"]][1])
                  for e in led)
    row["violations_strict"] = sum(v - 1 for v in raw.values() if v > 1)
    seqs = []
    for g in task.goals:
        seqs.append([e["seq"] for e in led
                     if e["tool"] == g.tool and all(
                         normalize_value(e["args"].get(k)) == normalize_value(v)
                         for k, v in g.canon_args.items())])
    pairs = [(i, j) for i in range(len(seqs)) for j in range(i + 1, len(seqs))] \
        if task.ordered else []
    row["stale_reexec"] = any(len(seqs[i]) > 1 and seqs[j]
                              and max(seqs[i]) > min(seqs[j]) for i, j in pairs)
    row["order_violation"] = any(seqs[i] and seqs[j] and max(seqs[i]) > min(seqs[j])
                                 for i, j in pairs)
    if t["retry_mode"] == "at_least_once":
        d1, d2 = t["deliveries"]
        inv = invalid
        e1 = [b for b, _ in _calls(d1["turns"])
              if b["name"] in EFFECTFUL_TOOLS and b["id"] not in inv]
        c2 = [b for b, _ in _calls(d2["turns"])]
        e2 = [b for b in c2 if b["name"] in EFFECTFUL_TOOLS and b["id"] not in inv]
        row["d2_readback_first"] = bool(c2) and any(
            b["name"] == "list_effects" for b in
            c2[:next((i for i, b in enumerate(c2) if b["name"] in EFFECTFUL_TOOLS), len(c2))])
        s1 = {_sid(b["name"], b["input"]) for b in e1}
        row["d2_reexecuted"] = any(_sid(b["name"], b["input"]) in s1 for b in e2)
        cat = ""
        if e1:
            sfirst = _sid(e1[0]["name"], e1[0]["input"])
            redo = next((b for b in e2 if _sid(b["name"], b["input"]) == sfirst), None)
            cat = _key_cat(_key(e1[0]), _key(redo)) if redo is not None else "not_redone"
        row["d2_key_category"] = cat
        # duplicates already present after delivery 1 (spontaneous, no failure)
        c = Counter(_sid(e["tool"], e["args"]) for e in led[:t["d1_ledger_len"]])
        row["d1_violations"] = sum(v - 1 for v in c.values() if v > 1)
    return row


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def _anthropic_client(models):
    if any(provider_of(m) == "anthropic" for m in models):
        import anthropic
        return anthropic.Anthropic(max_retries=6)   # SDK backoff on 429/5xx/529
    return None


def _client_for(model, anthropic_client):
    from providers import provider_for
    return provider_for(model, anthropic_client=anthropic_client)


def launch(cfg, models=None) -> int:
    models = [m for m in (models or cfg["models"])
              if provider_of(m) not in SKIP_PROVIDERS and m not in SKIP_MODELS]
    if not models:
        print(f"{cfg['exp']}: no models to run (skipped providers: "
              f"{', '.join(sorted(SKIP_PROVIDERS)) or '-'})")
        return 0
    from runlock import acquire
    if not acquire(cfg["out_dir"], "tier1", models):
        return 0
    import providers
    providers.RECORD_COST = True   # log OpenRouter usage.cost per turn
    if cfg.get("prompt_cache"):
        providers.PROMPT_CACHE = True
        print(f"{cfg['exp']}: Anthropic prompt caching ON (cache_control on tools, "
              f"system and the latest user turn; cache tokens logged per call)")
    episodes = plan(cfg, models)
    done = done_ids(cfg)
    remaining = [e for e in episodes if eid_of(e) not in done]
    print(f"{cfg['exp']}: {len(episodes)} episodes planned, "
          f"{len(episodes) - len(remaining)} done, {len(remaining)} remaining")
    if not remaining:
        aggregate(cfg)
        return 0
    client = _anthropic_client({e[0] for e in remaining})
    os.makedirs(cfg["out_dir"], exist_ok=True)
    spent = spent_by_provider(cfg["tier1_root"])
    caps = {"anthropic": BUDGET_USD, "openrouter": OR_BUDGET_USD}
    for p, cap in caps.items():
        if cap:
            print(f"{p} spend so far under {cfg['tier1_root']}: "
                  f"${spent[p]:.2f} of ${cap:.2f}")
    exp_caps = {"anthropic": float(os.environ.get("BENCH_EXP_BUDGET_USD")
                                   or cfg.get("exp_budget_usd") or 0),
                "openrouter": float(os.environ.get("BENCH_EXP_OR_BUDGET_USD")
                                    or cfg.get("exp_or_budget_usd") or 0)}
    spent_exp = spent_in_exp(cfg)
    for p, cap in exp_caps.items():
        if cap:
            print(f"{p} spend so far in {cfg['exp']}: "
                  f"${spent_exp[p]:.2f} of ${cap:.2f} "
                  f"({'exp_budget_usd' if p == 'anthropic' else 'exp_or_budget_usd'})")
    workers = max(1, WORKERS or int(cfg.get("workers") or 1))
    if LIMIT:
        workers = min(workers, LIMIT)
    # model_agent.MAX_TOKENS (and sweep2's copy) are process-global: models
    # with different max_tokens run in successive phases, in plan order
    phases = []
    for e in remaining:
        mt = max_tokens_for(cfg, e[0])
        if not phases or phases[-1][0] != mt:
            phases.append((mt, []))
        phases[-1][1].append(e)
    if workers > 1:
        import model_agent
        model_agent.MAX_TURNS = int(cfg["max_turns"])
        for m in {e[0] for e in remaining}:
            _client_for(m, client)       # build providers before the threads
        print(f"{cfg['exp']}: {workers} episodes in flight at a time"
              + (f"; {len(phases)} phases by max_tokens "
                 f"{[mt for mt, _ in phases]}" if len(phases) > 1 else ""))
    st = {"max_ep": {"anthropic": 0.0, "openrouter": 0.0}, "max_dur": 0.0,
          "stopped": {}, "broken": set(), "tries": {}, "fails": {},
          "ran": 0, "failed": 0}
    t0 = time.time()
    N = len(remaining)

    def admit(e, inflight):
        """Decide before starting e: None = run; "skip" = not this provider /
        model; "stop" = deadline (nothing more is started)."""
        model = e[0]
        prov = provider_of(model)
        if prov in st["stopped"] or model in st["broken"]:
            return "skip"
        if DEADLINE_S:
            margin = max(DEADLINE_MARGIN_S, st["max_dur"] if DEADLINE_MARGIN_S else 0.0)
            if time.time() - t0 + margin > DEADLINE_S:
                print(f"deadline {DEADLINE_S:.0f}s reached"
                      + (f" (margin {margin:.0f}s)" if margin else "")
                      + "; rerun to continue")
                return "stop"
        reserve = (inflight + 1) * max(st["max_ep"][prov], 0.5)
        cap = caps[prov]
        if cap and spent[prov] + reserve > cap:
            print(f"!! {prov} budget ${cap:.2f} would be exceeded "
                  f"(spent ${spent[prov]:.2f}); stopping {prov} episodes")
            st["stopped"][prov] = "budget"
            return "skip"
        if exp_caps[prov] and spent_exp[prov] + reserve > exp_caps[prov]:
            print(f"!! {cfg['exp']} {prov} budget ${exp_caps[prov]:.2f} would be exceeded "
                  f"(spent ${spent_exp[prov]:.2f} in this experiment); stopping")
            st["stopped"][prov] = "exp_budget"
            return "skip"
        return None

    def work(n, e):
        t = time.time()
        try:
            tr = run_one(cfg, _client_for(e[0], client), e)
            return n, e, tr, None, time.time() - t
        except Exception as ex:   # isolate one bad episode; no row => retried next run
            return n, e, None, ex, time.time() - t

    def finish(n, e, tr, ex, dur):
        model = e[0]
        prov = provider_of(model)
        st["max_dur"] = max(st["max_dur"], dur)
        if ex is not None:
            msg = str(ex)
            if prov == "anthropic" and "credit balance" in msg.lower():
                print(f"!! Anthropic credit balance too low: {msg:.300}")
                st["stopped"][prov] = "credits"
                return
            if prov == "openrouter" and "HTTP 402" in msg:
                print(f"!! OpenRouter HTTP 402 (key limit / credits): {msg:.300}")
                print("!! stopping openrouter episodes")
                st["stopped"][prov] = "402"
                return
            if cfg.get("refusal_rows") and any(mk in msg for mk in REFUSAL_MARKERS):
                row = refusal_row(cfg, e, msg)
                with open(cfg["raw_path"], "a") as f:
                    f.write(json.dumps(row) + "\n")
                st["tries"][model] = st["tries"].get(model, 0) + 1
                st["ran"] += 1
                st["refused"] = st.get("refused", 0) + 1
                print(f"[{n}/{N}] {row['episode_id']} REFUSED by the provider "
                      f"(row with refusal=true, no score): {msg:.200}")
                return
            st["failed"] += 1
            st["tries"][model] = st["tries"].get(model, 0) + 1
            st["fails"][model] = st["fails"].get(model, 0) + 1
            print(f"!! [{n}/{N}] {eid_of(e)} failed, will retry on next run: {msg:.300}")
            if st["fails"][model] >= BROKEN_AFTER and st["fails"][model] == st["tries"][model]:
                print(f"!! {model}: first {BROKEN_AFTER} episodes all failed "
                      f"(wrong slug or API change?); skipping it this run")
                st["broken"].add(model)
            return
        st["tries"][model] = st["tries"].get(model, 0) + 1
        row = tr if is_restart(e) else episode_row(tr, e[4])
        with open(cfg["raw_path"], "a") as f:
            f.write(json.dumps(row) + "\n")
        c = row["cost_usd"]
        spent[prov] += c
        spent_exp[prov] += c
        st["max_ep"][prov] = max(st["max_ep"][prov], c)
        st["ran"] += 1
        if is_restart(e):
            print(f"[{n}/{N}] {row['episode_id']} givr={row['givr_hit']} "
                  f"ovr={row['order_violation']} done={row['completion']} ${c:.4f}")
        else:
            print(f"[{n}/{N}] {row['episode_id']} v={row['violations']} "
                  f"ok={row['achieved']} ${c:.4f}")

    def set_max_tokens(mt):
        import model_agent
        model_agent.MAX_TOKENS = mt
        if any(is_restart(e) for e in remaining):
            _sweep2().MAX_TOKENS = mt

    halt = False          # deadline or BENCH_LIMIT: nothing more is started
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for mt, phase in phases:
            if halt:
                break
            set_max_tokens(mt)
            it = iter(enumerate(phase, n + 1))
            n += len(phase)
            more = True
            pending = set()
            while True:
                while more and len(pending) < workers:
                    try:
                        k, e = next(it)
                    except StopIteration:
                        more = False
                        break
                    d = admit(e, len(pending))
                    if d == "skip":
                        continue
                    if d == "stop":
                        more = False
                        halt = True
                        break
                    pending.add(pool.submit(work, k, e))
                if not pending:
                    break
                ready, pending = _wait(pending, return_when=FIRST_COMPLETED)
                for fut in ready:
                    finish(*fut.result())
                if LIMIT and st["ran"] >= LIMIT and more:
                    print(f"BENCH_LIMIT={LIMIT} reached")
                    more = False
                    halt = True
    ran, failed, stopped, broken = st["ran"], st["failed"], st["stopped"], st["broken"]
    print(f"{cfg['exp']}: {ran} new episodes -> {cfg['raw_path']}"
          + (f"; {st['refused']} provider refusals (rows with refusal=true)" if st.get("refused") else "")
          + (f"; {failed} failed (rerun to retry)" if failed else ""))
    for p in ("anthropic", "openrouter"):
        if spent[p]:
            print(f"{p} spend under {cfg['tier1_root']}: ${spent[p]:.2f}")
        if exp_caps[p]:
            print(f"{p} spend in {cfg['exp']}: ${spent_exp[p]:.2f} of ${exp_caps[p]:.2f}")
    aggregate(cfg)
    if stopped:
        print(f"stopped: {stopped}")
    if "anthropic" in stopped:
        return 2          # Claude cap: the launcher stops
    if "openrouter" in stopped:
        return 3          # OpenRouter key/cap: launcher drops OpenRouter models
    return 1 if (failed or broken) else 0


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def _rate(rows, key):
    return round(sum(1 for r in rows if r.get(key)) / len(rows), 4) if rows else None


def read_rows(raw_path) -> list:
    by_id = {}
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue          # blank, or cut off by a kill mid-write
                by_id[r["episode_id"]] = r   # last row wins
    return list(by_id.values())


def cell_stats(eps: list) -> dict:
    """Metrics of the registered metric definitions for one cell."""
    n = len(eps)
    kv = sum(1 for e in eps if e["violations"] > 0)
    p = kv / n
    lo, hi = sweep.wilson_ci(kv, n)
    by_task = {}
    for e in eps:
        by_task.setdefault(e["task_id"], []).append(e)
    nt = len(by_task)
    tlo, thi = sweep.wilson_ci(p * nt, nt)      # task-level (rho = 1) CI
    kt = sum(1 for es in by_task.values() if any(x["violations"] > 0 for x in es))
    pairs = [es for es in by_task.values() if len(es) >= 2]
    ret = [e for e in eps if e["retried"]]
    fired = [e for e in eps if e["failure_fired"]]
    cats = ("same_key", "new_key", "dropped_key", "added_key", "keyless")
    out = {"n_episodes": n, "n_tasks": nt, "ivr": round(p, 4),
           "ivr_ci_lo": round(lo, 4), "ivr_ci_hi": round(hi, 4),
           "ivr_ci_tlo": round(tlo, 4), "ivr_ci_thi": round(thi, 4),
           "ivr_task_any": round(kt / nt, 4),
           "ivr_strict": _strict_rate(eps),
           "violation_depth": round(sum(e["violations"] for e in eps) / n, 4),
           "success": _rate(eps, "achieved"), "omission": _rate(eps, "omission"),
           "misorder": _rate(eps, "misorder"),
           "stale_reexec": _rate(eps, "stale_reexec"),
           "order_violation": _rate(eps, "order_violation"),
           "truncated": _rate(eps, "truncated"),
           "turn_cap_hit": _rate(eps, "turn_cap_hit"),
           "invalid_call_eps": round(sum(1 for e in eps if e["invalid_calls"]) / n, 4),
           "failure_fired": _rate(eps, "failure_fired"),
           "readback_after_failure": _rate(fired, "readback_after_failure"),
           "key_attach_rate": _rate(eps, "key_attached_first"),
           "retry_rate": round(len(ret) / n, 4),
           "key_reuse_rate": round(sum(1 for e in ret if e["retry_key_category"] == "same_key")
                                   / len(ret), 4) if ret else None}
    for c in cats:
        out[f"retry_{c}"] = sum(1 for e in ret if e["retry_key_category"] == c)
    out["identical_pairs"] = sum(1 for es in pairs if len({x["sig"] for x in es[:2]}) == 1)
    out["identical_call_pairs"] = sum(1 for es in pairs
                                      if len({x.get("sig_calls") for x in es[:2]}) == 1)
    out["sample_pairs"] = len(pairs)
    if any(e.get("retry_mode") == "at_least_once" for e in eps):
        out["d1_spontaneous_dup"] = round(sum(1 for e in eps if e.get("d1_violations")) / n, 4)
        out["d2_readback_first"] = _rate(eps, "d2_readback_first")
        out["d2_reexecuted"] = _rate(eps, "d2_reexecuted")
        for c in cats + ("not_redone",):
            out[f"d2_{c}"] = sum(1 for e in eps if e.get("d2_key_category") == c)
    served = sorted({str(e.get("model_served")) for e in eps})
    temps = sorted({str(e.get("temperature")) for e in eps})
    out["model_served"] = ";".join(served)
    out["temperature_effective"] = ";".join(temps)
    ts = sorted(x for e in eps for x in (e.get("started_utc"), e.get("finished_utc")) if x)
    out["first_utc"], out["last_utc"] = (ts[0], ts[-1]) if ts else ("", "")
    out["cost_usd"] = round(sum(e.get("cost_usd") or 0 for e in eps), 4)
    return out


def _released_fname(model, prompt, mit, fm, s, task_id):
    return f"{model}_{prompt}_{mit}_{fm}__s{s}__{task_id}.json".replace("/", "_")


def _import_one(cfg, spec) -> list:
    import glob
    import rescore
    src = spec["dir"] if os.path.isabs(spec["dir"]) else os.path.join(HERE, spec["dir"])
    want = {tuple(c) for c in spec["cells"]}
    models = set(spec.get("models") or cfg["models"])
    if not os.path.isdir(src):
        print(f"import: {os.path.relpath(src, HERE)} not there yet (0 episodes)")
        return []
    tasks = {t.task_id: t for t in generate_tasks(per_domain=spec.get("per_template", 4))}
    v2 = {}
    if spec.get("score_from"):   # e.g. released cells: use the v2 rescore
        with open(os.path.join(HERE, spec["score_from"])) as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    v2[r["episode_id"]] = r
    rows = []
    for f in sorted(glob.glob(os.path.join(src, "*.json"))):
        m = rescore.FNAME.match(os.path.basename(f))
        if not m or m["task"] not in tasks or (m["prompt"], m["mit"], m["fm"]) not in want:
            continue
        try:
            with open(f) as fh:
                t = json.load(fh)
        except Exception:
            continue
        if t.get("model") not in models or "score" not in t:
            continue
        cell, s = (m["prompt"], m["mit"], m["fm"]), int(m["s"])
        eid = sweep.episode_id(t["model"], *cell, m["task"], s)
        if v2:
            if eid not in v2:
                continue
            for k in ("violations", "achieved", "omission", "misorder"):
                t["score"][k] = v2[eid][k]
        t.update({"exp": cfg["exp"], "prompt": cell[0], "mitigation": cell[1],
                  "failure_mode": cell[2], "sample": s, "retry_mode": "agent_retry",
                  "fail_on_call": 1, "max_turns": spec.get("max_turns", 8),
                  "episode_id": eid})
        r = episode_row(t, tasks[m["task"]])
        if v2:
            # The stored ledger of a released v1 transcript was built by the
            # pre-fix world (keyless retries deduped), so a strict count from
            # it would be wrong (Haiku idem_keys 0.27, not 0.92). Not available.
            r["violations_strict"] = None
        r["source"] = spec.get("role", "baseline")
        r["source_file"] = os.path.relpath(f, HERE)
        rows.append(r)
    print(f"imported {len(rows)} {r['source'] if rows else ''} episodes from "
          f"{os.path.relpath(src, HERE)}")
    return rows


def _strict_rate(eps):
    """Exact-string-match IVR over episodes where it is computable; None if
    none are (e.g. released v1 transcripts, whose ledgers predate the fix)."""
    known = [e for e in eps if e.get("violations_strict") is not None]
    if not known:
        return None
    return round(sum(1 for e in known if e["violations_strict"] > 0) / len(known), 4)


def import_rows(cfg) -> list:
    """Comparison rows rebuilt (read-only) from transcripts in other out
    dirs, written to <out_dir>/imported.jsonl and summarized as separate
    cells (column `source` = the entry's role). Never counted as done and
    never re-run. Each "import_from" entry: dir (a transcript dir), cells,
    optional models,
    per_template, score_from (a v2 raw file whose scores override the
    transcript's, for the released Claude cells), role (a label):
      "crossprovider"  E1-1 cells run_crossprovider.sh runs for the
                       OpenRouter models (same tasks, r, retry mode, cap);
      "released_v2"    released Claude cells, v2 rescore."""
    rows = []
    for spec in cfg.get("import_from", []):
        rows += _import_one(cfg, spec)
    if rows:
        os.makedirs(cfg["out_dir"], exist_ok=True)
        with open(os.path.join(cfg["out_dir"], "imported.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return rows


def compare_released(cfg, rows) -> None:
    """E1-0: per episode, is the rerun tool-call-identical to the released v1
    transcript, and does its verdict agree with the v2 rescore? Read-only."""
    spec = cfg.get("compare_released")
    if not spec:
        return
    tdir = os.path.join(HERE, spec.get("transcripts", "results/sweep_transcripts"))
    v2 = {}
    with open(os.path.join(HERE, spec.get("raw_v2", "results/sweep_raw_v2.jsonl"))) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                v2[r["episode_id"]] = r
    for r in rows:
        fn = os.path.join(tdir, _released_fname(r["model"], r["prompt"], r["mitigation"],
                                                r["failure_mode"], r["sample"], r["task_id"]))
        r["released_identical"] = r["released_calls_identical"] = None
        if os.path.exists(fn):
            with open(fn) as fh:
                old_turns = json.load(fh)["turns"]
            r["released_identical"] = transcript_sig(old_turns) == r["sig"]
            r["released_calls_identical"] = \
                transcript_sig(old_turns, calls_only=True) == r["sig_calls"]
        old = v2.get(r["episode_id"])
        r["released_v2_violated"] = None if old is None else old["violations"] > 0
        r["verdict_agrees_v2"] = None if old is None else \
            (old["violations"] > 0) == (r["violations"] > 0)


def _released_stats(eps):
    out = {}
    if any("released_identical" in e for e in eps):
        known = [e for e in eps if e.get("released_identical") is not None]
        out["released_matched"] = len(known)
        out["released_identical"] = sum(1 for e in known if e["released_identical"])
        out["released_calls_identical"] = sum(1 for e in known
                                              if e["released_calls_identical"])
        v2 = [e for e in eps if e.get("released_v2_violated") is not None]
        out["released_v2_ivr"] = round(sum(1 for e in v2 if e["released_v2_violated"])
                                       / len(v2), 4) if v2 else None
        out["verdict_agrees_v2"] = sum(1 for e in v2 if e["verdict_agrees_v2"])
    return out


def pooled_with_released(cfg, rows) -> list:
    """E1-4: pool the new tasks (r=1) with sample 0 of the released 32 tasks
    (v2 rescore) -> one sample per task, distinct tasks per cell."""
    spec = cfg.get("pool_released")
    if not spec:
        return []
    old = []
    with open(os.path.join(HERE, spec["raw_v2"])) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["sample"] == spec.get("sample", 0):
                    old.append(r)
    out = []
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["prompt"], r["mitigation"], r["failure_mode"]),
                         {"new": [], "old": []})["new"].append(r)
    for r in old:
        k = (r["model"], r["prompt"], r["mitigation"], r["failure_mode"])
        if k in cells:
            cells[k]["old"].append(r)
    for (model, p, mit, fm), d in sorted(cells.items()):
        new_ids = {r["task_id"] for r in d["new"]}
        oldr = [r for r in d["old"] if r["task_id"] not in new_ids]
        pool = d["new"] + oldr
        n, nn, no = len(pool), len(d["new"]), len(oldr)
        k = sum(1 for r in pool if r["violations"] > 0)
        kn = sum(1 for r in d["new"] if r["violations"] > 0)
        ko = sum(1 for r in oldr if r["violations"] > 0)
        lo, hi = sweep.wilson_ci(k, n)
        nlo, nhi = sweep.wilson_ci(kn, nn)
        olo, ohi = sweep.wilson_ci(ko, no)
        out.append({"exp": cfg["exp"], "model": model, "prompt": p, "mitigation": mit,
                    "failure_mode": fm, "n_tasks_pooled": n, "ivr_pooled": round(k / n, 4),
                    "ivr_pooled_ci_lo": round(lo, 4), "ivr_pooled_ci_hi": round(hi, 4),
                    "n_new": nn, "ivr_new": round(kn / nn, 4) if nn else None,
                    "ivr_new_ci_lo": round(nlo, 4), "ivr_new_ci_hi": round(nhi, 4),
                    "n_released_s0": no, "ivr_released_s0": round(ko / no, 4) if no else None,
                    "ivr_released_s0_ci_lo": round(olo, 4), "ivr_released_s0_ci_hi": round(ohi, 4),
                    "new_within_released_ci": (olo <= kn / nn <= ohi) if nn and no else None})
    if out:
        path = os.path.join(cfg["out_dir"], "summary_pooled.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
            w.writeheader(); w.writerows(out)
        print(f"{len(out)} pooled cells -> {path}")
    return out


def _write_cells(cfg, rows, by, path) -> list:
    cells = {}
    for r in rows:
        cells.setdefault(tuple(r[k] for k in by), []).append(r)
    out = []
    any_ref = any(r.get("refusal") for r in rows)
    for key, eps in sorted(cells.items(), key=lambda kv: tuple(map(str, kv[0]))):
        d = dict(zip(by, key))
        d["exp"] = cfg["exp"]
        ref = [r for r in eps if r.get("refusal")]
        eps = [r for r in eps if not r.get("refusal")]
        if eps:
            d.update(cell_stats(eps))
            d.update(_released_stats(eps))
        else:
            d["n_episodes"] = 0
        if any_ref:            # column only in runs that have refusal rows
            d["refusals"] = len(ref)
        out.append(d)
    fields = []
    for d in out:
        fields += [k for k in d if k not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(out)
    print(f"{len(out)} cells -> {path}")
    return out


def restart_cell_stats(eps: list) -> dict:
    """One restart cell (model x resume_context x mitigation): gIVR with
    episode- and graph-level Wilson intervals, order violations, completion,
    session-2 look-ups, refusals, truncation, served model, temperature,
    UTC window, cost. Refusals (a session that stopped with stop_reason
    "refusal") are their own outcome and excluded from the rates."""
    ref = [e for e in eps if e.get("refusal")]
    sc = [e for e in eps if not e.get("refusal")]
    n = len(sc)
    G = len({e["graph_id"] for e in sc})
    gk = sum(1 for e in sc if e["givr_hit"])
    p = gk / n if n else None
    lo, hi = sweep.wilson_ci(gk, n)
    glo, ghi = sweep.wilson_ci(p * G, G) if n else (0.0, 1.0)
    look = [e for e in sc if e.get("session2_lookup") is not None]
    out = {"n_episodes": n, "n_graphs": G, "refusals": len(ref),
           "givr": round(p, 4) if n else None,
           "givr_ci_lo": round(lo, 4), "givr_ci_hi": round(hi, 4),
           "givr_ci_glo": round(glo, 4), "givr_ci_ghi": round(ghi, 4),
           "violation_depth": round(sum(e["violations"] for e in sc) / n, 4) if n else None,
           "ovr": _rate(sc, "order_violation"), "completion": _rate(sc, "completion"),
           "interrupted": _rate(sc, "interrupted"),
           "session2_lookup": round(sum(1 for e in look if e["session2_lookup"]) / len(look), 4)
           if look else None,
           "truncated": _rate(sc, "truncated"), "turn_cap_hit": _rate(sc, "turn_cap_hit"),
           "invalid_call_eps": round(sum(1 for e in sc if e.get("invalid_calls")) / n, 4) if n else None}
    out["model_served"] = ";".join(sorted({str(e.get("model_served")) for e in eps}))
    out["temperature_effective"] = ";".join(sorted({str(e.get("temperature")) for e in eps}))
    ts = sorted(x for e in eps for x in (e.get("started_utc"), e.get("finished_utc")) if x)
    out["first_utc"], out["last_utc"] = (ts[0], ts[-1]) if ts else ("", "")
    out["cost_usd"] = round(sum(e.get("cost_usd") or 0 for e in eps), 4)
    return out


def aggregate_restart(cfg, rows) -> list:
    cells = {}
    for r in rows:
        cells.setdefault((r["model"], r["resume_context"], r["mitigation"]), []).append(r)
    out = []
    for key, eps in sorted(cells.items()):
        d = dict(zip(("model", "resume_context", "mitigation"), key))
        d["exp"] = cfg["exp"]
        d.update(restart_cell_stats(eps))
        out.append(d)
    path = os.path.join(cfg["out_dir"], "summary_restart.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    print(f"{len(out)} restart cells -> {path}")
    return out


def aggregate(cfg, by=("model", "prompt", "mitigation", "failure_mode")) -> list:
    """summary.csv (one row per cell) + optional extras from the config:
    import_from (E1-1), compare_released (E1-0), pool_released (E1-4),
    extra_groupings (e.g. by fail_on_call for E1-5, by domain for E1-4).
    Restart rows (kind = "restart", R2) go to summary_restart.csv."""
    os.makedirs(cfg["out_dir"], exist_ok=True)
    rows = read_rows(cfg["raw_path"])
    restart = [r for r in rows if r.get("kind") == RESTART]
    rows = [r for r in rows if r.get("kind") != RESTART]
    if restart:
        aggregate_restart(cfg, restart)
    for r in rows:
        r["source"] = "run"
    own = {r["episode_id"] for r in rows}
    rows += [r for r in import_rows(cfg) if r["episode_id"] not in own]
    if not rows:
        if not restart:
            print(f"no raw data at {cfg['raw_path']}")
        return []
    compare_released(cfg, rows)
    out = _write_cells(cfg, rows, by + ("source",), cfg["summary_path"])
    for extra in cfg.get("extra_groupings", []):
        _write_cells(cfg, [r for r in rows if r["source"] == "run"], tuple(extra),
                     os.path.join(cfg["out_dir"], f"summary_by_{extra[-1]}.csv"))
    pooled_with_released(cfg, rows)
    target = cfg.get("expect_per_cell")
    if target:
        short = [d for d in out if d["n_episodes"] + d.get("refusals", 0) < target]
        if short:
            print(f"   {len(short)} cell(s) below {target} episodes (rerun the "
                  f"launcher; imported cells fill as their own runner finishes):")
        for d in short:
            print(f"   note: {d['model']} {d['prompt']}/{d['mitigation']}/"
                  f"{d['failure_mode']} has {d['n_episodes']}/{target} episodes")
    return out


# ---------------------------------------------------------------------------
# Key checks (cheap): Anthropic = 8-token request; OpenRouter = GET /key (free)
# ---------------------------------------------------------------------------

def check_keys(cfg_paths) -> int:
    models = set()
    for p in cfg_paths:
        models |= set(load_config(p)["models"])
    rc = 0
    if any(provider_of(m) == "anthropic" for m in models) and not os.environ.get("BENCH_CHECK_SKIP_ANTHROPIC"):
        try:
            import anthropic
            r = anthropic.Anthropic().messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=8,
                messages=[{"role": "user", "content": "Say OK"}])
            print("anthropic key OK:", getattr(r, "model", "?"))
        except Exception as e:
            print(f"!! anthropic key check failed: {e!s:.300}")
            rc |= 1
    if any(provider_of(m) == "openrouter" for m in models):
        import urllib.error, urllib.request
        from providers import OpenRouterProvider
        key = os.environ.get("OPENROUTER_API_KEY")
        try:
            if not key:
                raise RuntimeError("OPENROUTER_API_KEY not set")
            url = OpenRouterProvider.BASE.rsplit("/chat/completions", 1)[0] + "/key"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read()).get("data", {})
            print(f"openrouter key OK: limit {d.get('limit')}, "
                  f"remaining {d.get('limit_remaining')}, used {d.get('usage')}")
            if d.get("limit_remaining") is not None and float(d["limit_remaining"]) <= 0:
                raise RuntimeError("the key's remaining limit is 0 (no credit left on this key)")
            try:   # account credits (GET /credits): total_credits - total_usage
                req = urllib.request.Request(url[:-len("/key")] + "/credits",
                                             headers={"Authorization": f"Bearer {key}"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    c = json.loads(r.read()).get("data", {})
                left = float(c.get("total_credits", 0)) - float(c.get("total_usage", 0))
                print(f"openrouter credits: ${left:.2f} remaining")
                if left <= 0:
                    raise RuntimeError(f"OpenRouter credit is ${left:.2f}: top up before running")
            except urllib.error.URLError:
                pass            # /credits not reachable: the key check above stands
        except Exception as e:
            print(f"!! openrouter key check failed: {e!s:.300}")
            rc |= 2
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", action="append", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--check-keys", action="store_true")
    ap.add_argument("--status", action="store_true",
                    help="remaining episodes per config; exit 1 if any remain")
    ap.add_argument("--models", default="", help="comma subset of the config's models")
    a = ap.parse_args(argv)
    if a.check_keys:
        return check_keys(a.config)
    if a.status:
        left = 0
        for path in a.config:
            c = load_config(path)
            eps = plan(c)
            done = done_ids(c)
            rem = [e for e in eps if eid_of(e) not in done]
            by = {}
            for e in rem:
                by[e[0]] = by.get(e[0], 0) + 1
            left += len(rem)
            print(f"  {c['exp']:6s} {len(eps) - len(rem):5d}/{len(eps):<5d} done"
                  + (f"   remaining: {by}" if by else ""))
        if a.config:
            sp = spent_by_provider(load_config(a.config[0])["tier1_root"])
            print(f"  logged spend under {load_config(a.config[0])['tier1_root']}: "
                  f"Anthropic ${sp['anthropic']:.2f} (list prices), "
                  f"OpenRouter ${sp['openrouter']:.2f} (usage.cost)")
        return 1 if left else 0
    cfg = load_config(a.config[0])
    models = [m for m in a.models.split(",") if m] or None
    if a.launch:
        return launch(cfg, models)
    if a.aggregate:
        aggregate(cfg)
        return 0
    eps = plan(cfg, models)
    done = done_ids(cfg)
    rem = [e for e in eps if eid_of(e) not in done]
    est = estimate(cfg, rem)
    tasks = tasks_for(cfg)
    n_retry = sum(1 for e in eps if not is_restart(e))
    if cfg["groups"]:
        for g in cfg["groups"]:
            nr = len(g["models"]) * len(g["cells"]) * len(tasks) * cfg["samples"]
            ns = len(g["models"]) * sum(len(restart_graphs(c[2])) for c in g["restart_cells"]) \
                * cfg["restart_samples"]
            print(f"{cfg['exp']} part {g.get('part', '?')}: {len(g['models'])} models x "
                  f"({len(g['cells'])} retry cells x {len(tasks)} tasks x r={cfg['samples']}"
                  f" + {len(g['restart_cells'])} restart cells x graphs x r={cfg['restart_samples']})"
                  f" = {nr + ns} episodes   [{', '.join(g['models'])}]")
        print(f"{cfg['exp']}: {len(eps)} episodes in all; done {len(eps) - len(rem)}, "
              f"remaining {len(rem)}")
    else:
        print(f"{cfg['exp']}: {len(cfg['models'])} models x {len(cfg['cells'])} cells x "
              f"{len(tasks)} tasks x r={cfg['samples']} = {n_retry} episodes "
              f"({cfg['retry_mode']})"
              + (f" + {len(eps) - n_retry} restart episodes ({len(cfg['restart_cells'])} "
                 f"restart cells, r={cfg['restart_samples']})" if cfg["restart_cells"] else "")
              + f"; done {len(eps) - len(rem)}, remaining {len(rem)}")
    print(f"{cfg['exp']}: est. cost of remaining: Anthropic ${est['anthropic']:.2f}, "
          f"OpenRouter ${est['openrouter']:.2f}"
          + (f" (+ unpriced: {', '.join(est['unpriced_models'])})"
             if est["unpriced_models"] else ""))
    if a.dry_run:
        print(f"out_dir: {cfg['out_dir']}  (no API calls made)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
