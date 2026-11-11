"""Session-resume model sweep: real-model session-interruption/resume episodes.

Pre-registered grid (the session-resume experiment plan):
  3 models x 4 cells x 16 graphs (8 chain4 + 8 fanout, interrupts balanced
  across depths, after_commit) x r=2  =  384 episodes.

Cells (resume_context, mitigation):
  M1  amnesiac        / none              seam behavior, no help
  M2a plan_only       / none              plan memory: fixes order, not seam?
  M2b ledger_readback / none              state memory: fixes both?
  M3  amnesiac        / runtime_receipts  substrate floor (checkpoint analogue)

Session-1 interruption for a MODEL agent: the tool loop is cut immediately
after the interrupt node's semantic effect commits -- the model never sees
that tool result (committed-but-unacknowledged seam == after_commit).

Usage (run locally; requires an API key and incurs cost):
  python3 src2/sweep2.py --dry-run        # validate grid+prompts, no API, $0
  python3 src2/sweep2.py --smoke          # Fable-only, r=1: 64 eps (~$10)
  python3 src2/sweep2.py --launch         # full 384-episode grid (~$35-60)
Resumable: rows append to results/sweep2_raw.jsonl keyed by episode_id.
NEVER launched automatically -- requires an explicit go (budget).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from world import World, ToolTimeout, ToolServerError, ToolCallError  # noqa: E402
from graph import _chain4, _fanout  # noqa: E402
from resume import score_graph, _node_commits, RESUME_CONTEXTS  # noqa: E402
from model_agent import (_tool_schemas, _serialize_content,  # noqa: E402
                         _temperature_unsupported, TEMPERATURE, MAX_TOKENS,
                         TRUNCATED_STOPS)

DEFAULT_MODELS = "claude-fable-5,claude-sonnet-5,claude-haiku-4-5-20251001"
MODELS = [m for m in os.environ.get(
    "BENCH_SWEEP_MODELS", DEFAULT_MODELS).split(",") if m]

# (resume_context, mitigation) -- pre-registered, do not reorder
CELLS = (
    ("amnesiac", "none"),
    ("plan_only", "none"),
    ("ledger_readback", "none"),
    ("amnesiac", "runtime_receipts"),
)
# W2 ablation + W3 cross-model cells, run separately via --fixes
FIX_CELLS = (
    ("bare_hint", "none"),      # W2: continuity hint with no progress content
    ("amnesiac", "none"),       # W3: cross-model baseline
    ("plan_only", "none"),      # W3: cross-model self-verification check
)
FIX_MODEL = os.environ.get("BENCH_FIX_MODEL", "claude-haiku-4-5-20251001")
R = 2                    # repetitions per (model, cell, graph)
MAX_TURNS_SESSION = 10   # graph episodes are longer than the single-session 8

# BENCH_OUT_DIR: write a new run outside results/ (default unchanged).
OUT_DIR = os.environ.get("BENCH_OUT_DIR", os.path.join(HERE, "results"))
RAW_PATH = os.path.join(OUT_DIR, "sweep2_raw.jsonl")
SUMMARY_PATH = os.path.join(OUT_DIR, "sweep2_summary.csv")
TRANSCRIPT_DIR = os.path.join(OUT_DIR, "transcripts2")
# BENCH_CELLS: override the cell list for --launch, e.g.
# "amnesiac:none,plan_only:none,bare_hint:none" (default: pre-registered CELLS)
if os.environ.get("BENCH_CELLS"):
    CELLS = tuple(tuple(c.split(":")) for c in
                  os.environ["BENCH_CELLS"].split(",") if c)

BASE_SYSTEM = (
    "You are an assistant that completes one multi-step task using the tools "
    "provided. Some steps depend on earlier steps; respect the stated order. "
    "Complete the task, then reply DONE."
)
# R2 (wording robustness) paraphrases of the bare_hint note. Like bare_hint
# they signal continuity only: no progress content, no list_effects
# instruction. Appended to the instruction exactly as bare_hint is.
HINT_PARAPHRASES = {
    "hint_p1": "NOTE: this task was started earlier in another session, which "
               "stopped before it finished. Complete the task.",
    "hint_p2": "This is a resumed task: an earlier attempt was cut off partway "
               "through. Please finish it.",
    "hint_p3": "Heads up: someone already began working on this task before "
               "being interrupted. Carry on and finish it.",
}
READBACK_HINT = (
    " IMPORTANT: a previous session may have already performed some of these "
    "actions. Before performing any action, use list_effects to check what "
    "has already been done, and never perform the same action twice."
)


def benchmark_graphs():
    """16 graphs: 8 chain4 + 8 fanout, interrupt depth balanced by index.
    Interrupt node for graph i cycles over the interior nodes so the seam
    lands at every depth equally (after_commit only, per the plan)."""
    out = []
    for i in range(8):
        g = _chain4(i)
        out.append((g, i % len(g.nodes)))
    for i in range(8):
        g = _fanout(i)
        out.append((g, i % len(g.nodes)))
    return out


def episode_id(model, ctx, mit, graph_id, k, sample) -> str:
    return f"{model}__{ctx}__{mit}__{graph_id}__k{k}__s{sample}"


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


# ---------------------------------------------------------------------------
# Model session with mid-plan interruption
# ---------------------------------------------------------------------------

def run_model_session(client, model, world, graph, system, user_msg,
                      interrupt_node=None, include_readback_tool=True):
    """One session of the tool loop. If interrupt_node is not None, the
    session is cut the moment that node's semantic effect commits: the
    pending tool result is NOT returned to the model (after_commit seam).
    `client` may be a legacy anthropic client or a src/providers.py provider.
    Returns (turns, interrupted, acknowledged_nodes, meta) where meta holds
    provider / served model / effective temperature / token usage."""
    from model_agent import _as_provider
    prov = _as_provider(client, model)
    turns = []
    acknowledged = set()
    meta = {"provider": prov.name, "model_served": None,
            "temperature_requested": TEMPERATURE, "temperature": None,
            "usage": {"input": 0, "output": 0},
            "truncated": False, "invalid_calls": []}
    messages = [{"role": "user", "content": user_msg}]
    for _ in range(MAX_TURNS_SESSION):
        comp = prov.complete(model, system, _tool_schemas(), messages,
                             MAX_TOKENS, TEMPERATURE)
        meta["temperature"] = comp.temperature_effective
        meta["model_served"] = comp.model_served
        for kk, v in comp.usage.items():      # input, output (+ cache_* when caching is on)
            if v is not None:
                meta["usage"][kk] = meta["usage"].get(kk, 0) + v
        turn = {"role": "assistant", "content": comp.blocks,
                "at_utc": comp.timestamp_utc, "stop_reason": comp.stop_reason}
        if comp.provider_meta:
            turn["provider_meta"] = comp.provider_meta
        turns.append(turn)
        tool_uses = comp.tool_uses
        if not tool_uses:
            meta["truncated"] = comp.stop_reason in TRUNCATED_STOPS
            return turns, False, acknowledged, meta
        messages.append(prov.history_assistant(comp))
        results = []
        for tu in tool_uses:
            args = dict(tu["input"])
            key = args.pop("idempotency_key", "") or ""
            try:
                out = world.call(tu["name"], args, key, graph.graph_id)
                # after-commit interruption check: did the seam node commit?
                if interrupt_node is not None and \
                        _node_commits(world, graph, interrupt_node):
                    turns.append({"role": "interrupt",
                                  "at_node": interrupt_node,
                                  "note": "session cut after commit; result "
                                          "never returned to model"})
                    return turns, True, acknowledged, meta
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": out})
            except (ToolTimeout, ToolServerError) as e:
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": f"ERROR: {e}", "is_error": True})
            except ToolCallError as e:
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": f"ERROR: {e}", "is_error": True})
                meta["invalid_calls"].append(
                    {"tool_use_id": tu["id"], "name": tu["name"], "error": str(e)})
        turns.append({"role": "tool_results", "content": results})
        messages.append(prov.history_tool_results(results))
        # Acknowledged = committed AND its result delivered to the model. Done
        # only once the whole turn's results are returned: if the seam node
        # commits mid-turn, earlier calls of that turn were never acknowledged.
        for idx in range(len(graph.nodes)):
            if _node_commits(world, graph, idx):
                acknowledged.add(idx)
    return turns, False, acknowledged, meta


def resume_user_msg(graph, ctx, acknowledged):
    """The resume-context prompt handed to session 2 (the crux axis)."""
    if ctx == "amnesiac":
        return graph.instruction  # whole_run analogue: no hint of session 1
    if ctx == "bare_hint":
        # W2 ablation: continuity signalled, ZERO progress content. Isolates
        # "a prior session existed" from "here is what it did".
        return (f"{graph.instruction}\n\nNOTE: a previous session was working "
                f"on this task and was interrupted. Finish the task.")
    if ctx in HINT_PARAPHRASES:
        return f"{graph.instruction}\n\n{HINT_PARAPHRASES[ctx]}"
    if ctx == "plan_only":
        done = ", ".join(f"step {i + 1}" for i in sorted(acknowledged)) or "none"
        return (f"{graph.instruction}\n\nNOTE: a previous session was working "
                f"on this task and was interrupted. Its saved plan notes say "
                f"it believed it had completed: {done}. It had no record of "
                f"anything else. Finish the task.")
    if ctx == "ledger_readback":
        return (f"{graph.instruction}\n\nNOTE: a previous session was working "
                f"on this task and was interrupted. Use list_effects to see "
                f"exactly which actions already happened, then finish the "
                f"task without repeating any of them.")
    # amnesiac/receipts cell uses the amnesiac prompt; the substrate dedupes
    return graph.instruction


def run_episode(client, model, ctx, mit, graph, k, sample, transcript_dir=None):
    """transcript_dir: where the transcript goes (default TRANSCRIPT_DIR;
    src/tier1.py passes its own out_dir for the R2 restart cells)."""
    transcript_dir = transcript_dir or TRANSCRIPT_DIR
    world = World(mitigation=mit, failure_mode="none")
    system = BASE_SYSTEM + (READBACK_HINT if ctx == "ledger_readback" else "")
    transcript = {"model": model, "resume_context": ctx, "mitigation": mit,
                  "graph_id": graph.graph_id, "topology": graph.topology,
                  "interrupt_node": k, "sample": sample,
                  "instruction": graph.instruction, "sessions": []}
    # session 1: interrupted after node k commits
    import time as _time
    transcript["started_utc"] = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
    t1, interrupted, acked, meta1 = run_model_session(
        client, model, world, graph, BASE_SYSTEM, graph.instruction,
        interrupt_node=k)
    transcript["sessions"].append({"n": 1, "turns": t1,
                                   "interrupted": interrupted,
                                   "acknowledged": sorted(acked)})
    if interrupted:  # failure-conditioned resume (single-session discipline)
        # seam node committed but never acknowledged -> drop it from notes
        acked.discard(k)
        t2, _, _, meta2 = run_model_session(
            client, model, world, graph, system,
            resume_user_msg(graph, ctx, acked))
        transcript["sessions"].append({"n": 2, "turns": t2, "meta": meta2})
    transcript["sessions"][0]["meta"] = meta1
    transcript["provider"] = meta1["provider"]
    transcript["model_served"] = meta1["model_served"]
    transcript["temperature_requested"] = meta1["temperature_requested"]
    transcript["temperature"] = meta1["temperature"]
    row = score_graph(world, graph, interrupted=interrupted,
                      resume_context=ctx, mitigation=mit,
                      key_posture="model")
    row["resume_context"] = ctx  # includes non-substrate ctx like bare_hint
    row["interrupt_node"] = k
    row["interrupt_timing"] = "after_commit"
    transcript["score"] = {kk: row[kk] for kk in
                           ("violations", "givr_hit", "completion",
                            "order_violation")}
    os.makedirs(transcript_dir, exist_ok=True)
    eid = episode_id(model, ctx, mit, graph.graph_id, k, sample)
    with open(os.path.join(transcript_dir, f"{eid}.json".replace("/", "_")),
              "w") as f:
        json.dump(transcript, f, indent=1)
    row.update({"episode_id": eid, "model": model, "sample": sample})
    metas = [s["meta"] for s in transcript["sessions"]]
    row["truncated"] = any(m.get("truncated") for m in metas)
    row["invalid_calls"] = sum(len(m.get("invalid_calls", [])) for m in metas)
    return row


# ---------------------------------------------------------------------------
# Grid, resumability, summary
# ---------------------------------------------------------------------------

# BENCH_TOPOLOGY: restrict --launch to one topology (e.g. "fanout").
TOPOLOGY = os.environ.get("BENCH_TOPOLOGY", "")
# BENCH_DEADLINE_S: stop starting new episodes after this many seconds
# (lets a run proceed in short resumable chunks); 0 = no deadline.
DEADLINE_S = float(os.environ.get("BENCH_DEADLINE_S", "0"))
# BENCH_BUDGET_USD: hard spend cap for this OUT_DIR, from logged token usage
# (all transcripts already in OUT_DIR count); 0 = no cap.
BUDGET_USD = float(os.environ.get("BENCH_BUDGET_USD", "0"))
# USD per million tokens (input, output). Anthropic list prices 2026-10-25.
PRICES = {"claude-fable-5": (10.0, 50.0), "claude-sonnet-5": (2.0, 10.0),
          "claude-haiku-4-5": (1.0, 5.0)}


def _price(model):
    for prefix, pr in PRICES.items():
        if model.startswith(prefix):
            return pr
    return None


def spent_usd():
    """Spend so far in OUT_DIR, from per-session token usage in transcripts.
    Unknown-price models count as 0 (OpenRouter runs are capped by the key)."""
    total = 0.0
    if not os.path.isdir(TRANSCRIPT_DIR):
        return total
    for fn in os.listdir(TRANSCRIPT_DIR):
        try:
            with open(os.path.join(TRANSCRIPT_DIR, fn)) as f:
                t = json.load(f)
        except Exception:
            continue
        pr = _price(t.get("model", "")) or (0.0, 0.0)
        for sess in t.get("sessions", []):
            u = (sess.get("meta") or {}).get("usage") or {}
            total += ((u.get("input") or 0) * pr[0]
                      + (u.get("output") or 0) * pr[1]) / 1e6
    return total


def build_grid(models, reps):
    grid = []
    for model in models:
        for (ctx, mit) in CELLS:
            for (g, k) in benchmark_graphs():
                if TOPOLOGY and g.topology != TOPOLOGY:
                    continue
                for s in range(reps):
                    grid.append((model, ctx, mit, g, k, s))
    return grid


def summarize():
    # Defensive de-duplication: overlapping runs (e.g. --fixes started while
    # --launch was in flight) can append the same episode_id twice from a
    # stale `done` snapshot. Duplicates would inflate n and shrink CIs. The last
    # row per episode_id wins, as in sweep.aggregate() and analysis/ (e78e23d).
    seen = {}
    if not os.path.exists(RAW_PATH):
        print(f"no raw data at {RAW_PATH}"); return
    with open(RAW_PATH) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                seen[r["episode_id"]] = r
    rows = list(seen.values())
    print(f"summarize: {len(rows)} unique episodes")
    cells = {}
    for r in rows:
        key = (r["model"], r["resume_context"], r["mitigation"])
        cells.setdefault(key, []).append(r)
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "resume_context", "mitigation", "n",
                    "givr", "givr_lo", "givr_hi",
                    "ovr", "ovr_lo", "ovr_hi", "completion"])
        for (model, ctx, mit), rs in sorted(cells.items()):
            n = len(rs)
            gk = sum(r["givr_hit"] for r in rs)
            ok = sum(r["order_violation"] for r in rs)
            glo, ghi = wilson_ci(gk, n)
            olo, ohi = wilson_ci(ok, n)
            w.writerow([model, ctx, mit, n,
                        round(gk / n, 4), round(glo, 4), round(ghi, 4),
                        round(ok / n, 4), round(olo, 4), round(ohi, 4),
                        round(sum(r["completion"] for r in rs) / n, 4)])
    print(f"wrote {SUMMARY_PATH} ({len(cells)} cells, {len(rows)} episodes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="validate grid + prompts offline; no API calls")
    ap.add_argument("--smoke", action="store_true",
                    help="Fable-only, r=1 (64 episodes, ~$10)")
    ap.add_argument("--launch", action="store_true",
                    help="full pre-registered grid (384 episodes)")
    ap.add_argument("--summarize", action="store_true",
                    help="rebuild summary CSV from raw jsonl only")
    ap.add_argument("--fixes", action="store_true",
                    help="W2 bare_hint ablation (Fable) + "
                         "W3 cross-model cells (Haiku). ~48 episodes, ~$5")
    args = ap.parse_args()

    if args.summarize:
        summarize()
        return

    if args.fixes:
        grid = []
        gs = benchmark_graphs()
        # W2: bare_hint ablation on the same model as the main sweep
        for (g, k) in gs:
            grid.append((MODELS[0], "bare_hint", "none", g, k, 0))
        # W3: cross-model amnesiac + plan_only (the capability question)
        for ctx in ("amnesiac", "plan_only"):
            for (g, k) in gs:
                grid.append((FIX_MODEL, ctx, "none", g, k, 0))
        models, reps = [MODELS[0], FIX_MODEL], 1
    else:
        models = MODELS[:1] if args.smoke else MODELS
        reps = 1 if args.smoke else R
        grid = build_grid(models, reps)

    if args.dry_run or not (args.smoke or args.launch or args.fixes):
        gs = benchmark_graphs()
        print(f"graphs: {len(gs)} "
              f"({sum(1 for g, _ in gs if g.topology == 'chain4')} chain4, "
              f"{sum(1 for g, _ in gs if g.topology == 'fanout')} fanout)")
        print(f"interrupt depths: "
              f"{sorted(set(k for _, k in gs))} (balanced)")
        print(f"grid: {len(models)} models x {len(CELLS)} cells x "
              f"{len(gs)} graphs x r={reps} = {len(grid)} episodes")
        g, k = gs[0]
        print("\n--- sample resume prompts (graph", g.graph_id,
              "interrupt after node", k, ") ---")
        for ctx in RESUME_CONTEXTS:
            if ctx == "checkpoint":
                continue
            print(f"\n[{ctx}]\n{resume_user_msg(g, ctx, {0})}")
        print("\ndry run only -- no API calls made. "
              "Use --smoke or --launch to run.")
        return

    from runlock import acquire
    if not acquire(OUT_DIR, "sweep2", models):
        return
    import anthropic
    client = anthropic.Anthropic()
    done = set()
    if os.path.exists(RAW_PATH):
        with open(RAW_PATH) as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["episode_id"])
    remaining = [e for e in grid if episode_id(
        e[0], e[1], e[2], e[3].graph_id, e[4], e[5]) not in done]
    print(f"episodes: {len(grid)} total, {len(done)} done, "
          f"{len(remaining)} remaining")
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    failed = 0
    import time as _time
    t0 = _time.time()
    spent = spent_usd() if BUDGET_USD else 0.0
    if BUDGET_USD:
        print(f"spent so far in {OUT_DIR}: ${spent:.2f} of ${BUDGET_USD:.2f}")
    # Worst-case cost of one more episode, used to stop BEFORE the cap.
    max_ep = 0.0
    for n, (model, ctx, mit, g, k, s) in enumerate(remaining, 1):
        if DEADLINE_S and _time.time() - t0 > DEADLINE_S:
            print(f"deadline {DEADLINE_S:.0f}s reached; rerun to continue")
            break
        if BUDGET_USD and spent + max(max_ep, 0.5) > BUDGET_USD:
            print(f"!! budget ${BUDGET_USD:.2f} would be exceeded "
                  f"(spent ${spent:.2f}); stopping")
            break
        before = spent_usd() if BUDGET_USD else 0.0
        try:
            row = run_episode(client, model, ctx, mit, g, k, s)
        except Exception as e:  # isolate one bad episode; no row => retried next run
            if "HTTP 402" in str(e):
                raise  # out of credits: stop the whole run
            failed += 1
            print(f"!! [{n}/{len(remaining)}] {model} {ctx} {g.graph_id} k{k} s{s} "
                  f"failed, will retry on next run: {e!s:.300}")
            continue
        with open(RAW_PATH, "a") as f:
            f.write(json.dumps(row) + "\n")
        if BUDGET_USD:
            spent = spent_usd()
            max_ep = max(max_ep, spent - before)
        print(f"[{n}/{len(remaining)}] {row['episode_id']} "
              f"givr={row['givr_hit']} ovr={row['order_violation']} "
              f"done={row['completion']}")
    summarize()
    if failed:
        print(f"{failed} episodes failed (rerun to retry)")
        sys.exit(1)


if __name__ == "__main__":
    main()
