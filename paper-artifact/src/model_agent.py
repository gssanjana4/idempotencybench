"""Real-model subject agent (Claude Fable 5) for the benchmark.

The model runs a natural tool-use loop against the simulated world. When a
tool call returns an error (including injected timeouts), THE MODEL decides
whether to retry and what idempotency key to send — that decision is what
IVR measures. Nothing here is scripted.

Every episode writes a full transcript to
results/transcripts/, including the final ledger and scoring outcome, so
the semantic matcher can be validated against what the model actually did.

Requires: pip install anthropic ; ANTHROPIC_API_KEY set.
Costs nothing until you actually run it.
"""
from __future__ import annotations

import csv
import json
import os

from world import (World, ToolTimeout, ToolServerError, ToolCallError,
                   EFFECTFUL_TOOLS)
from tasks import generate_tasks, TEMPLATE_COUNT

MODEL_ID = os.environ.get("BENCH_MODEL", "claude-fable-5")  # pin dated id for camera-ready
MAX_TURNS = 8          # hard cap per episode (cost control)
MAX_TOKENS = int(os.environ.get("BENCH_MAX_TOKENS", "1024"))  # default unchanged
# Completion stop reasons meaning "cut off at max_tokens" (Anthropic
# stop_reason / OpenRouter finish_reason). A reply with no tool call that
# stopped for one of these is flagged `truncated`, not treated as a clean DONE.
TRUNCATED_STOPS = ("max_tokens", "length", "error")  # "error": OpenRouter in-band error
# Pinned for reproducibility where the model supports it; some models
# (e.g. claude-fable-5) deprecate `temperature` and reject requests that
# set it — detected at runtime per model, recorded per-transcript as null.
TEMPERATURE = 0.0
_temperature_unsupported: set = set()  # model ids that reject temperature

BASE_SYSTEM = (
    "You are an assistant that completes one task using the tools provided. "
    "Complete the task, then reply DONE."
)
CAREFUL_SYSTEM = BASE_SYSTEM + (
    " IMPORTANT: tool calls can fail AFTER the action has already been "
    "executed. Before retrying any failed action, use list_effects to check "
    "whether it already happened. Never perform the same action twice."
)


def _tool_schemas():
    schemas = []
    for tool, (_, canon) in EFFECTFUL_TOOLS.items():
        props = {f: {"type": "string"} for f in canon}
        props["idempotency_key"] = {
            "type": "string",
            "description": "Optional idempotency key of your choosing.",
        }
        schemas.append({
            "name": tool,
            "description": f"Execute {tool}. Has a real side effect.",
            "input_schema": {"type": "object", "properties": props,
                             "required": list(canon)},
        })
    schemas.append({
        "name": "list_effects",
        "description": "Read-only: list side effects already recorded for this task.",
        "input_schema": {"type": "object", "properties": {}},
    })
    return schemas


def _serialize_content(content) -> list:
    out = []
    for b in content:
        if b.type == "text":
            out.append({"type": "text", "text": b.text})
        elif b.type == "tool_use":
            out.append({"type": "tool_use", "id": b.id, "name": b.name,
                        "input": dict(b.input)})
        else:
            out.append({"type": b.type})
    return out


def _as_provider(client_or_provider, model):
    """Accept a legacy anthropic.Anthropic client or a provider object."""
    if hasattr(client_or_provider, "complete"):
        return client_or_provider
    from providers import provider_for
    return provider_for(model, anthropic_client=client_or_provider)


def run_model_episode(client, world: World, task, system: str,
                      model: str = None) -> dict:
    """One episode: model converses with the tool world until DONE/cap.
    `client` may be an anthropic.Anthropic client (legacy) or any provider
    from src/providers.py. Returns the full transcript (turn-by-turn log)
    for audit, with per-episode reproducibility fields."""
    import time
    model = model or MODEL_ID
    prov = _as_provider(client, model)
    transcript = {
        "model": model,
        "provider": prov.name,
        "model_served": None,
        "temperature_requested": TEMPERATURE,
        "temperature": None,            # effective; filled from first completion
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "usage": {"input": 0, "output": 0},
        "task_id": task.task_id, "family": task.family,
        "instruction": task.instruction, "system": system, "turns": [],
        "truncated": False, "invalid_calls": [],
    }
    messages = [{"role": "user", "content": task.instruction}]
    for _ in range(MAX_TURNS):
        comp = prov.complete(model, system, _tool_schemas(), messages,
                             MAX_TOKENS, TEMPERATURE)
        transcript["temperature"] = comp.temperature_effective
        transcript["model_served"] = comp.model_served
        for k, v in comp.usage.items():      # input, output (+ cache_* when caching is on)
            if v is not None:
                transcript["usage"][k] = transcript["usage"].get(k, 0) + v
        turn = {"role": "assistant", "content": comp.blocks,
                "at_utc": comp.timestamp_utc, "stop_reason": comp.stop_reason}
        if comp.provider_meta:
            turn["provider_meta"] = comp.provider_meta
        transcript["turns"].append(turn)
        tool_uses = comp.tool_uses
        if not tool_uses:
            # model considers itself done -- unless the reply was cut off
            transcript["truncated"] = comp.stop_reason in TRUNCATED_STOPS
            return transcript
        messages.append(prov.history_assistant(comp))
        results = []
        for tu in tool_uses:
            args = dict(tu["input"])
            key = args.pop("idempotency_key", "") or ""
            if not isinstance(key, str):  # dict/list keys are unhashable
                key = json.dumps(key, sort_keys=True)
            try:
                out = world.call(tu["name"], args, key, task.task_id)
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": out})
            except (ToolTimeout, ToolServerError) as e:
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": f"ERROR: {e}", "is_error": True})
            except ToolCallError as e:
                # unknown tool / malformed args: nothing committed; tell the
                # model and keep going (the episode is not aborted)
                results.append({"type": "tool_result", "tool_use_id": tu["id"],
                                "content": f"ERROR: {e}", "is_error": True})
                transcript["invalid_calls"].append(
                    {"tool_use_id": tu["id"], "name": tu["name"], "error": str(e)})
        transcript["turns"].append({"role": "tool_results", "content": results})
        messages.append(prov.history_tool_results(results))
    return transcript


def _score_and_log(world: World, task, transcript: dict,
                   transcript_dir: str, cell: str) -> tuple[int, bool]:
    violations = world.ledger.violations(task.task_id)
    out = world.ledger.outcome(task.task_id, task.goals, task.ordered)
    achieved = out["success"]
    transcript["ledger"] = [
        {"tool": e.tool, "args": e.args, "key": e.key, "seq": e.seq}
        for e in world.ledger.for_task(task.task_id)
    ]
    transcript["score"] = {"violations": violations, "achieved": achieved,
                           "omission": out["omission"], "misorder": out["misorder"],
                           "goals": [{"tool": g.tool, "canon_args": g.canon_args}
                                     for g in task.goals]}
    os.makedirs(transcript_dir, exist_ok=True)
    # "/" in OpenRouter slugs (openrouter:openai/gpt-5.5) would be read as a
    # directory; keep one flat file per episode.
    path = os.path.join(transcript_dir,
                        f"{cell}__{task.task_id}.json".replace("/", "_"))
    with open(path, "w") as f:
        json.dump(transcript, f, indent=1)
    return violations, achieved


def run_fable5_pilot(out_path: str,
                     n_tasks: int = TEMPLATE_COUNT,  # one per template
                     prompts=("base", "careful"),
                     mitigations=("none", "runtime_receipts"),
                     failure_mode: str = "timeout_after_commit"):
    """Small cost-controlled grid: 2 prompts x 2 mitigations x n tasks
    (default 8 = one task per template, covering all four families).
    At Fable 5 pricing this is a few dollars, not more."""
    import anthropic
    client = anthropic.Anthropic()

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transcript_dir = os.path.join(here, "results", "transcripts")

    tasks = generate_tasks(per_domain=1)[:n_tasks]
    rows = []
    for prompt in prompts:
        system = CAREFUL_SYSTEM if prompt == "careful" else BASE_SYSTEM
        for mitigation in mitigations:
            cell = f"{prompt}_{mitigation}_{failure_mode}"
            violations = successes = 0
            for task in tasks:
                world = World(mitigation=mitigation, failure_mode=failure_mode)
                transcript = run_model_episode(client, world, task, system)
                v, ok = _score_and_log(world, task, transcript,
                                       transcript_dir, cell)
                violations += 1 if v else 0
                successes += 1 if ok else 0
            rows.append({
                "model": MODEL_ID, "prompt": prompt, "mitigation": mitigation,
                "failure_mode": failure_mode, "n_tasks": len(tasks),
                "ivr": round(violations / len(tasks), 4),
                "success": round(successes / len(tasks), 4),
            })
            print(json.dumps(rows[-1]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"pilot -> {out_path}  (transcripts -> {transcript_dir})")
