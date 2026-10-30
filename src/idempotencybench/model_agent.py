"""Real-model subject agent (Claude Fable 5) for IdempotencyBench.

The model runs a natural tool-use loop against the simulated world. When a
tool call returns an error (including injected timeouts), THE MODEL decides
whether to retry and what idempotency key to send — that decision is what
IVR measures. Nothing here is scripted.

Every episode writes a full transcript (mock-review W3) to
results/transcripts/, including the final ledger and scoring outcome, so
the semantic matcher can be validated against what the model actually did.

Requires: pip install anthropic ; ANTHROPIC_API_KEY set.
Costs nothing until you actually run it.
"""
from __future__ import annotations

import csv
import json
import os

from .world import World, ToolTimeout, ToolServerError, EFFECTFUL_TOOLS
from .tasks import generate_tasks, TEMPLATE_COUNT

MODEL_ID = os.environ.get("IBENCH_MODEL", "claude-fable-5")  # pin dated id for camera-ready
MAX_TURNS = 8          # hard cap per episode (cost control)
MAX_TOKENS = 1024
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


def run_model_episode(client, world: World, task, system: str,
                      model: str = None) -> dict:
    """One episode: model converses with the tool world until DONE/cap.
    Returns the full transcript (turn-by-turn log) for audit."""
    model = model or MODEL_ID
    transcript = {
        "model": model,
        "temperature": None if model in _temperature_unsupported else TEMPERATURE,
        "task_id": task.task_id, "family": task.family,
        "instruction": task.instruction, "system": system, "turns": [],
    }
    messages = [{"role": "user", "content": task.instruction}]
    for _ in range(MAX_TURNS):
        import anthropic
        kwargs = dict(model=model, max_tokens=MAX_TOKENS, system=system,
                      tools=_tool_schemas(), messages=messages)
        if model not in _temperature_unsupported:
            kwargs["temperature"] = TEMPERATURE
        try:
            resp = client.messages.create(**kwargs)
        except anthropic.BadRequestError as e:
            if model not in _temperature_unsupported and "temperature" in str(e):
                _temperature_unsupported.add(model)
                transcript["temperature"] = None
                kwargs.pop("temperature", None)
                resp = client.messages.create(**kwargs)
            else:
                raise
        transcript["turns"].append(
            {"role": "assistant", "content": _serialize_content(resp.content)})
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        if not tool_uses:
            return transcript  # model considers itself done
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            args = dict(tu.input)
            key = args.pop("idempotency_key", "") or ""
            try:
                out = world.call(tu.name, args, key, task.task_id)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": out})
            except (ToolTimeout, ToolServerError) as e:
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": f"ERROR: {e}", "is_error": True})
        transcript["turns"].append({"role": "tool_results", "content": results})
        messages.append({"role": "user", "content": results})
    return transcript


def _score_and_log(world: World, task, transcript: dict,
                   transcript_dir: str, cell: str) -> tuple[int, bool]:
    violations = world.ledger.violations(task.task_id)
    achieved = all(world.ledger.achieved(task.task_id, g) for g in task.goals)
    transcript["ledger"] = [
        {"tool": e.tool, "args": e.args, "key": e.key, "seq": e.seq}
        for e in world.ledger.for_task(task.task_id)
    ]
    transcript["score"] = {"violations": violations, "achieved": achieved,
                           "goals": [{"tool": g.tool, "canon_args": g.canon_args}
                                     for g in task.goals]}
    os.makedirs(transcript_dir, exist_ok=True)
    path = os.path.join(transcript_dir, f"{cell}__{task.task_id}.json")
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

    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
