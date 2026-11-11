"""Scripted (zero-cost) agents and retry orchestration.

Agents decide WHETHER to retry and WHAT key to send. Retry modes decide
WHO re-executes. Mitigations live in the world (see world.py).

Retry modes:
  no_retry       nobody retries.
  tool_retry     transport layer transparently reissues a failed call,
                 same key, below the agent's visibility.
  agent_retry    the agent itself retries on a surfaced error.
  whole_run      the orchestrator re-executes the episode from scratch,
                 but ONLY if the run reported failure — matching workflow
                 engines that retry failed runs, never successful ones.
  at_least_once  the episode is delivered/executed twice unconditionally,
                 modeling at-least-once delivery (e.g., standard SQS
                 queues, at-least-once activity execution): a consumer
                 can receive a duplicate even after a successful run.
"""
from __future__ import annotations

import hashlib
import uuid

from world import World, ToolTimeout, ToolServerError
from tasks import Task

RETRY_MODES = ("no_retry", "tool_retry", "agent_retry", "whole_run",
               "at_least_once")


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class NaiveAgent:
    """Retries on any error; mints a fresh UUID key on every call —
    which is what unprompted real agents do."""

    name = "naive"
    max_attempts = 3

    def key_for(self, task: Task, tool: str, args: dict, attempt: int) -> str:
        return uuid.uuid4().hex  # fresh every time

    def run(self, world: World, task: Task, retry_mode: str) -> bool:
        """Execute the task's steps in order. Returns True iff the episode
        ends without an unresolved failure — the run-level status an
        orchestrator sees."""
        for tool, args in task.steps:
            if not self._do_step(world, task, tool, args, retry_mode):
                return False
        return True

    def _do_step(self, world: World, task: Task, tool: str, args: dict,
                 retry_mode: str) -> bool:
        attempts = self.max_attempts if retry_mode == "agent_retry" else 1
        for attempt in range(attempts):
            try:
                self._call(world, task, tool, args, attempt, retry_mode)
                return True
            except (ToolTimeout, ToolServerError):
                if retry_mode == "agent_retry":
                    continue  # naive: just try again
                return False
        return False

    def _call(self, world: World, task: Task, tool: str, args: dict,
              attempt: int, retry_mode: str):
        key = self.key_for(task, tool, args, attempt)
        if retry_mode == "tool_retry":
            # transport layer transparently reissues on failure, same key
            try:
                world.call(tool, args, key, task.task_id)
            except (ToolTimeout, ToolServerError):
                world.call(tool, args, key, task.task_id)
        else:
            world.call(tool, args, key, task.task_id)


class CarefulAgent(NaiveAgent):
    """Prompt-level mitigation: on an ambiguous failure, read back the
    world state before retrying. Catches per-call duplicates — but an
    at-least-once redelivery presents no failure, so it never thinks
    to check."""

    name = "careful"

    def _do_step(self, world: World, task: Task, tool: str, args: dict,
                 retry_mode: str) -> bool:
        attempts = self.max_attempts if retry_mode == "agent_retry" else 1
        for attempt in range(attempts):
            try:
                self._call(world, task, tool, args, attempt, retry_mode)
                return True
            except (ToolTimeout, ToolServerError):
                # read-back: did my action actually happen?
                state = world.call("list_effects", {}, "", task.task_id)
                first_arg = str(args.get(list(args)[0]))
                if tool in state and first_arg in state:
                    return True  # effect exists; do NOT retry — step done
                if retry_mode != "agent_retry":
                    return False
        return False

    def _call(self, world: World, task: Task, tool: str, args: dict,
              attempt: int, retry_mode: str):
        key = self.key_for(task, tool, args, attempt)
        if retry_mode == "tool_retry":
            try:
                world.call(tool, args, key, task.task_id)
            except (ToolTimeout, ToolServerError):
                # transport retries BELOW the agent: the model never sees
                # the failure, so prompt-level care cannot intervene here.
                world.call(tool, args, key, task.task_id)
        else:
            world.call(tool, args, key, task.task_id)


class StableKeyAgent(NaiveAgent):
    """Derives the idempotency key from task identity + intended action,
    so retries reuse the same key. Models the 'well-engineered client'."""

    name = "stable_key"

    def key_for(self, task: Task, tool: str, args: dict, attempt: int) -> str:
        raw = task.task_id + tool + str(sorted(args.items()))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


AGENTS = {a.name: a for a in (NaiveAgent(), CarefulAgent(), StableKeyAgent())}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(agent, world: World, task: Task, retry_mode: str) -> None:
    """One task under one retry mode, against the SAME world.

    whole_run: the orchestrator re-executes the episode from scratch only
    if run 1 reported failure (no production workflow engine retries a run
    that reported success). The re-run agent starts fresh, with no memory
    of run 1.

    at_least_once: the episode executes twice unconditionally — the
    at-least-once delivery condition, where a duplicate delivery can follow
    even a successful run.
    """
    if retry_mode == "whole_run":
        ok = agent.run(world, task, "no_retry")
        if not ok:
            agent.run(world, task, "no_retry")
    elif retry_mode == "at_least_once":
        agent.run(world, task, "no_retry")
        agent.run(world, task, "no_retry")
    else:
        agent.run(world, task, retry_mode)
