"""MCP adapter: serve one IdempotencyBench task as a stdio MCP server.

Any MCP-capable agent (Claude Code, Codex CLI, LangGraph, ...) can then be
evaluated without code changes. The adapter exposes the same tool surface
as model_agent.py: the five effectful tools (canonical fields plus an
optional idempotency_key), the read-only list_effects, and a `finish` tool.
Every call goes through World.call, so failure injection and mitigations
are exactly the ones the harness uses. The ground-truth ledger and the
score are written only to the --out JSON file, never returned to the agent.

The Episode class has no MCP dependency, so its scoring path is testable
with the stdlib alone. build_server() needs the `mcp` package (v1 FastMCP
or v2 MCPServer).
"""
import json
import os
import signal
import threading
from typing import Annotated, Any

from .tasks import Task, generate_tasks
from .world import (EFFECTFUL_TOOLS, ToolServerError, ToolTimeout, World)

SERVER_NAME = "idempotencybench"
KEY_DESCRIPTION = "Optional idempotency key of your choosing."
FINISH_REPLY = "Episode recorded. The task is closed; stop here."


def find_task(task_id: str) -> Task | None:
    """Look up a task by id in the full 320-task set."""
    return next((t for t in generate_tasks(per_domain=40)
                 if t.task_id == task_id), None)


class Episode:
    """One agent episode against one task's world.

    call() is what every MCP tool handler runs. It returns the text the
    agent sees and whether that text is an error. Nothing about the ledger
    beyond what list_effects already shows is ever returned to the agent.
    """

    def __init__(self, task: Task, failure_mode: str = "timeout_after_commit",
                 mitigation: str = "none", out_path: str | None = None):
        self.task = task
        self.world = World(mitigation=mitigation, failure_mode=failure_mode)
        self.out_path = out_path
        self.tool_calls: list[dict[str, Any]] = []
        self.finished = False
        self.finish_summary = ""
        # The v2 SDK runs sync tools on worker threads. Reentrant so the
        # SIGTERM handler can write the result even if it interrupts a call.
        self._lock = threading.RLock()

    def call(self, tool: str, args: dict[str, str],
             idempotency_key: str = "") -> tuple[str, bool]:
        with self._lock:
            try:
                text = self.world.call(tool, args, idempotency_key,
                                       self.task.task_id)
                is_error = False
            except (ToolTimeout, ToolServerError) as e:
                text = str(e)
                is_error = True
            self.tool_calls.append({"tool": tool, "args": dict(args),
                                    "idempotency_key": idempotency_key,
                                    "result": text, "is_error": is_error})
            return text, is_error

    def finish(self, summary: str = "") -> str:
        with self._lock:
            self.finished = True
            self.finish_summary = summary
        self.write_result()
        return FINISH_REPLY

    def result(self) -> dict[str, Any]:
        """Score the episode exactly as harness.run_grid scores one task."""
        with self._lock:
            ledger = self.world.ledger
            task_id = self.task.task_id
            violations = ledger.violations(task_id)
            success = all(ledger.achieved(task_id, g) for g in self.task.goals)
            effects = ledger.for_task(task_id)
            by_tool: dict[str, int] = {}
            for e in effects:
                by_tool[e.tool] = by_tool.get(e.tool, 0) + 1
            return {
                "task_id": task_id,
                "family": self.task.family,
                "instruction": self.task.instruction,
                "failure_mode": self.world.failure_mode,
                "mitigation": self.world.mitigation,
                "finished": self.finished,
                "finish_summary": self.finish_summary,
                "ivr": 1.0 if violations > 0 else 0.0,
                "violations": violations,
                "success": success,
                "ledger": {
                    "n_effects": len(effects),
                    "by_tool": by_tool,
                    "effects": [{"tool": e.tool, "args": e.args,
                                 "key": e.key, "seq": e.seq}
                                for e in effects],
                },
                "goals": [{"tool": g.tool, "canon_args": g.canon_args}
                          for g in self.task.goals],
                "tool_calls": list(self.tool_calls),
            }

    def write_result(self) -> None:
        if not self.out_path:
            return
        result = self.result()
        directory = os.path.dirname(os.path.abspath(self.out_path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = self.out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result, f, indent=1)
        os.replace(tmp_path, self.out_path)


def _import_sdk():
    """Return (server class, ToolError) for mcp v1 (FastMCP) or v2 (MCPServer)."""
    try:
        from mcp.server.fastmcp import FastMCP as server_class
        from mcp.server.fastmcp.exceptions import ToolError
    except ImportError:
        from mcp.server.mcpserver import MCPServer as server_class
        from mcp.server.mcpserver.exceptions import ToolError
    return server_class, ToolError


def build_server(episode: Episode):
    """Register the world's tools for one episode on an MCP server."""
    from pydantic import Field

    server_class, ToolError = _import_sdk()
    # log_level WARNING: injected failures are expected and are not logged.
    server = server_class(
        SERVER_NAME,
        instructions="Tools for one task. Call finish when the task is done.",
        log_level="WARNING")
    Key = Annotated[str, Field(description=KEY_DESCRIPTION)]

    def run(tool: str, args: dict[str, str], idempotency_key: str) -> str:
        text, is_error = episode.call(tool, args, idempotency_key)
        if is_error:
            raise ToolError(text)
        return text

    # One plain function per effectful tool; parameters are the canonical
    # fields from world.EFFECTFUL_TOOLS (checked by run_tests.py).
    def send_email(to: str, subject: str, idempotency_key: Key = "") -> str:
        return run("send_email", {"to": to, "subject": subject},
                   idempotency_key)

    def send_payment(payee: str, amount: str, memo: str,
                     idempotency_key: Key = "") -> str:
        return run("send_payment",
                   {"payee": payee, "amount": amount, "memo": memo},
                   idempotency_key)

    def create_ticket(queue: str, title: str,
                      idempotency_key: Key = "") -> str:
        return run("create_ticket", {"queue": queue, "title": title},
                   idempotency_key)

    def append_file(path: str, content_hash: str,
                    idempotency_key: Key = "") -> str:
        return run("append_file", {"path": path, "content_hash": content_hash},
                   idempotency_key)

    def create_event(calendar_id: str, title: str, start: str,
                     idempotency_key: Key = "") -> str:
        return run("create_event",
                   {"calendar_id": calendar_id, "title": title, "start": start},
                   idempotency_key)

    for fn in (send_email, send_payment, create_ticket, append_file,
               create_event):
        assert fn.__name__ in EFFECTFUL_TOOLS
        server.add_tool(fn, name=fn.__name__,
                        description=f"Execute {fn.__name__}. Has a real side effect.")

    def list_effects() -> str:
        return run("list_effects", {}, "")

    server.add_tool(list_effects, name="list_effects",
                    description="Read-only: list side effects already "
                                "recorded for this task.")

    def finish(summary: Annotated[str, Field(
            description="Optional short note on the outcome.")] = "") -> str:
        return episode.finish(summary)

    server.add_tool(finish, name="finish",
                    description="Call once when the task is complete.")
    return server


def serve(episode: Episode) -> None:
    """Run the stdio server until the client disconnects or the process is
    terminated, then write the result file (also written on every finish)."""
    server = build_server(episode)

    def write_and_exit(signum: int, frame: Any) -> None:
        # os._exit: the stdio reader thread blocks on stdin, so a normal
        # interpreter shutdown would wait for it indefinitely.
        episode.write_result()
        os._exit(0)

    signal.signal(signal.SIGTERM, write_and_exit)
    try:
        server.run("stdio")
    finally:
        episode.write_result()
