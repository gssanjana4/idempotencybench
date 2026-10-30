"""Command-line entry point: `idempotencybench <command>`.

  scripted    run the scripted (agent x retry x mitigation) grid under every
              failure mode and print IVR per failure mode
  tasks       list task ids and instructions (or print one instruction)
  serve-mcp   serve one task's tool world as a stdio MCP server
"""
from __future__ import annotations

import argparse
import sys

from .harness import MITIGATIONS, run_failure_sweep, write_csv
from .tasks import TEMPLATE_COUNT, generate_tasks
from .world import FAILURE_MODES

# Short column headers for the failure modes, in FAILURE_MODES order.
FAILURE_HEADERS = {
    "none": "none",
    "timeout_before_commit": "to_before",
    "timeout_after_commit": "to_after",
    "error_500_after_commit": "e500_after",
}


def format_ivr_table(rows: list[dict]) -> str:
    """One line per (agent, retry_mode, mitigation), one IVR column per
    failure mode. `rows` are run_failure_sweep() output."""
    failure_modes = [fm for fm in FAILURE_MODES
                     if any(r["failure_mode"] == fm for r in rows)]
    ivr: dict[tuple, float] = {}
    cells: list[tuple] = []
    for r in rows:
        cell = (r["agent"], r["retry_mode"], r["mitigation"])
        if cell not in cells:
            cells.append(cell)
        ivr[cell + (r["failure_mode"],)] = r["ivr"]

    lines = [f"{'agent':<12}{'retry_mode':<15}{'mitigation':<18}"
             + "".join(f"{FAILURE_HEADERS[fm]:>11}" for fm in failure_modes)]
    for cell in cells:
        lines.append(f"{cell[0]:<12}{cell[1]:<15}{cell[2]:<18}"
                     + "".join(f"{ivr[cell + (fm,)]:>11.2f}"
                               for fm in failure_modes))
    return "\n".join(lines)


def cmd_scripted(args: argparse.Namespace) -> int:
    tasks = generate_tasks(per_domain=args.per_template)
    rows = run_failure_sweep(tasks)
    print(f"IdempotencyBench scripted grid: {len(tasks)} tasks "
          f"({TEMPLATE_COUNT} templates x {args.per_template}), "
          f"IVR by injected failure mode")
    print(format_ivr_table(rows))
    print("\nIVR = fraction of tasks with at least one duplicated side effect.")
    print("Columns: " + ", ".join(f"{FAILURE_HEADERS[fm]} = {fm}"
                                  for fm in FAILURE_MODES))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"{len(rows)} rows (IVR, violation depth, success, omission) "
              f"-> {args.csv}")
    return 0


def cmd_tasks(args: argparse.Namespace) -> int:
    tasks = generate_tasks(per_domain=40)
    if args.id:
        task = next((t for t in tasks if t.task_id == args.id), None)
        if task is None:
            print(f"unknown task id: {args.id}", file=sys.stderr)
            return 2
        print(task.instruction)
        return 0
    for t in tasks:
        print(f"{t.task_id}\t{t.family}\t{t.instruction}")
    return 0


def cmd_serve_mcp(args: argparse.Namespace) -> int:
    from .mcp_server import Episode, find_task, serve

    task = find_task(args.task)
    if task is None:
        print(f"unknown task id: {args.task} "
              f"(see `idempotencybench tasks`)", file=sys.stderr)
        return 2
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("serve-mcp needs the MCP SDK: "
              "pip install 'idempotencybench[mcp]'", file=sys.stderr)
        return 2
    episode = Episode(task, failure_mode=args.failure,
                      mitigation=args.mitigation, out_path=args.out)
    serve(episode)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idempotencybench",
        description="Measure duplicate side effects of tool-using agents "
                    "under retries.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("scripted",
                       help="run the scripted grid and print IVR per failure mode")
    p.add_argument("--per-template", type=int, default=40,
                   help="tasks per template (default 40 -> 320 tasks)")
    p.add_argument("--csv", default=None,
                   help="also write the full metric rows to this CSV path")
    p.set_defaults(func=cmd_scripted)

    p = sub.add_parser("tasks", help="list the 320 task ids and instructions")
    p.add_argument("--id", default=None,
                   help="print only this task's instruction")
    p.set_defaults(func=cmd_tasks)

    p = sub.add_parser("serve-mcp",
                       help="serve one task's tools as a stdio MCP server")
    p.add_argument("--task", required=True,
                   help="task id (see `idempotencybench tasks`)")
    p.add_argument("--failure", choices=FAILURE_MODES,
                   default="timeout_after_commit",
                   help="failure injected on the first effectful call "
                        "(default: timeout_after_commit)")
    p.add_argument("--mitigation", choices=MITIGATIONS, default="none",
                   help="world-side mitigation (default: none)")
    p.add_argument("--out", required=True,
                   help="path for the episode result JSON, written on "
                        "finish and on exit")
    p.set_defaults(func=cmd_serve_mcp)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
