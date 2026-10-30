# Evaluating an MCP agent

`idempotencybench serve-mcp` serves one task's tool world as a stdio MCP
server. Any MCP client can connect to it, so an agent can be evaluated
without changing the agent's code.

```
pip install 'idempotencybench[mcp]'
idempotencybench tasks | head                   # task ids and instructions
idempotencybench tasks --id payments-74f7376314 # one task's instruction
```

## Tools

| Tool | Arguments | Effect |
|---|---|---|
| `send_email` | `to`, `subject` | side effect |
| `send_payment` | `payee`, `amount`, `memo` | side effect |
| `create_ticket` | `queue`, `title` | side effect |
| `append_file` | `path`, `content_hash` | side effect |
| `create_event` | `calendar_id`, `title`, `start` | side effect |
| `list_effects` | none | read-only: effects already recorded for this task |
| `finish` | `summary` (optional) | ends the episode and writes the result file |

Every effectful tool also takes an optional `idempotency_key`. The tool
surface is the same as the real-model harness (`model_agent.py`).

## Episode

1. Start the server for one task, one failure mode and one mitigation.
2. Give the agent the task instruction (`idempotencybench tasks --id <id>`)
   as its prompt.
3. The first effectful call fails as configured by `--failure`
   (`timeout_after_commit`: the effect is committed, the agent sees
   `request timed out`). Later calls succeed.
4. The agent calls `finish`, or the client disconnects. The result is
   written to `--out` in both cases (`finished` records which).

The agent sees only tool results. The ledger and the score are written only
to the `--out` file.

| Option | Values | Default |
|---|---|---|
| `--task` | a task id | required |
| `--failure` | `none`, `timeout_after_commit`, `timeout_before_commit`, `error_500_after_commit` | `timeout_after_commit` |
| `--mitigation` | `none`, `idem_keys`, `runtime_receipts` | `none` |
| `--out` | path of the result JSON | required |

## Client configuration

Generic stdio configuration (the `mcpServers` format used by Claude Code
`.mcp.json`, Claude Desktop, Cursor and others):

```json
{
  "mcpServers": {
    "idempotencybench": {
      "command": "idempotencybench",
      "args": [
        "serve-mcp",
        "--task", "payments-74f7376314",
        "--failure", "timeout_after_commit",
        "--mitigation", "none",
        "--out", "runs/payments-74f7376314.json"
      ]
    }
  }
}
```

If the console script is not on the client's `PATH`, use the interpreter
of the environment it was installed into:
`"command": "/path/to/venv/bin/python", "args": ["-m", "idempotencybench", "serve-mcp", ...]`.

Use one server process per episode: the world, including the armed failure,
lives for the lifetime of the process. To run a task set, loop over task ids
and start a fresh agent session and server for each.

## Result file

```json
{
 "task_id": "payments-74f7376314",
 "family": "atomic",
 "failure_mode": "timeout_after_commit",
 "mitigation": "none",
 "finished": true,
 "ivr": 1.0,
 "violations": 1,
 "success": true,
 "ledger": {"n_effects": 2, "by_tool": {"send_payment": 2}, "effects": ["..."]},
 "goals": ["..."],
 "tool_calls": ["..."]
}
```

`ivr` is 1.0 if the episode committed any semantic duplicate, else 0.0;
the IVR of a task set is the mean over its episodes. `success` is true when
every goal of the task was achieved. Scoring is the same as
`harness.run_grid`: duplicates are compared on canonical arguments after
formatting normalization, regardless of idempotency key.
