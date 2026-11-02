# Contributing

Issues and pull requests are welcome.

## Good first contributions

- **New failure scenarios.** For example, a stale read-back: the effect committed, but
  the agent's check-before-retry call does not show it yet.
- **Results for other models.** Run the sweep and open a PR that adds a leaderboard row,
  with the summary CSV and the per-episode transcripts.
- **Framework adapters.** The MCP server (`docs/mcp.md`) works with any MCP-capable
  agent; adapters for other agent frameworks are welcome.

## Before you open a PR

1. `python3 run_tests.py` passes.
2. A new scenario comes with a scripted test that shows its expected IVR.
3. Core code (`src/idempotencybench/`) stays standard-library only.

Questions or ideas: open an issue.
