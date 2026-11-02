# IdempotencyBench

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23813526.svg)](https://doi.org/10.5281/zenodo.23813526)

**Do LLM agents act exactly once?** IdempotencyBench measures how often
tool-using LLM agents cause *duplicate side effects* when infrastructure
retries — timeouts, transport re-issues, orchestrator re-runs.

📄 Paper: [PDF in this repo](paper.pdf) (under review, ARR August 2026; arXiv link coming) · 📊 Metric: **IVR** (Idempotency Violation Rate)

## Why

Success-rate metrics (e.g. τ-bench's pass^k) can't see duplication: a run
that sends the same payment twice still "succeeds". IVR measures
over-execution — the failure mode that matters once agents touch payments,
email, and tickets.

## Quickstart (no API key, no dependencies)

```
python3 run_tests.py        # 43 tests (2 skipped without the mcp package)
python3 src/harness.py      # full 320-task scripted grid -> results/summary.csv
```

As a package (Python 3.10+):

```
pip install .
idempotencybench scripted           # scripted grid, IVR per failure mode
pip install '.[mcp]'
idempotencybench serve-mcp --task <id> --failure timeout_after_commit --out result.json
```

`serve-mcp` evaluates any MCP-capable agent; see [docs/mcp.md](docs/mcp.md).

Real-model pilot (Claude Fable 5, cost-capped):

```
pip install anthropic
export ANTHROPIC_API_KEY=sk-...
python3 src/harness.py --fable5
```

## Design

- **8 task templates, 4 structural families** (5 atomic domains — email,
  payments, tickets, files, calendar — plus multi-step, read-then-act, and
  ambiguous-goal), 320 deterministic tasks (40 parameterizations per
  template), ground-truth effect ledger invisible to the agent.
- **Failure injection**: the crux is *timeout-after-commit* — "no response"
  looks identical whether the action happened or not.
- **5 retry modes**: none / transport-level / agent-level /
  failure-conditioned whole-run / at-least-once redelivery.
- **3 mitigations**: none / idempotency keys / runtime receipts.
- **Violation = semantic duplicate** (same canonical action for the same
  task, args compared after formatting normalization), not key equality —
  agents that mint fresh keys are scored honestly.
- **Per-episode transcripts** for all real-model runs, so scoring is
  auditable.

## Substrate map (scripted harness validation)

| | transport retry | agent retry | whole-run (on failure) | at-least-once |
|---|---|---|---|---|
| prompt-only ("check first") | ✗ fails | ✓ works | ✓ works | ✗ fails |
| idempotency keys | ✓ (if keys stable) | ✓ (if keys stable) | ✓ (if keys stable) | ✓ (if keys stable) |
| runtime receipts | ✓ | ✓ | ✓ | ✓ |

Prompt-level mitigations only help where the *model sees the failure*.
Exactly-once behavior is a property of the execution substrate, not the model.

## Leaderboard

IVR with 95% Wilson intervals, base prompt, 64 episodes per cell (32 tasks ×
2 samples, `results/sweep_summary.csv`). The first two columns use no
mitigation. Task success is 1.00 in every cell. Haiku 4.5 ran at temperature 0,
so its two samples per task are often identical and its intervals are too
narrow; Sonnet 5 and Fable 5 ran at the API default temperature.

| Model | IVR ↓ (HTTP 500 after commit) | IVR ↓ (timeout after commit) | IVR ↓ (timeout after commit, runtime receipts) |
|---|---|---|---|
| Claude Haiku 4.5 | 0.69 [0.57, 0.79] | 0.94 [0.85, 0.98] | 0.00 [0.00, 0.06] |
| Claude Sonnet 5 | 0.64 [0.52, 0.75] | 0.02 [0.00, 0.08] | 0.00 [0.00, 0.06] |
| Claude Fable 5 | 0.58 [0.46, 0.69] | 0.14 [0.08, 0.25] | 0.00 [0.00, 0.06] |

PRs adding other models, scenarios or framework adapters are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Author

Sanjana Gopnal Swamy — [github.com/gssanjana4](https://github.com/gssanjana4)

## Citation

```bibtex
@misc{gopnalswamy2026idempotencybench,
  title  = {Do LLM Agents Act Exactly Once? Measuring Idempotency Violations Under Retries},
  author = {Gopnal Swamy, Sanjana},
  year   = {2026},
  doi    = {10.5281/zenodo.23813526},
  url    = {https://doi.org/10.5281/zenodo.23813526}
}
```

## License

Code: MIT. Task data and transcripts: CC BY 4.0.
