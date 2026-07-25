# IdempotencyBench

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
python3 run_tests.py        # 31 tests
python3 src/harness.py      # full 320-task scripted grid -> results/summary.csv
```

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

## Substrate map (scripted harness validation; model results TBD)

| | transport retry | agent retry | whole-run (on failure) | at-least-once |
|---|---|---|---|---|
| prompt-only ("check first") | ✗ fails | ✓ works | ✓ works | ✗ fails |
| idempotency keys | ✓ (if keys stable) | ✓ (if keys stable) | ✓ (if keys stable) | ✓ (if keys stable) |
| runtime receipts | ✓ | ✓ | ✓ | ✓ |

Prompt-level mitigations only help where the *model sees the failure*.
Exactly-once behavior is a property of the execution substrate, not the model.

## Leaderboard

| Model | IVR ↓ (agent retry) | IVR ↓ (whole-run) | Success ↑ |
|---|---|---|---|
| *results pending — PRs welcome* | | | |

## Citation

```bibtex
@misc{idempotencybench2026,
  title={Do LLM Agents Act Exactly Once? Measuring Idempotency Violations Under Retries},
  author={[Author]},
  year={2026},
  eprint={[arXiv ID TBD]},
  archivePrefix={arXiv}
}
```

## License

MIT (code) / CC-BY 4.0 (tasks)


## Author

Sanjana Gopnal Swamy — [github.com/gssanjana4](https://github.com/gssanjana4)

## Citation

```bibtex
@misc{gopnalswamy2026idempotencybench,
  title  = {Do LLM Agents Act Exactly Once? Measuring Idempotency Violations Under Retries},
  author = {Gopnal Swamy, Sanjana},
  year   = {2026},
  note   = {Under review at ACL Rolling Review; code and transcripts at github.com/gssanjana4/idempotencybench}
}
```

## License

Code: MIT. Task data and transcripts: CC BY 4.0.
