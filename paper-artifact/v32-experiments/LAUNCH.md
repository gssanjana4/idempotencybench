# Launching R5 (real error texts)

This runs on a Linux machine. Set `REPO` to the repository root, for example `REPO=~/benchmark`.

- **Size:** 2,560 episodes in about 1.5 h.
- **Cost:** expected **$28.17**, worst case **$39.90**.
- **Hard stop:** **$50 cumulative**, enforced in code.
- **Design:** see `v32-experiments/prereg_r5.md`.

Calls are synchronous with prompt caching on, as in R2/R3/R4. There is no Message Batches mode.

## 0. Once: prerequisites

```
REPO=~/benchmark          # <- your path
cd "$REPO"
python3 -c "import anthropic, numpy; print('ok')"   # anthropic SDK needed; numpy only for power_r5.py
```

- **Python:** if `python3` lacks the SDK, use a venv: `python3 -m venv .venv && .venv/bin/pip install anthropic numpy`. The launcher picks `.venv-claude`, `.venv-tier1`, `.venv` or `python3`, in that order, or whatever you set as `PY=...`.
- **API keys:** `.env` in `$REPO` must define `ANTHROPIC_API_KEY` and `OPENROUTER_API_KEY`. The launcher sources it and never prints it.
- **Backstops at the provider** (recommended, in addition to the code guard):
  - set a monthly spend limit of about $45 on the Anthropic workspace (Console → Limits);
  - set the OpenRouter key's own limit to about $8.

## 1. Check the eight texts against their pages (before registering)

```
python3 v32-experiments/src/run_r5.py --verify-texts
```

- Open each URL it prints and compare the text character by character: apostrophes, final period, spacing.
- Six of the eight were read through a quoting tool (`retrieval: webfetch`), so this step matters.
- If a character differs, fix it in `v32-experiments/error_texts/catalogue.json` and rerun step 2. A test fails if the selection rule then picks a different subset; in that case update `r5.injected` and the cells in the config to what `select_subset` gives.

## 2. Offline checks (no API call, no spend)

```
python3 v32-experiments/tests/test_r5.py                        # must end with OK (26 tests)
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --estimate   # tests + plan + estimate + spend so far
```

The estimate must say `fits, nothing cut`. If it exits with code 6, the worst case is above $45: apply the cut order in the prereg (drop Fable 5, then Sonnet 5) before going on.

## 3. Register (before any episode)

1. Fill in the date in the first line of `v32-experiments/prereg_r5.md`.
2. Do the blind classification check it describes.
3. Commit and push:

```
git add v32-experiments
git commit -m "R5 registration: real error texts (prereg_r5.md, runner, catalogue, tests)"
git push          # external timestamp; or upload prereg_r5.md to OSF
```

## 4. Smoke test (1 Haiku episode, about $0.005; it is kept and counts toward R5)

```
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --smoke
```

Exit 0 means the keys, the harness and the guard work. The ledger line shows the spend.

## 5. Run

```
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --yes
```

- Rerun the **same line** after any stop. Completed episodes are never rerun.
- `--yes --or-only` runs the Qwen cells only, with no Anthropic call.
- Progress and spend at any time: `bash "$REPO/v32-experiments/run_r5.sh" --status`.

| Exit | Meaning |
|---|---|
| 0 | All 2,560 episodes done; summary and analysis printed |
| 2 | Anthropic stopped: spend guard, $40 config cap or credit. Completed episodes are kept |
| 3 | OpenRouter stopped: spend guard, $6 config cap, key limit or credit |
| 4 | Pass finished with episodes left (API errors). Rerun the same line |
| 5 / 6 | Offline tests failed / estimate above $45. Nothing was spent |

## 6. Analysis

```
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --aggregate
```

Writes `v32-experiments/results/r5_real_errors/analysis/r5_real_errors.json`, with `r5_cells.csv` and `r5_tests.csv`. The R5-P1 verdicts for Haiku 4.5 and Qwen are printed.

## The $50 spend guard (how it works)

- **Ledger.** One shared ledger, `v32-experiments/results/spend_ledger.jsonl`, counts **all** v32 spend across every launch and process.
- **Reserve first.** Before each API call, the call's worst-case cost is reserved in the ledger: `max_tokens` output, plus input tokens bounded at 1 per 2 characters of the request plus 600, with Claude input priced as cache writes. The call is **refused if the cumulative total would pass $50**.
- **Settle after.** After the call, the reservation is settled at the logged cost: OpenRouter `usage.cost`, or Anthropic list price × tokens with cache tokens at 1.25x / 0.1x.
- **Crashes.** A reservation left open by a crash keeps counting at its worst case, at most about $0.06.
- **Stopping a provider.** A refusal stops that provider's episodes through the runner's existing stop paths. Before every attempt the launcher also aborts if less than $0.01 is left.
- **Inside $50,** the runner's own per-provider caps apply: Anthropic $40, OpenRouter $6.
- **Never edit or delete the ledger** to free budget. Starting a new ledger to spend more than $50 would be a deviation from the registration.

## Files

| Path | What |
|---|---|
| `v32-experiments/error_texts/catalogue.json` | 30 verbatim texts with URL, access date and class; selection rule |
| `v32-experiments/configs/r5_real_errors.json` | Cells, models, caps (tier1 config + `r5` block) |
| `v32-experiments/src/run_r5.py` | Runner: registers the texts, wraps providers with the guard, calls `src/tier1.py` |
| `v32-experiments/src/spend_guard.py` | Cumulative $50 guard (ledger, reserve/settle) |
| `v32-experiments/src/estimate_r5.py`, `out/estimate_r5.json` | Cost estimate from logged R2/R3 spend |
| `v32-experiments/src/power_r5.py`, `out/power_r5.json` | MDE / power by simulation |
| `v32-experiments/src/analysis_r5.py` | Registered analysis |
| `v32-experiments/tests/test_r5.py` | 26 offline tests |
| `v32-experiments/run_r5.sh` | Launcher |
| `v32-experiments/results/` | Created at run time: raw.jsonl, transcripts, run.log, summary, analysis, ledger |

No tracked file is modified. Results are written only under `v32-experiments/results/`, which `paper-*/make_numbers.py` does not scan.
