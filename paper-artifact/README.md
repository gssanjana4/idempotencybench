# Paper artifact: code, data and analysis

This folder holds the code as it was run, every result file and per-episode transcript,
and the scripts that compute the paper's numbers. It was built for anonymous review, so
project and path names inside the code are neutral (the environment variables start
with `BENCH_`, for example). Four files of the review archive are left out of this
public copy; see "Not in this copy" below.

## What this is

A measurement study, not a benchmark. A deterministic tool world (email, payments,
tickets, files, calendar) keeps an effect ledger the agent cannot see. Failures fire
either before the action commits (nothing is written; a retry is correct) or after it
has committed (timeout, HTTP 500, and wording variants of both). The study measures
whether a tool-using LLM agent repeats a side effect it has already caused (IVR, the
idempotency-violation rate). A second substrate interrupts and resumes a session over a
dependency graph of actions.

- `src/`: tool world, tasks, scripted agents, real-model loop, provider
  adapters (Anthropic SDK, OpenRouter), sweep runners, and `tier1.py` (the
  pre-registered follow-up experiments, one config per experiment).
- `src2/`: the session-interruption / resume substrate.
- `src_real/`: the Stripe test-mode check (R1). Stripe object ids and API request
  ids in `results/` are replaced by keyed hashes (same prefix and length, body
  starting with `anon`; one id maps to one hash, so every count and join is unchanged).
- `run_*.sh`: the launchers used for the real-model runs (resumable, cost-capped).
  Time stamps in their logs are in UTC.
- `configs/`: one JSON per follow-up experiment (models, cells, tasks, samples).
- `analysis/`: the scripts that compute every test and estimate.
  `run_all_v3.sh` writes `results/analysis_v3/numbers.json`.
  Pre-registration tags in the keys of `numbers.json`: `.P.` = registered
  primary test (uncorrected), `.S.` = registered secondary test (Holm within
  the experiment), `.X.` = exploratory, `sens.` = sensitivity analysis added
  after the results (deviation 8; not confirmatory). The main-grid `bare_hint`
  contrasts in `bare_hint_tests.csv` are post hoc (not registered). The registration's
  commit times are in `paper-merged/provenance.json`.
- `paper-merged/make_numbers.py`: reads the raw results and the analysis
  output and writes `paper-merged/numbers.tex`. `provenance.json` holds
  the facts the script normally reads from version control (when the
  pre-registration, R1, R2, R3 and R4 were committed; the SDK version used), frozen
  at build time.
- `analysis_v27/v27_numbers.py`, `analysis_v27/v28_numbers.py`,
  `analysis_v27/v29_numbers.py` and `analysis_v27/v31_numbers.py`: the later re-analysis
  (template-level tests, simulated minimum detectable effects, the key-honouring rescore of
  retries and, in `v31_numbers.py`, of restarts, which replays every restart transcript
  through the harness's world, the Stripe cell tests with Holm correction, the abstract's
  ranges, the strict-matcher cells of the registered runs and the registered verdicts by
  kind). They read `results/` and write `paper-v27/numbers_v27.tex`,
  `paper-v28/numbers_v28.tex`, `paper-v29/numbers_v29.tex` and `paper-v31/numbers_v31.tex`.
  `paper-v28/make_figures_v28.py` draws the results figure from `results/analysis_v3/`.
- `v32-experiments/`: R5, the registered experiment with error texts from API documentation.
  `prereg_r5.md` is its registration (committed and pushed to the private repository before the
  first R5 episode; `provenance/r5_registration.txt` gives that commit's hash and UTC time and the
  sha256 of each registered R5 file at it, which the shipped copies match). `error_texts/catalogue.json`
  holds the 30 verbatim texts with URL, access date, class and flags, and the selection rule;
  `classification_check/` the blind second coding (an AI model session given only the texts and the
  rule) and the check of the eight injected texts against their sources; `configs/r5_real_errors.json`
  the cells; `src/` the runner (`run_r5.py`, which registers the texts as failure modes of the
  unchanged `src/tier1.py` harness), the cumulative spend guard, the estimate, the power simulation and
  the registered analysis (`analysis_r5.py`); `tests/test_r5.py` the offline tests; `run_r5.sh` and
  `LAUNCH.md` the launcher and its instructions. `results/r5_real_errors/` holds `raw.jsonl` (2,560
  episodes), one transcript per episode, `run.log`, `summary.csv` and the registered analysis output
  (`analysis/`); `results/spend_ledger.jsonl` is the spend guard's ledger. In this copy only local paths
  in `run.log` and the example repository path and the name of the machine in `LAUNCH.md` and
  `run_r5.sh` differ from the run (the registered files are unchanged); `appendix_e_r6_optional.md`, a sketch of
  an experiment that was not run, is not included.
- `analysis_v27/v32_numbers.py` recomputes every R5 number the paper prints from
  `v32-experiments/results/` without the registered analysis script, compares each value with
  `analysis/r5_real_errors.json` (it exits non-zero on a mismatch; differences below 1e-4 come from that
  script rounding before averaging) and writes `paper-v32/numbers_v32.tex`.
- Every result number the paper prints comes from one of these six files
  (`numbers.tex`, `numbers_v27.tex`, `numbers_v28.tex`, `numbers_v29.tex`, `numbers_v31.tex`, `numbers_v32.tex`), which the
  scripts above write from `results/` and `v32-experiments/results/`; the exceptions are provenance values (registration
  commit times, the SDK version), taken from `provenance.json`, and numbers quoted from
  other papers, which are kept with their sources in `paper-merged/lit_numbers.tex`.
- `results/`: raw per-episode rows (`*raw*.jsonl`), the per-episode
  transcripts, run logs, summaries and analysis outputs, so scoring can be
  audited rather than trusted. Every transcript is included except the
  144 first-run transcripts of episodes that an overlapping second launch
  re-ran and overwrote; both runs' rows are kept (paper, appendix "Episodes Run Twice").

- `controls/`: the content-free note control for restarts (QX1): its plan, pushed to this
  repository before the first episode, the launch files, the launcher and the analysis
  script (`qx1_analysis.py`). Results are in `controls/results_qx1_note_control/` (the run's
  `out_dir`, kept there so that the file counts of `results/` stay as printed in the paper):
  raw rows, one transcript per episode, the run log and the analysis output. The run covered
  Haiku 4.5 (128 episodes). Fable 5 stopped after 42 of its 128 episodes when the API
  account ran out of credit, and the three OpenRouter models were not launched. It ran on
  the harness as used for R2 to R5, whose environment variables start with `IBENCH_`.
- `paper-v34/`: `make_figures_v34.py` draws the results figure of the current version of
  the paper; `lit_numbers_v34.tex` holds the numbers quoted from other papers.

## Not in this copy

The registration file of the experiments before R5 (`analysis/preregistration.md`), its two
section drafts and the private repository's commit log (`provenance/registration_log.txt`)
are not in this public copy. They are available from the author on request. R5's
registration (`v32-experiments/prereg_r5.md`) and the plan of the control
(`controls/QX1_PLAN_content-free-note.md`) are here.

Some scripts read registration dates and registered constants from that file and stop
without it: `paper-merged/make_numbers.py`, `tests_mutants3.py`, and one test in each of
`tests_mutants2.py`, `tests_mutants5.py`, `tests_mutants6.py` and `tests_r4.py`. Everything
else runs, including `tests_independent.py` (`independent/check.py`), which recomputes the
printed numeric values from the raw data, and the `analysis_v27/` scripts. The number files
in this folder are the ones the full run wrote.

About 5,200 transcript file names contain a colon (`openrouter:...`). Windows file systems
do not accept them, so clone on Linux, macOS or WSL.

## Run the tests (Python >= 3.9, no network, no API key)

Inside this repository, prefix the commands with `GIT_DIR=/dev/null`, for example
`GIT_DIR=/dev/null python3 tests_independent.py`. The review archive had no git history,
and two scripts would otherwise read this repository's commit dates as the registration
dates. See "Not in this copy" for the checks that need the registration file.

```
python3 run_tests.py          # tool world, tasks, scripted agents, sweep
python3 run_tests2.py         # resume substrate
python3 tests_providers.py    # provider adapters (needs: pip install anthropic)
python3 tests_tier1.py        # follow-up runner, configs, spend caps (fake models)
python3 tests_analysis.py     # analysis layer: cell selection, dedupe, tests, printed cells
python3 tests_mutants.py      # mutation-killing tests: statistics, numbers, scorer edge cases
python3 tests_mutants2.py     # mutation-killing tests: Stripe check, payment-only comparison, R1 macros
python3 tests_mutants3.py     # mutation-killing tests: permutation protocol, loaders, scorer, macros, registration constants
python3 tests_mutants4.py     # mutation-killing tests: appendix macros, loaders, Stripe client guards, providers
python3 tests_mutants5.py     # mutation-killing tests: formatting, belief coder, C1 rescore, design macros
python3 tests_mutants6.py     # random-mutant round: emit_* invariants, descriptive fields, helpers
python3 tests_independent.py  # independent recomputation of every numeric printed macro from the raw data
python3 tests_r2.py; python3 tests_r2_extra.py; python3 tests_r3.py; python3 tests_r4.py; python3 tests_real.py
python3 v32-experiments/tests/test_r5.py   # R5: catalogue, selection rule, injection, spend guard, analysis (fake providers)
```

## Regenerate the numbers (standard library only, about 20 s)

```
bash analysis/run_all_v3.sh          # -> results/analysis_v3/numbers.json
python3 paper-merged/make_numbers.py # -> paper-merged/numbers.tex, numbers.json
```

`paper-merged/numbers.tex` ships as the paper reads it (its two launch-time values in
UTC, as in the logs); the second command regenerates it byte for byte. The later
re-analysis needs `numpy` (and `matplotlib` for the figure):

```
python3 analysis_v27/v27_numbers.py      # -> paper-v27/numbers_v27.tex (about 5 s)
python3 analysis_v27/v28_numbers.py      # -> paper-v28/numbers_v28.tex
python3 analysis_v27/v29_numbers.py      # -> paper-v29/numbers_v29.tex
python3 analysis_v27/v31_numbers.py      # -> paper-v31/numbers_v31.tex
python3 analysis_v27/v32_numbers.py      # -> paper-v32/numbers_v32.tex (R5; standard library only)
python3 paper-v28/make_figures_v28.py    # -> paper-v28/fig_results_v28.pdf
```

All five number files ship as printed in the paper and regenerate byte for byte.

## Rerun real models (API keys, costs money)

```
pip install anthropic
export ANTHROPIC_API_KEY=...  OPENROUTER_API_KEY=...
python3 src/tier1.py --config configs/e1_2_at_least_once.json --dry-run   # no calls
BENCH_BUDGET_USD=20 python3 src/tier1.py --config configs/e1_2_at_least_once.json --launch
```

Runs are resumable (completed episodes are skipped), write only to their own
`out_dir`, and stop at the spend cap. Model sampling is not seedable, so a rerun
reproduces the distribution, not the exact transcripts.

## License

Our code (MIT, `LICENSE`), tasks and ledgers (CC BY 4.0). Model outputs (the
transcripts) are released for research use under the providers' terms.
