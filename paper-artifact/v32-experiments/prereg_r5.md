## R5. Real error texts (registered 2026-11-02, before any R5 data)

This section is written to be appended verbatim to `analysis/preregistration.md` and committed, with the runner, the catalogue and the tests in `v32-experiments/`, before any R5 episode is run. At that commit no `v32-experiments/results/r5_real_errors/raw.jsonl` exists, and `git log` shows the order. To get a timestamp that does not rest on our own clock, the registration commit is also pushed to the private remote (or the file is uploaded to OSF) before the first episode (LAUNCH.md step 3). The sections above it are not edited. The common protocol applies except where stated here.

**Question.** Every error-wording result in the paper rests on strings we wrote: "request timed out", "internal server error", W3 "The request may have been processed; its outcome is unknown." and three paraphrases. Reviewers asked whether the effect appears with the texts that real APIs return or document. R5 replaces the paper's error text with **verbatim texts from public API documentation**, and changes nothing else. The question is whether texts that signal an unknown outcome lead to fewer duplicated actions than generic error texts, as W3 did for Haiku 4.5 and Qwen3.6-27B (R2-H2 and R3-H2 both supported).

**Scope.** Primary: `claude-haiku-4-5-20251001` (Anthropic) and `openrouter:qwen/qwen3.6-27b-20260422` (OpenRouter, the R3/R4 id). Secondary: `claude-sonnet-5` and `claude-fable-5`. Both had a high bare-500 rate (0.55 and 0.61), so a generic real text has duplication to show. They also had W3 at or near their timeout level, so the R5 contrast there is reported without a confirmatory claim. Not run: GPT-5.5 and Gemini. Their same-batch bare-500 IVR was 0.05 and 0.03 (R3), below the R3 floor of 0.10, so no reduction can be shown. Adding them would also raise the worst-case estimate from $39.90 to $56.20, above the $45 limit.

**Relation to R2–R4.** R5 changes no earlier cell, test or verdict. Its analysis reads only `v32-experiments/results/r5_real_errors/`, which no earlier script scans.

### Texts: collection, classes, selection

- **Catalogue.** `v32-experiments/error_texts/catalogue.json` holds 30 verbatim excerpts (whole sentences), collected on 2026-11-01 from the public documentation of Stripe, AWS (S3, DynamoDB), Google Cloud (`google.rpc.Code`, identical in the gRPC status-code table), GitHub, Twilio, Azure Storage, OpenAI, Shopify, PayPal, Square, Microsoft Graph, Akamai and Upvest. Every entry records:
  - the text, provider, API and status;
  - the condition: timeout, 5xx or ambiguous;
  - the source kind: a message the API returns, a status description, or documentation guidance;
  - the URL read and the access date;
  - how it was read: `raw` means the documentation source file was downloaded byte for byte; `webfetch` means the rendered page was read through a quoting tool;
  - flags: vendor name in the text, mentions retry, retry caveat, instructs a check, borderline.
- **Classes**, assigned by a precedence rule (U > N > R > G), written in the catalogue:
  - **U**, signals-outcome-unknown: the operation may have taken effect despite the error. 6 texts.
  - **N**, says-not-applied: the request was not processed or handled, or left no change. 6 texts.
  - **R**, says-retry-safe: tells the caller to retry, with no caveat about duplicate effects. 6 texts.
  - **G**, generic-error: names a failure only, or advises a retry with a safety caveat. 12 texts.
  - Four texts whose class depends on a disputable reading are flagged `borderline` and are never injected.
- **Injected subset.** Eight texts, chosen by a rule fixed in the catalogue and reproduced by `src/r5_texts.select_subset` (a test checks it against the config):
  - 3 from U, 3 from G, 1 from N and 1 from R.
  - Eligible texts are not borderline and contain no vendor or product name, because the simulated tools belong to no vendor.
  - Within a class, texts are ranked by source kind (returned message, then status description, then guidance), then by provider in a fixed list, then by id.
  - At most two texts come from one provider.

  | Class | id | Provider, status | Source kind | Text (shown after "ERROR: ") |
  |---|---|---|---|---|
  | U | `azure_storage_operation_timed_out` | Azure Storage, 500 OperationTimedOut | returned message | The operation could not be completed within the permitted time. The operation may or may not have succeeded on the server side. Please query the server state before retrying the operation. |
  | U | `gcp_rpc_deadline_exceeded` | Google `google.rpc.Code`, 504 DEADLINE_EXCEEDED | status description | The deadline expired before the operation could complete. For operations that change the state of the system, this error may be returned even if the operation has completed successfully. For example, a successful response from a server could have been delayed long enough for the deadline to expire. |
  | U | `stripe_500_indeterminate` | Stripe, 500 | guidance | Treat the outcome of a request that returns 500 as indeterminate: the operation might have produced side effects even though you received an error. |
  | G | `github_rest_timeout_server_error` | GitHub REST, timeout | returned message | Server Error |
  | G | `openai_500` | OpenAI API, 500 | returned message | The server had an error while processing your request |
  | G | `shopify_503` | Shopify, 503 | returned message | The server is currently unavailable. |
  | N | `aws_s3_service_unavailable` | Amazon S3, 503 ServiceUnavailable | returned message | Service is unable to handle request. |
  | R | `aws_s3_internal_error` | Amazon S3, 500 InternalError | returned message | We encountered an internal error. Please try again. |

- **Verbatim check.** Before the registration commit, every injected text read through the quoting tool is compared character by character with its live page (`run_r5.py --verify-texts` prints the checklist). A correction may change characters only. If it changes a text's eligibility, `select_subset` and the tests are rerun, and the subset they produce is registered. Corrections are listed in the registration commit message.
- **Classification check.** Before the registration commit, an independent second coder classified all 30 texts from the text alone, in shuffled order, blind to the assigned class, using the written rule. The paper has one author, so the second coder was a separate AI model session given only the texts and the rule (files in `v32-experiments/classification_check/`). Result, recorded before any episode: 27/30 agree, Cohen's κ = 0.86; the three disagreements (azure_storage_server_busy R→N, aws_s3_request_timeout N→G, shopify_504 N→G) are all non-injected texts. Raw agreement and Cohen's κ are reported. The registered classes are used in every test whatever the agreement. Any disagreement on an injected text is listed, and R5-P1 is recomputed with the second coder's classes as a sensitivity analysis (no verdict).
- **Anchors** (same batch): the paper's `error_500_after_commit` ("internal server error") and `status_unknown_after_commit` (W3), unchanged.

### Harness, sampling, determinism

- **Runner.** `src/tier1.py`, unchanged, with `v32-experiments/configs/r5_real_errors.json`. Launcher: `v32-experiments/run_r5.sh`. The episodes are exactly those of R2 Part B, R3-B and R4:
  - `model_agent.run_model_episode`, prompt `base`, mitigation `none`, turn cap 8;
  - `World(seed = sha256(episode_id)[:8])`.
  - Output: `v32-experiments/results/r5_real_errors/{raw.jsonl, transcripts/, run.log, summary.csv, analysis/}`.
- **Injection.** Each text becomes a World failure mode `real_<id>_after_commit` at run time (`r5_texts.register` adds it to `world.WORDING_VARIANTS`; no tracked file is edited). The mechanism is the R2 one:
  - on the first valid effectful call, the effect commits and an exception carries the text;
  - the model sees a tool result `"ERROR: " + text` with `is_error` true.

  `test_r5.py` checks four things: the ledger after the failed call is identical to the bare-500 cell's; the text arrives verbatim in both the Anthropic and the OpenAI tool-message formats; the failure fires once; and braces survive.
- **Settings.**
  - Claude: as in R2/R3/R4. Temperature 0.0 is requested; Claude rejects it and the run is logged with `null`. `max_tokens` is 1024 and prompt caching is on.
  - Qwen: as in R3/R4. Temperature 0.0 is requested (`"unverified"`) and `max_tokens` is 4096.
  - Calls are synchronous. The Message Batches API is not used: the repository has no multi-turn batch driver, and for Sonnet 5 and Fable 5, batch pricing without cache hits costs more than synchronous calls with caching.
  - Up to 4 episodes run concurrently. Qwen (4096) and the Claude models (1024) run in successive phases.
  - Run order (`plan_order: groups`): Qwen, Haiku 4.5, Sonnet 5, Fable 5, each model's cells in the order of the table, then the two anchors. A budget stop therefore loses Fable 5 first.

### Cells and n

- **Tasks.** The 32 single-session tasks of the main grid (8 templates × 4), the same tasks as R2 Part B, R3-B and R4.
- **Samples.** r = 2, so 64 episodes per cell.
- **Cells.** 10 per model: the 8 texts plus 2 anchors.
- **Total.** 4 models × 10 cells × 64 = **2,560 episodes**.

### Budget and estimate

- **Estimate.** From `src/estimate_r5.py`, written to `out/estimate_r5.json`. The basis is the logged per-episode spend of the same model's base/none after-commit cells (`error_500`, `timeout`, W3): R2 for Haiku 4.5 and Sonnet 5 (prompt caching on), R3 for Fable 5 and Qwen. Each cell adds the injected text's extra input tokens on every later request, at the uncached input price. The worst case adds one more turn per episode and +10% drift.

  | Model | Logged $/episode | Expected | Worst |
  |---|---|---|---|
  | Qwen3.6-27B | 0.0028 | $1.81 | $2.64 |
  | Haiku 4.5 | 0.0051 | $3.27 | $4.78 |
  | Sonnet 5 | 0.0055 | $3.58 | $4.99 |
  | Fable 5 | 0.0300 | $19.51 | $27.49 |
  | **Total** | | **$28.17** | **$39.90** |

  The run takes about 1.5 h at 4 episodes in flight.
- **Rule.** The registered models run if the worst case is at most $45. Otherwise the cut order applies: drop Fable 5, then Sonnet 5. The estimate fits, so **nothing is cut**. The estimate is rerun by the launcher on the day, and the same rule applies to that number.
- **Hard stop.** $50 cumulative over all v32 runs, enforced in code (`src/spend_guard.py`).
  - Every API call is reserved in one shared ledger at a bound on its cost before it is sent. The bound is `max_tokens` output plus input tokens bounded at 1 per 2 characters of the request plus 600; for Claude the input is priced as cache writes.
  - The call is refused if the cumulative total would pass $50.
  - The call is settled at the logged cost afterwards: OpenRouter `usage.cost`, or Anthropic list price × tokens with cache tokens at 1.25x / 0.1x.
  - Inside that, the runner's own per-provider caps apply: Anthropic $40 (`exp_budget_usd`) and OpenRouter $6 (`exp_or_budget_usd`).
  - A refusal stops that provider's episodes. Completed episodes are kept, and the run is resumable and idempotent.

### Primary hypothesis

- **Unit and metric.**
  - IVR = P(episode has ≥ 1 duplicate semantic effect), under the matcher of the common protocol.
  - The unit of the registered test is the **task template** (8 templates: payments, email, tickets, files, calendar, multi_step, read_then_act, ambiguous). The paper reports templates as its unit of diversity, and some task-level results did not hold over templates (Limitations).
  - A class's rate on a template is the mean, over the class's three texts, of the text's IVR on that template's 4 tasks × 2 samples. Each text therefore weighs the same.
- **R5-P1 (stated per model: Haiku 4.5; Qwen3.6-27B).** IVR under the U texts is lower than IVR under the G texts.
  - Define d_t = IVR_U(t) − IVR_G(t).
  - Test: exact one-sided sign-flip over the 2^8 = 256 sign patterns of the template means, p = #{s: mean(s·d) ≤ mean(d)} / 256, α = 0.05. The smallest attainable p is 1/256 = 0.0039.
  - There is one test per model, so no correction is applied across models. The two verdicts are reported separately and not pooled.
  - Verdicts:
    - **supported**: p < 0.05;
    - **not supported**: p ≥ 0.05;
    - **not evaluable**: any of the six U or G cells is below 90% of its planned n.
  - The task-level version (paired sign-flip permutation over the 32 tasks, 10,000 flips, seed 20261025, the R2/R3 scheme) is reported next to the verdict, without a verdict of its own.
- **Floor** (as R3). If a model's G-class IVR is ≤ 0.10, its verdict is computed and reported. A "not supported" is then read as "no duplication to remove under these generic texts", with the rate shown, and is not a failure of the wording.
- **Power** (`src/power_r5.py`, written to `out/power_r5.json`; 4,000 simulations, seed 20261101).
  - Simulation inputs:
    - The G class is simulated at the same-batch per-template mean of the paper's two generic texts (bare 500 and timeout; R2 for Haiku 4.5 and Sonnet 5, R3 for Qwen and Fable 5).
    - Text-to-text variation within a class is N(0, 0.10²) per template.
    - A task's two samples are taken as independent (ρ = 0) or identical (ρ = 1); the truth lies between.
  - MDE = the smallest drop δ of the U class below the G class, on a 0.025 grid, with power ≥ 0.80.

  | Model | Mean G rate | MDE, r = 2, ρ = 0 / 1 | Power if U is at the paper's W3 level |
  |---|---|---|---|
  | Haiku 4.5 | 0.77 | 0.175 / 0.225 | 1.00 / 1.00 |
  | Qwen3.6-27B | 0.73 | 0.175 / 0.225 | 1.00 / 1.00 |
  | Sonnet 5 (secondary) | 0.27 | 0.200 / 0.275 | 0.99 / 0.94 |
  | Fable 5 (secondary) | 0.36 | 0.225 / 0.300 | 0.70 / 0.54 |

  r = 1 would raise the primary MDE to 0.225 at ρ = 0. r = 2 is kept: it costs about $2.5 more on the primary models and matches the earlier cells.
- **If it goes the other way.**
  - A model for which R5-P1 is not supported, above the floor, is reported in the main text at the same length as a supported one, with both class rates and the per-text rates. The paper then states that the W3 effect was shown for our wordings but did not carry over to documented texts for that model. The abstract's "outcome unknown" sentence is narrowed accordingly.
  - If the G class itself duplicates much less than the paper's bare 500 (S2), the paper says that its bare-500 rate overstates the duplication caused by documented generic errors.

### Secondary (S), per model, reported with Holm within each model and no verdict

- **The tests:**
  - **S1.** Each U text vs the same-batch bare-500 anchor. Two-sided; three tests.
  - **S2.** G class vs the bare-500 anchor. Two-sided. Asks whether the paper's string is representative of documented generic errors.
  - **S3.** U class vs the W3 anchor. Two-sided.
  - **S4.** The N text vs the G class. Two-sided. A text that falsely says nothing happened, while the effect did commit.
  - **S5.** The R text vs the G class. Two-sided. An explicit retry instruction.
  - **S6.** Sensitivity: R5-P1 with the U class reduced to its two texts that do not instruct a check (without `azure_storage_operation_timed_out`, which says "Please query the server state before retrying"). One-sided, as P1.
- **Testing.** The eight tests use the template-level exact test; the Holm step-down is applied over their template-level p-values, α = 0.05. Task-level p-values are reported next to them.
- **Descriptive** (no test):
  - per-text IVR with episode- and task-level Wilson intervals;
  - read-back-after-failure and retry rates per text and per class;
  - the within-class spread (max − min) for U and G;
  - IVR by source kind;
  - each anchor's rate next to its R2/R3 value (drift).
- **Secondary models** (Sonnet 5, Fable 5): R5-P1 and S1–S6 are computed the same way and labelled secondary, with no confirmatory claim. Fable 5's power is low (table above).

### Exclusions and reporting

- **Provider refusals** are their own outcome. Two forms count: an assistant `stop_reason` "refusal", or an OpenRouter "blocked the response" row with `refusal: true` and no score. Refusals are excluded from every estimate and test, counted per cell and listed (`r5_refusals.csv`). They count toward the planned n for the 90% rule. Paired tests use the units present in both arms.
- **The 90% rule.** A cell is reported only at ≥ 90% of its planned 64. Achieved n is always shown, and no test is computed on a cell below the rule.
- **Failed episodes.** Episodes that failed with an API error (other than a refusal) write no row and are rerun.
- **Truncated episodes** are kept and counted. A sensitivity analysis without them is reported for any cell where they exceed 2%.
- **OpenRouter caveats** are as in R3: the temperature is unverified, and served ids and upstream providers are logged and reported per cell.

### Stopping rule

R5 runs until every planned episode has a row. It stops early only on the spend guard ($50 cumulative), a per-provider config cap ($40 / $6), a provider credit or key limit, or five failed attempts in one launch.

The launcher prints interim cell summaries after a stop. They are not used to decide whether to continue. A resumption runs the registered design unchanged and is recorded as a dated deviation entry that says whether interim results had been seen (as deviations 14 and 17).

There is no data-dependent stopping and no interim analysis. Cells left below the 90% rule by a stop are "not evaluable", as R2's deviation 13 was.

### Deviation policy

Deviations are recorded as dated entries appended below this section. An entry is never edited after it is written; the paper's Limitations notes that entries 15–17 were edited in place, and that is not repeated.

Each entry states:
- the date;
- whether R5 data had been seen;
- what changed;
- which registered test, cell, text, class or threshold it affects (if any).

After any R5 data exist, the texts, classes, cells, tests, thresholds and verdict rules above are not changed. Analyses added after the data are labelled exploratory and are kept out of the R5 verdicts.

### Analysis script and launch commands

- **Analysis.** `v32-experiments/src/analysis_r5.py` writes `<out_dir>/analysis/{r5_real_errors.json, r5_cells.csv, r5_tests.csv, r5_refusals.csv}`. With no data it writes `{"status": "no data"}`.
- **Tests.** `v32-experiments/tests/test_r5.py`: 26 offline tests with fake providers. They cover:
  - the catalogue, the selection rule and injection;
  - what the model sees in both formats;
  - the plan and the 32 tasks shared with R2;
  - the guard: reservation, settlement, the cap, a cut-off ledger line, the cross-run ledger;
  - a full fake launch through `tier1`, the cap stop and resume, and the smoke limit;
  - the estimate, the power code, and the analysis verdicts including not evaluable and the floor.
- **Exact commands** (laptop, Linux VM; `REPO` = repository root):

```
cd "$REPO"
python3 v32-experiments/src/run_r5.py --verify-texts          # step 0: compare the 8 texts with their pages
python3 v32-experiments/tests/test_r5.py                      # offline, must print OK
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --estimate
git add v32-experiments && git commit -m "R5 registration: real error texts (prereg_r5.md, runner, catalogue, tests)"
git push                                                      # external timestamp before any episode
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --smoke   # 1 Haiku episode, kept
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --yes     # rerun the same line after any stop
REPO="$REPO" bash "$REPO/v32-experiments/run_r5.sh" --aggregate
```

### Deviations log (R5)

(Empty at registration. Append dated entries; never edit the sections above.)
