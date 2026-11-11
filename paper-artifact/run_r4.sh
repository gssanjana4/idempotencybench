#!/bin/bash
# R4: outcome unknown BEFORE commit (analysis/preregistration.md, section R4).
# One line from your own Terminal:
#
#   bash ./run_r4.sh --yes
#
#   bash run_r4.sh                 plan + cost estimate + caps, ask y/N, then run to completion
#   bash run_r4.sh --yes           same, no question (resumable: rerun the same line after any stop)
#   bash run_r4.sh --estimate      plan, estimate and caps only; no API call
#   bash run_r4.sh --status        episodes done / remaining and logged spend
#   bash run_r4.sh --yes --or-only OpenRouter cells only (Qwen; no Anthropic call or credit needed)
#   bash run_r4.sh --yes --workers N   N episodes in flight (default: the config's 4)
#
# Design  configs/r4_precommit.json: the failure fires on the same call as the
#         after-commit cells and NOTHING commits. Cells (prompt base, mitigation
#         none, the 32 main-grid tasks x r=2 = 64 episodes each):
#         timeout_before_commit, error_500_before_commit, status_unknown_before_commit,
#         for Qwen3.6-27B (OpenRouter), Claude Haiku 4.5 and Sonnet 5 (Anthropic).
#         576 episodes. Output: results/r4_precommit/.
# Money   Anthropic: hard stop at $8 of R4 spend (exp_budget_usd; list prices x
#         logged tokens, cache tokens at 1.25x / 0.1x). OpenRouter: hard stop at
#         $4 (exp_or_budget_usd; usage.cost as returned). Checked before every
#         episode. Prompt caching stays on for Anthropic.
#         A missing key, a failed key or no credit stops before any episode.
# - Resumable and idempotent: completed episodes are never run again (last row per id).
# - A provider refusal ("blocked the response") is a row with refusal=true.
# - .env is sourced, never printed.
set -uo pipefail
cd "$(dirname "$0")"

YES=0; MODE="run"; WORKERS=""; OR_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --estimate) MODE="estimate" ;;
    --status) MODE="status" ;;
    --or-only) OR_ONLY=1; export BENCH_CHECK_SKIP_ANTHROPIC=1 ;;   # OpenRouter cells only; no Anthropic call
    --workers) WORKERS="${2:-}"; shift ;;
    -h|--help) sed -n 2,27p "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 64 ;;
  esac
  shift
done
CFG="configs/r4_precommit.json"
[ -f "$CFG" ] || { echo "missing $CFG"; exit 64; }

PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in .venv-claude/bin/python .venv-tier1/bin/python .venv/bin/python python3; do
    if "$cand" -c "import anthropic" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with the anthropic SDK found (pip install anthropic, or set PY=...)"; exit 1; }
export PY
CAP_A=$($PY -c "import json;print(float(json.load(open('$CFG'))['exp_budget_usd']))")
CAP_OR=$($PY -c "import json;print(float(json.load(open('$CFG'))['exp_or_budget_usd']))")

OUT="results/r4_precommit"
if [ "$MODE" = run ]; then
  mkdir -p "$OUT"
  exec > >(tee -a "$OUT/run.log") 2>&1
fi
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ)  mode=$MODE or_only=$OR_ONLY  python=$PY"

set -a; [ -f .env ] && source .env; set +a
if [ "$MODE" != status ]; then
  if [ "$OR_ONLY" = 1 ]; then
    echo "ANTHROPIC_API_KEY: not needed (--or-only)"
  else
    [ -n "${ANTHROPIC_API_KEY:-}" ] && echo "ANTHROPIC_API_KEY: set (not shown)" || { echo "!! ANTHROPIC_API_KEY missing in .env - stopping"; exit 3; }
  fi
  [ -n "${OPENROUTER_API_KEY:-}" ] && echo "OPENROUTER_API_KEY: set (not shown)" || { echo "!! OPENROUTER_API_KEY missing in .env - stopping"; exit 3; }
fi
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &

export BENCH_BUDGET_USD=0 BENCH_OR_BUDGET_USD=0          # the config exp caps are the guard
export BENCH_TIER1_SKIP_PROVIDERS="" BENCH_TIER1_SKIP_MODELS=""
unset BENCH_EXP_BUDGET_USD BENCH_EXP_OR_BUDGET_USD        # the registered caps only ($8 / $4)
[ -n "$WORKERS" ] && export BENCH_WORKERS="$WORKERS"

if [ "$MODE" = status ]; then
  $PY src/tier1.py --status --config "$CFG"; exit $?
fi

echo "== plan (R4)"
$PY src/tier1.py --config "$CFG" --dry-run
echo "== cost and runtime estimate (src/r4_estimate.py: October logged tokens; registered rule for GPT-5.5 / Gemini)"
$PY src/r4_estimate.py --optional
echo "== caps: Anthropic \$$CAP_A, OpenRouter \$$CAP_OR (R4 spend, enforced before every episode); prompt caching ON (Anthropic)"
[ "$MODE" = estimate ] && exit 0

if [ "$YES" != 1 ]; then
  read -r -p "Run R4 now (bills the keys in .env)? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "not started"; exit 0 ;; esac
fi

echo "== key and credit check (one 8-token Anthropic request unless --or-only; OpenRouter GET /key and /credits)"
$PY src/tier1.py --check-keys --config "$CFG" || { echo "!! a key failed or has no credit - stopping before any episode (see the line above)"; exit 3; }

if [ "$OR_ONLY" = 1 ]; then
  echo "== --or-only: the Claude cells are skipped; no Anthropic request is made"
  export BENCH_TIER1_SKIP_PROVIDERS="anthropic"
fi

RC=1
for attempt in 1 2 3 4 5; do
  echo "== attempt $attempt  $(date -u +%H:%M:%SZ)"
  $PY src/tier1.py --config "$CFG" --launch; RC=$?
  case $RC in
    0) break ;;
    2) echo "!! Anthropic stopped (\$$CAP_A cap or credit). Completed episodes are kept; the OpenRouter cells finished in this pass unless noted above."; break ;;
    3) echo "!! OpenRouter stopped (\$$CAP_OR cap / key limit / credit). Completed episodes are kept."; break ;;
  esac
  echo "   some episodes failed (exit $RC); retrying in ${BENCH_TIER1_RETRY_SLEEP:-60}s (completed episodes are kept)"
  sleep "${BENCH_TIER1_RETRY_SLEEP:-60}"
done

echo
echo "=== summary  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
$PY src/tier1.py --status --config "$CFG"; ST=$?
if [ -f "$OUT/summary.csv" ]; then
  echo "--- $OUT/summary.csv"
  $PY - "$OUT/summary.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keys = [k for k in ("model", "failure_mode", "n_episodes", "refusals", "omission", "success", "ivr",
                    "readback_after_failure", "retry_rate", "truncated", "model_served", "cost_usd")
        if rows and k in rows[0]]
print("  ".join(keys))
for r in rows:
    print("  ".join(str(r[k])[:60] for k in keys))
PYEOF
fi
echo "== analysis"
$PY analysis/r4_precommit.py | tail -30
if [ $ST -eq 0 ]; then
  echo "=== ALL DONE. Results in $OUT/ - tell Claude it finished."
  exit 0
fi
case $RC in
  2|3) exit $RC ;;
  *) echo "=== NOT COMPLETE. Rerun the same line to continue; completed episodes are kept."; exit 4 ;;
esac
