#!/bin/bash
# Tier-1 "Must" experiments (E1-0 ... E1-6).
# Run from your own Terminal:   bash ./run_tier1.sh
# - Resumable: if it stops (sleep, crash, Ctrl-C), run the same line again.
# - Money: hard stop at $120 of Claude usage (list prices x logged tokens,
#   all of results/tier1/). OpenRouter models stop at the $50 key limit
#   (HTTP 402); the Claude experiments then carry on without them.
# - Writes ONLY to results/tier1/<experiment>/ (released data is never touched).
# - Safe next to run_crossprovider.sh / run_claude_batch.sh (own folder, own venv,
#   per-model locks).
#
# One added (non-registered) experiment at a time:
#   bash ./run_tier1.sh --only e1_6_oct
# runs only configs/<name>.json: key check and smoke for that config's
# models only (the smoke is the experiment's own first episode per model, so
# nothing is paid twice), log in results/tier1/<name>/run.log (the registered
# runs' results/tier1/run.log is not touched). e1_6_oct = October bare-500 /
# timeout baselines for E1-6, a SENSITIVITY analysis (preregistration.md,
# deviations log), capped by exp_budget_usd in its config.
set -uo pipefail
cd "$(dirname "$0")"
OUT="results/tier1"
ONLY=""
if [ "${1:-}" = "--only" ]; then
  ONLY="${2:-}"
  case "$ONLY" in ""|*/*|*.*) echo "usage: run_tier1.sh [--only <config name, e.g. e1_6_oct>]"; exit 64 ;; esac
  [ -f "configs/$ONLY.json" ] || { echo "no configs/$ONLY.json"; exit 64; }
elif [ $# -gt 0 ]; then
  echo "usage: run_tier1.sh [--only <config name, e.g. e1_6_oct>]"; exit 64
fi
LOGDIR="$OUT${ONLY:+/$ONLY}"
mkdir -p "$LOGDIR"
exec > >(tee -a "$LOGDIR/run.log") 2>&1
echo "=== start $(date)${ONLY:+   (--only $ONLY)}"

set -a; [ -f .env ] && source .env; set +a

VENV=.venv-tier1     # own env, so it never races the other scripts' setup
if [ ! -x "$VENV/bin/python" ]; then
  echo "Setting up $VENV (one time)..."
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q anthropic || { echo "venv setup failed"; exit 1; }
fi
PY="$VENV/bin/python"
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &

CAP="${BENCH_TIER1_CAP_USD:-120}"
export BENCH_BUDGET_USD="$CAP"
export BENCH_TIER1_SKIP_PROVIDERS="" BENCH_TIER1_SKIP_MODELS=""

# Order = priority.
#   E1-1  209-AC + 249-AC: models from another provider (cell 8 here; cells 1-7
#         come from run_crossprovider.sh and are merged into the summary)
#   E1-2  209-AC: real models under at-least-once redelivery
#   E1-0  209-AC: keyless-dedup fix, confirmation rerun
#   E1-3  209-AC: key reuse vs server dedup; keys under a 500 (T26)
#   E1-5  209-AC + 249-AC: longer multi-action workflows, failure mid-plan
#   E1-4  qkC4/dXix/GFU6 + 209-AC task-level CIs: 96 distinct tasks
#   E1-6  249-AC realism / dXix: realistic error payloads, outcome-unknown wording
EXPS="e1_1_crossprovider_core e1_2_at_least_once e1_0_key_rerun e1_3_keys_under_500 e1_5_long_chains e1_4_task_pool e1_6_signal_wording"
[ -n "$ONLY" ] && EXPS="$ONLY"
CFGS=""; for E in $EXPS; do CFGS="$CFGS --config configs/$E.json"; done

# 1. Key checks: Anthropic = one 8-token request; OpenRouter = free GET /key.
$PY src/tier1.py --check-keys $CFGS
KC=$?
if [ $((KC & 1)) -ne 0 ]; then
  echo "!! Anthropic key check failed: Claude experiments skipped this run. Fix ANTHROPIC_API_KEY in .env and rerun."
  BENCH_TIER1_SKIP_PROVIDERS="anthropic"
fi
if [ $((KC & 2)) -ne 0 ]; then
  echo "!! OpenRouter key check failed: OpenRouter models skipped this run. Fix OPENROUTER_API_KEY in .env and rerun."
  BENCH_TIER1_SKIP_PROVIDERS="${BENCH_TIER1_SKIP_PROVIDERS:+$BENCH_TIER1_SKIP_PROVIDERS,}openrouter"
fi
if [ $((KC & 3)) -eq 3 ]; then echo "!! no working key - stopping."; exit 3; fi
export BENCH_TIER1_SKIP_PROVIDERS

stop_openrouter() {
  echo "!! OpenRouter stopped (key limit / credits) - continuing with Claude models only."
  BENCH_TIER1_SKIP_PROVIDERS="${BENCH_TIER1_SKIP_PROVIDERS:+$BENCH_TIER1_SKIP_PROVIDERS,}openrouter"
  export BENCH_TIER1_SKIP_PROVIDERS
}

# 2. Smoke: 1 episode per model (wrong slug / no tool calls => model skipped).
#    Paid once: a model whose smoke transcript exists is not re-smoked.
SMOKE=configs/e1_smoke.json; SMOKE_DIR="$OUT/smoke"; SMOKE_LOG="$OUT"
if [ -n "$ONLY" ]; then   # smoke = first episode of the experiment itself (kept)
  SMOKE="configs/$ONLY.json"; SMOKE_DIR="$OUT/$ONLY"; SMOKE_LOG="$OUT/$ONLY"
fi
for M in $($PY -c "import json;print(' '.join(json.load(open('$SMOKE'))['models']))"); do
  SAFE="${M//\//_}"
  case "$M" in openrouter:*) P=openrouter ;; *) P=anthropic ;; esac
  case ",$BENCH_TIER1_SKIP_PROVIDERS," in *",$P,"*) continue ;; esac
  if ! ls "$SMOKE_DIR/transcripts/${SAFE}_"* >/dev/null 2>&1; then
    echo "== smoke $M"
    BENCH_LIMIT=1 $PY src/tier1.py --config $SMOKE --launch --models "$M" > "$SMOKE_LOG/smoke_${SAFE//:/_}.log" 2>&1
    SC=$?
    [ $SC -eq 2 ] && { echo "!! Claude spend cap reached (\$$CAP overall, or the config's exp_budget_usd) - stopping."; exit 2; }
    [ $SC -eq 3 ] && { stop_openrouter; continue; }
  fi
  if ! grep -lq '"tool_use"' "$SMOKE_DIR/transcripts/${SAFE}_"* 2>/dev/null; then
    echo "!! $M failed the smoke test (see $SMOKE_LOG/smoke_${SAFE//:/_}.log) - skipping this model"
    BENCH_TIER1_SKIP_MODELS="${BENCH_TIER1_SKIP_MODELS:+$BENCH_TIER1_SKIP_MODELS,}$M"
  else
    echo "== smoke $M ok"
  fi
done
export BENCH_TIER1_SKIP_MODELS

run_exp() {   # $1 = experiment config name
  local cfg="configs/$1.json"
  echo
  echo "=================== $1   $(date)"
  $PY src/tier1.py --config "$cfg" --estimate
  for attempt in 1 2 3 4 5; do
    echo "== $1  (attempt $attempt)  $(date)"
    $PY src/tier1.py --config "$cfg" --launch
    local status=$?
    case $status in
      0) return 0 ;;
      2) echo "!! Claude spend cap reached (\$$CAP overall, or the config's exp_budget_usd) - stopping. Completed episodes are kept."
         $PY src/tier1.py --status $CFGS; exit 2 ;;
      3) stop_openrouter; continue ;;
    esac
    echo "   some episodes failed (exit $status); retrying in 60s (completed episodes are kept)"
    sleep "${BENCH_TIER1_RETRY_SLEEP:-60}"
  done
  echo "!! $1: episodes still failing after 5 tries - moving on (rerun later)"; return 1
}

for E in $EXPS; do run_exp "$E"; done

echo
echo "=== status"
if $PY src/tier1.py --status $CFGS; then
  echo "=== ALL DONE $(date). Results in $OUT/ - tell Claude it finished."
else
  echo "=== FINISHED THIS PASS, NOT COMPLETE $(date): episodes above remain"
  echo "    (skipped: providers '${BENCH_TIER1_SKIP_PROVIDERS}' models '${BENCH_TIER1_SKIP_MODELS}')."
  echo "    Rerun the same line to continue; tell Claude what it printed."
  exit 4
fi
