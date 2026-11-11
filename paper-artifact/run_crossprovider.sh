#!/bin/bash
# Cross-provider runs for the paper (T01 + bare_hint on all models).
# Run from your own Terminal:   bash ./run_crossprovider.sh
# - Resumable: if it stops (sleep, crash, Ctrl-C), run the same line again.
# - Money: the OpenRouter key has a $50 limit; when it is hit the script stops.
# - Writes ONLY to results/crossprovider/ (released data is never touched).
set -uo pipefail
cd "$(dirname "$0")"
OUT="results/crossprovider"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== start $(date)"

set -a; source .env; set +a
if [ -z "${OPENROUTER_API_KEY:-}" ]; then echo "OPENROUTER_API_KEY missing in .env"; exit 1; fi

# Private Python env with the one dependency (anthropic SDK; the harness imports it).
if [ ! -x .venv/bin/python ]; then
  echo "Setting up .venv (one time)..."
  python3 -m venv .venv && .venv/bin/pip install -q anthropic || { echo "venv setup failed"; exit 1; }
fi
PY=.venv/bin/python
# Reasoning tokens count against max_tokens on OpenRouter; 1024 would cut retry turns.
export BENCH_MAX_TOKENS=4096

# Keep the Mac awake while this script runs (macOS).
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &

# Dated slugs (dated model ids). Cheapest first.
MODELS=(
  "openrouter:qwen/qwen3.6-27b-20260422"
  "openrouter:openai/gpt-5.5-20260423"
  "openrouter:google/gemini-4-argon-20261013"
)
CELLS2="amnesiac:none,plan_only:none,ledger_readback:none,bare_hint:none"

budget_hit() { grep -qiE "OpenRouter HTTP 402|insufficient credits|openrouter_key_limit|openrouter_credits" "$1"; }

run_resumable() {   # $1 = label, rest = command
  local label=$1; shift
  for attempt in 1 2 3 4 5; do
    echo "== $label  (attempt $attempt)  $(date)"
    "$@" 2>&1 | tee "$OUT/last.log"
    local status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] && return 0
    if budget_hit "$OUT/last.log"; then echo "!! \$50 key limit reached - stopping."; exit 2; fi
    echo "   failed (exit $status); retrying in 60s (completed episodes are kept)"
    sleep 60
  done
  echo "!! $label failed 5 times - skipping it"; return 1
}

for M in "${MODELS[@]}"; do
  SAFE="${M//\//_}"
  # Smoke: 2 episodes. Skip the model if the slug is wrong or it never calls a tool.
  if ls "$OUT/smoke/sweep_transcripts/${SAFE}_"* >/dev/null 2>&1; then
    echo "== smoke $M already done"
  else
    echo "== smoke $M"
    BENCH_SWEEP_MODELS="$M" BENCH_OUT_DIR="$OUT/smoke" BENCH_LIMIT=2 \
      $PY src/sweep.py --launch > "$OUT/smoke_${SAFE//:/_}.log" 2>&1
  fi
  if budget_hit "$OUT/smoke_${SAFE//:/_}.log"; then echo "!! \$50 key limit reached - stopping."; exit 2; fi
  if ! grep -lq '"tool_use"' "$OUT/smoke/sweep_transcripts/${SAFE}_"* 2>/dev/null; then
    echo "!! $M failed the smoke test (see $OUT/smoke_${SAFE//:/_}.log) - skipping this model"
    continue
  fi
  run_resumable "paper-209 cells  $M" env BENCH_SWEEP_MODELS="$M" BENCH_OUT_DIR="$OUT" \
    $PY src/sweep.py --launch
  run_resumable "paper-249 cells  $M" env BENCH_SWEEP_MODELS="$M" BENCH_OUT_DIR="$OUT" \
    BENCH_CELLS="$CELLS2" $PY src2/sweep2.py --launch
done
echo "=== ALL DONE $(date). Results in $OUT/ - tell Claude it finished."
