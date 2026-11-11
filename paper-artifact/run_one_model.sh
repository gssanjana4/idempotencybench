#!/bin/bash
# Run ONE OpenRouter model (paper-209 + paper-249 cells) in its own Terminal tab,
# in parallel with run_crossprovider.sh. Same output folder; a per-model lock
# guarantees no episode is ever run twice. Resumable: rerun the same line.
#   bash ./run_one_model.sh gpt      (or: gemini, qwen)
set -uo pipefail
cd "$(dirname "$0")"
case "${1:-}" in
  gpt)    M="openrouter:openai/gpt-5.5-20260423" ;;
  gemini) M="openrouter:google/gemini-4-argon-20261013" ;;
  qwen)   M="openrouter:qwen/qwen3.6-27b-20260422" ;;
  *) echo "usage: bash run_one_model.sh gpt|gemini|qwen"; exit 1 ;;
esac
OUT="results/crossprovider"
SAFE="${M//\//_}"; TAG="${SAFE//:/_}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run_$1.log") 2>&1
echo "=== start $1 $(date)"
set -a; source .env; set +a
if [ -z "${OPENROUTER_API_KEY:-}" ]; then echo "OPENROUTER_API_KEY missing in .env"; exit 1; fi
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv && .venv/bin/pip install -q anthropic || { echo "venv setup failed"; exit 1; }
fi
PY=.venv/bin/python
export BENCH_MAX_TOKENS=4096
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &
CELLS2="amnesiac:none,plan_only:none,ledger_readback:none,bare_hint:none"
budget_hit() { grep -qiE "OpenRouter HTTP 402|insufficient credits|openrouter_key_limit|openrouter_credits" "$1"; }
LAST="$OUT/last_$1.log"

run_resumable() {
  local label=$1; shift
  for attempt in 1 2 3 4 5; do
    echo "== $label  (attempt $attempt)  $(date)"
    "$@" 2>&1 | tee "$LAST"
    local status=${PIPESTATUS[0]}
    [ "$status" -eq 0 ] && return 0
    if budget_hit "$LAST"; then echo "!! \$50 key limit reached - stopping."; exit 2; fi
    echo "   some episodes failed; retrying in 60s (completed episodes are kept)"
    sleep 60
  done
  echo "!! $label: still failing after 5 tries - skipping"; return 1
}

if ls "$OUT/smoke/sweep_transcripts/${SAFE}_"* >/dev/null 2>&1; then
  echo "== smoke $M already done"
else
  echo "== smoke $M"
  BENCH_SWEEP_MODELS="$M" BENCH_OUT_DIR="$OUT/smoke" BENCH_LIMIT=2 \
    $PY src/sweep.py --launch > "$OUT/smoke_${TAG}.log" 2>&1
  if budget_hit "$OUT/smoke_${TAG}.log"; then echo "!! \$50 key limit reached - stopping."; exit 2; fi
  if ! grep -lq '"tool_use"' "$OUT/smoke/sweep_transcripts/${SAFE}_"* 2>/dev/null; then
    echo "!! $M failed the smoke test (see $OUT/smoke_${TAG}.log)"; exit 1
  fi
fi
run_resumable "paper-209 cells  $M" env BENCH_SWEEP_MODELS="$M" BENCH_OUT_DIR="$OUT" \
  $PY src/sweep.py --launch
run_resumable "paper-249 cells  $M" env BENCH_SWEEP_MODELS="$M" BENCH_OUT_DIR="$OUT" \
  BENCH_CELLS="$CELLS2" $PY src2/sweep2.py --launch
echo "=== $1 DONE $(date)"
