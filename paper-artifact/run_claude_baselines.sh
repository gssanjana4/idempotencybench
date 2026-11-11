#!/bin/bash
# Same-month baselines for the bare_hint comparison.
# Run from your own Terminal:   bash ./run_claude_baselines.sh
# Safe to run at the same time as run_crossprovider.sh (separate output folder).
# Resumable: if it stops, run the same line again. Hard stop at $60 of Claude usage.
set -uo pipefail
cd "$(dirname "$0")"
OUT="results/claude_batch_2026-10"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== start $(date)"

set -a; source .env; set +a
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then echo "ANTHROPIC_API_KEY missing in .env"; exit 1; fi

VENV=.venv-claude    # own env, so it never races the other script's setup
if [ ! -x "$VENV/bin/python" ]; then
  echo "Setting up $VENV (one time)..."
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q anthropic || { echo "venv setup failed"; exit 1; }
fi
PY="$VENV/bin/python"
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &

export BENCH_OUT_DIR="$OUT" BENCH_BUDGET_USD=60
export BENCH_SWEEP_MODELS="claude-fable-5,claude-sonnet-5,claude-haiku-4-5-20251001"

# 1. Key check: one tiny request. Stops here with a clear message if the key is dead.
$PY - <<'PYEOF' || { echo "!! Anthropic key check failed (see above). Fix ANTHROPIC_API_KEY in .env and rerun."; exit 3; }
import anthropic
c = anthropic.Anthropic()
r = c.messages.create(model="claude-haiku-4-5-20251001", max_tokens=8,
                      messages=[{"role": "user", "content": "Say OK"}])
print("key OK:", r.model)
PYEOF

run_resumable() {   # $1 = label, rest = command
  local label=$1; shift
  for attempt in 1 2 3 4 5; do
    echo "== $label  (attempt $attempt)  $(date)"
    "$@" 2>&1 | tee "$OUT/last.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] && return 0
    if grep -q "budget .* would be exceeded" "$OUT/last.log"; then echo "!! \$60 cap reached - stopping."; exit 2; fi
    echo "   some episodes failed; retrying in 60s (completed episodes are kept)"
    sleep 60
  done
  echo "!! $label: episodes still failing after 5 tries - skipping"; return 1
}

# 2. Same-month, same-settings baselines for bare_hint (Oct 2026, provider-default
#    temperature, same SDK/harness): amnesiac + ledger_readback, 3 Claude models,
#    16 graphs x 2 samples = 192 episodes. Same output folder as the bare_hint runs.
run_resumable "same-month baselines (amnesiac, ledger_readback)" \
  env BENCH_CELLS="amnesiac:none,ledger_readback:none" $PY src2/sweep2.py --launch

echo "=== ALL DONE $(date). Results in $OUT/ - tell Claude it finished."
