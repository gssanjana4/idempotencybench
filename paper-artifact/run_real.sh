#!/bin/bash
# Real-API validation slice: the payment tool on Stripe TEST mode
# (preregistration.md section R1; answers 209-dXix W2, 249-AC, 249-GFU6).
# Run from your own Terminal:   bash ./run_real.sh
#
# One-time setup: add your Stripe test secret key to .env (next to the
# Anthropic/OpenRouter keys):
#     STRIPE_SECRET_KEY=sk_test_...
# The script refuses to start unless the key starts with sk_test_ (test mode:
# no money moves) and never prints it.
#
# - Resumable: if it stops (sleep, crash, Ctrl-C), run the same line again.
# - Money: Claude stops at $20 (list prices x logged tokens, results/real_stripe/
#   only; override with BENCH_REAL_CAP_USD). OpenRouter stops at its key limit
#   (HTTP 402); the Claude models then carry on. Stripe test mode costs $0.
# - Writes ONLY to results/real_stripe/.
# - The three models run in parallel (one process each, per-model locks).
#   BENCH_REAL_SERIAL=1 runs them one after another instead.
set -uo pipefail
cd "$(dirname "$0")"
OUT="results/real_stripe"
CFG="configs/real_stripe.json"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "=== start $(date)"

set -a; [ -f .env ] && source .env; set +a

VENV=.venv-real      # own env, so it never races the other scripts' setup
if [ ! -x "$VENV/bin/python" ]; then
  echo "Setting up $VENV (one time)..."
  python3 -m venv "$VENV" && "$VENV/bin/pip" install -q anthropic || { echo "venv setup failed"; exit 1; }
fi
PY="$VENV/bin/python"
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &

# 1. Stripe key: must be a test-mode secret key (hard check, exit 5).
if ! $PY src_real/run_real.py --config $CFG --check-stripe-key; then
  echo "!! Add your Stripe TEST secret key to .env as  STRIPE_SECRET_KEY=sk_test_...  and rerun."
  exit 5
fi

CAP="${BENCH_REAL_CAP_USD:-20}"
export BENCH_BUDGET_USD="$CAP"
export BENCH_REAL_SKIP_PROVIDERS="" BENCH_REAL_SKIP_MODELS=""

# 2. Model keys: Anthropic = one 8-token request; OpenRouter = free GET /key.
$PY src/tier1.py --check-keys --config $CFG
KC=$?
if [ $((KC & 1)) -ne 0 ]; then
  echo "!! Anthropic key check failed: Claude models skipped this run. Fix ANTHROPIC_API_KEY in .env and rerun."
  BENCH_REAL_SKIP_PROVIDERS="anthropic"
fi
if [ $((KC & 2)) -ne 0 ]; then
  echo "!! OpenRouter key check failed: Qwen skipped this run. Fix OPENROUTER_API_KEY in .env and rerun."
  BENCH_REAL_SKIP_PROVIDERS="${BENCH_REAL_SKIP_PROVIDERS:+$BENCH_REAL_SKIP_PROVIDERS,}openrouter"
fi
if [ $((KC & 3)) -eq 3 ]; then echo "!! no working model key - stopping."; exit 3; fi
export BENCH_REAL_SKIP_PROVIDERS

# 3. One Stripe round trip: charge $0.50, replay the same Idempotency-Key
#    (must be the same PaymentIntent, one charge), list it back, reuse the key
#    with other parameters (must be refused). Result: results/real_stripe/stripe_check.json
if ! $PY src_real/run_real.py --config $CFG --stripe-check; then
  echo "!! Stripe round-trip check failed (see above) - nothing was run. Tell Claude what it printed."
  exit 7
fi

$PY src_real/run_real.py --config $CFG --estimate

# 4. Smoke: 1 episode per model (its first grid episode, kept; paid once).
for M in $($PY -c "import json;print(' '.join(json.load(open('$CFG'))['models']))"); do
  SAFE="${M//\//_}"
  case "$M" in openrouter:*) P=openrouter ;; *) P=anthropic ;; esac
  case ",$BENCH_REAL_SKIP_PROVIDERS," in *",$P,"*) continue ;; esac
  if ! ls "$OUT/transcripts/${SAFE}_"* >/dev/null 2>&1; then
    echo "== smoke $M"
    BENCH_LIMIT=1 $PY src_real/run_real.py --config $CFG --launch --models "$M" > "$OUT/smoke_${SAFE//:/_}.log" 2>&1
    SC=$?
    [ $SC -eq 5 ] && { echo "!! Stripe key problem (see $OUT/smoke_${SAFE//:/_}.log) - stopping."; exit 5; }
    [ $SC -eq 2 ] && { echo "!! Claude spend cap reached (\$$CAP) - stopping."; exit 2; }
  fi
  if ! grep -lq '"tool_use"' "$OUT/transcripts/${SAFE}_"* 2>/dev/null; then
    echo "!! $M failed the smoke test (see $OUT/smoke_${SAFE//:/_}.log) - skipping this model"
    BENCH_REAL_SKIP_MODELS="${BENCH_REAL_SKIP_MODELS:+$BENCH_REAL_SKIP_MODELS,}$M"
  else
    echo "== smoke $M ok"
  fi
done
export BENCH_REAL_SKIP_MODELS

# 5. Grid: one resumable process per model, retried up to 5 times.
run_model() {   # $1 = model id
  for attempt in 1 2 3 4 5; do
    echo "== $1  (attempt $attempt)  $(date)"
    $PY src_real/run_real.py --config $CFG --launch --models "$1"
    local status=$?
    case $status in
      0|2|3|5) return $status ;;
      6) echo "   Stripe unreachable; retrying in 120s"; sleep 120 ;;
      *) echo "   some episodes failed (exit $status); retrying in 60s (completed episodes are kept)"; sleep 60 ;;
    esac
  done
  return 1
}
PIDS=""; RCS=""
for M in $($PY -c "import json;print(' '.join(json.load(open('$CFG'))['models']))"); do
  case "$M" in openrouter:*) P=openrouter ;; *) P=anthropic ;; esac
  case ",$BENCH_REAL_SKIP_PROVIDERS," in *",$P,"*) continue ;; esac
  case ",$BENCH_REAL_SKIP_MODELS," in *",$M,"*) continue ;; esac
  LOG="$OUT/run_${M//[\/:]/_}.log"
  if [ "${BENCH_REAL_SERIAL:-0}" = "1" ]; then
    run_model "$M" >> "$LOG" 2>&1; RCS="$RCS $M=$?"
  else
    echo "== $M running in the background; progress: tail -f $LOG"
    run_model "$M" >> "$LOG" 2>&1 &
    PIDS="$PIDS $!:$M"
  fi
done
for PM in $PIDS; do
  wait "${PM%%:*}"; RCS="$RCS ${PM#*:}=$?"
done
echo "== model runs finished:$RCS   (0 done, 1 failures left, 2 Claude cap, 3 OpenRouter stopped, 5 Stripe key)"
case "$RCS" in *=5*) echo "!! Stripe key problem - see the run_*.log files"; exit 5 ;; esac
case "$RCS" in *=2*) echo "!! Claude spend cap reached (\$$CAP). Completed episodes are kept." ;; esac

# 6. Delayed cross-check of Stripe's record with the Search API (waits ~70 s).
$PY src_real/run_real.py --config $CFG --verify-search
$PY src_real/run_real.py --config $CFG --aggregate > /dev/null

echo
echo "=== status"
if $PY src_real/run_real.py --config $CFG --status; then
  echo "=== ALL DONE $(date). Results in $OUT/ - tell Claude it finished."
else
  echo "=== FINISHED THIS PASS, NOT COMPLETE $(date): episodes above remain"
  echo "    (skipped: providers '${BENCH_REAL_SKIP_PROVIDERS}' models '${BENCH_REAL_SKIP_MODELS}')."
  echo "    Rerun the same line to continue; tell Claude what it printed."
  exit 4
fi
