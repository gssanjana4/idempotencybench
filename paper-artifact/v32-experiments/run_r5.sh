#!/bin/bash
# R5: real error texts (v32-experiments/prereg_r5.md). One line from a terminal
# on a Linux machine, with REPO = the repository root:
#
#   REPO=~/benchmark bash ~/benchmark/v32-experiments/run_r5.sh --yes
#
#   run_r5.sh                plan + offline tests + estimate + guard, ask y/N, then run to completion
#   run_r5.sh --yes          same, no question (resumable: rerun the same line after any stop)
#   run_r5.sh --estimate     plan, estimate and spend so far; no API call
#   run_r5.sh --smoke        exactly 1 Haiku episode (key + harness check; kept, counts toward R5)
#   run_r5.sh --status       episodes done / remaining, cumulative spend
#   run_r5.sh --aggregate    summary.csv + the registered analysis (no API call)
#   run_r5.sh --yes --or-only   Qwen cells only (no Anthropic request)
#
# Money   ONE cumulative cap for all v32 runs: $50 (config r5.total_cap_usd), kept in
#         the ledger v32-experiments/results/spend_ledger.jsonl. Every API call is
#         reserved there at its worst case BEFORE it is sent and refused if the
#         cumulative total would pass $50 (src/spend_guard.py); this script also
#         refuses to start an attempt when less than $0.01 is left. Inside that,
#         the runner's own per-provider caps apply: Anthropic $40, OpenRouter $6.
#         Estimate: expected $28.17, worst $39.90 (src/estimate_r5.py).
# Writes  ONLY under v32-experiments/results/ (raw.jsonl, transcripts/, run.log,
#         summary.csv, analysis/, the ledger). No tracked file is touched.
# .env is sourced, never printed.
set -uo pipefail
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO" || { echo "REPO=$REPO not found"; exit 64; }
V32="v32-experiments"
CFG="$V32/configs/r5_real_errors.json"
[ -f "$CFG" ] && [ -f src/tier1.py ] || { echo "REPO=$REPO is not the repository root (no $CFG or src/tier1.py)"; exit 64; }

YES=0; MODE="run"; SMOKE=0; OR_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --estimate) MODE="estimate" ;;
    --status) MODE="status" ;;
    --aggregate) MODE="aggregate" ;;
    --smoke) SMOKE=1 ;;
    --or-only) OR_ONLY=1 ;;
    -h|--help) sed -n 2,26p "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 64 ;;
  esac
  shift
done

PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in .venv-claude/bin/python .venv-tier1/bin/python .venv/bin/python python3; do
    if "$cand" -c "import anthropic" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with the anthropic SDK found (pip install anthropic, or set PY=...)"; exit 1; }
R5="$PY $V32/src/run_r5.py --config $CFG"
OUT="$V32/results/r5_real_errors"

if [ "$MODE" = run ]; then
  mkdir -p "$OUT"
  exec > >(tee -a "$OUT/run.log") 2>&1
fi
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ)  repo=$REPO mode=$MODE smoke=$SMOKE or_only=$OR_ONLY python=$PY"

guard_left() {   # dollars left under the cumulative cap (ledger), printed with 4 decimals
  $PY - "$CFG" <<'PYEOF'
import json, os, sys
sys.path.insert(0, "v32-experiments/src")
from spend_guard import SpendGuard
c = json.load(open(sys.argv[1]))["r5"]
g = SpendGuard(c["ledger"], c["total_cap_usd"])
print("%.4f" % g.remaining())
PYEOF
}

case "$MODE" in
  status) $R5 --status; exit $? ;;
  aggregate) $R5 --aggregate; exit $? ;;
esac

echo "== offline tests (fake providers; no network)"
$PY $V32/tests/test_r5.py -q >/tmp/r5_tests.$$ 2>&1 || { tail -30 /tmp/r5_tests.$$; echo "!! R5 tests failed - stopping before any API call"; exit 5; }
tail -3 /tmp/r5_tests.$$; rm -f /tmp/r5_tests.$$
echo "== plan and injected texts"
$R5 --dry-run | tail -40
echo "== cost estimate (logged R2/R3 per-episode spend)"
$R5 --estimate || { echo "!! estimate above the registered \$45: apply the cut order in the config first"; exit 6; }
LEFT=$(guard_left)
echo "== cumulative cap: \$$LEFT left of \$$($PY -c "import json;print(json.load(open('$CFG'))['r5']['total_cap_usd'])")"
[ "$MODE" = estimate ] && exit 0

if [ "$YES" != 1 ] && [ "$SMOKE" != 1 ]; then
  read -r -p "Run R5 now (bills the keys in .env; hard stop at \$50 cumulative)? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "not started"; exit 0 ;; esac
fi

set -a; [ -f .env ] && source .env; set +a
export BENCH_BUDGET_USD=0 BENCH_OR_BUDGET_USD=0           # the guard and the config caps apply
unset BENCH_EXP_BUDGET_USD BENCH_EXP_OR_BUDGET_USD
export BENCH_TIER1_SKIP_PROVIDERS="" BENCH_TIER1_SKIP_MODELS=""
if [ "$OR_ONLY" = 1 ]; then
  export BENCH_CHECK_SKIP_ANTHROPIC=1 BENCH_TIER1_SKIP_PROVIDERS="anthropic"
  echo "ANTHROPIC_API_KEY: not needed (--or-only)"
else
  [ -n "${ANTHROPIC_API_KEY:-}" ] && echo "ANTHROPIC_API_KEY: set (not shown)" || { echo "!! ANTHROPIC_API_KEY missing in .env"; exit 3; }
fi
[ -n "${OPENROUTER_API_KEY:-}" ] && echo "OPENROUTER_API_KEY: set (not shown)" || { echo "!! OPENROUTER_API_KEY missing in .env"; exit 3; }

echo "== key and credit check (one 8-token Anthropic request unless --or-only; OpenRouter GET /key, /credits)"
$R5 --check-keys || { echo "!! a key failed or has no credit - stopping before any episode"; exit 3; }

if [ "$SMOKE" = 1 ]; then
  echo "== smoke: 1 Haiku episode (the first planned Haiku episode; kept)"
  $R5 --launch --limit 1 --models claude-haiku-4-5-20251001; SC=$?
  $R5 --status
  echo "== smoke exit $SC (0 = keys, harness and guard OK)"
  exit $SC
fi

RC=1
for attempt in 1 2 3 4 5; do
  LEFT=$(guard_left)
  if $PY -c "import sys; sys.exit(0 if $LEFT < 0.01 else 1)"; then
    echo "!! cumulative spend cap reached (\$$LEFT left) - ABORT. Completed episodes are kept."; RC=2; break
  fi
  echo "== attempt $attempt  $(date -u +%H:%M:%SZ)  (\$$LEFT left under the cap)"
  $R5 --launch; RC=$?
  case $RC in
    0) break ;;
    2) echo "!! Anthropic stopped (spend guard, \$40 config cap or credit). Completed episodes are kept."; break ;;
    3) echo "!! OpenRouter stopped (spend guard, \$6 config cap, key limit or credit). Completed episodes are kept."; break ;;
  esac
  echo "   some episodes failed (exit $RC); retrying in ${BENCH_TIER1_RETRY_SLEEP:-60}s (completed episodes are kept)"
  sleep "${BENCH_TIER1_RETRY_SLEEP:-60}"
done

echo
echo "=== summary  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
$R5 --status; ST=$?
$R5 --aggregate | tail -12
if [ $ST -eq 0 ]; then
  echo "=== ALL DONE. Results in $OUT/ - tell Claude it finished."
  exit 0
fi
case $RC in
  2|3) exit $RC ;;
  *) echo "=== NOT COMPLETE. Rerun the same line to continue; completed episodes are kept."; exit 4 ;;
esac
