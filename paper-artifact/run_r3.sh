#!/bin/bash
# R3: cross-provider wording (analysis/prereg_r3_draft.md) + completion of R2.
# One line from your own Terminal:
#
#   bash ./run_r3.sh --yes
#
#   bash run_r3.sh                 plan + cost estimate + caps, ask y/N, then run both parts to completion
#   bash run_r3.sh --yes           same, no question (resumable: rerun the same line after any stop)
#   bash run_r3.sh --estimate      plan, estimate and caps only; no API call
#   bash run_r3.sh --status        episodes done / remaining and logged spend, both parts
#   bash run_r3.sh --yes --skip-r2 R3 only (Part 1 skipped)
#   bash run_r3.sh --yes --only-r2 Part 1 only (R2 completion)
#   bash run_r3.sh --yes --or-only OpenRouter R3 cells only (no Anthropic call or credit needed)
#
# Part 1  R2 completion as registered (configs/r2_wording.json, run_r2.sh --yes):
#         the remaining Fable 5 Part-A paraphrase cells and Part C. No design change.
# Part 2  R3 (configs/r3_wording.json): R3-B error wording for GPT-5.5, Gemini,
#         Qwen3.6-27B (OpenRouter) and Fable 5 (Anthropic); R3-A restart notes
#         for the three OpenRouter models. Output: results/r3_wording/.
# Money   OpenRouter: hard stop at $15 (exp_or_budget_usd, usage.cost as returned).
#         Anthropic: ONE $25 cap shared by Part 1 and the Fable R3-B cells: Part 1
#         stops at (R2 spend when this script first ran) + $25, and the Fable cells
#         get $25 minus what Part 1 spent (list prices x logged tokens, cache
#         tokens at 1.25x / 0.1x). Prompt caching stays on for Anthropic.
#         A missing key, a failed key or no credit stops before any episode.
# - Resumable and idempotent: completed episodes are never run again (last row per id).
# - A provider refusal (Gemini "blocked the response") is a row with refusal=true.
# - .env is sourced, never printed.
set -uo pipefail
cd "$(dirname "$0")"

YES=0; MODE="run"; SKIP_R2=0; ONLY_R2=0; WORKERS=""; OR_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --estimate) MODE="estimate" ;;
    --status) MODE="status" ;;
    --skip-r2) SKIP_R2=1 ;;
    --only-r2) ONLY_R2=1 ;;
    --or-only) OR_ONLY=1; SKIP_R2=1; export BENCH_CHECK_SKIP_ANTHROPIC=1 ;;   # OpenRouter cells only; no Anthropic call
    --workers) WORKERS="${2:-}"; shift ;;
    -h|--help) sed -n 2,27p "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 64 ;;
  esac
  shift
done
CFG3="configs/r3_wording.json"; CFG2="configs/r2_wording.json"
[ -f "$CFG3" ] && [ -f "$CFG2" ] || { echo "missing $CFG3 or $CFG2"; exit 64; }

PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in .venv-claude/bin/python .venv-tier1/bin/python .venv/bin/python python3; do
    if "$cand" -c "import anthropic" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with the anthropic SDK found (pip install anthropic, or set PY=...)"; exit 1; }
export PY
SHARED_CAP=$($PY -c "import json;print(float(json.load(open('$CFG3'))['exp_budget_usd']))")

OUT="results/r3_wording"
if [ "$MODE" = run ]; then
  mkdir -p "$OUT"
  exec > >(tee -a "$OUT/run.log") 2>&1
fi
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ)  mode=$MODE skip_r2=$SKIP_R2 only_r2=$ONLY_R2  python=$PY"

set -a; [ -f .env ] && source .env; set +a
if [ "$MODE" != status ]; then
  [ -n "${ANTHROPIC_API_KEY:-}" ] && echo "ANTHROPIC_API_KEY: set (not shown)" || { echo "!! ANTHROPIC_API_KEY missing in .env - stopping"; exit 3; }
  [ -n "${OPENROUTER_API_KEY:-}" ] && echo "OPENROUTER_API_KEY: set (not shown)" || { echo "!! OPENROUTER_API_KEY missing in .env - stopping"; exit 3; }
fi
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &

export BENCH_BUDGET_USD=0 BENCH_OR_BUDGET_USD=0          # the config exp caps are the guard
export BENCH_TIER1_SKIP_PROVIDERS="" BENCH_TIER1_SKIP_MODELS=""
[ -n "$WORKERS" ] && export BENCH_WORKERS="$WORKERS"

r2_spend() { $PY -c "import sys;sys.path.insert(0,'src');import tier1;print('%.4f'%tier1.spent_in_exp(tier1.load_config('$CFG2'))['anthropic'])"; }
r3_spend() { $PY -c "import sys;sys.path.insert(0,'src');import tier1;s=tier1.spent_in_exp(tier1.load_config('$CFG3'));print('%.4f %.4f'%(s['anthropic'],s['openrouter']))"; }

if [ "$MODE" = status ]; then
  echo "== Part 1 (R2 completion)"; $PY src/tier1.py --status --config "$CFG2"
  echo "== Part 2 (R3)"; $PY src/tier1.py --status --config "$CFG3"; ST=$?
  [ -f "$OUT/r2_spend_at_start.usd" ] && echo "R2 Anthropic spend when run_r3.sh first ran: \$$(cat "$OUT/r2_spend_at_start.usd") (now \$$(r2_spend))"
  exit $ST
fi

echo "== plan: Part 1 (R2 completion, registered design)"
$PY src/tier1.py --status --config "$CFG2"
echo "== plan: Part 2 (R3)"
$PY src/tier1.py --config "$CFG3" --dry-run
echo "== cost estimate (src/r3_estimate.py: October logged tokens; cut order applied to the OpenRouter cap)"
$PY src/r3_estimate.py --cuts
echo "== caps: OpenRouter \$$($PY -c "import json;print(json.load(open('$CFG3'))['exp_or_budget_usd'])") (R3); Anthropic \$$SHARED_CAP shared by Part 1 and the Fable R3-B cells; prompt caching ON (Anthropic)"
[ "$MODE" = estimate ] && exit 0

if [ "$YES" != 1 ]; then
  read -r -p "Run Part 1 and Part 2 now (bills both keys in .env)? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "not started"; exit 0 ;; esac
fi

echo "== key and credit check (one 8-token Anthropic request; OpenRouter GET /key and /credits)"
$PY src/tier1.py --check-keys --config "$CFG3" --config "$CFG2" || { echo "!! a key failed or has no credit - stopping before any episode (see the line above)"; exit 3; }

# The shared Anthropic cap needs the R2 spend at the FIRST launch of this script (persisted).
[ -f "$OUT/r2_spend_at_start.usd" ] || r2_spend > "$OUT/r2_spend_at_start.usd"
S0=$(cat "$OUT/r2_spend_at_start.usd")
echo "R2 Anthropic spend at first launch: \$$S0"

SKIP_FABLE=0
[ "$OR_ONLY" = 1 ] && { SKIP_FABLE=1; echo "== --or-only: R2 completion and the Fable cells are skipped; no Anthropic request is made"; }
if [ "$SKIP_R2" != 1 ]; then
  CAP1=$($PY -c "print('%.2f' % min(45.0, $S0 + $SHARED_CAP))")
  echo
  echo "=== PART 1: R2 completion (run_r2.sh --yes; Anthropic stop at \$$CAP1 of R2 spend = registered \$45 or start + shared \$$SHARED_CAP)  $(date -u +%H:%M:%SZ)"
  BENCH_EXP_BUDGET_USD="$CAP1" bash run_r2.sh --yes; RC1=$?
  case $RC1 in
    0) echo "=== Part 1 complete: R2 has all planned episodes." ;;
    2) echo "!! Part 1 stopped: Anthropic cap or credit (exit 2). The Fable R3-B cells are skipped; the OpenRouter cells run."; SKIP_FABLE=1 ;;
    4) echo "!! Part 1 pass finished with episodes remaining (exit 4). Rerun this line later to finish them." ;;
    *) echo "!! Part 1 not complete (exit $RC1). Rerun this line later; completed episodes are kept." ;;
  esac
  [ "$ONLY_R2" = 1 ] && exit $RC1
fi

S1=$(r2_spend)
PART1=$($PY -c "print('%.4f' % max(0.0, $S1 - $S0))")
R3A=$(r3_spend | cut -d' ' -f1)
FABLE_CAP=$($PY -c "print('%.2f' % max(0.0, $SHARED_CAP - $PART1))")
echo
echo "=== PART 2: R3  $(date -u +%H:%M:%SZ)   Part 1 spent \$$PART1 of the shared \$$SHARED_CAP; Fable R3-B cap \$$FABLE_CAP (spent so far \$$R3A); OpenRouter cap from the config"
if [ "$SKIP_FABLE" = 1 ] || $PY -c "import sys; sys.exit(0 if $FABLE_CAP - $R3A < 0.5 else 1)"; then
  echo "!! no Anthropic budget left for the Fable R3-B cells (shared cap); running the OpenRouter cells only"
  export BENCH_TIER1_SKIP_MODELS="claude-fable-5"
  export BENCH_EXP_BUDGET_USD="0.01"          # nothing may start on Anthropic
else
  export BENCH_EXP_BUDGET_USD="$FABLE_CAP"
fi

RC=1
for attempt in 1 2 3 4 5; do
  echo "== attempt $attempt  $(date -u +%H:%M:%SZ)"
  $PY src/tier1.py --config "$CFG3" --launch; RC=$?
  case $RC in
    0) break ;;
    2) echo "!! Anthropic stopped (shared cap or credit). Completed episodes are kept; OpenRouter cells finished in this pass unless noted above."; break ;;
    3) echo "!! OpenRouter stopped (cap \$15 / key limit / credit). Completed episodes are kept."; break ;;
  esac
  echo "   some episodes failed (exit $RC); retrying in ${BENCH_TIER1_RETRY_SLEEP:-60}s (completed episodes are kept)"
  sleep "${BENCH_TIER1_RETRY_SLEEP:-60}"
done

echo
echo "=== summary  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "== Part 1 (R2)"; $PY src/tier1.py --status --config "$CFG2"; ST2=$?
echo "== Part 2 (R3)"; $PY src/tier1.py --status --config "$CFG3"; ST3=$?
for f in "$OUT/summary.csv" "$OUT/summary_restart.csv"; do
  if [ -f "$f" ]; then
    echo "--- $f"
    $PY - "$f" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keys = [k for k in ("model", "failure_mode", "resume_context", "n_episodes", "refusals", "ivr", "ivr_ci_tlo", "ivr_ci_thi",
                    "givr", "givr_ci_glo", "givr_ci_ghi", "readback_after_failure", "session2_lookup", "truncated", "model_served", "cost_usd")
        if rows and k in rows[0]]
print("  ".join(keys))
for r in rows:
    print("  ".join(str(r[k])[:60] for k in keys))
PYEOF
  fi
done
echo "== analysis"
$PY analysis/r2_wording.py | tail -3
$PY analysis/r3_wording.py | tail -40
if [ $ST2 -eq 0 ] && [ $ST3 -eq 0 ]; then
  echo "=== ALL DONE. Results in results/r2_wording/ and $OUT/ - tell Claude it finished."
  exit 0
fi
case $RC in
  2|3) exit $RC ;;
  *) echo "=== NOT COMPLETE. Rerun the same line to continue; completed episodes are kept."; exit 4 ;;
esac
