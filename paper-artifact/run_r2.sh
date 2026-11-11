#!/bin/bash
# R2: wording robustness (analysis/prereg_r2_draft.md). Claude-only default run
# through the Tier-1 runner (src/tier1.py, configs/r2_wording.json).
#
#   bash run_r2.sh                         plan + cost estimate, ask y/N, run to completion
#   bash run_r2.sh --yes                   same, no question
#   bash run_r2.sh --yes --smoke           exactly 1 cheap Haiku episode (key check), then exit
#   bash run_r2.sh --yes --max-seconds 150 run episodes for ~150 s, exit cleanly; rerun to continue
#   bash run_r2.sh --yes --only r2_wording_or   OPTIONAL OpenRouter extension (not run by default)
#   bash run_r2.sh --estimate              plan + cost estimate only, no API calls
#   bash run_r2.sh --status                episodes done / remaining, logged spend
#
# - Resumable and idempotent: completed episodes (rows in raw.jsonl, last row per
#   id) are never run again; a per-model-set runlock stops two copies at once.
# - Money: hard stop at the config's exp_budget_usd ($45 Anthropic, list prices x
#   logged tokens incl. cached tokens at 1.25x / 0.1x). No OpenRouter spend in the
#   default run. The optional r2_wording_or config stops at exp_or_budget_usd ($40).
# - Writes ONLY to results/<out_dir>/ {raw.jsonl, transcripts/, run.log,
#   summary.csv, summary_restart.csv}. Released data is never touched.
# - .env is sourced, never printed.
set -uo pipefail
cd "$(dirname "$0")"

YES=0; SMOKE=0; MAXS=""; ONLY="r2_wording"; MODE="run"; WORKERS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --yes|-y) YES=1 ;;
    --smoke) SMOKE=1 ;;
    --max-seconds) MAXS="${2:-}"; shift ;;
    --only) ONLY="${2:-}"; shift ;;
    --workers) WORKERS="${2:-}"; shift ;;
    --estimate) MODE="estimate" ;;
    --status) MODE="status" ;;
    -h|--help) sed -n 2,20p "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)"; exit 64 ;;
  esac
  shift
done
CFG="configs/$ONLY.json"
[ -f "$CFG" ] || { echo "no $CFG"; exit 64; }
case "$MAXS" in ""|*[!0-9]*) [ -n "$MAXS" ] && { echo "--max-seconds needs a whole number"; exit 64; } ;; esac

# Python with the anthropic SDK (the Mac venvs are not runnable from a Linux VM and vice versa).
PY="${PY:-}"
if [ -z "$PY" ]; then
  for cand in .venv-claude/bin/python .venv-tier1/bin/python .venv/bin/python python3; do
    if "$cand" -c "import anthropic" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
[ -n "$PY" ] || { echo "no Python with the anthropic SDK found (pip install anthropic, or set PY=...)"; exit 1; }

OUT=$($PY -c "import json,sys;print(json.load(open('$CFG'))['out_dir'])")
if [ "$MODE" = run ]; then          # only a real run (or smoke) writes the log
  mkdir -p "$OUT"
  exec > >(tee -a "$OUT/run.log") 2>&1
fi
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ)  config=$CFG  mode=$MODE smoke=$SMOKE max_seconds=${MAXS:-none}  python=$PY"

set -a; [ -f .env ] && source .env; set +a
NEED_A=$($PY -c "import sys;sys.path.insert(0,'src');import tier1;print(int(any(tier1.provider_of(m)=='anthropic' for m in tier1.load_config('$CFG')['models'])))")
NEED_O=$($PY -c "import sys;sys.path.insert(0,'src');import tier1;print(int(any(tier1.provider_of(m)=='openrouter' for m in tier1.load_config('$CFG')['models'])))")
if [ "$NEED_A" = 1 ]; then
  [ -n "${ANTHROPIC_API_KEY:-}" ] && echo "ANTHROPIC_API_KEY: set (not shown)" || { echo "!! ANTHROPIC_API_KEY missing in .env"; exit 3; }
fi
if [ "$NEED_O" = 1 ]; then
  [ -n "${OPENROUTER_API_KEY:-}" ] && echo "OPENROUTER_API_KEY: set (not shown)" || { echo "!! OPENROUTER_API_KEY missing in .env"; exit 3; }
fi
command -v caffeinate >/dev/null && caffeinate -dimsu -w $$ &

export BENCH_BUDGET_USD=0 BENCH_OR_BUDGET_USD=0          # the config's exp caps are the guard
export BENCH_TIER1_SKIP_PROVIDERS="" BENCH_TIER1_SKIP_MODELS=""
[ -n "$WORKERS" ] && export BENCH_WORKERS="$WORKERS"
if [ -n "$MAXS" ]; then
  # stop STARTING episodes when elapsed + max(margin, longest episode so far) would pass the limit
  export BENCH_DEADLINE_S="$MAXS" BENCH_DEADLINE_MARGIN_S="${BENCH_DEADLINE_MARGIN_S:-30}"
fi

if [ "$MODE" = status ]; then $PY src/tier1.py --status --config "$CFG"; exit $?; fi

if [ "$SMOKE" = 1 ]; then
  echo "== smoke: 1 Haiku episode (the first planned Part-B episode; it is kept and counts toward R2)"
  BENCH_LIMIT=1 $PY src/tier1.py --config "$CFG" --launch --models claude-haiku-4-5-20251001
  SC=$?
  echo "== smoke exit $SC (0 = key and harness OK)"
  $PY src/tier1.py --status --config "$CFG"
  exit $SC
fi

echo "== plan"
$PY src/tier1.py --config "$CFG" --dry-run
echo "== cost estimate (October logged tokens; see src/r2_estimate.py)"
$PY src/r2_estimate.py --config "$CFG"
$PY - "$CFG" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1]))
print(f"caps: Anthropic exp_budget_usd ${c.get('exp_budget_usd', 0):.2f}; OpenRouter exp_or_budget_usd ${c.get('exp_or_budget_usd', 0):.2f}; "
      f"prompt_cache={c.get('prompt_cache')}; workers={c.get('workers')}")
for g in c["groups"]:
    n = len(g.get("cells", [])) * 32 * c["samples"] + sum((16 if rc[2] == "all" else 8) * c.get("restart_samples", c["samples"]) for rc in g.get("restart_cells", []))
    print(f"  Part {g['part']}: {len(g['models'])} models x {n} = {len(g['models']) * n} episodes   models: {', '.join(g['models'])}")
PYEOF
[ "$MODE" = estimate ] && exit 0

if [ "$YES" != 1 ]; then
  read -r -p "Run R2 now (bills the key in .env)? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) echo "not started"; exit 0 ;; esac
fi

if [ -z "$MAXS" ]; then
  echo "== key check"
  $PY src/tier1.py --check-keys --config "$CFG" || { echo "!! key check failed - stopping"; exit 3; }
fi

run_once() { $PY src/tier1.py --config "$CFG" --launch; return $?; }

if [ -n "$MAXS" ]; then
  echo "== chunk of up to ${MAXS}s  $(date -u +%H:%M:%SZ)"
  run_once; RC=$?
else
  RC=1
  for attempt in 1 2 3 4 5; do
    echo "== attempt $attempt  $(date -u +%H:%M:%SZ)"
    run_once; RC=$?
    case $RC in
      0) break ;;
      2) echo "!! Anthropic cap reached (config exp_budget_usd). Completed episodes are kept."; break ;;
      3) echo "!! OpenRouter stopped (cap / key limit). Completed episodes are kept."; break ;;
    esac
    echo "   some episodes failed (exit $RC); retrying in ${BENCH_TIER1_RETRY_SLEEP:-60}s (completed episodes are kept)"
    sleep "${BENCH_TIER1_RETRY_SLEEP:-60}"
  done
fi

echo
echo "=== summary  $(date -u +%Y-%m-%dT%H:%M:%SZ)"
$PY src/tier1.py --status --config "$CFG"; ST=$?
for f in "$OUT/summary.csv" "$OUT/summary_restart.csv"; do
  if [ -f "$f" ]; then
    echo "--- $f"
    $PY - "$f" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
keys = [k for k in ("model", "prompt", "mitigation", "failure_mode", "resume_context", "n_episodes", "ivr", "ivr_ci_tlo", "ivr_ci_thi",
                    "givr", "givr_ci_glo", "givr_ci_ghi", "readback_after_failure", "session2_lookup", "refusals", "truncated", "cost_usd")
        if rows and k in rows[0]]
print("  ".join(keys))
for r in rows:
    print("  ".join(str(r[k]) for k in keys))
PYEOF
  fi
done
if [ $ST -eq 0 ] && [ "$ONLY" = r2_wording ]; then
  echo "=== all planned episodes done. Analysis (analysis/r2_wording.py -> results/analysis_v3/r2_wording.json):"
  $PY analysis/r2_wording.py | tail -40
  echo "=== ALL DONE. Results in $OUT/ - tell Claude it finished."
  exit 0
fi
case $RC in
  0) echo "=== this pass finished; episodes remain (see status). Rerun the same line to continue."; exit 4 ;;
  2) exit 2 ;;
  *) echo "=== NOT COMPLETE (exit $RC). Rerun the same line to continue; completed episodes are kept."; exit $RC ;;
esac
