#!/usr/bin/env bash
# Reproduce every v3 analysis of the Tier-1 runs from the raw data.
#   bash analysis/run_all_v3.sh
# Python >= 3.9, standard library only, no network, no API calls.
# Reads results/ (read-only) except that the C1 re-score step writes
# results/**/<raw>_v2.jsonl next to any raw file whose rows it changes (the
# original raw file is never edited); everything else goes to results/analysis_v3/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
mkdir -p results/analysis_v3
echo "== C1 re-score: replay all episodes under the fixed matcher (ISO date-time seconds)"
$PY src/rescore_c1.py > /dev/null
echo "== E0-3 matcher audit over every episode"
$PY analysis/v3_matcher_audit_all.py > /dev/null
echo "== pre-registered E1-0 .. E1-6"
$PY analysis/v3_tier1_prereg.py > /dev/null
echo "== bare_hint vs same-month baselines"
$PY analysis/v3_bare_hint.py > /dev/null
echo "== E1-6 belief coding (rule-based; provisional until hand-check)"
$PY analysis/v3_e1_6_belief.py > /dev/null
echo "== E1-6 coder vs hand-check agreement (PENDING until human_code is filled)"
$PY analysis/v3_e1_6_kappa.py
echo "== SENSITIVITY (added after results, not confirmatory): E1-6 vs October bare baselines"
$PY analysis/v3_sens_oct.py
$PY analysis/v3_sens_oct_fix.py
if [ -f results/real_stripe/raw.jsonl ]; then
  echo "== OPTIONAL: real-API validation slice (Stripe test mode, preregistration R1)"
  $PY analysis/real_stripe.py
  echo "== EXPLORATORY (post hoc, not registered): real slice vs payment-only simulator cells (real_extra.json)"
  $PY analysis/v3_real_extra.py > /dev/null
fi
echo "== POST HOC extras for the iteration-1 writing review (not registered; iter1_extra.json)"
$PY analysis/v3_iter1_extra.py > /dev/null
echo "== numbers.json"
$PY analysis/v3_numbers.py
echo "== done: results/analysis_v3/"
