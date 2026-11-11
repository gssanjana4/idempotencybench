"""Merge the v3 analysis outputs into results/analysis_v3/numbers.json.

Schema (read by paper-merged/make_numbers.py or any other consumer):
  {"meta": {...},
   "numbers": {name: {"value": v, "ci": [lo, hi]?, "ci_method": str?,
                      "n": int?, "k": int?, "T"/"G": int?, "source": [paths],
                      "status": str?, "verdict": str?, "criterion": str?, ...}}}
Name prefixes:
  <src>.<model>.<prompt>.<mit>.<fm>:ivr   cell IVR, ci = task-level Wilson (primary)
      src in v2 (released July, v2 rescore), xp (run_crossprovider.sh, last
      row per id), e1_0..e1_6 (Tier-1 runs); :ivr_ci_ep, :ivr_ci_boot,
      :ivr_strict, :ivr_task_any, :readback_after_failure, :refusals, :ivr_bound
  e1_X.P.* / e1_X.S.* / e1_X.X.*   pre-registered primary / secondary /
      exploratory tests and estimates (:d, :p, :p holm_p, :d_bound, :p_bound)
  sens.first_row.*                 first-row dedupe sensitivity (Qwen duplicates)
  refusal.*, hygiene.*             refusal counts, data checks
  bh.*                             bare_hint vs same-month baselines (249)
  e1_6.belief.*                    PROVISIONAL belief coding (+ kappa when done)
  real.*                           real-API validation slice on Stripe test mode
      (preregistration R1; present only once results/real_stripe/ exists)
  sens.oct.*                       SENSITIVITY added after the results (not
      confirmatory): E1-6 contrasts against October bare baselines
      (results/tier1/e1_6_oct/); present only once that run exists
Values are raw (proportions in [0, 1], p-values unrounded); formatting is the
consumer's job.
"""
import hashlib, json, os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import OUT, ROOT, PERM_N, PERM_SEED, SAMPLE_SEED

PARTS = ["tier1_prereg.json", "bare_hint.json", "e1_6_belief.json", "matcher_audit_all.json"]
OPTIONAL_PARTS = ["sens_oct.json", "sens_oct_fix.json",   # only when their data exists (v3_sens_oct.py, v3_sens_oct_fix.py)
                  "real_stripe.json"]   # real-API slice (analysis/real_stripe.py, preregistration R1)
INPUTS = ["results/sweep_raw_v2.jsonl", "results/sweep_raw.jsonl", "results/sweep2_raw_v2.jsonl",
          "results/crossprovider/sweep_raw.jsonl", "results/crossprovider/sweep2_raw.jsonl",
          "results/crossprovider/run.log", "results/claude_batch_2026-10/sweep2_raw.jsonl",
          "results/tier1/run.log"] + [f"results/tier1/e1_{i}/raw.jsonl" for i in range(7)] + \
         [f"results/tier1/e1_{i}/imported.jsonl" for i in (1, 3, 6)] + \
         ["results/tier1/e1_6_oct/raw.jsonl"]   # sensitivity; hashed only if present
# C1 re-score outputs (src/rescore_c1.py), read in place of raw.jsonl when present
INPUTS += [p[:-len(".jsonl")] + "_v2.jsonl" for p in INPUTS if p.startswith("results/tier1/") and p.endswith("/raw.jsonl")]
INPUTS += ["results/analysis_v3/rescore_c1.csv"]
INPUTS += ["results/real_stripe/raw.jsonl", "results/real_stripe/stripe_ground_truth.jsonl",
           "results/real_stripe/search_check.jsonl"]   # hashed only if present


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def main():
    nums = {}
    for p in PARTS + [q for q in OPTIONAL_PARTS if os.path.exists(os.path.join(OUT, q))]:
        d = json.load(open(os.path.join(OUT, p)))
        clash = set(d) & set(nums)
        assert not clash, clash
        nums.update(d)
    kp = os.path.join(OUT, "e1_6_kappa.json")
    if os.path.exists(kp):
        k = json.load(open(kp))
        nums["e1_6.belief.kappa"] = {"value": k.get("kappa_6way"), "n": k.get("n"),
                                     "status": k.get("status"), "kappa_binary": k.get("kappa_binary"),
                                     "agreement": k.get("agreement"), "source": ["results/analysis_v3/e1_6_kappa.json"]}
    try:
        head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        head = None
    meta = {"generated_by": "analysis/run_all_v3.sh", "preregistration": "analysis/preregistration.md",
            "git_head_at_run": head, "perm_n": PERM_N, "perm_seed": PERM_SEED, "handcheck_seed": SAMPLE_SEED,
            "primary_ci": "task-level Wilson(p*T, T), rho = 1 (graph-level for bare_hint)",
            "dedupe_rule": "last row per episode_id (e78e23d); first-row values under sens.first_row.*",
            "refusal_rule": "provider block = own outcome; primary estimates exclude; *:ivr_bound and "
                            "*:d_bound/*:p_bound give refusal-as-no-violation / as-violation",
            "input_sha256_16": {p: sha(p) for p in INPUTS if os.path.exists(os.path.join(ROOT, p))},
            "n_numbers": len(nums)}
    with open(os.path.join(OUT, "numbers.json"), "w") as fh:
        json.dump({"meta": meta, "numbers": dict(sorted(nums.items()))}, fh, indent=1)
    print(f"numbers.json: {len(nums)} values")


if __name__ == "__main__":
    main()
