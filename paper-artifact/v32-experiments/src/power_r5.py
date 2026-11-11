"""Power and minimum detectable effect (MDE) for the R5 primary test, by
simulation from the paper's same-batch baseline rates. No network.

Primary test (prereg_r5.md, R5-P1): per model, d_t = IVR_U(t) - IVR_G(t) over
the 8 task templates t, where IVR_C(t) is the mean over the three texts of
class C of the text's IVR on template t (4 tasks x r samples each); exact
one-sided sign-flip over the 2^8 sign patterns of the template means,
p = #{s : mean(s*d) <= mean(d)} / 256; reject at p < 0.05.

Generative model (stated in the registration):
  generic class   p_G(t) = mean of the paper's two generic texts on template t,
                  same batch: (bare 500 + timeout) / 2  (Haiku 4.5, Sonnet 5: R2
                  Part B; Qwen3.6-27B, Fable 5: R3-B; results/*/raw.jsonl)
  unknown class   p_U(t) = max(0, p_G(t) - delta)          (MDE search), or
                  p_U(t) = the paper's W3 rate on template t ("paper effect")
  text effect     each of the 3 texts of a class gets p + N(0, sd_text^2) on
                  each template, clipped to [0, 1]; sd_text = 0.10 (the spread
                  of the R2 unknown paraphrases around W3 for Haiku 4.5 is 0.08)
  samples         rho = 0: a task's r samples independent Bernoulli(p);
                  rho = 1: a task's r samples identical (one draw repeated);
                  the truth lies between (identical-pair rates in R2 ~ 0.6-0.9).
MDE = smallest delta on a 0.025 grid with power >= 0.80 (nsim simulations).

  python3 v32-experiments/src/power_r5.py [--nsim 4000] [--write out.json]
Requires numpy (as analysis_v27/v27_common.mde_sim).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.dirname(HERE)
REPO = os.path.dirname(V32)
RES = os.path.join(REPO, "results")

TEMPLATES = ["ambiguous", "calendar", "email", "files", "multi_step", "payments", "read_then_act", "tickets"]
BASIS = {"claude-haiku-4-5-20251001": "r2_wording", "openrouter:qwen/qwen3.6-27b-20260422": "r3_wording",
         "claude-sonnet-5": "r2_wording", "claude-fable-5": "r3_wording"}
SEED = 20261101
SD_TEXT = 0.10
TASKS_PER_TEMPLATE = 4
TEXTS_PER_CLASS = 3
ALPHA = 0.05
POWER = 0.80


def template_rates(model, res_dir=RES) -> dict:
    rows = {}
    with open(os.path.join(res_dir, BASIS[model], "raw.jsonl")) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows[r["episode_id"]] = r
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows.values():
        if r.get("model") != model or r.get("refusal") or r.get("kind") == "restart":
            continue
        if (r.get("prompt"), r.get("mitigation")) != ("base", "none"):
            continue
        acc[r["failure_mode"]][r["template"]].append(1.0 if r["violations"] > 0 else 0.0)
    out = {}
    for fm in ("error_500_after_commit", "timeout_after_commit", "status_unknown_after_commit"):
        out[fm] = [sum(acc[fm][t]) / len(acc[fm][t]) for t in TEMPLATES]
    out["generic"] = [(a + b) / 2 for a, b in zip(out["error_500_after_commit"], out["timeout_after_commit"])]
    return out


def _signs():
    import numpy as np
    return np.array(list(itertools.product([1.0, -1.0], repeat=len(TEMPLATES))))


def exact_p_less(d, signs=None):
    """One-sided exact sign-flip p over template means (d: array (..., 8))."""
    import numpy as np
    S = _signs() if signs is None else signs
    d = np.asarray(d, dtype=float)
    obs = d.mean(axis=-1)
    null = d @ S.T / d.shape[-1]                       # (..., 256)
    return (null <= obs[..., None] + 1e-12).mean(axis=-1)


def simulate_power(pG, pU, r, rho, nsim, rng, signs=None) -> float:
    import numpy as np
    pG, pU = np.asarray(pG, float), np.asarray(pU, float)
    T, K, J = len(pG), TASKS_PER_TEMPLATE, TEXTS_PER_CLASS

    def class_means(p):
        pt = np.clip(p[None, None, :] + rng.normal(0, SD_TEXT, size=(nsim, J, T)), 0, 1)   # text x template
        if rho == 0:
            x = rng.binomial(r, pt[..., None].repeat(K, -1)) / r                           # (nsim, J, T, K)
        else:
            x = (rng.random((nsim, J, T, K)) < pt[..., None]).astype(float)
        return x.mean(axis=(1, 3))                                                        # (nsim, T)
    d = class_means(pU) - class_means(pG)
    return float((exact_p_less(d, signs) < ALPHA).mean())


def mde(pG, r, rho, nsim, seed=SEED, step=0.025):
    import numpy as np
    S = _signs()
    k = 1
    while True:
        delta = round(step * k, 4)
        if delta > 1.0:
            return None, None
        rng = np.random.default_rng(seed)
        pU = np.clip(np.asarray(pG) - delta, 0, 1)
        pw = simulate_power(pG, pU, r, rho, nsim, rng, S)
        if pw >= POWER:
            return delta, pw
        k += 1


def run(nsim=4000, models=None, res_dir=RES) -> dict:
    import numpy as np
    out = {"seed": SEED, "nsim": nsim, "sd_text": SD_TEXT, "alpha": ALPHA, "power_target": POWER,
           "test": "exact one-sided sign-flip over 8 template means (256 patterns)", "models": {}}
    for m in models or list(BASIS):
        tr = template_rates(m, res_dir)
        pG = tr["generic"]
        res = {"basis": BASIS[m], "p_generic_by_template": [round(x, 3) for x in pG],
               "p_generic_mean": round(sum(pG) / len(pG), 3),
               "p_w3_by_template": [round(x, 3) for x in tr["status_unknown_after_commit"]]}
        for r in (1, 2):
            for rho in (0, 1):
                d, pw = mde(pG, r, rho, nsim)
                rng = np.random.default_rng(SEED + 1)
                paper = simulate_power(pG, tr["status_unknown_after_commit"], r, rho, nsim, rng)
                res[f"r{r}_rho{rho}"] = {"mde_delta": d, "power_at_mde": None if pw is None else round(pw, 3),
                                         "power_paper_effect": round(paper, 3)}
        out["models"][m] = res
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nsim", type=int, default=4000)
    ap.add_argument("--write", default="")
    a = ap.parse_args(argv)
    out = run(a.nsim)
    if a.write:
        with open(a.write, "w") as f:
            json.dump(out, f, indent=1)
    for m, res in out["models"].items():
        print(f"{m}: p_G mean {res['p_generic_mean']} (templates {res['p_generic_by_template']})")
        for k in ("r2_rho0", "r2_rho1", "r1_rho0", "r1_rho1"):
            v = res[k]
            print(f"   {k}: MDE delta {v['mde_delta']} (power {v['power_at_mde']}); "
                  f"power at the paper's W3 effect {v['power_paper_effect']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
