"""Tests added by the code-verifier loop (iteration 2): each kills a mutant of the
code added since iteration 1 (analysis/real_stripe.py, analysis/v3_real_extra.py,
analysis/v3_iter1_extra.py, the R1/cost/PII sections of paper-merged/make_numbers.py,
src_real/) that survived every existing suite. Run from the repo root:
  python3 tests_mutants2.py"""
import glob, importlib.util, json, os, shutil, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("analysis", "src", "src2", "src_real"):
    sys.path.insert(0, os.path.join(ROOT, d))
import v3_common                                   # noqa: E402
from v3_common import Numbers, wilson              # noqa: E402
import v3_bare_hint as bh                          # noqa: E402
import v3_load                                     # noqa: E402
import v3_real_extra as xr                         # noqa: E402
import v3_iter1_extra as x1                        # noqa: E402
import real_stripe                                 # noqa: E402
import run_real, sweep2                            # noqa: E402
from world import Goal, ToolServerError            # noqa: E402
from stripe_world import StripeWorld, stripe_ground_truth   # noqa: E402

_spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(ROOT, "paper-merged", "make_numbers.py"))
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)

HAIKU, SONNET, QWEN = "claude-haiku-4-5-20251001", "claude-sonnet-5", "openrouter:qwen/qwen3.6-27b-20260422"


def jl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def jd(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    return path


def retry_row(i, model=HAIKU, mit="none", fm="error_500_after_commit", ivr=False, **kw):
    r = {"episode_id": f"{model}__base__{mit}__{fm}__rpay-{i // 2:02d}__s{i % 2}", "kind": "retry", "model": model,
         "prompt": "base", "mitigation": mit, "failure_mode": fm, "task_id": f"rpay-{i // 2:02d}", "sample": i % 2,
         "ivr_stripe": ivr, "ivr_ledger": ivr, "violations_stripe": int(ivr), "violations_ledger": int(ivr),
         "agree": True, "agree_exact": True, "refusal": False, "failure_fired": True, "readback_after_failure": False,
         "replayed": 0, "forwarded_keys": 0, "stripe_errors": 0, "cost_usd": 0.01, "attempt_id": f"att{i}",
         "key_attached_first": False, "retried": True, "truncated": False, "turn_cap_hit": False}
    r.update(kw)
    return r


def restart_row(i, model=HAIKU, ctx="amnesiac", ivr=False, **kw):
    r = {"episode_id": f"{model}__restart__{ctx}__none__rplan-{i // 2:02d}__k0__s{i % 2}", "kind": "restart",
         "model": model, "prompt": "restart", "resume_context": ctx, "mitigation": "none",
         "failure_mode": "restart_after_commit", "task_id": f"rplan-{i // 2:02d}", "sample": i % 2,
         "ivr_stripe": ivr, "ivr_ledger": ivr, "violations_stripe": int(ivr), "violations_ledger": int(ivr),
         "agree": True, "agree_exact": True, "refusal": False, "interrupted": True, "s2_lookup": False,
         "cost_usd": 0.02, "attempt_id": f"ratt{i}"}
    r.update(kw)
    return r


# ---------------------------------------------------------------------------
# analysis/real_stripe.py  (R1: Table 11, section 7.3)
# ---------------------------------------------------------------------------
class RealStripeBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name
        self.cfg = {"models": [HAIKU, SONNET], "n_tasks": 3, "samples": 2, "expect_per_cell": 6,
                    "raw_path": os.path.join(d, "raw.jsonl"), "gt_path": os.path.join(d, "gt.jsonl"),
                    "out_dir": d, "search_path": os.path.join(d, "search_check.jsonl"),
                    "sim_compare": {HAIKU: {"base.none.error_500_after_commit": "sim.haiku.e500",
                                            "restart.amnesiac": "sim.haiku.amn"},
                                    SONNET: {"base.none.error_500_after_commit": "sim.sonnet.e500"}}}
        self.out = os.path.join(d, "real_stripe.json")

    def tearDown(self):
        self.tmp.cleanup()

    def build(self, rows, sim, search=None):
        jl(self.cfg["raw_path"], rows); jl(self.cfg["gt_path"], [])
        jl(self.cfg["search_path"], search or [])
        return real_stripe.build(self.cfg, sim_numbers=sim, out_path=self.out)

    def test_cell_stats_use_stripe_verdict_task_level_ci_and_exclude_refusals(self):
        # survivors RSA01 (refusals counted), RSA05 (k from the shadow ledger), RSA08 (T = n), RSA04 (episode CI)
        rows = [retry_row(i, ivr=(i in (0, 1, 2))) for i in range(6)]
        rows[5]["refusal"] = True; rows[5]["ivr_stripe"] = True         # refused: not scored
        rows[3]["ivr_ledger"] = True                                    # ledger disagrees: k follows Stripe
        rows[3]["agree"] = False
        N = self.build(rows, {})
        e = N["real.haiku.base.none.e500:ivr"]
        self.assertEqual((e["n"], e["k"], e["T"], e["refusals"]), (5, 3, 3, 1))
        self.assertAlmostEqual(e["value"], 0.6)
        self.assertEqual(e["ci"], list(wilson(0.6 * 3, 3)))
        self.assertNotEqual(e["ci"], list(wilson(3, 5)))
        self.assertEqual(e["ci_ep"], list(wilson(3, 5)))
        self.assertEqual(N["real.haiku.base.none.e500:ivr_ledger"]["k"], 4)
        self.assertEqual(N["real.haiku.base.none.e500:agree"]["k"], 4)
        self.assertEqual(N["real.agreement.ledger_vs_stripe"]["k"], 4)
        self.assertEqual(N["real.agreement.ledger_vs_stripe"]["n"], 5)

    def test_within_is_real_point_inside_simulator_interval_inclusive(self):
        # survivors RSA03 (simulator value inside the real interval instead), RSA02 (strict bounds), RSA11 (sign of d)
        rows = [retry_row(i, ivr=(i < 2)) for i in range(6)]          # p = 1/3; real CI is wide
        sim = {"sim.haiku.e500": {"value": 0.6, "ci": [0.5, 0.7]}}
        e = self.build(rows, sim)["real.haiku.base.none.e500:ivr"]
        self.assertFalse(e["within_sim_ci"]); self.assertAlmostEqual(e["d"], 1 / 3 - 0.6)
        self.assertTrue(e["ci"][0] <= 0.6 <= e["ci"][1])            # the swapped test would say yes
        for ci in ([1 / 3, 0.9], [0.0, 1 / 3]):                       # real p exactly on a bound counts as inside
            sim = {"sim.haiku.e500": {"value": 0.6, "ci": ci}}
            self.assertTrue(self.build(rows, sim)["real.haiku.base.none.e500:ivr"]["within_sim_ci"], ci)

    def test_h1_counts_restart_cells_and_simulator_cells_are_matched_per_model(self):
        # survivors RSA06 (restart cells dropped from H1), RSA15 (every model compared to the first model's cells)
        rows = [retry_row(i, ivr=True) for i in range(6)] + [retry_row(i, model=SONNET, ivr=False) for i in range(6)] \
            + [restart_row(i, ivr=True) for i in range(6)]
        sim = {"sim.haiku.e500": {"value": 0.9, "ci": [0.7, 1.0]}, "sim.sonnet.e500": {"value": 0.1, "ci": [0.0, 0.3]},
               "sim.haiku.amn": {"value": 0.95, "ci": [0.8, 1.0]}}
        N = self.build(rows, sim)
        self.assertEqual((N["real.H1.within"]["value"], N["real.H1.within"]["n"]), (3, 3))
        self.assertEqual(N["real.sonnet.base.none.e500:ivr"]["sim_key"], "sim.sonnet.e500")
        self.assertEqual(N["real.haiku.restart.amnesiac:givr"]["sim_key"], "sim.haiku.amn")
        self.assertTrue(N["real.haiku.restart.amnesiac:givr"]["within_sim_ci"])
        sim["sim.sonnet.e500"] = {"value": 0.9, "ci": [0.7, 1.0]}       # Sonnet now outside, Haiku still inside
        N = self.build(rows, sim)
        self.assertEqual(N["real.H1.within"]["value"], 2)
        self.assertFalse(N["real.sonnet.base.none.e500:ivr"]["within_sim_ci"])

    def test_last_row_per_episode_wins_and_readback_denominator_is_fired_episodes(self):
        # survivors RSA07 (first row kept), RSA13 (read-back over every episode)
        rows = [retry_row(i, ivr=False, readback_after_failure=(i < 2)) for i in range(6)]
        rows[4]["failure_fired"] = False; rows[5]["failure_fired"] = False
        rows.append(dict(rows[0], ivr_stripe=True, violations_stripe=1))   # rerun of episode 0: last row counts
        N = self.build(rows, {})
        e = N["real.haiku.base.none.e500:ivr"]
        self.assertEqual((e["n"], e["k"]), (6, 1))
        self.assertAlmostEqual(N["real.haiku.base.none.e500:readback_after_failure"]["value"], 2 / 4)

    def test_search_check_counts_only_the_scored_attempt(self):
        # survivor RSA09: rows of a crashed earlier attempt entered the Search-API agreement count
        rows = [retry_row(i) for i in range(6)]
        search = [{"episode_id": r["episode_id"], "attempt_id": r["attempt_id"], "agree": True} for r in rows]
        search += [{"episode_id": "never-finished-episode", "attempt_id": "stale", "agree": False}]
        N = self.build(rows, {}, search)
        self.assertEqual((N["real.search_check.agree"]["k"], N["real.search_check.agree"]["n"]), (6, 6))

    def test_payments_subset_counts_any_duplicate(self):
        # survivor RSA14: violations > 1 instead of > 0 in the descriptive payment-only subset
        d = self.tmp.name
        src = jl(os.path.join(d, "sim_raw.jsonl"),
                 [{"episode_id": f"e{i}", "model": HAIKU, "prompt": "base", "mitigation": "none",
                   "failure_mode": "error_500_after_commit", "task_id": f"payments-{i}", "violations": 1} for i in range(4)]
                 + [{"episode_id": "o", "model": HAIKU, "prompt": "base", "mitigation": "none",
                     "failure_mode": "error_500_after_commit", "task_id": "email-1", "violations": 3}])
        old = real_stripe.ROOT
        try:
            real_stripe.ROOT = d
            ps = real_stripe.payments_subset({"source": [os.path.relpath(src, d)]}, HAIKU, "base", "none",
                                             "error_500_after_commit")
        finally:
            real_stripe.ROOT = old
        self.assertEqual((ps["k"], ps["n"], ps["T"]), (4, 4, 4))


# ---------------------------------------------------------------------------
# analysis/v3_real_extra.py  (exploratory payment-only comparison, Table 11 lower block)
# ---------------------------------------------------------------------------
class RealExtra(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = self.tmp.name
        self.saved = {k: getattr(xr, k) for k in ("ROOT", "OUT", "REAL_RAW", "REAL_JSON", "N")}
        self.saved_bh = (bh.OCT, bh.XP)
        xr.ROOT = self.d; xr.OUT = os.path.join(self.d, "analysis_v3")
        xr.REAL_RAW = os.path.join(self.d, "real_stripe", "raw.jsonl")
        xr.REAL_JSON = os.path.join(xr.OUT, "real_stripe.json")
        xr.N = Numbers()
        bh.OCT = os.path.join(self.d, "oct"); bh.XP = os.path.join(self.d, "xp")

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(xr, k, v)
        bh.OCT, bh.XP = self.saved_bh
        self.tmp.cleanup()

    def sim_row(self, i, model=HAIKU, prompt="base", mit="none", fm="error_500_after_commit", tmpl="payments", v=1, **kw):
        r = {"episode_id": f"{model}__{prompt}__{mit}__{fm}__{tmpl}-{i:02d}__s0", "model": model, "prompt": prompt,
             "mitigation": mit, "failure_mode": fm, "task_id": f"{tmpl}-{i:02d}", "violations": v, "sample": 0}
        r.update(kw)
        return r

    def test_payment_subset_filters_every_factor(self):
        # survivors XR01 (prompt), XR02 (mitigation), XR03 (failure mode), XR04 (template), XR05 (refusal), XR06 (model)
        rows = [self.sim_row(i, v=1) for i in range(4)]                       # the cell: 4/4
        rows += [self.sim_row(i, prompt="hint", v=0) for i in range(4)]       # other prompt
        rows += [self.sim_row(i, mit="idem_keys", v=0) for i in range(4)]     # other mitigation
        rows += [self.sim_row(i, fm="timeout_after_commit", v=0) for i in range(4)]
        rows += [self.sim_row(i, tmpl="email", v=0) for i in range(4)]        # other template
        rows += [self.sim_row(i, model=SONNET, v=0) for i in range(4)]        # other model
        rows += [self.sim_row(9, v=0, refusal=True)]                           # refused
        src = jl(os.path.join(self.d, "sim_raw.jsonl"), rows)
        jd(os.path.join(xr.OUT, "tier1_prereg.json"), {"v2.haiku.base.none.e500:ivr": {"source": [os.path.relpath(src, self.d)]}})
        jd(os.path.join(xr.OUT, "bare_hint.json"), {})
        jd(xr.REAL_JSON, {"real.haiku.base.none.e500:ivr": {"value": 0.5, "k": 16, "n": 32, "model": HAIKU,
                                                      "sim_key": "v2.haiku.base.none.e500:ivr"}})
        xr.payments()
        e = xr.N.d["xr.pay.haiku.none.e500:ivr"]
        self.assertEqual((e["k"], e["n"], e["T"]), (4, 4, 4))
        self.assertEqual(e["ci"], [round(c, 6) for c in wilson(4, 4)])
        self.assertFalse(e["within"]); self.assertAlmostEqual(e["d"], -0.5)
        self.assertEqual(xr.N.d["xr.pay.within"]["value"], 0)

    def test_within_inclusive_at_the_bound(self):
        # survivor XR13: strict inequality flipped 5 "within" cells and xr.rst.within 6 -> 1
        xr.put_cell("xr.t:ivr", 0, 4, 4, {"value": 0.0, "k": 0, "n": 32}, [], "task", "")
        self.assertTrue(xr.N.d["xr.t:ivr"]["within"])
        xr.put_cell("xr.u:ivr", 4, 4, 4, {"value": 1.0, "k": 32, "n": 32}, [], "task", "")
        self.assertTrue(xr.N.d["xr.u:ivr"]["within"])

    def rst_row(self, model, ctx, gid, k, s, v, topo="chain4"):
        return {"episode_id": f"{model}__{ctx}__none__{gid}__k{k}__s{s}", "model": model, "resume_context": ctx,
                "mitigation": "none", "key_posture": "model", "interrupt_timing": "after_commit", "topology": topo,
                "graph_id": gid, "interrupt_node": k, "sample": s, "givr_hit": v, "violations": v}

    def test_restart_subset_is_chain4_cut_after_node_0_only(self):
        # survivors XR07 (every cut point), XR08 (any chain topology; equivalent on the released data)
        real = {}
        for m in ("haiku", "sonnet", "qwen"):
            for ctx in ("amnesiac", "bare_hint"):
                real[f"real.{m}.restart.{ctx}:givr"] = {"value": 1.0, "k": 32, "n": 32}
        jd(xr.REAL_JSON, real)
        for model, d in ((HAIKU, bh.OCT), (SONNET, bh.OCT), (QWEN, bh.XP)):
            rows = []
            for ctx in ("amnesiac", "bare_hint"):
                for g in ("chain4-a", "chain4-b"):
                    rows += [self.rst_row(model, ctx, g, 0, s, 1) for s in (0, 1)]          # the subset: 4/4
                    rows += [self.rst_row(model, ctx, g, k, 0, 0) for k in (1, 2, 3)]       # other cut points
                rows += [self.rst_row(model, ctx, "fanout-a", 0, 0, 0, "fanout")]            # other topology
            jl(os.path.join(d, "sweep2_raw.jsonl"), rows if d == bh.XP else rows)
        # both Claude models share the October file: write them together
        rows = []
        for model in (HAIKU, SONNET):
            for ctx in ("amnesiac", "bare_hint"):
                for g in ("chain4-a", "chain4-b"):
                    rows += [self.rst_row(model, ctx, g, 0, s, 1) for s in (0, 1)]
                    rows += [self.rst_row(model, ctx, g, k, 0, 0) for k in (1, 2, 3)]
                rows += [self.rst_row(model, ctx, "fanout-a", 0, 0, 0, "fanout")]
        jl(os.path.join(bh.OCT, "sweep2_raw.jsonl"), rows)
        os.makedirs(os.path.join(bh.OCT, "transcripts2"), exist_ok=True); os.makedirs(os.path.join(bh.XP, "transcripts2"), exist_ok=True)
        xr.restart()
        for m in ("haiku", "sonnet", "qwen"):
            e = xr.N.d[f"xr.rst.{m}.amnesiac:givr"]
            self.assertEqual((e["k"], e["n"], e["G"]), (4, 4, 2), m)
            self.assertTrue(e["within"])
        self.assertEqual(xr.N.d["xr.rst.within"]["value"], 6)

    def test_real_contrast_is_paired_by_task_over_the_keyless_cells_with_stripe_truth(self):
        # survivors XR09 (paired by episode: T = 0), XR10 (keyed cell pooled into the 500 arm),
        # XR15 (shadow-ledger verdict instead of Stripe's), XR11 (key-attach count from the retry flag), XR14
        rows = []
        for i in range(8):
            rows.append(retry_row(i, fm="timeout_after_commit", ivr=False, retried=True))
            rows.append(retry_row(i, fm="error_500_after_commit", ivr=(i % 2 == 0), retried=True))
            rows.append(retry_row(i, mit="idem_keys", fm="error_500_after_commit", ivr=True, retried=True,
                                  key_attached_first=(i < 3)))
        for r in rows:
            r["ivr_ledger"] = not r["ivr_stripe"]                    # the ledger is wrong on purpose
        rows += [restart_row(i) for i in range(4)]
        jl(xr.REAL_RAW, rows)
        xr.real_contrast(); xr.hygiene()
        d = xr.N.d["xr.real.haiku.e500_minus_toA:d"]
        self.assertEqual(d["n"], 4); self.assertAlmostEqual(d["value"], 0.5)
        self.assertEqual((d["plus"], d["minus"]), (4, 0))
        self.assertEqual(xr.N.d["xr.real.haiku.keys.e500:keyfirst"]["value"], "3/8")
        self.assertEqual(xr.N.d["xr.real.hygiene"]["failure_fired"], "24/24")

    def test_outside_direction_ignores_cells_without_a_simulator_number(self):
        # survivor XR12 (crashed the pipeline on None values; keep it explicit)
        jd(xr.REAL_JSON, {"a:ivr": {"value": 0.9, "sim_value": 0.5, "within_sim_ci": False},
                          "b:ivr": {"value": 0.1, "sim_value": 0.5, "within_sim_ci": False},
                          "c:ivr": {"value": 0.5, "sim_value": 0.5, "within_sim_ci": True},
                          "d:ivr": {"value": 0.0, "sim_value": None, "within_sim_ci": None}})
        xr.outside_direction()
        e = xr.N.d["xr.h1.outside"]
        self.assertEqual((e["value"], e["above"], e["below"]), (2, 1, 1))


# ---------------------------------------------------------------------------
# analysis/v3_iter1_extra.py  (misorder denominators, July vs October, key-naming retries,
# read-back contrast, per-template IVR)
# ---------------------------------------------------------------------------
class Iter1Extra(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = self.tmp.name
        self.saved = {k: getattr(x1, k) for k in ("RES", "OUT", "V2_RAW", "XP_RAW", "N")}
        self.saved_bh = (bh.OCT, bh.XP, bh.JUL_RAW)
        x1.RES = self.d; x1.OUT = os.path.join(self.d, "analysis_v3")
        x1.V2_RAW = os.path.join(self.d, "sweep_raw_v2.jsonl")
        x1.XP_RAW = os.path.join(self.d, "crossprovider", "sweep_raw.jsonl")
        x1.N = Numbers()
        bh.OCT = os.path.join(self.d, "claude_batch_2026-10"); bh.XP = os.path.join(self.d, "crossprovider")
        bh.JUL_RAW = os.path.join(self.d, "sweep2_raw_v2.jsonl")
        for d in (bh.OCT, bh.XP, self.d):
            os.makedirs(os.path.join(d, "transcripts2"), exist_ok=True)

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(x1, k, v)
        bh.OCT, bh.XP, bh.JUL_RAW = self.saved_bh
        self.tmp.cleanup()

    def core_row(self, model, fm, fam, i, mis, **kw):
        r = {"episode_id": f"{model}__base__none__{fm}__{fam}-{i}__s0", "model": model, "prompt": "base", "mitigation": "none",
             "failure_mode": fm, "family": fam, "task_id": f"{fam}-{i}", "sample": 0, "violations": 0, "misorder": mis}
        r.update(kw)
        return r

    def test_misorder_denominator_is_two_step_timeout_before_episodes_of_the_core_grid(self):
        # survivors X01 (every family), X02 (every failure mode), X14 (Claude file only); X03 is equivalent
        v2, xp = [], []
        for model, dest in ((HAIKU, v2), (SONNET, v2), ("claude-fable-5", v2), ("openrouter:openai/gpt-5.5-20260423", xp),
                            ("openrouter:google/gemini-4-argon-20261013", xp), (QWEN, xp)):
            dest += [self.core_row(model, "timeout_before_commit", "multi_step", i, i == 0) for i in range(8)]
            dest += [self.core_row(model, "timeout_before_commit", "atomic", i, False) for i in range(8)]
            dest += [self.core_row(model, "error_500_after_commit", "multi_step", i, False) for i in range(8)]
        v2.append(self.core_row(QWEN, "timeout_before_commit", "multi_step", 99, True))   # non-Claude row in the Claude file
        jl(x1.V2_RAW, v2); jl(x1.XP_RAW, xp)
        x1.misorder()
        for m in ("haiku", "sonnet", "fable", "gpt", "gemini", "qwen"):
            self.assertEqual((x1.N.d[f"x.misorder.{m}"]["k"], x1.N.d[f"x.misorder.{m}"]["n"]), (1, 8), m)
        self.assertEqual((x1.N.d["x.misorder.all"]["k"], x1.N.d["x.misorder.all"]["n"]), (6, 48))
        self.assertTrue(x1.N.d["x.misorder.all"]["all_toB_twostep"])

    def rst_row(self, model, ctx, gid, s, v):
        return {"episode_id": f"{model}__{ctx}__none__{gid}__k1__s{s}", "model": model, "resume_context": ctx,
                "mitigation": "none", "key_posture": "model", "interrupt_timing": "after_commit", "topology": "chain4",
                "graph_id": gid, "interrupt_node": 1, "sample": s, "givr_hit": v, "violations": v}

    def test_amnesiac_month_is_october_minus_july_amnesiac(self):
        # survivors X04 (sign), X05 (October bare_hint in place of amnesiac)
        jul, octo = [], []
        for model, _ in bh.MODELS:
            for g in range(16):
                jul += [self.rst_row(model, "amnesiac", f"g{g}", s, 1) for s in (0, 1)]           # July: 1.0
                octo += [self.rst_row(model, "amnesiac", f"g{g}", s, int(g < 8)) for s in (0, 1)]  # October: 0.5
                octo += [self.rst_row(model, "bare_hint", f"g{g}", s, 0) for s in (0, 1)]         # hint: 0.0
        jl(bh.JUL_RAW, jul); jl(os.path.join(bh.OCT, "sweep2_raw.jsonl"), octo)
        x1.amnesiac_month()
        for _, sm in bh.MODELS:
            e = x1.N.d[f"x.amn_month.{sm}:d"]
            self.assertAlmostEqual(e["value"], -0.5, msg=sm)
            self.assertEqual((e["plus"], e["minus"], e["n"]), (0, 8, 16))
            self.assertLess(x1.N.d[f"x.amn_month.{sm}:p"]["value"], 0.05)

    def tr(self, name, blocks_by_turn):
        turns = []
        for blocks in blocks_by_turn:
            turns.append({"role": "assistant", "content": blocks})
            turns.append({"role": "tool_results", "content": []})
        p = os.path.join(self.d, "sweep_transcripts", name)
        jd(p, {"turns": turns})

    def call(self, memo="m1", key=None, tool="send_payment", **over):
        args = {"payee": "v", "amount": "10.00", "memo": memo}
        args.update(over)
        if key is not None:
            args["idempotency_key"] = key
        return {"type": "tool_use", "id": memo + str(key), "name": tool, "input": args}

    def text(self, s):
        return {"type": "text", "text": s}

    def test_key_naming_retries_count_text_that_names_the_key_and_key_reuse(self):
        # survivors X06 (k = any same-key retry), X07 (n = any text), X13 (retry = any later effectful call)
        pat = "claude-fable-5_base_none_error_500_after_commit__s0__payments-%d.json"
        self.tr(pat % 0, [[self.call(key="A")], [self.text("The idempotency key protects the retry."), self.call(key="A")]])   # k, n
        self.tr(pat % 1, [[self.call(key="A")], [self.text("Retrying with a fresh idempotency key."), self.call(key="B")]])    # n only
        self.tr(pat % 2, [[self.call(key="A")], [self.text("Retrying now."), self.call(key="A")]])                            # same key, key not named
        self.tr(pat % 3, [[self.call(key="A")], [self.call(key="A")]])                                                         # no text
        # the retry is the first later call with the SAME identity; an unrelated payment in between does not count
        self.tr(pat % 4, [[self.call(key="A")], [self.text("Paying the other invoice first."), self.call(memo="other", key="C")],
                          [self.text("Now the idempotency key is reused."), self.call(key="A")]])                            # k, n
        self.tr(pat % 5, [[self.call(key="A")], [self.text("done")]])                                                           # no retry
        x1.key_naming()
        e = x1.N.d["x.rtext.fable.e500"]
        self.assertEqual((e["k"], e["n"]), (2, 3))
        self.assertEqual((e["retries"], e["text"], e["same_key"]), (5, 4, 4))

    def test_readback_contrast_is_500_minus_timeout_on_the_base_prompt(self):
        # survivors X08 (sign), X09 (other prompts pooled in)
        cells = {}
        for m, src in (("haiku", "v2"), ("gpt", "xp")):
            model = {"haiku": HAIKU, "gpt": "openrouter:openai/gpt-5.5-20260423"}[m]
            cells[(src, model, "base", "none", "timeout_after_commit")] = [{"task": f"t{i}", "rb": 0} for i in range(8)]
            cells[(src, model, "base", "none", "error_500_after_commit")] = [{"task": f"t{i}", "rb": 1} for i in range(8)]
            cells[(src, model, "hint", "none", "timeout_after_commit")] = [{"task": f"t{i}", "rb": 1} for i in range(8)]
            cells[(src, model, "hint", "none", "error_500_after_commit")] = [{"task": f"t{i}", "rb": 0} for i in range(8)]
        old = v3_load.load_all
        try:
            v3_load.load_all = lambda: (cells, {})
            x1.readback_contrast()
        finally:
            v3_load.load_all = old
        for m in ("haiku", "gpt"):
            e = x1.N.d[f"x.rb500.{m}:d"]
            self.assertAlmostEqual(e["value"], 1.0, msg=m); self.assertEqual((e["plus"], e["minus"], e["n"]), (8, 0, 8))

    def test_e1_4_templates_pool_new_tasks_with_sample_0_of_the_released_ones(self):
        # survivors X10 (both released samples pooled), X11 (violations > 1)
        tm = ["ambiguous", "calendar", "email", "files", "multi_step", "payments", "read_then_act", "tickets"]
        new, rel = [], []
        for m in (HAIKU, SONNET):
            for fm in ("timeout_after_commit", "error_500_after_commit"):
                for t in tm:
                    new += [{"episode_id": f"{m}__base__none__{fm}__{t}-n{i}__s0", "model": m, "prompt": "base", "mitigation": "none",
                             "failure_mode": fm, "task_id": f"{t}-n{i}", "sample": 0, "violations": 1, "template": t} for i in range(8)]
                    for i in range(4):
                        for s in (0, 1):
                            rel.append({"episode_id": f"{m}__base__none__{fm}__{t}-r{i}__s{s}", "model": m, "prompt": "base",
                                        "mitigation": "none", "failure_mode": fm, "task_id": f"{t}-r{i}", "sample": s,
                                        "violations": 1 if s == 0 and i == 0 else 0})
        jl(os.path.join(self.d, "tier1", "e1_4", "raw.jsonl"), new); jl(x1.V2_RAW, rel)
        x1.e1_4_templates()
        e = x1.N.d["x.e1_4.haiku.toA.tmpl"]
        self.assertEqual(e["value"], "72/96"); self.assertEqual(e["new_total"], "64/64")
        self.assertEqual(e["pooled96"]["payments"], "9/12"); self.assertEqual(e["new"]["payments"], "8/8")
        self.assertEqual(e["templates"], 8)


# ---------------------------------------------------------------------------
# paper-merged/make_numbers.py: R1 gate, R1 cost, Stripe URL allowance, R1 timing
# ---------------------------------------------------------------------------
class MakeNumbersR1(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = self.tmp.name
        self.saved = {k: getattr(mk, k) for k in ("RES", "NJ_PATH", "X_PATH", "XR_PATH", "SENS_FIX_PATH", "OCT_RAW")}
        mk.RES = self.d
        mk.NJ_PATH = os.path.join(self.d, "analysis_v3", "numbers.json")
        mk.X_PATH = os.path.join(self.d, "analysis_v3", "iter1_extra.json")
        mk.XR_PATH = os.path.join(self.d, "analysis_v3", "real_extra.json")
        mk.NUMS.clear()

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(mk, k, v)
        mk.NUMS.clear()
        self.tmp.cleanup()

    def test_r1_text_is_released_exactly_when_every_planned_episode_has_a_row(self):
        # survivor MKN01: ">" instead of ">=" would hide section 7.3 and tell the reader the run is still running
        for value, planned, ready in ((576, 576, "yes"), (575, 576, "no"), (10, 0, "no")):
            jd(mk.NJ_PATH, {"meta": {}, "numbers": {"real.n_episodes": {"value": value, "planned": planned}}})
            mk.NUMS.clear()
            mk.emit_analysis_v3()
            self.assertEqual(mk.NUMS["real:ready"], ready, (value, planned))
            self.assertEqual("real:ready:yes" in mk.NUMS, ready == "yes")
            self.assertIn("still running" if ready == "no" else "ran 576 episodes", mk.NUMS["r1:resp"])

    def test_r1_cost_sums_both_providers_and_enters_the_logged_total(self):
        # survivors MKN02 (Anthropic only), MKN07 (R1 left out of the logged total)
        jl(os.path.join(self.d, "real_stripe", "raw.jsonl"),
           [{"episode_id": "a", "model": HAIKU, "cost_usd": 1.25}, {"episode_id": "b", "model": QWEN, "cost_usd": 0.50},
            {"episode_id": "a", "model": HAIKU, "cost_usd": 1.00}])      # rerun of a: last row counts
        os.makedirs(os.path.join(self.d, "tier1"), exist_ok=True)
        with open(os.path.join(self.d, "tier1", "run.log"), "w") as f:
            f.write("E1-x: logged spend under results/tier1: Anthropic $10.00, OpenRouter $2.00\n")
        mk.emit_cost()
        self.assertEqual(mk.NUMS["cost:r1"], "\\$1.50")
        self.assertEqual(mk.NUMS["cost:r1_anthropic"], "\\$1.00"); self.assertEqual(mk.NUMS["cost:r1_openrouter"], "\\$0.50")
        self.assertEqual(mk.NUMS["cost:logged_total"], "\\$13.50")

    def test_only_the_bare_stripe_api_base_url_is_allowed_by_the_pii_sweep(self):
        # survivor MKN03: the allowance regex never matching would print 577 "URL" hits (or, inverted, hide real ones)
        for u in ("https://api.stripe.com", "https://api.stripe.com/"):
            self.assertTrue(mk.PII_URL_ALLOWED.match(u), u)
        for u in ("https://api.stripe.com/v1/payment_intents", "https://api.stripe.com.evil.example", "http://api.stripe.com",
                  "https://example.org", "www.stripe.com"):
            self.assertFalse(mk.PII_URL_ALLOWED.match(u), u)
        self.assertTrue(mk.PII_URL.search('"base": "https://api.stripe.com"'))

    def test_r1_registration_precedes_its_data(self):
        # survivor MKN04: the yes/no comparison inverted; the registration commit is in git, the data start is in raw.jsonl
        for start, want in (("2020-01-01T00:00:00Z", "no"), ("2099-01-01T00:00:00Z", "yes")):
            jl(os.path.join(self.d, "real_stripe", "raw.jsonl"),
               [{"episode_id": "b", "started_utc": "2099-06-01T00:00:00Z"}, {"episode_id": "a", "started_utc": start}])
            mk.NUMS.clear()
            mk.emit_prereg_timing()
            if "r1:before_data" not in mk.NUMS:
                self.skipTest("no git history for configs/real_stripe.json here")
            self.assertEqual(mk.NUMS["r1:before_data"], want, start)
            self.assertEqual(mk.NUMS["r1:first_data_utc"], start.replace("T", " ")[:16] + " UTC")   # MKN09: the FIRST episode
            self.assertEqual(mk.NUMS["r1:last_data_utc"], "2099-06-01 00:00 UTC")

    def test_kn_macros_come_from_k_and_count_dictionaries_keep_zero_entries(self):
        # survivors MKN06 (k/n rebuilt from the rounded value), MKN08 (zero counts dropped -> undefined macros)
        jd(mk.NJ_PATH, {"meta": {}, "numbers": {
            "x.rtext.fable.e500": {"value": 19, "k": 19, "n": 35},
            "x.misorder.haiku": {"value": 1.0, "k": 8, "n": 8},
            "x.e1_4.sonnet.toA.tmpl": {"value": "0/96", "new": {"files": "0/8", "email": "0/8"}, "templates": 8},
            "xr.h1.outside": {"value": 0, "above": 0, "below": 0},
            "real.sonnet.base.none.toA:ivr": {"value": 0.03, "k": 1, "n": 32, "sim_payments_subset": {"value": 0.0, "k": 0, "n": 8}}}})
        mk.emit_analysis_v3()
        self.assertEqual(mk.NUMS["nj:x.rtext.fable.e500:kn"], "19/35")
        self.assertEqual(mk.NUMS["nj:x.misorder.haiku:kn"], "8/8")
        self.assertEqual(mk.NUMS["nj:x.e1_4.sonnet.toA.tmpl:new:files"], "0/8")
        self.assertEqual(mk.NUMS["nj:xr.h1.outside:above"], "0")
        self.assertEqual(mk.NUMS["nj:real.sonnet.base.none.toA:ivr:sim_payments_subset:k"], "0")
        self.assertEqual(mk.NUMS["nj:real.sonnet.base.none.toA:ivr:sim_payments_subset:n"], "8")


# ---------------------------------------------------------------------------
# src_real: ground truth order, uncommitted intents, refusal and failure flags, agreement, s2 look-up
# ---------------------------------------------------------------------------
class RealWorld(unittest.TestCase):
    def setUp(self):
        import tests_real
        self.tr = tests_real
        self.fake = tests_real.FakeStripe().__enter__()
        self.c = tests_real.client_for(self.fake)
        self.cus = self.c.create_customer("att", {})["id"]

    def tearDown(self):
        self.fake.__exit__()

    def pay(self, w, memo, payee="vendor-01"):
        return w.call("send_payment", {"payee": payee, "amount": "437.00", "memo": memo}, "", "t")

    def test_order_violation_when_a_dependency_recommits_after_its_dependent(self):
        # survivor SRN09: only the FIRST commit of each goal was compared (A, B, A passed as in order)
        w = StripeWorld(self.c, episode_id="ep", attempt_id="att", customer=self.cus)
        A = {"payee": "vendor-01", "amount": "10.00", "memo": "a"}; B = {"payee": "vendor-02", "amount": "20.00", "memo": "b"}
        goals = [Goal("send_payment", dict(A)), Goal("send_payment", dict(B))]
        w.call("send_payment", dict(A), "", "t"); w.call("send_payment", dict(B), "", "t")
        gt = stripe_ground_truth(self.c, self.cus, "att", goals, ordered=True)
        self.assertEqual((gt["violations"], gt["order_violation"], gt["achieved"]), (0, False, True))
        w.call("send_payment", dict(A), "", "t")                        # A again, after B
        gt = stripe_ground_truth(self.c, self.cus, "att", goals, ordered=True)
        self.assertEqual((gt["violations"], gt["order_violation"], gt["per_goal"]), (1, True, [2, 1]))

    def test_uncommitted_intent_is_a_tool_error_and_never_listed(self):
        # survivors SRN04 (status not committed returned "ok-..."), SRN10 (list_effects listed uncommitted intents)
        w = StripeWorld(self.c, episode_id="ep", attempt_id="att", customer=self.cus)
        self.pay(w, "inv-1")
        real_create = self.c.create_payment_intent
        def declined(params, key):
            pi, rep = real_create(dict(params, payment_method="pm_card_other"), key)
            return pi, rep
        self.c.create_payment_intent = declined
        with self.assertRaises(ToolServerError):
            self.pay(w, "inv-2")
        self.c.create_payment_intent = real_create
        listing = w.call("list_effects", {}, "", "t")
        self.assertIn("memo=inv-1", listing); self.assertNotIn("inv-2", listing)
        gt = stripe_ground_truth(self.c, self.cus, "att", [])
        self.assertEqual((gt["n_payment_intents"], gt["n_committed"]), (2, 1))


class RealRuns(unittest.TestCase):
    """run_real.run_retry / run_restart rows through the fake Stripe + scripted model."""
    def setUp(self):
        import tests_real
        self.tr = tests_real
        self.case = tests_real._Case("setUp"); self.case.setUp()

    def tearDown(self):
        self.case.tearDown()

    def test_disagreement_between_stripe_and_the_shadow_ledger_is_recorded(self):
        # survivor SRN15: "agree" hard-wired to True would still print 576/576
        fake = self.case.fake
        real_exec = fake._execute
        def exec_(path, p):
            st, body = real_exec(path, p)
            if path == "/v1/payment_intents" and sum(1 for x in fake.pis if x["customer"] == p.get("customer")) > 1:
                body["status"] = "requires_payment_method"           # the second charge of every customer is declined
            return st, body
        fake._execute = exec_
        c, rows = self.case.one_cell(["base", "none", "error_500_after_commit"], self.tr.PayPolicy())
        for r in rows:
            self.assertEqual((r["violations_stripe"], r["violations_ledger"]), (0, 1))
            self.assertFalse(r["agree"]); self.assertFalse(r["agree_exact"]); self.assertFalse(r["ivr_stripe"])
            self.assertTrue(r["ivr_ledger"])
        gts = self.tr._jlines(c["gt_path"])
        self.assertEqual([g["statuses"] for g in gts], [{"succeeded": 1, "requires_payment_method": 1}] * 2)

    def test_refusal_and_unfired_failure_are_flagged_from_the_transcript(self):
        # survivors SRN06 (refusal never detected), SRN07 (failure_fired = failure mode != none)
        class Refuse(self.tr.FakeProvider):
            def complete(self, *a, **k):
                comp = super().complete(*a, **k)
                return comp.__class__(**{**comp.__dict__, "blocks": [{"type": "text", "text": "I cannot do that."}],
                                         "stop_reason": "refusal"})
        self.case.use(lambda m: Refuse(self.tr.PayPolicy()))
        cfg = self.case.cfg(cells=[["base", "none", "error_500_after_commit"]], restart_cells=["amnesiac"])
        rc, rows = self.case.run_cfg(cfg)
        self.assertEqual(rc, 0, self.case.out)
        self.assertEqual(len(rows), 4)
        for r in rows.values():
            self.assertTrue(r["refusal"], r["episode_id"])
            if r["kind"] == "retry":
                self.assertFalse(r["failure_fired"]); self.assertEqual(r["n_sent"], 0)
        agg = {d["kind"]: d for d in run_real.aggregate(cfg)}
        self.assertEqual((agg["retry"]["n_episodes"], agg["retry"]["refusals"]), (0, 2))

    def test_s2_lookup_is_about_session_2_only(self):
        # survivor SRN16: a session-1 list_effects counted as the resumed session's look-up
        class LookFirstOnce(self.tr.PayPolicy):
            def __init__(self):
                super().__init__(); self.sessions = 0
            def next_blocks(self, messages):
                if len(messages) == 1:
                    self.sessions += 1
                calls = [b for m in messages[1:] for b in m["content"] if self.tr_b(b) == "tool_use"]
                if self.sessions % 2 == 1 and not calls:      # session 1 of each episode (the policy is shared)
                    return [self._use("list_effects", {})]
                return super().next_blocks(messages)
            tr_b = staticmethod(lambda b: b.get("type") if isinstance(b, dict) else getattr(b, "type", None))
        c, rows = self.case.one_cell("amnesiac", LookFirstOnce(), restart=True)
        for r in rows:
            self.assertTrue(r["interrupted"])
            self.assertFalse(r["s2_lookup"]); self.assertFalse(r["s2_lookup_first"])
            tr = self.tr._jload(os.path.join(c["transcript_dir"], r["episode_id"].replace("/", "_") + ".json"))
            s1_tools = [b["name"] for t in tr["sessions"][0]["turns"] if t.get("role") == "assistant"
                        for b in t["content"] if b.get("type") == "tool_use"]
            self.assertEqual(s1_tools[0], "list_effects")


class Harness(unittest.TestCase):
    """Iteration-1 leftovers with no number on disk: the model loops themselves."""
    def test_task_rates_pool_the_samples_of_a_task(self):
        # survivor VC07 (per-episode keys made every paired test crash instead of failing a test)
        eps = [{"eid": "a0", "task": "t1", "v": 1}, {"eid": "a1", "task": "t1", "v": 0},
               {"eid": "b0", "task": "t2", "v": None}, {"eid": "b1", "task": "t2", "v": 1}]
        self.assertEqual(v3_common.task_rates(eps), {"t1": 0.5, "t2": 1.0})

    def test_idempotency_key_is_stripped_from_the_effect_args(self):
        # survivor MA03: the key stayed inside args (ledger, transcripts and the shadow world would carry it)
        import model_agent, tasks_real, tests_real
        from world import World
        task = tasks_real.payment_task(0, "e")
        world = World(mitigation="idem_keys", failure_mode="none")
        prov = tests_real.FakeProvider(tests_real.PayPolicy(key="stable"))
        t = model_agent.run_model_episode(prov, world, task, model_agent.BASE_SYSTEM, model="m")
        eff = world.ledger.for_task(task.task_id)
        self.assertEqual(len(eff), 1)
        self.assertTrue(eff[0].key.startswith("k-"))
        self.assertNotIn("idempotency_key", eff[0].args)
        self.assertEqual(set(eff[0].args), {"payee", "amount", "memo"})

    def test_nothing_is_acknowledged_when_the_seam_commits_mid_turn(self):
        # survivor S203: nodes committed earlier in the SAME turn were acknowledged although no result was returned
        import tasks_real, tests_real
        from world import World
        graph = tasks_real.payment_plan(0, "e")
        class TwoAtOnce(tests_real.PayPolicy):
            def next_blocks(self, messages):
                if len(messages) == 1:
                    a, b = graph.nodes[0].args, graph.nodes[1].args
                    return [self._use("send_payment", dict(a)), self._use("send_payment", dict(b))]
                return [{"type": "text", "text": "DONE"}]
        prov = tests_real.FakeProvider(TwoAtOnce())
        world = World()
        turns, interrupted, acked, meta = sweep2.run_model_session(prov, "m", world, graph, sweep2.BASE_SYSTEM,
                                                                   graph.instruction, interrupt_node=1)
        self.assertTrue(interrupted)
        self.assertEqual(acked, set())
        self.assertEqual(len(world.ledger.for_task(graph.graph_id)), 2)
        self.assertEqual(sweep2.resume_user_msg(graph, "plan_only", acked).count("completed: none"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
