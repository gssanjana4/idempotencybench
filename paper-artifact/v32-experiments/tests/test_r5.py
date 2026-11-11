"""Offline tests for R5 (real error texts): catalogue and selection rule,
injection into the World, what the model sees, the config and plan, the
spend guard (reserve / settle / cap / cross-run ledger), a full fake launch
through src/tier1.py, the stop on the cap and the resume, the estimate, the
power simulation and the registered analysis. No network: the scripted fake
provider of tests_tier1.py plays the model.

  python3 v32-experiments/tests/test_r5.py
"""
import copy, json, os, re, sys, tempfile, unittest

V32 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(V32)
for d in (REPO, os.path.join(REPO, "src"), os.path.join(REPO, "src2"), os.path.join(V32, "src")):
    if d not in sys.path:
        sys.path.insert(0, d)

import tests_tier1 as tt                                  # noqa: E402
import tier1, world, model_agent, providers, sweep        # noqa: E402
import r5_texts, run_r5, spend_guard, estimate_r5, analysis_r5   # noqa: E402
from spend_guard import SpendGuard, SpendCapReached, GuardedProvider   # noqa: E402
from world import World, ToolTimeout, ToolServerError     # noqa: E402

CFG = os.path.join(V32, "configs", "r5_real_errors.json")
CFG_R2 = os.path.join(REPO, "configs", "r2_wording.json")
QWEN, HAIKU = "openrouter:qwen/qwen3.6-27b-20260422", "claude-haiku-4-5-20251001"
SONNET, FABLE = "claude-sonnet-5", "claude-fable-5"
VENDORS = ("stripe", "aws", "amazon", "dynamodb", "google", "github", "twilio", "azure", "microsoft",
           "openai", "shopify", "paypal", "square", "akamai", "upvest")


def cat():
    return r5_texts.load()


class TestCatalogue(unittest.TestCase):
    def test_size_fields_dates_classes(self):
        c = cat()
        self.assertTrue(20 <= len(c["texts"]) <= 30, len(c["texts"]))
        for t in c["texts"]:
            self.assertEqual(t["accessed"], "2026-11-01", t["id"])
            self.assertIn(t["retrieval"], ("raw", "webfetch"))
            self.assertIn(t["source_kind"], c["source_kind_order"])
            self.assertIn(t["condition"], ("timeout", "5xx", "ambiguous"))
            self.assertTrue(t["url"].startswith("https://"))
        self.assertEqual({t["class"] for t in c["texts"]}, {"U", "N", "R", "G"})
        self.assertEqual(len({t["text"] for t in c["texts"]}), len(c["texts"]), "duplicate texts")

    def test_vendor_flag_matches_text(self):
        for t in cat()["texts"]:
            has = any(re.search(r"\b" + v + r"\b", t["text"].lower()) for v in VENDORS)
            if has:
                self.assertTrue(t["vendor_name_in_text"], t["id"])

    def test_selection_rule_reproduces_config(self):
        c = cat()
        sel = r5_texts.select_subset(c)
        cfgj = json.load(open(CFG))
        self.assertEqual(sel, cfgj["r5"]["injected"])
        ids = r5_texts.by_id(c)
        self.assertEqual([ids[i]["class"] for i in sel], ["U"] * 3 + ["G"] * 3 + ["N", "R"])
        for i in sel:
            t = ids[i]
            self.assertFalse(t["borderline"] or t["vendor_name_in_text"], i)
            self.assertFalse(any(re.search(r"\b" + v + r"\b", t["text"].lower()) for v in VENDORS), i)
        provs = [ids[i]["provider"] for i in sel]
        self.assertLessEqual(max(provs.count(p) for p in provs), 2)

    def test_rule_is_deterministic_under_reordering(self):
        c = cat()
        c2 = copy.deepcopy(c)
        c2["texts"].reverse()
        self.assertEqual(r5_texts.select_subset(c), r5_texts.select_subset(c2))


class TestInjection(unittest.TestCase):
    def setUp(self):
        self.saved = dict(world.WORDING_VARIANTS)
        self.cfg = json.load(open(CFG))
        self.map = r5_texts.register(self.cfg["r5"])

    def tearDown(self):
        world.WORDING_VARIANTS.clear()
        world.WORDING_VARIANTS.update(self.saved)

    def test_existing_modes_untouched(self):
        for k, v in self.saved.items():
            self.assertEqual(world.WORDING_VARIANTS[k], v)
        self.assertEqual(world.FAILURE_MODES, ("none", "timeout_after_commit", "timeout_before_commit",
                                               "error_500_after_commit"))
        self.assertEqual(len(world.WORDING_VARIANTS), len(self.saved) + 8)
        r5_texts.register(self.cfg["r5"])        # idempotent
        self.assertEqual(len(world.WORDING_VARIANTS), len(self.saved) + 8)

    def test_commits_once_then_raises_the_verbatim_text_same_ledger_as_500(self):
        ids = r5_texts.by_id(cat())
        args = {"payee": "Acme", "amount": "120.00", "memo": "inv 7"}
        ref = World(failure_mode="error_500_after_commit", seed=3)
        with self.assertRaises(ToolServerError):
            ref.call("send_payment", args, "", "t")
        for fm, tid in self.map.items():
            w = World(failure_mode=fm, seed=3)
            with self.assertRaises((ToolTimeout, ToolServerError)) as cm:
                w.call("send_payment", args, "", "t")
            self.assertEqual(str(cm.exception), ids[tid]["text"])
            self.assertEqual([(e.tool, e.args) for e in w.ledger.effects],
                             [(e.tool, e.args) for e in ref.ledger.effects])
            self.assertEqual(w.call("send_payment", args, "", "t")[:3], "ok-")   # fires once
            self.assertEqual(w.ledger.violations("t"), 1)

    def test_braces_survive_format(self):
        c = cat()
        c["texts"].append({**c["texts"][0], "id": "zz_brace", "text": 'HTTP 500 {"error":"x"}'})
        r5_texts.register({"injected": ["zz_brace"]}, c)
        w = World(failure_mode="real_zz_brace_after_commit")
        with self.assertRaises(ToolTimeout) as cm:
            w.call("send_email", {"to": "a@b.c", "subject": "s"}, "", "t")
        self.assertEqual(str(cm.exception), 'HTTP 500 {"error":"x"}')

    def test_model_sees_ERROR_prefix_and_text_both_formats(self):
        ids = r5_texts.by_id(cat())
        task = tt.ALL_TASKS[next(iter(tt.ALL_TASKS))]
        for fm, tid in self.map.items():
            prov = tt.FakeProvider(tt.Policy())
            tr = model_agent.run_model_episode(prov, World(failure_mode=fm, seed=1), task,
                                               model_agent.BASE_SYSTEM, model="m")
            errs = [b for t in tr["turns"] if t["role"] == "tool_results" for b in t["content"] if b.get("is_error")]
            self.assertEqual(errs[0]["content"], "ERROR: " + ids[tid]["text"])
            oa = providers.messages_to_openai("s", [{"role": "user", "content": [errs[0]]}])
            self.assertEqual(oa[-1]["content"], "ERROR: " + ids[tid]["text"])


class TestConfigAndPlan(unittest.TestCase):
    def test_plan_counts_tasks_and_ids(self):
        cfg = run_r5.load(CFG)
        eps = tier1.plan(cfg)
        self.assertEqual(len(eps), 2560)
        by = {}
        for e in eps:
            by[e[0]] = by.get(e[0], 0) + 1
        self.assertEqual(by, {QWEN: 640, HAIKU: 640, SONNET: 640, FABLE: 640})
        self.assertEqual(len({tier1.eid_of(e) for e in eps}), 2560)
        self.assertEqual([m for m in dict.fromkeys(e[0] for e in eps)], [QWEN, HAIKU, SONNET, FABLE])
        r2 = tier1.load_config(CFG_R2)
        self.assertEqual([t.task_id for t in tier1.tasks_for(cfg)], [t.task_id for t in tier1.tasks_for(r2)])
        self.assertEqual(len({t.domain if t.family == "atomic" else t.family for t in tier1.tasks_for(cfg)}), 8)

    def test_settings_match_r2_r3(self):
        cfg = run_r5.load(CFG)
        self.assertEqual((cfg["samples"], cfg["max_turns"], cfg["prompt_cache"], cfg["retry_mode"]), (2, 8, True, "agent_retry"))
        self.assertEqual(tier1.max_tokens_for(cfg, QWEN), 4096)
        self.assertEqual(tier1.max_tokens_for(cfg, FABLE), 1024)
        self.assertTrue(cfg["out_dir"].startswith(os.path.join(REPO, "v32-experiments", "results")))
        self.assertEqual(cfg["r5"]["total_cap_usd"], 50)
        self.assertLessEqual(cfg["exp_budget_usd"] + cfg["exp_or_budget_usd"], cfg["r5"]["total_cap_usd"])
        self.assertEqual(set(cfg["r5"]["primary_models"]), {HAIKU, QWEN})

    def test_cell_list_is_enforced(self):
        c = json.load(open(CFG))
        c["groups"][0]["cells"] = c["groups"][0]["cells"][:-1]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            json.dump(c, open(p, "w"))
            with self.assertRaises(AssertionError):
                run_r5.load(p)


class TestGuard(unittest.TestCase):
    def test_reserve_settle_charge_cap(self):
        with tempfile.TemporaryDirectory() as d:
            g = SpendGuard(os.path.join(d, "l.jsonl"), 1.0)
            self.assertEqual(g.committed(), 0.0)
            a = g.reserve(0.6)
            self.assertAlmostEqual(g.committed(), 0.6)
            with self.assertRaises(SpendCapReached) as cm:
                g.reserve(0.5)
            self.assertIn("HTTP 402", str(cm.exception))
            self.assertIn("credit balance", str(cm.exception).lower())
            g.settle(a, 0.1)
            self.assertAlmostEqual(g.committed(), 0.1)
            g.charge(0.2, provider="anthropic")
            self.assertAlmostEqual(g.committed(), 0.3)
            g2 = SpendGuard(g.path, 1.0)                     # another process / a later run
            self.assertAlmostEqual(g2.committed(), 0.3)
            with open(g.path, "a") as f:
                f.write('{"type": "settle", "id": "x", "us')  # cut off by a kill
            self.assertAlmostEqual(g2.committed(), 0.3)
            g2.reserve(0.7)                                   # exactly to the cap
            with self.assertRaises(SpendCapReached):
                g2.reserve(0.001)

    def test_no_ledger_file_until_first_write(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sub", "l.jsonl")
            g = SpendGuard(p, 1.0)
            self.assertEqual(g.remaining(), 1.0)
            self.assertFalse(os.path.exists(p))

    def test_guarded_provider_settles_failures_at_zero_and_bounds_cost(self):
        with tempfile.TemporaryDirectory() as d:
            g = SpendGuard(os.path.join(d, "l.jsonl"), 10.0)
            bad = GuardedProvider(tt.FakeProvider(tt.Policy(), fail="boom"), g, {"m": (1.0, 5.0)})
            with self.assertRaises(RuntimeError):
                bad.complete("m", "s", [], [{"role": "user", "content": "x"}], 1024, 0.0)
            self.assertEqual(g.committed(), 0.0)
            ok = GuardedProvider(tt.FakeProvider(tt.Policy(), usage=(1000, 100)), g, {HAIKU: (1.0, 5.0)},
                                 cost_fn=spend_guard.anthropic_call_cost, worst_in_mult=1.25)
            task = tt.ALL_TASKS[next(iter(tt.ALL_TASKS))]
            msgs = [{"role": "user", "content": task.instruction}]
            ok.complete(HAIKU, "s", model_agent._tool_schemas(), msgs, 1024, 0.0)
            s = g.summarize(g.entries())
            self.assertAlmostEqual(s["actual"], (1000 * 1.0 + 100 * 5.0) / 1e6)
            res = [e for e in g.entries() if e["type"] == "reserve"][-1]
            self.assertGreater(res["usd"], s["actual"])        # worst case bounds the bill

    def test_anthropic_cost_equals_sweep_episode_cost(self):
        comp = providers.Completion(blocks=[], provider="anthropic", model_requested=FABLE, model_served=FABLE,
                                    temperature_requested=0.0, temperature_effective=None,
                                    usage={"input": 7, "output": 400, "cache_write": 500, "cache_read": 4000})
        t = {"model": FABLE, "usage": comp.usage}
        self.assertAlmostEqual(spend_guard.anthropic_call_cost(FABLE, comp, {FABLE: sweep.price_for(FABLE)}),
                               sweep.episode_cost(t))


class _LaunchCase(unittest.TestCase):
    """run_r5.launch with tier1's providers replaced by FakeProviders (wrapped
    by the guard exactly as in a real run)."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.saved = (tier1._client_for, tier1._anthropic_client, tier1.LIMIT, tier1.SKIP_PROVIDERS,
                      dict(world.WORDING_VARIANTS))
        tier1._anthropic_client = lambda models: None
        tier1.SKIP_PROVIDERS = set()
        self.fakes = {}

        def base_client_for(model, _c):
            if model not in self.fakes:
                cost = 0.002 if model.startswith("openrouter:") else None
                self.fakes[model] = tt.FakeProvider(tt.Policy(), name=tier1.provider_of(model),
                                                    usage=(1000, 100), cost=cost)
            return self.fakes[model]
        tier1._client_for = base_client_for

    def tearDown(self):
        (tier1._client_for, tier1._anthropic_client, tier1.LIMIT, tier1.SKIP_PROVIDERS, wv) = self.saved
        world.WORDING_VARIANTS.clear()
        world.WORDING_VARIANTS.update(wv)
        model_agent.MAX_TURNS, model_agent.MAX_TOKENS = 8, 1024
        providers.RECORD_COST = False
        providers.PROMPT_CACHE = False
        tt.release_locks()
        self.tmp.cleanup()

    def cfg(self, cap=50.0, models=(QWEN, HAIKU), per_template=1, samples=1):
        c = json.load(open(CFG))
        c["groups"] = [{"part": "core", "label": "t", "models": list(models), "cells": c["groups"][0]["cells"]}]
        c.update(per_template=per_template, samples=samples, workers=1, expect_per_cell=8 * per_template * samples)
        c["out_dir"] = c["tier1_root"] = os.path.join(self.d, "r5")
        c["r5"]["ledger"] = os.path.join(self.d, "ledger.jsonl")
        c["r5"]["total_cap_usd"] = cap
        p = os.path.join(self.d, "r5.json")
        json.dump(c, open(p, "w"))
        return run_r5.load(p)

    def launch(self, cfg, **kw):
        try:
            return run_r5.launch(cfg, **kw)
        finally:
            tt.release_locks()


class TestFakeLaunch(_LaunchCase):
    def test_full_run_rows_costs_ledger(self):
        cfg = self.cfg()
        rc = self.launch(cfg)
        self.assertEqual(rc, 0)
        rows = tier1.read_rows(cfg["raw_path"])
        self.assertEqual(len(rows), 2 * 10 * 8)
        fms = {r["failure_mode"] for r in rows}
        self.assertEqual(len(fms), 10)
        self.assertTrue(all(r["failure_fired"] for r in rows))
        self.assertTrue(all(r["violations"] == 1 for r in rows))     # the fake retries blindly
        g = run_r5.guard_for(cfg)
        s = g.summarize(g.entries())
        self.assertEqual(s["open"], {})
        logged = sum(r["cost_usd"] for r in rows)
        self.assertAlmostEqual(s["actual"], logged, places=5)
        self.assertGreater(s["by_provider"]["openrouter"], 0)
        self.assertGreater(s["by_provider"]["anthropic"], 0)
        self.assertEqual(self.launch(cfg), 0)                          # resume: nothing new
        self.assertEqual(len(tier1.read_rows(cfg["raw_path"])), 160)
        out = analysis_r5.analyse(tier1.load_config(cfg["cfg_path"]))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["models"][HAIKU]["P1"]["verdict"], "not supported")   # U == G for a blind retrier

    def test_cap_stops_both_providers_never_exceeded_then_resume(self):
        cfg = self.cfg(cap=0.03)
        rc = self.launch(cfg)
        self.assertIn(rc, (2, 3))
        g = run_r5.guard_for(cfg)
        self.assertLessEqual(g.committed(), 0.03 + 1e-9)
        n1 = len(tier1.read_rows(cfg["raw_path"]))
        self.assertLess(n1, 160)
        with open(cfg["r5"]["ledger"]) as f:
            self.assertTrue(all(json.loads(l)["type"] in ("reserve", "settle") for l in f))
        # nothing is raised; a fresh ledger with room completes the rest, no duplicates
        cfg2 = self.cfg(cap=50.0)
        cfg2["r5"]["ledger"] = os.path.join(self.d, "ledger2.jsonl")
        self.assertEqual(self.launch(cfg2), 0)
        rows = tier1.read_rows(cfg2["raw_path"])
        self.assertEqual(len(rows), 160)
        ids = [json.loads(l)["episode_id"] for l in open(cfg2["raw_path"])]
        self.assertEqual(len(ids), len(set(ids)))

    def test_cap_already_reached_refuses_to_start(self):
        cfg = self.cfg(cap=1.0)
        run_r5.guard_for(cfg).charge(1.0, provider="anthropic")
        self.assertEqual(self.launch(cfg), 2)
        self.assertFalse(os.path.exists(cfg["raw_path"]))

    def test_smoke_limit_one(self):
        cfg = self.cfg()
        tier1.LIMIT = 0
        rc = self.launch(cfg, limit=1, models=[HAIKU])
        tier1.LIMIT = 0
        self.assertEqual(len(tier1.read_rows(cfg["raw_path"])), 1)
        self.assertIn(rc, (0, 1))


class TestEstimate(unittest.TestCase):
    def test_fits_registered_cap_nothing_cut(self):
        cfg = tier1.load_config(CFG)
        out = estimate_r5.apply_rule(cfg)
        self.assertTrue(out["fits"])
        self.assertEqual(out["cuts_applied"], [])
        self.assertLessEqual(out["estimate"]["worst_usd"], 45.0)
        self.assertLessEqual(out["estimate"]["expected_usd"], out["estimate"]["worst_usd"])
        self.assertGreater(out["if_gpt_gemini_added"]["worst_usd"], 45.0)
        rec = os.path.join(V32, "out", "estimate_r5.json")
        if os.path.exists(rec):
            self.assertEqual(json.load(open(rec))["estimate"]["worst_usd"], out["estimate"]["worst_usd"])


class TestPower(unittest.TestCase):
    def setUp(self):
        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy not installed")

    def test_exact_p_and_small_simulation(self):
        import numpy as np
        import power_r5
        self.assertAlmostEqual(float(power_r5.exact_p_less(np.full(8, -0.3))), 1 / 256)
        self.assertAlmostEqual(float(power_r5.exact_p_less(np.zeros(8))), 1.0)
        rng = np.random.default_rng(1)
        pG = [0.8] * 8
        self.assertGreater(power_r5.simulate_power(pG, [0.05] * 8, 2, 0, 200, rng), 0.95)
        self.assertLess(power_r5.simulate_power(pG, pG, 2, 0, 400, np.random.default_rng(2)), 0.10)


class TestAnalysis(unittest.TestCase):
    def _cfg(self, d):
        c = json.load(open(CFG))
        c["out_dir"] = c["tier1_root"] = d
        p = os.path.join(d, "c.json")
        json.dump(c, open(p, "w"))
        return tier1.load_config(p)

    def _write(self, d, rates, models=(HAIKU,), drop=None):
        cfg = self._cfg(d)
        tasks = tier1.tasks_for(cfg)
        cls = {r5_texts.fm_of(i): r5_texts.by_id(cat())[i]["class"] for i in cfg["r5"]["injected"]}
        cls.update({"error_500_after_commit": "anchor_e500", "status_unknown_after_commit": "anchor_w3"})
        with open(os.path.join(d, "raw.jsonl"), "w") as f:
            for m in models:
                for fm, c in cls.items():
                    for i, t in enumerate(tasks):
                        for s in range(2):
                            if drop and fm == drop and i < 6:
                                continue
                            v = 1 if ((i * 2 + s) % 8) < round(8 * rates[c]) else 0
                            tmpl = t.domain if t.family == "atomic" else t.family
                            f.write(json.dumps({"episode_id": f"{m}{fm}{t.task_id}{s}", "model": m, "failure_mode": fm,
                                                "task_id": t.task_id, "template": tmpl, "violations": v,
                                                "failure_fired": True, "readback_after_failure": not v,
                                                "retried": bool(v), "achieved": True}) + "\n")
        return cfg

    def test_supported_when_unknown_low(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._write(d, {"U": 0.0, "G": 0.75, "N": 0.75, "R": 0.75, "anchor_e500": 0.75, "anchor_w3": 0.0})
            out = analysis_r5.analyse(cfg)
            p1 = out["models"][HAIKU]["P1"]
            self.assertEqual(p1["verdict"], "supported")
            self.assertAlmostEqual(p1["contrast"]["template"]["p"], 1 / 256, places=5)
            self.assertAlmostEqual(out["models"][HAIKU]["class_ivr"]["G"], 0.75)
            s2 = [b for b in out["models"][HAIKU]["secondary"] if b["test"].startswith("S2")][0]
            self.assertEqual(s2["contrast"]["template"]["d"], 0.0)
            self.assertIn("holm_p_template", s2)

    def test_not_supported_when_equal_and_floor_reading(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = self._write(d, {k: 0.0 for k in ("U", "G", "N", "R", "anchor_e500", "anchor_w3")})
            p1 = analysis_r5.analyse(cfg)["models"][HAIKU]["P1"]
            self.assertEqual(p1["verdict"], "not supported")
            self.assertIn("floor", p1["reading"])

    def test_not_evaluable_below_90_percent(self):
        with tempfile.TemporaryDirectory() as d:
            drop = r5_texts.fm_of(json.load(open(CFG))["r5"]["injected"][0])
            cfg = self._write(d, {"U": 0.0, "G": 0.75, "N": 0.5, "R": 0.5, "anchor_e500": 0.75, "anchor_w3": 0.0}, drop=drop)
            out = analysis_r5.analyse(cfg)
            self.assertEqual(out["models"][HAIKU]["P1"]["verdict"], "not evaluable")
            self.assertFalse(out["models"][HAIKU]["cells"][drop]["ok90"])

    def test_no_data(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(analysis_r5.analyse(self._cfg(d)), {"status": "no data"})


class TestLauncherStatic(unittest.TestCase):
    def test_run_r5_sh(self):
        s = open(os.path.join(V32, "run_r5.sh")).read()
        self.assertIn("set -uo pipefail", s)
        self.assertIn("guard_left", s)
        self.assertIn("tests/test_r5.py", s)
        self.assertIn('REPO="${REPO:-', s)
        self.assertNotRegex(s, r"\b(cat|less|more|head|tail|grep)\s+[^\n|]*\.env\b")
        self.assertNotRegex(s, r"echo[^\n]*\$\{?(ANTHROPIC|OPENROUTER)_API_KEY")


if __name__ == "__main__":
    unittest.main()
