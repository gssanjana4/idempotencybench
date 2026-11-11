"""Tests for the Tier-1 runners (src/tier1.py, configs/e1_*.json) and the
shared-module additions they need (World fail_on_call / seed / signal
variants, long-chain tasks, sweep.py spend guard, OpenRouter cost logging).
No network: a scripted fake provider plays the model.

  python3 tests_tier1.py
"""
import copy, hashlib, json, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from providers import Completion, OpenRouterProvider  # noqa: E402
from world import (World, FAILURE_MODES, SIGNAL_VARIANTS, ToolTimeout,  # noqa: E402
                   ToolServerError, EFFECTFUL_TOOLS, normalize_value)
from tasks import generate_tasks, generate_long_tasks  # noqa: E402

ALL_TASKS = {t.instruction: t for t in generate_tasks(per_domain=12) + generate_long_tasks()}


def release_locks():
    """runlock holds its lock for the life of the process; a fresh process
    per launch is what the launcher does. Tests release between launches."""
    import runlock
    while runlock._HELD:
        runlock._HELD.pop().close()


def _b(x, k, default=None):
    return x.get(k, default) if isinstance(x, dict) else getattr(x, k, default)


class Policy:
    """Scripted 'model'. Executes the task's steps in order, one call per
    turn. On an error result it optionally reads back (list_effects) and
    retries the failed step once. key: "none" | "stable" | "fresh".
    readback_first: start every conversation with list_effects and skip
    steps already listed (a model that checks unprompted)."""

    def __init__(self, key="none", readback_on_error=False, readback_first=False,
                 retry=True, misorder=False):
        self.key, self.rb_err, self.rb_first = key, readback_on_error, readback_first
        self.retry, self.misorder = retry, misorder
        self.n = 0

    def _key(self, task, tool, args):
        if self.key == "stable":
            return "k-" + hashlib.sha256(f"{task.task_id}{tool}{sorted(args.items())}".encode()).hexdigest()[:10]
        if self.key == "fresh":
            self.n += 1
            return f"fresh-{self.n}"
        return None

    def next_blocks(self, messages):
        user = messages[0]["content"]
        task = ALL_TASKS[user]
        steps = [(t, dict(a)) for t, a in task.steps]
        if self.misorder:
            steps = steps[::-1]
        calls, results = [], {}
        for m in messages[1:]:
            for b in m["content"]:
                if _b(b, "type") == "tool_use":
                    calls.append(b)
                elif _b(b, "type") == "tool_result":
                    results[b["tool_use_id"]] = b
        done, errs, rb_text = [], {}, None
        for c in calls:
            r = results.get(_b(c, "id"), {})
            name, inp = _b(c, "name"), dict(_b(c, "input"))
            inp.pop("idempotency_key", None)
            if name == "list_effects":
                rb_text = str(r.get("content", ""))
                continue
            if r.get("is_error"):
                errs[(name, json.dumps(inp, sort_keys=True))] = errs.get(
                    (name, json.dumps(inp, sort_keys=True)), 0) + 1
            else:
                done.append((name, inp))
        if self.rb_first and not calls:
            return [self._use("list_effects", {})]
        for tool, args in steps:
            canon = {f: args[f] for f in EFFECTFUL_TOOLS[tool][1]}
            if (tool, canon) in done:
                continue
            if rb_text and tool in rb_text and all(str(v) in rb_text for v in canon.values()):
                continue      # read-back shows it already happened
            ne = errs.get((tool, json.dumps(canon, sort_keys=True)), 0)
            if ne:
                if self.rb_err and (rb_text is None or _b(calls[-1], "name") != "list_effects"):
                    return [self._use("list_effects", {})]
                if not self.retry or ne >= 2:
                    continue
            a = dict(canon)
            k = self._key(task, tool, canon)
            if k:
                a["idempotency_key"] = k
            return [self._use(tool, a)]
        return [{"type": "text", "text": "DONE"}]

    def _use(self, name, inp):
        self.n += 1
        return {"type": "tool_use", "id": f"tu{self.n}", "name": name, "input": inp}


class FakeProvider:
    """Provider-protocol fake driven by a Policy; fixed usage per turn.
    cost: OpenRouter-style usage.cost per turn (provider_meta["cost"])."""
    def __init__(self, policy, name="fake", usage=(1000, 100), cost=None, fail=None):
        self.policy, self.name, self.usage, self.cost, self.fail = policy, name, usage, cost, fail
        self.calls = 0

    def complete(self, model, system, tools, messages, max_tokens, temperature):
        self.calls += 1
        if self.fail:
            raise RuntimeError(self.fail)
        blocks = self.policy.next_blocks(copy.deepcopy(messages))
        meta = {"cost": self.cost} if self.cost is not None else {}
        return Completion(blocks=blocks, provider=self.name, model_requested=model,
                          model_served=model + "-served", temperature_requested=temperature,
                          temperature_effective=temperature,
                          usage={"input": self.usage[0], "output": self.usage[1]},
                          timestamp_utc="2026-10-25T00:00:00Z",
                          stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
                          provider_meta=meta)
    history_assistant = staticmethod(OpenRouterProvider.history_assistant)
    history_tool_results = staticmethod(OpenRouterProvider.history_tool_results)


def make_cfg(name, d, **over):
    import tier1
    c = json.load(open(os.path.join(HERE, "configs", name + ".json")))
    c.update(over)
    c["out_dir"] = os.path.join(d, c["exp"])
    c["tier1_root"] = d
    p = os.path.join(d, name + ".json")
    json.dump(c, open(p, "w"))
    return tier1.load_config(p)


class _Tier1Case(unittest.TestCase):
    """Runs tier1.launch with providers replaced by FakeProvider(s)."""
    def setUp(self):
        import tier1
        self.tier1 = tier1
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.saved = (tier1._client_for, tier1._anthropic_client, tier1.LIMIT,
                      tier1.BUDGET_USD, tier1.OR_BUDGET_USD, tier1.SKIP_PROVIDERS)
        tier1._anthropic_client = lambda models: None
        tier1.LIMIT = 0
        tier1.BUDGET_USD = tier1.OR_BUDGET_USD = 0.0
        tier1.SKIP_PROVIDERS = set()
        self.providers = {}

    def tearDown(self):
        (self.tier1._client_for, self.tier1._anthropic_client, self.tier1.LIMIT,
         self.tier1.BUDGET_USD, self.tier1.OR_BUDGET_USD, self.tier1.SKIP_PROVIDERS) = self.saved
        import model_agent, providers
        model_agent.MAX_TURNS, model_agent.MAX_TOKENS = 8, int(os.environ.get("BENCH_MAX_TOKENS", "1024"))
        providers.RECORD_COST = False
        self.tmp.cleanup()

    def use(self, factory):
        """factory(model) -> provider; cached per model."""
        def client_for(model, _c):
            if model not in self.providers:
                self.providers[model] = factory(model)
            return self.providers[model]
        self.tier1._client_for = client_for

    def run_cfg(self, cfg):
        try:
            rc = self.tier1.launch(cfg)
        finally:
            release_locks()
        rows = self.tier1.read_rows(cfg["raw_path"])
        import csv
        summ = list(csv.DictReader(open(cfg["summary_path"]))) if os.path.exists(cfg["summary_path"]) else []
        return rc, rows, summ


# ---------------------------------------------------------------------------
# World / tasks additions (defaults reproduce released behaviour)
# ---------------------------------------------------------------------------

class TestWorldFailOnCall(unittest.TestCase):
    def test_k2_on_multi_step_first_call_clean_second_times_out(self):
        task = next(t for t in generate_tasks(4) if t.family == "multi_step")
        w = World(failure_mode="timeout_after_commit", fail_on_call=2)
        (t1, a1), (t2, a2) = task.steps
        self.assertTrue(w.call(t1, dict(a1), "", task.task_id).startswith("ok-"))
        with self.assertRaises(ToolTimeout):
            w.call(t2, dict(a2), "", task.task_id)
        self.assertEqual(len(w.ledger.for_task(task.task_id)), 2)   # committed
        self.assertTrue(w.call(t2, dict(a2), "", task.task_id).startswith("ok-"))
        self.assertEqual(w.ledger.violations(task.task_id), 1)

    def test_invalid_call_does_not_count_toward_k(self):
        w = World(failure_mode="timeout_before_commit", fail_on_call=2)
        with self.assertRaises(ValueError):
            w.call("send_email", {"to": ""}, "", "t")
        w.call("send_email", {"to": "a", "subject": "1"}, "", "t")
        with self.assertRaises(ToolTimeout):
            w.call("send_email", {"to": "a", "subject": "2"}, "", "t")

    def test_default_is_first_call(self):
        w = World(failure_mode="timeout_before_commit")
        with self.assertRaises(ToolTimeout):
            w.call("send_email", {"to": "a", "subject": "1"}, "", "t")


class TestWorldSeed(unittest.TestCase):
    def test_same_seed_same_acks_and_default_unseeded(self):
        def acks(seed):
            w = World(mitigation="idem_keys", seed=seed)
            return [w.call("send_email", {"to": "a", "subject": str(i)}, f"k{i % 2}", "t")
                    for i in range(4)]
        self.assertEqual(acks(7), acks(7))
        self.assertNotEqual(acks(7), acks(8))
        self.assertNotEqual(acks(None), acks(None))
        self.assertTrue(all(a.startswith("ok-") and len(a) == 11 for a in acks(7)))


class TestSignalVariants(unittest.TestCase):
    TEXTS = {"error_500_body_after_commit": ("Retry-After: 2", '"error":"internal_error"', "HTTP 500"),
             "error_503_retry_after_after_commit": ("503 Service Unavailable", "Retry-After: 2"),
             "status_unknown_after_commit": ("may have been processed", "outcome is unknown"),
             "timeout_504_after_commit": ("504 Gateway Timeout",)}

    def test_each_variant_commits_once_and_raises_its_text(self):
        self.assertEqual(set(self.TEXTS), set(SIGNAL_VARIANTS))
        for fm, needles in self.TEXTS.items():
            w = World(failure_mode=fm)
            with self.assertRaises((ToolTimeout, ToolServerError)) as cm:
                w.call("send_email", {"to": "a", "subject": "s"}, "", "t")
            for nd in needles:
                self.assertIn(nd, str(cm.exception), fm)
            self.assertEqual(len(w.ledger.for_task("t")), 1, fm)
            self.assertTrue(w.call("send_email", {"to": "a", "subject": "s"}, "", "t").startswith("ok-"))
            self.assertEqual(w.ledger.violations("t"), 1, fm)

    def test_request_id_deterministic_and_released_modes_unchanged(self):
        def text():
            w = World(failure_mode="error_500_body_after_commit")
            try:
                w.call("send_email", {"to": "a", "subject": "s"}, "", "t")
            except ToolServerError as e:
                return str(e)
        self.assertEqual(text(), text())
        self.assertEqual(FAILURE_MODES, ("none", "timeout_after_commit",
                                         "timeout_before_commit", "error_500_after_commit"))
        w = World(failure_mode="error_500_after_commit")
        with self.assertRaises(ToolServerError) as cm:
            w.call("send_email", {"to": "a", "subject": "s"}, "", "t")
        self.assertEqual(str(cm.exception), "internal server error")

    def test_rescore_fname_parses_variants(self):
        import rescore
        for fm in SIGNAL_VARIANTS:
            m = rescore.FNAME.match(f"claude-sonnet-5_base_none_{fm}__s1__email-abc.json")
            self.assertEqual(m["fm"], fm)


class TestLongChainTasks(unittest.TestCase):
    def test_set_shape_order_and_balanced_k(self):
        ts = generate_long_tasks()
        self.assertEqual(len(ts), 32)
        self.assertEqual(len({t.task_id for t in ts}), 32)
        self.assertEqual(sorted({len(t.goals) for t in ts}), [3, 4])
        for t in ts:
            self.assertTrue(t.ordered)
            self.assertEqual(t.family, "long_chain")
            for g in t.goals:
                for v in g.canon_args.values():
                    self.assertIn(str(v), t.instruction)
            self.assertIn(t.fail_on_call, range(1, len(t.goals) + 1))
        for L in (3, 4):
            ks = [t.fail_on_call for t in ts if len(t.goals) == L]
            self.assertEqual({k: ks.count(k) for k in set(ks)}, {k: 16 // L + (k <= 16 % L) for k in range(1, L + 1)})
        self.assertEqual(len(generate_tasks()), 320)       # benchmark unchanged

    def test_out_of_order_is_misorder(self):
        t = generate_long_tasks()[0]
        w = World()
        for tool, a in t.steps[::-1]:
            w.call(tool, dict(a), "", t.task_id)
        out = w.ledger.outcome(t.task_id, t.goals, t.ordered)
        self.assertTrue(out["misorder"]); self.assertFalse(out["success"])


# ---------------------------------------------------------------------------
# Spend guard: src/sweep.py (ported from sweep2) and OpenRouter cost logging
# ---------------------------------------------------------------------------

class TestSweepBudgetGuard(unittest.TestCase):
    def _launch(self, d, cap, prov, model="claude-sonnet-5"):
        import types, sweep, model_agent, providers
        saved = (sweep.MODELS, sweep.CELLS, sweep.SAMPLES, sweep.RAW_PATH, sweep.SUMMARY_PATH,
                 sweep.TRANSCRIPT_DIR, sweep.LIMIT, sweep.BUDGET_USD, sweep.OUT_DIR,
                 providers.provider_for)
        real = sys.modules.get("anthropic")
        sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=lambda: None)
        try:
            sweep.MODELS, sweep.CELLS, sweep.SAMPLES = [model], [("base", "none", "none")], 1
            sweep.OUT_DIR = d
            sweep.RAW_PATH, sweep.SUMMARY_PATH = os.path.join(d, "raw.jsonl"), os.path.join(d, "s.csv")
            sweep.TRANSCRIPT_DIR, sweep.LIMIT, sweep.BUDGET_USD = os.path.join(d, "tr"), 0, cap
            providers.provider_for = lambda model, anthropic_client=None: prov
            sweep.launch()
            release_locks()
            return [json.loads(l) for l in open(sweep.RAW_PATH)]
        finally:
            (sweep.MODELS, sweep.CELLS, sweep.SAMPLES, sweep.RAW_PATH, sweep.SUMMARY_PATH,
             sweep.TRANSCRIPT_DIR, sweep.LIMIT, sweep.BUDGET_USD, sweep.OUT_DIR,
             providers.provider_for) = saved
            if real is not None:
                sys.modules["anthropic"] = real
            else:
                sys.modules.pop("anthropic", None)

    def test_stops_before_cap_and_resumes_without_duplicates(self):
        # Sonnet $2/$10 per MTok; 2 turns x (100k in, 10k out) = $0.60/episode.
        # Cap $2.0: the first episode is admitted against the 0.5 floor, then
        # each next one only while spent + 0.60 <= 2.0 -> exactly 3 episodes.
        with tempfile.TemporaryDirectory() as d:
            rows = self._launch(d, 2.0, FakeProvider(Policy(), usage=(100_000, 10_000)))
            self.assertEqual(len(rows), 3)
            rows = self._launch(d, 2.0, FakeProvider(Policy(), usage=(100_000, 10_000)))
            self.assertEqual(len(rows), 3)                       # nothing more, no dupes
            rows = self._launch(d, 4.5, FakeProvider(Policy(), usage=(100_000, 10_000)))
            self.assertEqual(len(rows), 7)
            self.assertEqual(len({r["episode_id"] for r in rows}), 7)

    def test_openrouter_usage_cost_counts_toward_sweep_cap(self):
        # fake OpenRouter-style provider: $0.40 per turn via usage.cost
        import providers
        try:
            with tempfile.TemporaryDirectory() as d:
                rows = self._launch(d, 2.0, FakeProvider(Policy(), usage=(1, 1), cost=0.40),
                                    model="openrouter:org/m")
                self.assertEqual(len(rows), 2)     # 0.8/episode: 0 -> 0.8 -> 1.6, stop
                self.assertTrue(providers.RECORD_COST)
        finally:
            providers.RECORD_COST = False

    def test_default_no_cap_runs_everything(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(self._launch(d, 0.0, FakeProvider(Policy(), usage=(10**6, 10**5)))), 32)

    def test_openrouter_cost_counts_and_unknown_is_zero(self):
        import sweep
        t = {"model": "openrouter:x/y", "turns": [{"role": "assistant", "provider_meta": {"cost": 0.25}},
                                                  {"role": "tool_results"},
                                                  {"role": "assistant", "provider_meta": {}}]}
        self.assertAlmostEqual(sweep.episode_cost(t), 0.25)
        self.assertEqual(sweep.episode_cost({"model": "openrouter:x/y", "turns": []}), 0.0)
        self.assertAlmostEqual(sweep.episode_cost({"model": "claude-fable-5-x",
                                                   "usage": {"input": 10**6, "output": 10**5}}), 15.0)
        self.assertAlmostEqual(sweep.episode_cost({"model": "claude-haiku-4-5-20251001",
                                                   "deliveries": [], "usage": {"input": 10**6, "output": 0}}), 1.0)


class TestOpenRouterCostLogging(unittest.TestCase):
    def _complete(self):
        class P(OpenRouterProvider):
            def _post(self, payload):
                return {"model": "m", "choices": [{"message": {"content": "DONE"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.0123}}
        return P(api_key="k").complete("openrouter:o/m", "s", [], [{"role": "user", "content": "x"}], 8, 0.0)

    def test_off_by_default_on_when_enabled(self):
        import providers
        self.assertNotIn("cost", self._complete().provider_meta)   # running scripts unchanged
        try:
            providers.RECORD_COST = True
            self.assertEqual(self._complete().provider_meta["cost"], 0.0123)
        finally:
            providers.RECORD_COST = False


# ---------------------------------------------------------------------------
# Config + plan
# ---------------------------------------------------------------------------

class TestConfigsAndPlans(unittest.TestCase):
    EXPECT = {"e1_0_key_rerun": 64, "e1_1_crossprovider_core": 192, "e1_2_at_least_once": 960,
              "e1_3_keys_under_500": 128, "e1_4_task_pool": 256, "e1_5_long_chains": 128,
              "e1_6_signal_wording": 256, "e1_6_oct": 256}

    def test_episode_counts_and_isolated_out_dirs(self):
        import tier1
        for name, n in self.EXPECT.items():
            c = tier1.load_config(os.path.join(HERE, "configs", name + ".json"))
            self.assertEqual(len(tier1.plan(c)), n, name)
            self.assertIn(os.path.join("results", "tier1"), c["out_dir"])
            self.assertEqual(len({tier1.eid_of(e) for e in tier1.plan(c)}), n)

    def test_e1_4_tasks_are_33_to_96_and_disjoint_from_released(self):
        import tier1
        c = tier1.load_config(os.path.join(HERE, "configs", "e1_4_task_pool.json"))
        new = {t.task_id for t in tier1.tasks_for(c)}
        old = {t.task_id for t in generate_tasks(per_domain=4)}
        self.assertEqual(len(new), 64)
        self.assertFalse(new & old)
        self.assertEqual(new | old, {t.task_id for t in generate_tasks(per_domain=12)})

    def test_e1_0_reruns_released_tasks_but_released_raw_is_not_done(self):
        import tier1
        c = tier1.load_config(os.path.join(HERE, "configs", "e1_0_key_rerun.json"))
        rel = os.path.join(HERE, "results", "sweep_raw.jsonl")
        if os.path.exists(rel):
            ids = {json.loads(l)["episode_id"] for l in open(rel)}
            self.assertTrue({tier1.eid_of(e) for e in tier1.plan(c)} <= ids)
        # done = this experiment's own raw.jsonl only (a fresh out_dir: the real
        # results/tier1/e1_0/ holds the completed rerun once it has been made)
        with tempfile.TemporaryDirectory() as d:
            c = make_cfg("e1_0_key_rerun", d)
            self.assertEqual(len(tier1.done_ids(c) & {tier1.eid_of(e) for e in tier1.plan(c)}), 0)

    def test_released_out_dir_refused(self):
        import tier1
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            json.dump({"exp": "x", "models": ["m"], "cells": [], "out_dir": "results"}, open(p, "w"))
            with self.assertRaises(ValueError):
                tier1.load_config(p)


# ---------------------------------------------------------------------------
# One end-to-end fake-provider test per runner
# ---------------------------------------------------------------------------

class TestE1_0KeyRerun(_Tier1Case):
    def test_rows_transcripts_comparison_and_resume(self):
        c = make_cfg("e1_0_key_rerun", self.d)
        self.use(lambda m: FakeProvider(Policy(key="none")))
        self.tier1.LIMIT = 6
        rc, rows, summ = self.run_cfg(c)
        self.assertEqual(rc, 0); self.assertEqual(len(rows), 6)
        self.assertEqual(len(os.listdir(c["transcript_dir"])), 6)
        # keyless retry under idem_keys must commit twice (the T08 fix)
        self.assertTrue(all(r["violations"] == 1 and r["retry_key_category"] == "keyless" for r in rows))
        self.assertTrue(all(r["key_attached_first"] is False for r in rows))
        s = summ[0]
        self.assertEqual(s["ivr"], "1.0"); self.assertEqual(s["key_attach_rate"], "0.0")
        self.assertIn("released_identical", s)
        self.assertIn("released_calls_identical", s)
        if os.path.isdir(os.path.join(HERE, "results", "sweep_transcripts")):
            self.assertEqual(s["released_matched"], "6")   # all 6 have a v1 transcript
            self.assertEqual(s["released_v2_ivr"], "1.0")  # those 6 violate in v2 too
        self.tier1.LIMIT = 0
        rc, rows, _ = self.run_cfg(c)
        self.assertEqual(len(rows), 64)
        self.assertEqual(len(os.listdir(c["transcript_dir"])), 64)
        # seeded acks: replaying an episode's calls through World(seed) gives
        # exactly the ack ids its model saw (ack ids are reproducible)
        for fn in sorted(os.listdir(c["transcript_dir"]))[:8]:
            t = json.load(open(os.path.join(c["transcript_dir"], fn)))
            w = World(mitigation="idem_keys", failure_mode="timeout_after_commit", seed=t["ack_seed"])
            seen = [r for tu in t["turns"] if tu["role"] == "tool_results" for r in tu["content"]]
            calls = [b for tu in t["turns"] if tu["role"] == "assistant"
                     for b in tu["content"] if b["type"] == "tool_use"]
            for b, r in zip(calls, seen):
                a = dict(b["input"]); k = a.pop("idempotency_key", "") or ""
                try:
                    out = w.call(b["name"], a, k, t["task_id"])
                except ToolTimeout as e:
                    out = f"ERROR: {e}"
                self.assertEqual(out, r["content"], fn)


class TestE1_1CrossProviderCore(_Tier1Case):
    def test_cell8_runs_imports_merge_and_402_stops_openrouter_only(self):
        c = make_cfg("e1_1_crossprovider_core", self.d)
        # fake crossprovider dir with one imported transcript (cell 3)
        imp = os.path.join(self.d, "xp", "sweep_transcripts")
        os.makedirs(imp)
        import model_agent
        task = generate_tasks(4)[0]
        w = World(failure_mode="timeout_after_commit")
        tr = model_agent.run_model_episode(FakeProvider(Policy()), w, task, model_agent.BASE_SYSTEM,
                                           model=c["models"][0])
        tr["sample"] = 0
        model_agent._score_and_log(w, task, tr, imp, f"{c['models'][0]}_base_none_timeout_after_commit__s0")
        c["import_from"][0]["dir"] = imp
        c["import_from"][1]["dir"] = os.path.join(HERE, "results", "sweep_transcripts")
        c["import_from"][2]["dir"] = os.path.join(self.d, "E1-3", "transcripts")
        gpt = c["models"][1]
        self.use(lambda m: FakeProvider(Policy(key="stable"), cost=0.01,
                                        fail="OpenRouter HTTP 402: key limit" if m == gpt else None))
        self.tier1.LIMIT = 0
        c["models"] = c["models"][:2]
        rc, rows, summ = self.run_cfg(c)
        self.assertEqual(rc, 3)                                  # OpenRouter stopped
        self.assertEqual(self.providers[gpt].calls, 1)           # not hammered after 402
        own = [r for r in rows if r["model"] == c["models"][0]]
        self.assertEqual(len(own), 64)
        self.assertTrue(all(r["retry_key_category"] == "same_key" and r["violations"] == 0 for r in own))
        n_turns = sum(sum(1 for tu in json.load(open(os.path.join(c["transcript_dir"], f)))["turns"]
                          if tu["role"] == "assistant") for f in os.listdir(c["transcript_dir"]))
        self.assertAlmostEqual(sum(r["cost_usd"] for r in own), n_turns * 0.01, places=6)
        self.assertGreater(n_turns, 128)
        srcs = {(s["model"], s["failure_mode"], s["source"]) for s in summ}
        self.assertIn((c["models"][0], "error_500_after_commit", "run"), srcs)
        self.assertIn((c["models"][0], "timeout_after_commit", "crossprovider"), srcs)
        if os.path.isdir(os.path.join(HERE, "results", "sweep_transcripts")):
            rel = [s for s in summ if s["source"] == "released_v2"]
            self.assertEqual(len(rel), 21)                       # 3 models x 7 cells
            h = next(s for s in rel if s["model"].startswith("claude-haiku") and
                     s["mitigation"] == "idem_keys")
            # E0-1 [measured]: 0.922, task-level CI [0.778, 0.975]
            self.assertEqual((h["ivr"], h["ivr_ci_tlo"], h["ivr_ci_thi"]),
                             ("0.9219", "0.7778", "0.9755"))     # = e0_1_corrected_table.csv


class TestE1_2AtLeastOnce(_Tier1Case):
    def _run(self, policy_kw, cells, models=("claude-haiku-4-5-20251001",)):
        c = make_cfg("e1_2_at_least_once", self.d, models=list(models), cells=cells)
        self.use(lambda m: FakeProvider(Policy(**policy_kw)))
        self.tier1.LIMIT = 8
        return c, self.run_cfg(c)

    def test_redelivery_none_duplicates_every_effect(self):
        c, (rc, rows, summ) = self._run({}, [["base", "none", "none"]])
        self.assertEqual(rc, 0)
        for r in rows:
            n_steps = len(next(t for t in generate_tasks(4) if t.task_id == r["task_id"]).steps)
            self.assertEqual(r["violations"], n_steps)
            self.assertTrue(r["d2_reexecuted"]); self.assertFalse(r["d2_readback_first"])
            self.assertEqual(r["d1_violations"], 0)
            self.assertEqual(r["retry_mode"], "at_least_once")
        t = json.load(open(os.path.join(c["transcript_dir"], os.listdir(c["transcript_dir"])[0])))
        self.assertEqual(len(t["deliveries"]), 2)
        self.assertEqual(t["usage"]["input"], sum(d["usage"]["input"] for d in t["deliveries"]))
        self.assertEqual(summ[0]["d2_reexecuted"], "1.0")

    def test_idem_keys_constant_key_dedups_fresh_key_does_not(self):
        _, (_, rows, _) = self._run({"key": "stable"}, [["base", "idem_keys", "none"]])
        self.assertTrue(all(r["violations"] == 0 and r["d2_key_category"] == "same_key" for r in rows))
        self.tearDown(); self.setUp()
        _, (_, rows, _) = self._run({"key": "fresh"}, [["base", "idem_keys", "none"]])
        self.assertTrue(all(r["violations"] >= 1 and r["d2_key_category"] == "new_key" for r in rows))

    def test_readback_first_model_does_not_duplicate(self):
        _, (_, rows, summ) = self._run({"readback_first": True}, [["careful", "none", "none"]])
        self.assertTrue(all(r["violations"] == 0 and r["d2_readback_first"] for r in rows))
        self.assertEqual(summ[0]["d2_readback_first"], "1.0")

    def test_rescore_replays_deliveries_in_order(self):
        c, _ = self._run({}, [["base", "none", "none"]])
        for fn in os.listdir(c["transcript_dir"]):
            t = json.load(open(os.path.join(c["transcript_dir"], fn)))
            w = World(seed=t["ack_seed"])
            for d in t["deliveries"]:
                for tu in d["turns"]:
                    if tu["role"] == "assistant":
                        for b in tu["content"]:
                            if b["type"] == "tool_use" and b["name"] in EFFECTFUL_TOOLS:
                                a = dict(b["input"]); k = a.pop("idempotency_key", "")
                                w.call(b["name"], a, k, t["task_id"])
            self.assertEqual(w.ledger.violations(t["task_id"]), t["score"]["violations"])


class TestE1_3KeysUnder500(_Tier1Case):
    def test_same_key_retry_dedups_keyless_does_not_and_baselines_imported(self):
        c = make_cfg("e1_3_keys_under_500", self.d)
        c["import_from"][0]["dir"] = os.path.join(HERE, "results", "sweep_transcripts")
        self.use(lambda m: FakeProvider(Policy(key="stable" if "sonnet" in m else "none")))
        rc, rows, summ = self.run_cfg(c)
        self.assertEqual(rc, 0); self.assertEqual(len([r for r in rows if r["source"] == "run"]) if False else len(rows), 128)
        son = [r for r in rows if "sonnet" in r["model"]]
        hai = [r for r in rows if "haiku" in r["model"]]
        self.assertTrue(all(r["violations"] == 0 and r["retry_key_category"] == "same_key" for r in son))
        self.assertTrue(all(r["violations"] > 0 and r["retry_key_category"] == "keyless" for r in hai))
        run = {s["model"]: s for s in summ if s["source"] == "run"}
        self.assertEqual(run["claude-sonnet-5"]["key_reuse_rate"], "1.0")
        self.assertEqual(run["claude-haiku-4-5-20251001"]["ivr"], "1.0")
        if os.path.isdir(os.path.join(HERE, "results", "sweep_transcripts")):
            base = [s for s in summ if s["source"] == "released_v2"]
            self.assertEqual(len(base), 4)
            f = next(s for s in base if s["model"] == "claude-sonnet-5" and s["mitigation"] == "none")
            # E0-1 [measured]: Sonnet 500 = 0.641, task-level CI [0.468, 0.783]
            self.assertEqual((f["ivr"], f["ivr_ci_tlo"], f["ivr_ci_thi"]),
                             ("0.6406", "0.4677", "0.7834"))     # = e0_1_corrected_table.csv


class TestE1_4TaskPool(_Tier1Case):
    def test_new_tasks_r1_pooled_with_released_s0(self):
        c = make_cfg("e1_4_task_pool", self.d)
        self.use(lambda m: FakeProvider(Policy()))
        rc, rows, summ = self.run_cfg(c)
        self.assertEqual(rc, 0); self.assertEqual(len(rows), 256)
        self.assertEqual({r["sample"] for r in rows}, {0})
        # r = 1 -> task-level CI equals the episode CI
        for s in summ:
            self.assertEqual((s["ivr_ci_lo"], s["ivr_ci_hi"]), (s["ivr_ci_tlo"], s["ivr_ci_thi"]))
        self.assertTrue(os.path.exists(os.path.join(c["out_dir"], "summary_by_template.csv")))
        pooled = os.path.join(c["out_dir"], "summary_pooled.csv")
        if os.path.exists(os.path.join(HERE, "results", "sweep_raw_v2.jsonl")):
            import csv
            p = list(csv.DictReader(open(pooled)))
            self.assertEqual(len(p), 4)
            self.assertTrue(all(x["n_tasks_pooled"] == "96" and x["n_new"] == "64" for x in p))


class TestE1_5LongChains(_Tier1Case):
    def test_mid_plan_failure_lands_on_call_k_and_stale_reexec(self):
        c = make_cfg("e1_5_long_chains", self.d, models=["claude-haiku-4-5-20251001"])
        self.use(lambda m: FakeProvider(Policy()))
        rc, rows, summ = self.run_cfg(c)
        self.assertEqual(rc, 0); self.assertEqual(len(rows), 64)
        tasks = {t.task_id: t for t in generate_long_tasks()}
        for fn in os.listdir(c["transcript_dir"]):
            t = json.load(open(os.path.join(c["transcript_dir"], fn)))
            eff = [r for tu in t["turns"] if tu["role"] == "tool_results" for r in tu["content"]]
            err = [i for i, r in enumerate(eff) if r.get("is_error")]
            self.assertEqual(err, [tasks[t["task_id"]].fail_on_call - 1], fn)
        # policy retries in place -> duplicate, no misorder, no stale re-exec
        self.assertTrue(all(r["violations"] == 1 and not r["misorder"] and not r["stale_reexec"] for r in rows))
        by_k = os.path.join(c["out_dir"], "summary_by_fail_on_call.csv")
        import csv
        ks = {(x["failure_mode"], x["fail_on_call"]) for x in csv.DictReader(open(by_k))}
        self.assertEqual(len(ks), 8)                         # 2 signals x k in 1..4

    def test_stale_reexec_and_order_violation_from_ledger(self):
        import tier1
        t = generate_long_tasks()[1]           # long3, k = 2
        led = [{"tool": tl, "args": dict(a), "seq": i + 1} for i, (tl, a) in enumerate(t.steps)]
        led.append({"tool": t.steps[0][0], "args": dict(t.steps[0][1]), "seq": 4})   # step 1 again
        tr = {"episode_id": "e", "exp": "E1-5", "model": "m", "prompt": "base", "mitigation": "none",
              "failure_mode": "timeout_after_commit", "retry_mode": "agent_retry", "sample": 0,
              "score": {"violations": 1, "achieved": True, "omission": False, "misorder": False},
              "ledger": led, "turns": [], "fail_on_call": 2, "max_turns": 12}
        r = tier1.episode_row(tr, t)
        self.assertTrue(r["stale_reexec"]); self.assertTrue(r["order_violation"])
        self.assertEqual(r["violations_strict"], 1)
        led[-1]["args"]["amount"] = "$" + led[-1]["args"]["amount"]     # "$121.00" vs "121.00"
        r = tier1.episode_row(tr, t)
        self.assertEqual(r["violations_strict"], 0)                     # strict matcher misses it

    def test_rescore_replay_honours_fail_on_call(self):
        import rescore
        c = make_cfg("e1_5_long_chains", self.d, models=["claude-haiku-4-5-20251001"])
        self.use(lambda m: FakeProvider(Policy()))
        self.tier1.LIMIT = 8
        self.run_cfg(c)
        for fn in os.listdir(c["transcript_dir"]):
            m = rescore.FNAME.match(fn)
            t = json.load(open(os.path.join(c["transcript_dir"], fn)))
            v, ok, _ = rescore.replay(t, m["mit"], m["fm"])
            self.assertEqual((v, ok), (t["score"]["violations"], t["score"]["achieved"]))


class TestE1_6SignalWording(_Tier1Case):
    def test_variant_text_reaches_model_readback_rate_and_baselines(self):
        c = make_cfg("e1_6_signal_wording", self.d)
        c["import_from"][0]["dir"] = os.path.join(HERE, "results", "sweep_transcripts")
        # Haiku: retries blindly; Sonnet: reads back after an error
        self.use(lambda m: FakeProvider(Policy(readback_on_error="sonnet" in m)))
        rc, rows, summ = self.run_cfg(c)
        self.assertEqual(rc, 0); self.assertEqual(len(rows), 256)
        fn = next(f for f in os.listdir(c["transcript_dir"]) if "status_unknown" in f)
        t = json.load(open(os.path.join(c["transcript_dir"], fn)))
        errs = [r["content"] for tu in t["turns"] if tu["role"] == "tool_results"
                for r in tu["content"] if r.get("is_error")]
        self.assertEqual(errs, ["ERROR: The request may have been processed; its outcome is unknown."])
        run = {(s["model"], s["failure_mode"]): s for s in summ if s["source"] == "run"}
        self.assertEqual(run[("claude-sonnet-5", "status_unknown_after_commit")]["readback_after_failure"], "1.0")
        self.assertEqual(run[("claude-sonnet-5", "status_unknown_after_commit")]["ivr"], "0.0")
        self.assertEqual(run[("claude-haiku-4-5-20251001", "error_500_body_after_commit")]["ivr"], "1.0")
        self.assertEqual(run[("claude-haiku-4-5-20251001", "error_500_body_after_commit")]["readback_after_failure"], "0.0")
        if os.path.isdir(os.path.join(HERE, "results", "sweep_transcripts")):
            self.assertEqual(len([s for s in summ if s["source"] == "released_v2"]), 4)


class TestE1_6OctoberBaselines(_Tier1Case):
    """Sensitivity run (preregistration deviations log, entry 8): the bare
    500 / timeout baselines rerun through the Tier-1 path, same 32 tasks."""
    def test_same_tasks_as_e1_6_and_bare_signal_text(self):
        import tier1
        c = make_cfg("e1_6_oct", self.d)
        e16 = make_cfg("e1_6_signal_wording", self.d)
        self.assertEqual({t.task_id for t in tier1.tasks_for(c)},
                         {t.task_id for t in tier1.tasks_for(e16)})
        self.assertEqual({fm for _, _, fm in c["cells"]},
                         {"error_500_after_commit", "timeout_after_commit"})
        self.assertFalse(c.get("import_from"))   # no imported.jsonl in its dir
        self.use(lambda m: FakeProvider(Policy()))
        rc, rows, summ = self.run_cfg(c)
        self.assertEqual(rc, 0); self.assertEqual(len(rows), 256)
        self.assertEqual({s["source"] for s in summ}, {"run"})
        fn = next(f for f in os.listdir(c["transcript_dir"]) if "error_500_after_commit" in f)
        t = json.load(open(os.path.join(c["transcript_dir"], fn)))
        errs = [r["content"] for tu in t["turns"] if tu["role"] == "tool_results"
                for r in tu["content"] if r.get("is_error")]
        self.assertTrue(errs and "internal server error" in errs[0].lower(), errs)

    def test_exp_budget_stops_this_experiment_only_and_resumes(self):
        # $0.30 per episode (100k in + 10k out at Sonnet 2/10 = 0.2+0.1)
        c = make_cfg("e1_6_oct", self.d, exp_budget_usd=3.0, models=["claude-sonnet-5"])
        self.use(lambda m: FakeProvider(Policy(), usage=(100_000, 10_000)))
        rc, rows, _ = self.run_cfg(c)
        self.assertEqual(rc, 2)
        self.assertTrue(1 <= len(rows) < 256, len(rows))
        self.assertLessEqual(self.tier1.spent_in_exp(c)["anthropic"], 3.0)
        n1 = len(rows)
        # spend in OTHER experiment dirs does not count toward exp_budget_usd
        other = make_cfg("e1_6_signal_wording", self.d, models=["claude-sonnet-5"])
        os.makedirs(other["transcript_dir"], exist_ok=True)
        json.dump({"model": "claude-sonnet-5", "turns": [], "usage": {"input": 10**7, "output": 0}},
                  open(os.path.join(other["transcript_dir"], "x.json"), "w"))
        c2 = make_cfg("e1_6_oct", self.d, exp_budget_usd=6.0, models=["claude-sonnet-5"])
        rc, rows, _ = self.run_cfg(c2)
        self.assertEqual(rc, 2)
        self.assertGreater(len(rows), n1)
        self.assertLessEqual(self.tier1.spent_in_exp(c2)["anthropic"], 6.0)
        self.assertEqual(len(rows), len({r["episode_id"] for r in rows}))


class TestTier1SpendGuard(_Tier1Case):
    def test_anthropic_cap_spans_experiments_and_returns_2(self):
        # Haiku $1/$5: 3 turns (fail, retry, DONE) x (100k in, 10k out) = $0.45.
        # Worst case per next episode = max(largest so far, $0.50 floor):
        # 0 -> 0.45 -> 0.90 -> 1.35, then 1.35 + 0.50 > 1.5 -> stop at 3.
        self.use(lambda m: FakeProvider(Policy(), usage=(100_000, 10_000)))
        self.tier1.BUDGET_USD = 1.5
        c0 = make_cfg("e1_0_key_rerun", self.d)
        rc, rows, _ = self.run_cfg(c0)
        n0 = len(rows)
        self.assertEqual(rc, 2)
        self.assertEqual(n0, 3)
        c3 = make_cfg("e1_3_keys_under_500", self.d, models=["claude-haiku-4-5-20251001"])
        rc, rows, _ = self.run_cfg(c3)        # cap is root-wide: $1.35 already spent
        self.assertEqual((rc, len(rows)), (2, 0))
        self.tier1.BUDGET_USD = 0.0
        rc, rows, _ = self.run_cfg(c0)        # resumes, no duplicates
        self.assertEqual((rc, len(rows)), (0, 64))

    def test_skip_models_env(self):
        self.tier1.SKIP_MODELS = {"claude-haiku-4-5-20251001"}
        try:
            c = make_cfg("e1_0_key_rerun", self.d)
            self.use(lambda m: FakeProvider(Policy()))
            rc, rows, _ = self.run_cfg(c)
            self.assertEqual((rc, len(rows)), (0, 0))
        finally:
            self.tier1.SKIP_MODELS = set()

    def test_broken_model_skipped_after_3_failures(self):
        c = make_cfg("e1_0_key_rerun", self.d)
        self.use(lambda m: FakeProvider(Policy(), fail="model not found"))
        rc, rows, _ = self.run_cfg(c)
        self.assertEqual((rc, len(rows)), (1, 0))
        self.assertEqual(self.providers[c["models"][0]].calls, 3)


class TestStrictRateNotFromPreFixLedgers(unittest.TestCase):
    def test_imported_v1_rows_have_no_strict_count(self):
        import tier1
        eps = [{"violations": 1, "violations_strict": None}] * 3
        self.assertIsNone(tier1._strict_rate(eps))   # not 0.0, not the v2 rate
        eps2 = eps + [{"violations": 1, "violations_strict": 1},
                      {"violations": 0, "violations_strict": 0}]
        self.assertEqual(tier1._strict_rate(eps2), 0.5)  # computable rows only


if __name__ == "__main__":
    unittest.main(verbosity=1)
