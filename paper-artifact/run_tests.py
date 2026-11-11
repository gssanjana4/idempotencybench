"""Benchmark test suite. Stdlib only: python3 run_tests.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from world import World, Ledger, Effect, ToolTimeout, Goal, normalize_value
from tasks import generate_tasks, TEMPLATE_COUNT
from agents import AGENTS, run_episode
from harness import run_grid, run_failure_sweep
from world import FAILURE_MODES


def make_task():
    return generate_tasks(per_domain=1)[0]  # one email task


def make_family_task(family):
    return next(t for t in generate_tasks(per_domain=1) if t.family == family)


class TestLedger(unittest.TestCase):
    def test_semantic_duplicate_detected_despite_different_keys(self):
        led = Ledger()
        args = {"to": "a@example.com", "subject": "s"}
        led.commit(Effect("send_email", args, "key1", "t1", 0))
        led.commit(Effect("send_email", args, "key2", "t1", 0))
        self.assertEqual(led.violations("t1"), 1)

    def test_different_args_not_duplicate(self):
        led = Ledger()
        led.commit(Effect("send_email", {"to": "a@x.com", "subject": "s1"}, "k", "t1", 0))
        led.commit(Effect("send_email", {"to": "a@x.com", "subject": "s2"}, "k", "t1", 0))
        self.assertEqual(led.violations("t1"), 0)


class TestMatcherNormalization(unittest.TestCase):
    # exact string equality is too fragile — formatting
    # variants of the same value must match; distinct values must not.

    def test_normalize_formatting_variants(self):
        self.assertEqual(normalize_value("$1,200.50"), normalize_value("1200.5"))
        self.assertEqual(normalize_value("450.00"), normalize_value("450"))
        self.assertEqual(normalize_value("Q3  Update "), normalize_value("q3 update"))
        self.assertEqual(normalize_value("'quoted'"), normalize_value("quoted"))

    def test_normalize_preserves_distinctions(self):
        self.assertNotEqual(normalize_value("450.01"), normalize_value("450"))
        self.assertNotEqual(normalize_value("a@x.com"), normalize_value("b@x.com"))

    def test_reformatted_amount_is_still_a_duplicate(self):
        led = Ledger()
        args1 = {"payee": "vendor-01", "amount": "450.00", "memo": "inv-1"}
        args2 = {"payee": "Vendor-01", "amount": "$450", "memo": "INV-1"}
        led.commit(Effect("send_payment", args1, "k1", "t1", 0))
        led.commit(Effect("send_payment", args2, "k2", "t1", 0))
        self.assertEqual(led.violations("t1"), 1)

    def test_iso_datetime_seconds_are_formatting_only(self):
        from world import ToolServerError
        self.assertEqual(normalize_value("2026-08-01T10:00"), normalize_value("2026-08-01T10:00:00"))
        w = World("none", "error_500_after_commit"); a = {"calendar_id": "team", "title": "Sync", "start": "2026-08-01T10:00"}
        self.assertRaises(ToolServerError, w.call, "create_event", a, "", "t")
        w.call("create_event", dict(a, start="2026-08-01T10:00:00"), "", "t")
        self.assertEqual(w.ledger.violations("t"), 1)

    def test_achieved_tolerates_reformatting(self):
        led = Ledger()
        led.commit(Effect("send_email",
                          {"to": "Alex@Example.com", "subject": "q3 status update #000"},
                          "k", "t1", 0))
        goal = Goal("send_email", {"to": "alex@example.com",
                                   "subject": "Q3 status update #000"})
        self.assertTrue(led.achieved("t1", goal))


class TestFailureInjection(unittest.TestCase):
    def test_timeout_after_commit_commits(self):
        w = World(failure_mode="timeout_after_commit")
        t = make_task()
        with self.assertRaises(ToolTimeout):
            w.call(t.tool, t.args, "k", t.task_id)
        self.assertEqual(len(w.ledger.for_task(t.task_id)), 1)

    def test_timeout_before_commit_commits_nothing(self):
        w = World(failure_mode="timeout_before_commit")
        t = make_task()
        with self.assertRaises(ToolTimeout):
            w.call(t.tool, t.args, "k", t.task_id)
        self.assertEqual(len(w.ledger.for_task(t.task_id)), 0)


class TestAgentsUnderRetry(unittest.TestCase):
    def _run(self, agent, retry_mode, mitigation="none"):
        t = make_task()
        w = World(mitigation=mitigation, failure_mode="timeout_after_commit")
        run_episode(AGENTS[agent], w, t, retry_mode)
        return w.ledger.violations(t.task_id), w.ledger.achieved(t.task_id, t.goal)

    def test_naive_duplicates_on_agent_retry(self):
        v, ok = self._run("naive", "agent_retry")
        self.assertGreater(v, 0); self.assertTrue(ok)

    def test_naive_duplicates_on_tool_retry(self):
        v, _ = self._run("naive", "tool_retry")
        self.assertGreater(v, 0)

    def test_naive_duplicates_on_whole_run(self):
        # run 1 surfaces a failure -> orchestrator legitimately re-runs ->
        # naive agent re-commits
        v, _ = self._run("naive", "whole_run")
        self.assertGreater(v, 0)

    def test_careful_stops_agent_retry_duplicates(self):
        v, ok = self._run("careful", "agent_retry")
        self.assertEqual(v, 0); self.assertTrue(ok)

    def test_careful_stops_failure_conditioned_whole_run(self):
        # W2 fix: with a failure-conditioned orchestrator, the careful agent
        # verifies via read-back, reports success, and no re-run happens.
        v, ok = self._run("careful", "whole_run")
        self.assertEqual(v, 0); self.assertTrue(ok)

    def test_careful_does_NOT_stop_at_least_once_duplicates(self):
        # the paper's key negative result, relocated to the honest condition:
        # an at-least-once redelivery presents no failure signal, so
        # prompt-level care never triggers.
        v, _ = self._run("careful", "at_least_once")
        self.assertGreater(v, 0)

    def test_stable_keys_plus_idem_keys_robust_everywhere(self):
        for mode in ("agent_retry", "tool_retry", "whole_run", "at_least_once"):
            v, ok = self._run("stable_key", mode, mitigation="idem_keys")
            self.assertEqual(v, 0, mode); self.assertTrue(ok, mode)

    def test_fresh_keys_defeat_idem_keys_mitigation(self):
        v, _ = self._run("naive", "whole_run", mitigation="idem_keys")
        self.assertGreater(v, 0)

    def test_receipts_robust_for_all_agents_and_modes(self):
        for agent in ("naive", "careful", "stable_key"):
            for mode in ("agent_retry", "tool_retry", "whole_run",
                         "at_least_once"):
                v, ok = self._run(agent, mode, mitigation="runtime_receipts")
                self.assertEqual(v, 0, (agent, mode))
                self.assertTrue(ok, (agent, mode))


class TestTaskFamilies(unittest.TestCase):
    # structurally distinct families, honestly counted.

    def _run_family(self, family, agent, retry_mode, mitigation="none",
                    failure_mode="timeout_after_commit"):
        t = make_family_task(family)
        w = World(mitigation=mitigation, failure_mode=failure_mode)
        run_episode(AGENTS[agent], w, t, retry_mode)
        ok = all(w.ledger.achieved(t.task_id, g) for g in t.goals)
        return w.ledger.violations(t.task_id), ok, t

    def test_multi_step_task_has_two_steps_and_goals(self):
        t = make_family_task("multi_step")
        self.assertEqual(len(t.steps), 2)
        self.assertEqual(len(t.goals), 2)
        self.assertNotEqual(t.steps[0][0], t.steps[1][0])

    def test_multi_step_naive_whole_run_duplicates_committed_step(self):
        # step 1 commits then times out; the failure-conditioned re-run
        # re-commits step 1 -> exactly one violation, but both goals achieved
        v, ok, _ = self._run_family("multi_step", "naive", "whole_run")
        self.assertEqual(v, 1); self.assertTrue(ok)

    def test_multi_step_careful_whole_run_clean(self):
        v, ok, _ = self._run_family("multi_step", "careful", "whole_run")
        self.assertEqual(v, 0); self.assertTrue(ok)

    def test_multi_step_receipts_robust_under_at_least_once(self):
        v, ok, _ = self._run_family("multi_step", "naive", "at_least_once",
                                    mitigation="runtime_receipts")
        self.assertEqual(v, 0); self.assertTrue(ok)

    def test_every_canonical_value_stated_in_instruction(self):
        # A model agent only sees the instruction. If a canonical goal value
        # is absent from it, the task is unachievable for models and scoring
        # is silently broken (pilot audit 2026-07-11: files template).
        for t in generate_tasks(per_domain=3):
            ni = normalize_value(t.instruction)
            for g in t.goals:
                for k, v in g.canon_args.items():
                    self.assertIn(normalize_value(v), ni,
                                  (t.family, t.task_id, k, v))

    def test_all_families_present_with_honest_counts(self):
        ts = generate_tasks(per_domain=2)
        from collections import Counter
        by_family = Counter(t.family for t in ts)
        self.assertEqual(by_family["atomic"], 5 * 2)   # 5 atomic templates
        self.assertEqual(by_family["multi_step"], 2)
        self.assertEqual(by_family["read_then_act"], 2)
        self.assertEqual(by_family["ambiguous"], 2)
        self.assertEqual(TEMPLATE_COUNT, 8)


class TestSweepStats(unittest.TestCase):
    def test_wilson_ci_bounds_and_known_values(self):
        from sweep import wilson_ci
        lo, hi = wilson_ci(0, 30)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1135, places=3)  # rule-of-three-ish
        lo, hi = wilson_ci(15, 30)
        self.assertTrue(lo < 0.5 < hi)
        for k, n in [(0, 1), (1, 1), (5, 64), (64, 64)]:
            lo, hi = wilson_ci(k, n)
            self.assertTrue(0.0 <= lo <= k / n <= hi <= 1.0, (k, n))

    def test_sweep_plan_counts(self):
        from sweep import plan, CELLS, SAMPLES, MODELS
        tasks, episodes = plan()
        self.assertEqual(len(tasks), 32)          # 4 x 8 templates, n >= 30
        self.assertEqual(len(episodes),
                         len(MODELS) * len(CELLS) * 32 * SAMPLES)


class TestTasksAndHarness(unittest.TestCase):
    def test_320_unique_tasks(self):
        ts = generate_tasks(per_domain=40)
        self.assertEqual(len(ts), 320)  # 8 templates x 40
        self.assertEqual(len({t.task_id for t in ts}), 320)

    def test_grid_produces_full_matrix(self):
        rows = run_grid(generate_tasks(per_domain=2))
        self.assertEqual(len(rows), 3 * 5 * 3)  # agents x retry x mitigation
        for r in rows:
            self.assertGreaterEqual(r["ivr"], 0.0)
            self.assertLessEqual(r["ivr"], 1.0)

    def test_failure_sweep_covers_all_modes(self):
        ts = generate_tasks(per_domain=2)
        rows = run_failure_sweep(ts)
        # one full grid (3 agents x 5 retry x 3 mitigation = 45) per mode
        self.assertEqual(len(rows), 45 * len(FAILURE_MODES))
        self.assertEqual({r["failure_mode"] for r in rows}, set(FAILURE_MODES))

    def test_timeout_before_commit_never_violates(self):
        # nothing commits on the failing call, so any retry is *correct*:
        # no cell should record an idempotency violation under this mode.
        ts = generate_tasks(per_domain=3)
        rows = run_grid(ts, failure_mode="timeout_before_commit")
        for r in rows:
            self.assertEqual(r["ivr"], 0.0, (r["agent"], r["retry_mode"]))

    def test_error_500_mirrors_timeout_after_commit(self):
        # both modes commit then signal failure, so the naive crux cell
        # (agent_retry, no mitigation) must violate identically.
        ts = generate_tasks(per_domain=3)
        a = run_grid(ts, failure_mode="error_500_after_commit")
        b = run_grid(ts, failure_mode="timeout_after_commit")
        def crux(rows):
            return next(r for r in rows if r["agent"] == "naive"
                        and r["retry_mode"] == "agent_retry"
                        and r["mitigation"] == "none")
        self.assertEqual(crux(a)["ivr"], crux(b)["ivr"])
        self.assertGreater(crux(a)["ivr"], 0.0)

    def test_no_retry_no_failure_is_clean(self):
        t = make_task()
        w = World(failure_mode="none")
        run_episode(AGENTS["naive"], w, t, "no_retry")
        self.assertEqual(w.ledger.violations(t.task_id), 0)
        self.assertTrue(w.ledger.achieved(t.task_id, t.goal))


class TestIdemKeysKeylessRegression(unittest.TestCase):
    """idem_keys must never deduplicate a call that carries no key."""

    def test_keyless_retry_under_idem_keys_is_a_violation(self):
        w = World(mitigation="idem_keys", failure_mode="timeout_after_commit")
        t = make_task()
        with self.assertRaises(ToolTimeout):
            w.call(t.tool, t.args, "", t.task_id)      # commits, then times out
        w.call(t.tool, t.args, "", t.task_id)          # keyless retry: must commit again
        self.assertEqual(len(w.ledger.for_task(t.task_id)), 2)
        self.assertEqual(w.ledger.violations(t.task_id), 1)

    def test_keyless_retry_matches_no_mitigation(self):
        t = make_task()
        results = []
        for mit in ("none", "idem_keys"):
            w = World(mitigation=mit, failure_mode="timeout_after_commit")
            with self.assertRaises(ToolTimeout):
                w.call(t.tool, t.args, "", t.task_id)
            w.call(t.tool, t.args, "", t.task_id)
            results.append(w.ledger.violations(t.task_id))
        self.assertEqual(results, [1, 1])

    def test_stable_nonempty_key_still_dedupes(self):
        w = World(mitigation="idem_keys", failure_mode="timeout_after_commit")
        t = make_task()
        with self.assertRaises(ToolTimeout):
            w.call(t.tool, t.args, "k1", t.task_id)
        w.call(t.tool, t.args, "k1", t.task_id)
        self.assertEqual(w.ledger.violations(t.task_id), 0)


class TestOutcomeOrderingAndOmission(unittest.TestCase):
    """Success requires every goal, committed in dependency order; omission
    and misorder are distinct failure types; the three partition outcomes."""

    def _ms(self):
        return make_family_task("multi_step")

    def _commit(self, led, task, order):
        for i in order:
            tool, args = task.steps[i]
            led.commit(Effect(tool, dict(args), "", task.task_id, 0))

    def test_multi_step_is_ordered_atomic_is_not(self):
        self.assertTrue(self._ms().ordered)
        self.assertFalse(make_task().ordered)

    def test_in_order_is_success(self):
        t, led = self._ms(), Ledger()
        self._commit(led, t, [0, 1])
        self.assertEqual(led.outcome(t.task_id, t.goals, t.ordered),
                         {"success": True, "omission": False, "misorder": False})

    def test_out_of_order_is_misorder_not_success(self):
        t, led = self._ms(), Ledger()
        self._commit(led, t, [1, 0])        # email before the ticket exists
        self.assertEqual(led.outcome(t.task_id, t.goals, t.ordered),
                         {"success": False, "omission": False, "misorder": True})

    def test_missing_step_is_omission(self):
        t, led = self._ms(), Ledger()
        self._commit(led, t, [0])
        self.assertEqual(led.outcome(t.task_id, t.goals, t.ordered),
                         {"success": False, "omission": True, "misorder": False})

    def test_omission_is_not_just_one_minus_success(self):
        t, led = self._ms(), Ledger()
        self._commit(led, t, [1, 0])
        out = led.outcome(t.task_id, t.goals, t.ordered)
        self.assertFalse(out["success"]); self.assertFalse(out["omission"])

    def test_outcomes_partition(self):
        t = self._ms()
        for order in ([], [0], [1], [0, 1], [1, 0], [1, 0, 1]):
            led = Ledger(); self._commit(led, t, order)
            out = led.outcome(t.task_id, t.goals, t.ordered)
            self.assertEqual(sum(out.values()), 1, order)

    def test_retry_duplicate_of_step1_after_step2_does_not_break_order(self):
        # first commit of each goal decides order; a later duplicate is an
        # IVR violation, not a misorder
        t, led = self._ms(), Ledger()
        self._commit(led, t, [0, 1, 0])
        out = led.outcome(t.task_id, t.goals, t.ordered)
        self.assertTrue(out["success"])
        self.assertEqual(led.violations(t.task_id), 1)


class TestInvalidToolCallsDoNotCommit(unittest.TestCase):
    # Round-2 audit F5: unknown tools crashed the sweep; partial/malformed
    # args were committed and two of them scored as a duplicate.
    def test_unknown_tool_is_a_tool_error_and_commits_nothing(self):
        w = World(failure_mode="timeout_after_commit")
        with self.assertRaises(ValueError):  # ToolCallError subclasses ValueError
            w.call("file_ticket", {"queue": "q", "title": "t"}, "", "t1")
        self.assertEqual(w.ledger.for_task("t1"), [])

    def test_partial_args_twice_is_not_a_violation(self):
        w = World(failure_mode="none")
        for _ in range(2):
            with self.assertRaises(ValueError):  # ToolCallError subclasses ValueError
                w.call("send_email", {"to": "a@example.com"}, "", "t1")
            with self.assertRaises(ValueError):  # ToolCallError subclasses ValueError
                w.call("send_email", {"_unparsed_arguments": "{bad"}, "", "t1")
        self.assertEqual(w.ledger.for_task("t1"), [])
        self.assertEqual(w.ledger.violations("t1"), 0)

    def test_invalid_call_does_not_consume_injected_failure(self):
        w = World(failure_mode="timeout_after_commit")
        with self.assertRaises(ValueError):  # ToolCallError subclasses ValueError
            w.call("send_email", {"to": "a@example.com", "subject": ""}, "", "t1")
        with self.assertRaises(ToolTimeout):
            w.call("send_email", {"to": "a@example.com", "subject": "s"}, "", "t1")
        self.assertEqual(len(w.ledger.for_task("t1")), 1)

    def test_valid_call_unchanged(self):
        w = World(failure_mode="none")
        out = w.call("send_payment", {"payee": "v", "amount": 450, "memo": "m"}, "", "t1")
        self.assertTrue(out.startswith("ok-"))


class TestLaunchRowCarriesOutcomeFields(unittest.TestCase):
    # Round-2 audit F3: sweep.py --launch wrote only violations/achieved, so
    # aggregate() reported omission = misorder = 0 for every new run.
    def test_launch_writes_omission_and_misorder(self):
        import json, tempfile, types
        import sweep, model_agent
        # launch() builds an anthropic client; stub it (no SDK/key needed)
        real_anthropic = sys.modules.get("anthropic")
        sys.modules["anthropic"] = types.SimpleNamespace(Anthropic=lambda: None)

        def fake_episode(client, world, task, system, model=None):
            steps = list(task.steps)
            if task.family == "multi_step":
                steps = steps[::-1]          # dependency order violated
            elif task.family == "atomic":
                steps = []                   # nothing done -> omission
            for tool, args in steps:
                world.call(tool, dict(args), "", task.task_id)
            return {"model": model, "task_id": task.task_id,
                    "family": task.family, "turns": []}

        saved = (sweep.MODELS, sweep.CELLS, sweep.SAMPLES, sweep.RAW_PATH,
                 sweep.SUMMARY_PATH, sweep.TRANSCRIPT_DIR, sweep.LIMIT,
                 model_agent.run_model_episode, sweep.OUT_DIR)
        with tempfile.TemporaryDirectory() as d:
            try:
                sweep.MODELS = ["openrouter:org/fake-model"]
                sweep.CELLS = [("base", "none", "none")]
                sweep.SAMPLES = 1
                sweep.LIMIT = 0
                sweep.RAW_PATH = os.path.join(d, "raw.jsonl")
                sweep.SUMMARY_PATH = os.path.join(d, "summary.csv")
                sweep.TRANSCRIPT_DIR = os.path.join(d, "tr")
                sweep.OUT_DIR = d   # run lock file goes here, not in results/
                model_agent.run_model_episode = fake_episode
                sweep.launch()
                rows = [json.loads(l) for l in open(sweep.RAW_PATH)]
                summary = open(sweep.SUMMARY_PATH).read().splitlines()
                tr_files = sorted(os.listdir(sweep.TRANSCRIPT_DIR))
                import rescore
                rescore_eids = set()
                for fn in tr_files:
                    mm = rescore.FNAME.match(fn)
                    t = json.load(open(os.path.join(sweep.TRANSCRIPT_DIR, fn)))
                    model = t.get("model", mm["model"])
                    rescore_eids.add(sweep.episode_id(model, mm["prompt"], mm["mit"],
                                                      mm["fm"], mm["task"], int(mm["s"])))
            finally:
                (sweep.MODELS, sweep.CELLS, sweep.SAMPLES, sweep.RAW_PATH,
                 sweep.SUMMARY_PATH, sweep.TRANSCRIPT_DIR, sweep.LIMIT,
                 model_agent.run_model_episode, sweep.OUT_DIR) = saved
                if real_anthropic is not None:
                    sys.modules["anthropic"] = real_anthropic
                else:
                    sys.modules.pop("anthropic", None)
        # model ids with "/" (OpenRouter): flat filename that rescore parses,
        # raw id kept inside the JSON and used for episode_id on rescore
        import rescore
        self.assertEqual(len(tr_files), len(rows))
        m = rescore.FNAME.match(tr_files[0]); self.assertTrue(m)
        self.assertEqual(rescore_eids, {r["episode_id"] for r in rows})
        ms = [r for r in rows if r["family"] == "multi_step"]
        at = [r for r in rows if r["family"] == "atomic"]
        self.assertTrue(ms and at)
        self.assertTrue(all(r["misorder"] for r in ms))
        self.assertTrue(all(r["omission"] for r in at))
        hdr = summary[0].split(","); vals = summary[1].split(",")
        self.assertGreater(float(vals[hdr.index("misorder")]), 0)
        self.assertGreater(float(vals[hdr.index("omission")]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
