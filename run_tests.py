"""IdempotencyBench test suite. Stdlib only: python3 run_tests.py"""
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
    # mock-review W3: exact string equality is too fragile — formatting
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
    # mock-review W4: structurally distinct families, honestly counted.

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
