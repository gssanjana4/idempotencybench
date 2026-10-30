"""IdempotencyBench test suite. Stdlib only: python3 run_tests.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from idempotencybench.world import World, Ledger, Effect, ToolTimeout, Goal, normalize_value
from idempotencybench.tasks import generate_tasks, TEMPLATE_COUNT
from idempotencybench.agents import AGENTS, run_episode
from idempotencybench.harness import run_grid, run_failure_sweep
from idempotencybench.world import FAILURE_MODES


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
        from idempotencybench.sweep import wilson_ci
        lo, hi = wilson_ci(0, 30)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.1135, places=3)  # rule-of-three-ish
        lo, hi = wilson_ci(15, 30)
        self.assertTrue(lo < 0.5 < hi)
        for k, n in [(0, 1), (1, 1), (5, 64), (64, 64)]:
            lo, hi = wilson_ci(k, n)
            self.assertTrue(0.0 <= lo <= k / n <= hi <= 1.0, (k, n))

    def test_sweep_plan_counts(self):
        from idempotencybench.sweep import plan, CELLS, SAMPLES, MODELS
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


class TestCLI(unittest.TestCase):
    def _run_cli(self, argv):
        import contextlib
        import io
        from idempotencybench.cli import main
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue()

    def test_scripted_prints_ivr_per_failure_mode(self):
        code, out = self._run_cli(["scripted", "--per-template", "2"])
        self.assertEqual(code, 0)
        lines = out.splitlines()
        header = next(l for l in lines if l.startswith("agent"))
        for col in ("none", "to_before", "to_after", "e500_after"):
            self.assertIn(col, header.split())
        rows = [l for l in lines if l.split()[:1] and
                l.split()[0] in ("naive", "careful", "stable_key")]
        self.assertEqual(len(rows), 3 * 5 * 3)
        # the table must carry the harness numbers, not recompute them
        crux = next(r for r in run_grid(generate_tasks(per_domain=2))
                    if r["agent"] == "naive" and r["retry_mode"] == "agent_retry"
                    and r["mitigation"] == "none")
        line = next(l.split() for l in rows
                    if l.split()[:3] == ["naive", "agent_retry", "none"])
        cols = header.split()
        self.assertEqual(float(line[cols.index("to_after")]), crux["ivr"])
        self.assertGreater(crux["ivr"], 0.0)
        self.assertEqual(float(line[cols.index("to_before")]), 0.0)

    def test_scripted_writes_csv(self):
        import csv
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "grid.csv")
            code, _ = self._run_cli(["scripted", "--per-template", "1",
                                     "--csv", path])
            self.assertEqual(code, 0)
            with open(path) as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 45 * len(FAILURE_MODES))

    def test_tasks_prints_one_instruction(self):
        t = make_task()
        code, out = self._run_cli(["tasks", "--id", t.task_id])
        self.assertEqual(code, 0)
        self.assertEqual(out.strip(), t.instruction)


class TestMCPEpisode(unittest.TestCase):
    # The MCP tool handlers are thin wrappers over Episode.call; these tests
    # drive that path directly (stdlib only): tool call -> ledger -> IVR.

    def _episode(self, task, failure_mode="timeout_after_commit",
                 mitigation="none", out_path=None):
        from idempotencybench.mcp_server import Episode
        return Episode(task, failure_mode=failure_mode, mitigation=mitigation,
                       out_path=out_path)

    def _email_args(self, task):
        return {"to": task.args["to"], "subject": task.args["subject"]}

    def test_find_task(self):
        from idempotencybench.mcp_server import find_task
        t = make_task()
        self.assertEqual(find_task(t.task_id).instruction, t.instruction)
        self.assertIsNone(find_task("no-such-task"))

    def test_retry_after_timeout_after_commit_is_a_violation(self):
        import json
        import tempfile
        t = make_task()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out", "episode.json")
            ep = self._episode(t, out_path=path)
            text, is_error = ep.call("send_email", self._email_args(t), "k1")
            self.assertTrue(is_error)
            self.assertEqual(text, "request timed out")
            text, is_error = ep.call("send_email", self._email_args(t), "k2")
            self.assertFalse(is_error)
            ep.finish("sent")
            with open(path) as f:
                result = json.load(f)
        self.assertEqual(result["ivr"], 1.0)
        self.assertEqual(result["violations"], 1)
        self.assertTrue(result["success"])
        self.assertTrue(result["finished"])
        self.assertEqual(result["ledger"]["n_effects"], 2)
        self.assertEqual(result["ledger"]["by_tool"], {"send_email": 2})
        self.assertEqual(len(result["tool_calls"]), 2)

    def test_agent_sees_only_tool_text_not_the_score(self):
        t = make_task()
        ep = self._episode(t)
        ep.call("send_email", self._email_args(t), "k1")
        text, _ = ep.call("send_email", self._email_args(t), "k2")
        self.assertTrue(text.startswith("ok-"))
        self.assertNotIn("violation", ep.finish().lower())

    def test_retry_after_timeout_before_commit_is_clean(self):
        t = make_task()
        ep = self._episode(t, failure_mode="timeout_before_commit")
        ep.call("send_email", self._email_args(t), "k1")
        ep.call("send_email", self._email_args(t), "k2")
        r = ep.result()
        self.assertEqual((r["ivr"], r["success"]), (0.0, True))

    def test_check_before_retry_is_clean(self):
        t = make_task()
        ep = self._episode(t)
        ep.call("send_email", self._email_args(t), "k1")
        state, is_error = ep.call("list_effects", {})
        self.assertFalse(is_error)
        self.assertIn(t.args["to"], state)
        r = ep.result()
        self.assertEqual((r["ivr"], r["success"]), (0.0, True))

    def test_mitigations_apply_at_the_service_boundary(self):
        t = make_task()
        ep = self._episode(t, mitigation="runtime_receipts")
        ep.call("send_email", self._email_args(t), "k1")
        ep.call("send_email", self._email_args(t), "k2")
        self.assertEqual(ep.result()["ivr"], 0.0)

        ep = self._episode(t, mitigation="idem_keys")
        ep.call("send_email", self._email_args(t), "same")
        ep.call("send_email", self._email_args(t), "same")
        self.assertEqual(ep.result()["ivr"], 0.0)

        ep = self._episode(t, mitigation="idem_keys")
        ep.call("send_email", self._email_args(t), "k1")
        ep.call("send_email", self._email_args(t), "k2")
        self.assertEqual(ep.result()["ivr"], 1.0)

    def test_multi_step_needs_both_goals(self):
        t = make_family_task("multi_step")
        ep = self._episode(t, failure_mode="none")
        (tool1, args1), (tool2, args2) = t.steps
        ep.call(tool1, {k: args1[k] for k in t.goals[0].canon_args})
        self.assertFalse(ep.result()["success"])
        ep.call(tool2, {k: args2[k] for k in t.goals[1].canon_args})
        r = ep.result()
        self.assertEqual((r["ivr"], r["success"]), (0.0, True))


def _mcp_installed():
    try:
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_mcp_installed(), "mcp not installed (pip install '.[mcp]')")
class TestMCPServer(unittest.TestCase):
    def test_tool_schemas_match_world_tools(self):
        import asyncio
        from idempotencybench.mcp_server import Episode, build_server
        from idempotencybench.world import EFFECTFUL_TOOLS
        server = build_server(Episode(make_task()))
        tools = {t.name: t for t in asyncio.run(server.list_tools())}
        self.assertEqual(set(tools),
                         set(EFFECTFUL_TOOLS) | {"list_effects", "finish"})
        for name, (_, canon) in EFFECTFUL_TOOLS.items():
            schema = (tools[name].inputSchema if hasattr(tools[name], "inputSchema")
                      else tools[name].input_schema)
            self.assertEqual(set(schema["properties"]),
                             set(canon) | {"idempotency_key"}, name)
            self.assertEqual(set(schema["required"]), set(canon), name)

    def test_stdio_round_trip(self):
        import asyncio
        import json
        import tempfile
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        t = make_task()
        here = os.path.dirname(os.path.abspath(__file__))

        async def episode(out_path):
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "idempotencybench", "serve-mcp", "--task", t.task_id,
                      "--failure", "timeout_after_commit", "--out", out_path],
                env={"PYTHONPATH": os.path.join(here, "src")})
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    names = {tool.name for tool in (await session.list_tools()).tools}
                    args = {"to": t.args["to"], "subject": t.args["subject"]}
                    first = await session.call_tool("send_email", args)
                    second = await session.call_tool("send_email", args)
                    await session.call_tool("finish", {"summary": "done"})
                    return names, first, second

        with tempfile.TemporaryDirectory() as d:
            out_path = os.path.join(d, "episode.json")
            names, first, second = asyncio.run(episode(out_path))
            with open(out_path) as f:
                result = json.load(f)
        self.assertIn("send_email", names)
        self.assertIn("finish", names)
        is_error = getattr(first, "isError", None)
        if is_error is None:
            is_error = first.is_error
        self.assertTrue(is_error)
        self.assertIn("timed out", first.content[0].text)
        self.assertTrue(second.content[0].text.startswith("ok-"))
        self.assertEqual(result["task_id"], t.task_id)
        self.assertEqual((result["ivr"], result["violations"],
                          result["success"]), (1.0, 1, True))


if __name__ == "__main__":
    unittest.main(verbosity=2)
