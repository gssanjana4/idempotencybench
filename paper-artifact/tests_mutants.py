"""Tests added by the code-verifier loop (iteration 1): each one kills a mutant that
survived every existing suite. Run from the repo root: python3 tests_mutants.py"""
import importlib.util, json, os, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("analysis", "src", "src2", "src_real"):
    sys.path.insert(0, os.path.join(ROOT, d))
import v3_common, v3_load                                     # noqa: E402
import sweep, sweep2, tier1, model_agent, resume              # noqa: E402
from world import World, Goal, Ledger, Effect, normalize_value  # noqa: E402
from tasks import generate_tasks                              # noqa: E402

_spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(ROOT, "paper-merged", "make_numbers.py"))
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)


def _wilson_ref(k, n, z=1.959964):
    p = k / n; a = 1 + z * z / n; b = -(2 * p + z * z / n); c = p * p
    d = (b * b - 4 * a * c) ** 0.5
    return ((-b - d) / (2 * a), (-b + d) / (2 * a))


class Stats(unittest.TestCase):
    def test_perm_p_uses_plus_one_correction_and_never_zero(self):
        # survivor VC03: p = hit/N printed 0.036 for the abstract's Haiku contrast instead of 0.037
        x = {f"t{i}": 0.0 for i in range(20)}; y = {f"t{i}": 1.0 for i in range(20)}
        old = v3_common.PERM_N
        try:
            v3_common.PERM_N = 1          # with one permutation p must be (hit+1)/(N+1) in {0.5, 1.0}
            r = v3_common.perm_test(x, y)
            self.assertIn(r["p"], (0.5, 1.0)); self.assertGreater(r["p"], 0.0)
        finally:
            v3_common.PERM_N = old
        old = mk.PERM_N
        try:
            mk.PERM_N = 1
            d, p = mk.perm_test([(0.0, 1.0)] * 20, "k")
            self.assertIn(p, (0.5, 1.0))
        finally:
            mk.PERM_N = old
        # the two implementations agree on a real-sized contrast (same seed)
        xs = {f"t{i}": (i % 3) / 2 for i in range(32)}; ys = {f"t{i}": ((i * 7) % 5) / 4 for i in range(32)}
        r = v3_common.perm_test(xs, ys)
        d2, p2 = mk.perm_test([(xs[t], ys[t]) for t in sorted(xs)], "k")
        self.assertAlmostEqual(r["p"], p2, places=12); self.assertAlmostEqual(r["d"], d2, places=12)

    def test_wilson_is_95pct_and_identical_in_every_copy(self):
        # survivor VC05: z=1.645 changed 416 printed CIs (e.g. every :ci and :gci) and no test failed
        for n in (16, 31, 32, 62, 64, 96, 128):
            for k in (0, 1, n // 3, n // 2, n - 1, n):
                lo, hi = _wilson_ref(k, n)
                for fn in (v3_common.wilson, sweep.wilson_ci, sweep2.wilson_ci, mk.wilson):
                    a, b = fn(k, n)
                    self.assertAlmostEqual(a, lo, places=4, msg=f"{fn.__module__} {k}/{n}")
                    self.assertAlmostEqual(b, hi, places=4, msg=f"{fn.__module__} {k}/{n}")
        self.assertAlmostEqual(v3_common.wilson(0, 64)[1], 0.0570, places=3)   # the paper's "<= 0.057"
        self.assertAlmostEqual(v3_common.wilson(1, 64)[0], 0.0028, places=3)


class Loaders(unittest.TestCase):
    def test_readback_is_undefined_when_the_failure_never_fired(self):
        # survivor VL01: read-back of episodes whose failure never fired was counted (E1-2 cells)
        base = {"episode_id": "e", "task_id": "t", "sample": 0, "violations": 0, "model": "m",
                "prompt": "base", "mitigation": "none", "failure_mode": "none"}
        self.assertIsNone(v3_load._ep(dict(base, failure_fired=False, readback_after_failure=True), "x")["rb"])
        self.assertEqual(v3_load._ep(dict(base, failure_fired=True, readback_after_failure=True), "x")["rb"], 1)
        self.assertEqual(v3_load._ep(dict(base, failure_fired=True, readback_after_failure=False), "x")["rb"], 0)
        self.assertIsNone(v3_load._ep(dict(base), "x")["rb"])


class MakeNumbers249(unittest.TestCase):
    """make_numbers.py's session (249) path had no test at all (survivors MK03, MK04, MK05)."""
    def rows(self):
        rows = []
        for g in range(16):
            for s in range(2):
                v = 3 if g < 5 and s == 0 else 0
                rows.append({"episode_id": f"m__amnesiac__none__g{g}__k1__s{s}", "model": "m", "resume_context": "amnesiac",
                             "mitigation": "none", "graph_id": f"g{g}", "interrupt_node": 1, "sample": s, "topology": "chain4",
                             "violations": v, "givr_hit": int(v > 0), "completion": 1, "order_violation": int(v > 0)})
        return rows

    def test_cell_counts_duplicating_episodes_not_duplicate_actions(self):
        rows = self.rows()
        trans = {}
        for r in rows:
            two = r["sample"] == 0   # sample 1 never resumed
            trans[("oct", "m", "amnesiac", "none", r["graph_id"], 1, r["sample"])] = {
                "sessions": [{"turns": []}] + ([{"turns": [{"role": "assistant", "content": [{"type": "tool_use", "name": "list_effects"}]}]}] if two else [])}
        mk.NUMS.clear()
        mk.emit_249_cell("oct:m:amnesiac:none", rows, "oct", trans, 32)
        N = mk.NUMS
        self.assertEqual(N["oct:m:amnesiac:none:kn"], "5/32")             # MK03 gave 15/32
        self.assertEqual(N["oct:m:amnesiac:none:givr"], "0.16")
        self.assertEqual(N["oct:m:amnesiac:none:meanviol"], "0.5"); self.assertEqual(N["oct:m:amnesiac:none:maxviol"], "3")
        self.assertEqual(N["oct:m:amnesiac:none:ci"], mk.ci_str(*mk.wilson(5, 32)))
        self.assertEqual(N["oct:m:amnesiac:none:gci"], mk.ci_str(*mk.wilson(5 / 32 * 16, 16)))   # MK04 used (k, n)
        self.assertNotEqual(N["oct:m:amnesiac:none:gci"], N["oct:m:amnesiac:none:ci"])
        self.assertEqual(N["oct:m:amnesiac:none:chk"], "16/16")             # only resumed episodes count (MK02)
        self.assertEqual(N["oct:m:amnesiac:none:resumed"], "16"); self.assertEqual(N["oct:m:amnesiac:none:notresumed"], "16")
        self.assertEqual(N["oct:m:amnesiac:none:stalekn"], "5/32"); self.assertEqual(N["oct:m:amnesiac:none:dupG"], "5")

    def test_smoke_runs_never_enter_the_paper(self):
        # survivor MK05: dropping the smoke filter changed total:all 6,798 -> 6,804 and nothing failed
        for pat_v2, pat in (("sweep_raw_v2.jsonl", "sweep_raw*.jsonl"), ("sweep2_raw_v2.jsonl", "sweep2_raw*.jsonl")):
            files = mk.raw_files(pat_v2, pat)
            self.assertTrue(files)
            for f in files:
                self.assertNotIn("smoke", f.split(os.sep), f)
                self.assertFalse(f.endswith(".bak"), f)
        if os.path.exists(os.path.join(mk.RES, "crossprovider", "smoke")):
            self.assertTrue(any(os.path.basename(os.path.dirname(f)) == "crossprovider"
                                for f in mk.raw_files("sweep_raw_v2.jsonl", "sweep_raw*.jsonl")))


class Scorer(unittest.TestCase):
    def test_violations_count_every_extra_commit(self):
        # survivor W06 (caught only by the rescore_c1 replay assertion, not by a unit test)
        w = World()
        a = {"to": "a@x.com", "subject": "s", "body": "b"}
        for _ in range(3):
            w.call("send_email", dict(a), "", "t")
        self.assertEqual(w.ledger.violations("t"), 2)
        w.call("create_ticket", {"queue": "q", "title": "T"}, "", "t")
        w.call("create_ticket", {"queue": "q", "title": "t "}, "", "t")
        self.assertEqual(w.ledger.violations("t"), 3)

    def test_truncation_stop_reasons_cover_every_provider(self):
        # survivor MA01: "length" (OpenRouter) dropped from TRUNCATED_STOPS and nothing failed
        for mod in (model_agent, sweep2):
            self.assertTrue({"max_tokens", "length", "error"} <= set(mod.TRUNCATED_STOPS), mod.__name__)

    def test_graph_completion_requires_every_node(self):
        # survivor RS04: completion = any(...) is indistinguishable from all(...) on the released data
        g, k = sweep2.benchmark_graphs()[0]
        w = World()
        n0 = g.nodes[0]
        w.call(n0.tool, dict(n0.args), "", g.graph_id)
        row = resume.score_graph(w, g)
        self.assertEqual((row["completion"], row["violations"], row["givr_hit"]), (0, 0, 0))
        for nd in g.nodes[1:]:
            w.call(nd.tool, dict(nd.args), "", g.graph_id)
        self.assertEqual(resume.score_graph(w, g)["completion"], 1)

    def test_stale_reexecution_is_not_plain_misorder(self):
        # survivor T103: stale_reexec >= 1 collapses into order_violation; both agree on every Tier-1 row
        task = next(t for t in generate_tasks(1) if t.ordered and len(t.goals) == 2)
        g0, g1 = task.goals
        led = [{"tool": g1.tool, "args": dict(g1.canon_args), "key": "", "seq": 1},
               {"tool": g0.tool, "args": dict(g0.canon_args), "key": "", "seq": 2}]
        t = {"episode_id": "e", "exp": "E", "model": "m", "prompt": "base", "mitigation": "none",
             "failure_mode": "timeout_before_commit", "retry_mode": "agent_retry", "sample": 0,
             "score": {"violations": 0, "achieved": False, "omission": False, "misorder": True},
             "turns": [], "max_turns": 8, "fail_on_call": 1, "failure_fired": True, "ledger": led}
        row = tier1.episode_row(t, task)
        self.assertTrue(row["order_violation"]); self.assertFalse(row["stale_reexec"])
        led.append({"tool": g0.tool, "args": dict(g0.canon_args), "key": "", "seq": 3})
        led.insert(0, {"tool": g0.tool, "args": dict(g0.canon_args), "key": "", "seq": 0})
        row = tier1.episode_row(dict(t, ledger=led), task)
        self.assertTrue(row["stale_reexec"]); self.assertTrue(row["order_violation"])


class RealStripe(unittest.TestCase):
    def setUp(self):
        import tests_real
        self.tr = tests_real
        self.fake = tests_real.FakeStripe().__enter__()
        self.c = tests_real.client_for(self.fake)
        self.cus = self.c.create_customer("att", {})["id"]

    def tearDown(self):
        self.fake.__exit__()

    def test_ground_truth_ignores_uncommitted_intents_and_normalizes_memo(self):
        from stripe_world import StripeWorld, stripe_ground_truth
        A = {"payee": "vendor-01", "amount": "437.00", "memo": "inv-1"}
        w = StripeWorld(self.c, episode_id="ep", attempt_id="att", customer=self.cus)
        w.call("send_payment", dict(A), "", "t")
        w.call("send_payment", dict(A, memo="INV-1 "), "", "t")                     # survivor SR02
        # a PaymentIntent that never committed (declined card) for the same attempt
        self.c.create_payment_intent(dict(self.tr._params(self.cus, att="att", memo="inv-1"), payment_method="pm_card_other"), "K-decl")
        gt = stripe_ground_truth(self.c, self.cus, "att", [Goal("send_payment", dict(A))])
        self.assertEqual(gt["n_payment_intents"], 3)
        self.assertEqual((gt["n_committed"], gt["violations"], gt["per_goal"]), (2, 1, [2]))   # survivor SR01 gave 3, 2
        self.assertIn("requires_payment_method", gt["statuses"])

    def test_aggregate_excludes_refusals_from_the_denominator(self):
        import run_real
        rows = [{"kind": "retry", "model": "m", "prompt": "base", "mitigation": "none", "failure_mode": "f",
                 "task_id": f"t{i}", "episode_id": f"e{i}", "ivr_stripe": i == 0, "ivr_ledger": i == 0,
                 "agree": True, "agree_exact": True, "achieved_stripe": True, "refusal": i == 3,
                 "failure_fired": True, "readback_after_failure": False, "retried": True,
                 "key_attached_first": False, "cost_usd": 0.0} for i in range(4)]
        with tempfile.TemporaryDirectory() as d:
            raw = os.path.join(d, "raw.jsonl"); summ = os.path.join(d, "summary.csv")
            with open(raw, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            out = run_real.aggregate({"raw_path": raw, "summary_path": summ})
        self.assertEqual((out[0]["n_episodes"], out[0]["refusals"], out[0]["n_tasks"]), (3, 1, 3))   # survivor SR04 gave 4, 0
        self.assertEqual(out[0]["ivr_stripe"], round(1 / 3, 4))


if __name__ == "__main__":
    unittest.main(verbosity=1)
