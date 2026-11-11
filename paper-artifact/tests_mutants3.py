"""Tests added by the code-verifier loop (iteration 3): each kills a mutant of code that
neither earlier round had targeted (src/world.py ledger/matcher/dedupe/failure timing,
src/sweep.py aggregate(), src2/sweep2.py, src2/resume.py, analysis/v3_common.py,
analysis/v3_bare_hint.py, analysis/v3_sens_oct*.py, analysis/v3_tier1_prereg.py,
src/tier1.py and the 209/249/summary sections of paper-merged/make_numbers.py) and
survived every existing suite. Run from the repo root:  python3 tests_mutants3.py"""
import contextlib, csv, glob, hashlib, importlib.util, io, json, math, os, random, shutil, sys, tempfile, unittest
ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("analysis", "src", "src2", "src_real"):
    sys.path.insert(0, os.path.join(ROOT, d))
import v3_common                                   # noqa: E402
from v3_common import Numbers, wilson              # noqa: E402
import v3_bare_hint as bh                          # noqa: E402
import v3_sens_oct as so                           # noqa: E402
import v3_sens_oct_fix as sof                      # noqa: E402
import v3_tier1_prereg as tp                       # noqa: E402
import v3_load                                     # noqa: E402
import sweep, sweep2, tier1, resume                # noqa: E402
from world import (World, Goal, Ledger, Effect, normalize_value, validate_call,   # noqa: E402
                   ToolCallError, ToolTimeout, ToolServerError, SIGNAL_VARIANTS)
from graph import _chain4                          # noqa: E402
from tasks import generate_long_tasks              # noqa: E402

_spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(ROOT, "paper-merged", "make_numbers.py"))
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)

HAIKU, SONNET, FABLE = "claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-fable-5"
PREREG = open(os.path.join(ROOT, "analysis", "preregistration.md")).read()


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


# ---------------------------------------------------------------------------
# src/world.py: matcher, ledger, dedupe, failure injection
# ---------------------------------------------------------------------------
class WorldMatcher(unittest.TestCase):
    def test_iso_fix_drops_zero_seconds_only(self):
        # survivor NW01: any seconds were dropped, so 10:00:30 == 10:00 (content, not formatting)
        self.assertEqual(normalize_value("2026-08-01T10:00:00"), normalize_value("2026-08-01T10:00"))
        self.assertNotEqual(normalize_value("2026-08-01T10:00:30"), normalize_value("2026-08-01T10:00"))
        self.assertEqual(normalize_value("2026-08-01T10:00:30"), "2026-08-01t10:00:30")
        # make_numbers.norm is the same rule (survivor NMK27)
        self.assertEqual(mk.norm("2026-08-01T10:00:00"), mk.norm("2026-08-01T10:00"))
        self.assertEqual(mk.norm("2026-08-01T10:00:00"), normalize_value("2026-08-01T10:00:00"))
        self.assertNotEqual(mk.norm("2026-08-01T10:00:30"), mk.norm("2026-08-01T10:00"))

    def test_validate_rejects_bool_fields(self):
        # survivor NW22
        with self.assertRaises(ToolCallError):
            validate_call("send_email", {"to": True, "subject": "s"})
        validate_call("send_email", {"to": "a@example.com", "subject": "s"})


class WorldLedger(unittest.TestCase):
    def test_ledger_is_per_task(self):
        # survivor NW06: for_task returned every task's effects
        w = World()
        a = {"to": "a@example.com", "subject": "s"}
        w.call("send_email", dict(a), "", "t1")
        w.call("send_email", dict(a), "", "t2")
        self.assertEqual(len(w.ledger.for_task("t1")), 1)
        self.assertEqual(w.ledger.violations("t1"), 0); self.assertEqual(w.ledger.violations("t2"), 0)
        self.assertEqual(w.ledger.outcome("t1", [Goal("send_email", a)]), {"success": True, "omission": False, "misorder": False})

    def test_achieved_checks_the_canonical_args(self):
        # survivor NW08: achieved() matched on the tool name alone
        w = World()
        w.call("send_email", {"to": "a@example.com", "subject": "other"}, "", "t")
        self.assertFalse(w.ledger.achieved("t", Goal("send_email", {"to": "a@example.com", "subject": "wanted"})))
        self.assertTrue(w.ledger.achieved("t", Goal("send_email", {"to": "A@example.com ", "subject": "OTHER"})))

    def test_semantic_identity_includes_the_tool(self):
        # survivor NW09: two tools with the same canonical VALUES were duplicates of each other
        w = World()
        w.call("send_email", {"to": "x", "subject": "y"}, "", "t")
        w.call("create_ticket", {"queue": "x", "title": "y"}, "", "t")
        self.assertEqual(w.ledger.violations("t"), 0)
        e = w.ledger.for_task("t")
        self.assertNotEqual(e[0].semantic_id(), e[1].semantic_id())
        self.assertIn("send_email", e[0].semantic_id())

    def test_commit_order_is_global_and_increasing(self):
        w = World()
        w.call("send_email", {"to": "x", "subject": "y"}, "", "t")
        w.call("create_ticket", {"queue": "q", "title": "y"}, "", "t")
        self.assertEqual([e.seq for e in w.ledger.for_task("t")], [1, 2])


class WorldDedupe(unittest.TestCase):
    def test_idem_key_is_scoped_to_the_tool(self):
        # survivor NW12: the same key on two different tools deduplicated the second
        w = World(mitigation="idem_keys")
        w.call("send_email", {"to": "x", "subject": "y"}, "K", "t")
        w.call("create_ticket", {"queue": "q", "title": "y"}, "K", "t")
        self.assertEqual(len(w.ledger.for_task("t")), 2)
        w.call("send_email", {"to": "x", "subject": "y"}, "K", "t")      # same tool, same key: deduped
        self.assertEqual(len(w.ledger.for_task("t")), 2)

    def test_receipts_are_scoped_to_the_task_and_use_every_canonical_field(self):
        # survivors NW13 (receipt ignored task_id) and NW14 (World._semantic_id dropped the last field)
        w = World(mitigation="runtime_receipts")
        w.call("send_payment", {"payee": "v", "amount": "10", "memo": "m1"}, "", "t1")
        w.call("send_payment", {"payee": "v", "amount": "10", "memo": "m1"}, "", "t2")
        self.assertEqual(len(w.ledger.effects), 2, "another task's receipt must not suppress a commit")
        w.call("send_payment", {"payee": "v", "amount": "10", "memo": "m2"}, "", "t1")
        self.assertEqual(len(w.ledger.for_task("t1")), 2, "a payment with another memo is a distinct action")
        self.assertEqual(w.ledger.violations("t1"), 0)
        r = w.call("send_payment", {"payee": "v", "amount": "10.0", "memo": "M1"}, "", "t1")
        self.assertEqual(r, "ok-deduped-receipt"); self.assertEqual(len(w.ledger.for_task("t1")), 2)
        # the two identity functions agree
        self.assertEqual(w._semantic_id("send_payment", {"payee": "v", "amount": "10", "memo": "m1"}, "t1"),
                         w.ledger.for_task("t1")[0].semantic_id())


class WorldFailure(unittest.TestCase):
    def test_w3_is_a_timeout_class_error_and_w1_a_server_error(self):
        # survivor NW18 (preregistration E1-6: W3 "raised as a timeout-class error")
        self.assertEqual(SIGNAL_VARIANTS["status_unknown_after_commit"][0], "timeout")
        self.assertEqual(SIGNAL_VARIANTS["error_500_body_after_commit"][0], "server")
        w = World(failure_mode="status_unknown_after_commit")
        with self.assertRaises(ToolTimeout):
            w.call("send_email", {"to": "x", "subject": "y"}, "", "t")
        self.assertEqual(len(w.ledger.for_task("t")), 1)
        w = World(failure_mode="error_500_body_after_commit")
        with self.assertRaises(ToolServerError) as cm:
            w.call("send_email", {"to": "x", "subject": "y"}, "", "t")
        self.assertIn("Retry-After", str(cm.exception)); self.assertIn("request_id", str(cm.exception))


# ---------------------------------------------------------------------------
# src/sweep.py aggregate()  (survivors NS01-NS05, NS10: no test touched aggregate at all)
# ---------------------------------------------------------------------------
class SweepAggregate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name
        self.saved = (sweep.RAW_PATH, sweep.SUMMARY_PATH)
        sweep.RAW_PATH = os.path.join(d, "sweep_raw.jsonl"); sweep.SUMMARY_PATH = os.path.join(d, "sweep_summary.csv")

    def tearDown(self):
        sweep.RAW_PATH, sweep.SUMMARY_PATH = self.saved
        self.tmp.cleanup()

    def row(self, task, s, v, ach=True):
        return {"episode_id": f"m__base__none__toA__{task}__s{s}", "model": "m", "prompt": "base", "mitigation": "none",
                "failure_mode": "toA", "task_id": task, "family": "atomic", "sample": s, "violations": v, "achieved": ach,
                "omission": False, "misorder": False}

    def test_aggregate_last_row_wins_counts_episodes_and_task_any(self):
        rows = [self.row("t0", 0, 0), self.row("t0", 1, 2), self.row("t1", 0, 0), self.row("t1", 1, 0),
                self.row("t2", 0, 1), self.row("t2", 1, 1), self.row("t3", 0, 1), self.row("t3", 1, 0)]
        rows.append(self.row("t1", 1, 1))        # the same episode run twice: the LAST row is the transcript on disk
        jl(sweep.RAW_PATH, rows)
        with contextlib.redirect_stdout(io.StringIO()):
            sweep.aggregate()
        r = list(csv.DictReader(open(sweep.SUMMARY_PATH)))[0]
        self.assertEqual(int(r["n_episodes"]), 8)                      # NS01: first row kept gave the same n, other k
        self.assertEqual(float(r["ivr"]), round(5 / 8, 4))             # NS01/NS02: violations > 0, last rows
        self.assertEqual(int(r["n_tasks"]), 4)
        self.assertEqual(float(r["ivr_task"]), 1.0)                     # NS03: a task violates if ANY sample does
        lo, hi = sweep.wilson_ci(4, 4)
        self.assertEqual((float(r["ivr_task_ci_lo"]), float(r["ivr_task_ci_hi"])), (round(lo, 4), round(hi, 4)))   # NS05
        self.assertEqual(float(r["violation_depth"]), round(6 / 8, 4))   # NS10: mean duplicate count, not episode rate

    def test_aggregate_refuses_to_overwrite_the_released_summary(self):
        # survivor NS04
        released = os.path.join(sweep.HERE, "results", "sweep_summary.csv")
        if not os.path.exists(released):
            self.skipTest("released summary not present")
        before = open(released, "rb").read()
        jl(sweep.RAW_PATH, [self.row("t0", 0, 1)])
        sweep.SUMMARY_PATH = released
        env = os.environ.pop("BENCH_WRITE_RELEASED", None)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                sweep.aggregate()
        finally:
            if env is not None:
                os.environ["BENCH_WRITE_RELEASED"] = env
        self.assertEqual(open(released, "rb").read(), before)


# ---------------------------------------------------------------------------
# src2/sweep2.py, src2/resume.py
# ---------------------------------------------------------------------------
class Sweep2Summary(unittest.TestCase):
    def test_summarize_givr_counts_episodes(self):
        # survivor NS209 (violations summed in place of givr_hit)
        tmp = tempfile.TemporaryDirectory(); d = tmp.name
        saved = (sweep2.RAW_PATH, sweep2.SUMMARY_PATH)
        sweep2.RAW_PATH = os.path.join(d, "r.jsonl"); sweep2.SUMMARY_PATH = os.path.join(d, "s.csv")
        try:
            rows = [{"episode_id": f"m__amnesiac__none__g{i}__k1__s0", "model": "m", "resume_context": "amnesiac",
                     "mitigation": "none", "givr_hit": 1, "violations": 3, "order_violation": 0, "completion": 1} for i in range(4)]
            jl(sweep2.RAW_PATH, rows)
            with contextlib.redirect_stdout(io.StringIO()):
                sweep2.summarize()
            r = list(csv.DictReader(open(sweep2.SUMMARY_PATH)))[0]
            self.assertEqual(float(r["givr"]), 1.0); self.assertEqual(int(r["n"]), 4)
        finally:
            sweep2.RAW_PATH, sweep2.SUMMARY_PATH = saved
            tmp.cleanup()

    def test_resume_prompts_differ_by_context(self):
        g = _chain4(0)
        amn = sweep2.resume_user_msg(g, "amnesiac", set())
        hint = sweep2.resume_user_msg(g, "bare_hint", set())
        self.assertEqual(amn, g.instruction)
        self.assertNotEqual(hint, amn); self.assertIn("interrupted", hint)
        self.assertNotIn("list_effects", hint); self.assertNotIn("step", hint.lower().replace(g.instruction.lower(), ""))
        self.assertIn("list_effects", sweep2.resume_user_msg(g, "ledger_readback", set()))
        self.assertIn("step 1", sweep2.resume_user_msg(g, "plan_only", {0}))
        self.assertIn("none", sweep2.resume_user_msg(g, "plan_only", set()))


class ResumeReadback(unittest.TestCase):
    def test_readback_skip_needs_tool_and_first_canonical_value(self):
        # survivor NR01: a node was skipped whenever its TOOL name appeared in the log
        g = _chain4(0)
        w = World()
        n0 = g.nodes[0]
        other = dict(n0.goal.canon_args); other[list(other)[0]] = "someone-else"
        w.call(n0.tool, other, "", g.graph_id)
        self.assertNotIn(0, resume._readback_skip(w, g))
        w.call(n0.tool, dict(n0.goal.canon_args), "", g.graph_id)
        self.assertIn(0, resume._readback_skip(w, g))


# ---------------------------------------------------------------------------
# analysis/v3_common.py
# ---------------------------------------------------------------------------
class CommonProtocol(unittest.TestCase):
    def test_permutation_constants_match_the_registration(self):
        # survivors NVC01 (seed), NVC02 (N), NMK24 (make_numbers seed): p values printed in the paper moved
        self.assertEqual(v3_common.PERM_SEED, 20261025); self.assertEqual(mk.PERM_SEED, 20261025)
        self.assertEqual(v3_common.PERM_N, 10000); self.assertEqual(mk.PERM_N, 10000)
        self.assertIn("10,000 permutations, seed 20261025", PREREG)
        self.assertEqual(v3_common.SAMPLE_SEED, 20261026); self.assertIn("seed 20261026", PREREG)

    def test_task_bootstrap_is_seeded_with_the_registered_seed(self):
        # survivor NTP06: the registered bootstrap (percentile, 10,000 resamples, seed 20261025) drew from another seed
        self.assertEqual(tp.BOOT_N, 10000)
        seeds = []
        real = tp.random.Random
        class Spy(real):
            def __init__(self, *a):
                seeds.append(a); super().__init__(*a)
        eps = [{"task": f"t{i}", "v": i % 2} for i in range(8)]
        try:
            tp.random.Random = Spy
            tp.boot(eps)
        finally:
            tp.random.Random = real
        self.assertEqual(seeds, [(20261025,)])

    def test_perm_test_uses_a_fresh_rng_per_call(self):
        # survivor NVC03: a shared RNG made p depend on which contrasts ran before
        xs = {f"t{i}": (i % 3) / 2 for i in range(32)}; ys = {f"t{i}": ((i * 7) % 5) / 4 for i in range(32)}
        r1 = v3_common.perm_test(xs, ys)
        v3_common.perm_test({f"t{i}": 0.0 for i in range(20)}, {f"t{i}": float(i % 2) for i in range(20)})
        r2 = v3_common.perm_test(xs, ys)
        self.assertEqual(r1["p"], r2["p"])
        self.assertAlmostEqual(r1["p"], mk.perm_test([(xs[t], ys[t]) for t in sorted(xs)], "k")[1], places=12)

    def test_perm_p_is_hits_plus_one_over_n_plus_one(self):
        # survivor NVC13: p = min(1, (hit+1)/N) survived the iteration-1 check
        old = v3_common.PERM_N
        try:
            v3_common.PERM_N = 1
            r = v3_common.perm_test({f"t{i}": 0.0 for i in range(20)}, {f"t{i}": 1.0 for i in range(20)})
            self.assertEqual(r["p"], 0.5)          # the one permutation cannot reach |sum| = 20 unless all signs agree
            v3_common.PERM_N = 3
            r = v3_common.perm_test({"a": 0.0}, {"a": 1.0})   # one task: every permutation hits
            self.assertEqual(r["p"], 1.0)
        finally:
            v3_common.PERM_N = old
        r = v3_common.perm_test({f"t{i}": 0.0 for i in range(32)}, {f"t{i}": float(i < 6) for i in range(32)})
        hits = r["p"] * (v3_common.PERM_N + 1) - 1
        self.assertAlmostEqual(hits, round(hits), places=6); self.assertTrue(0 <= hits <= v3_common.PERM_N)

    def test_rate_stats_T_counts_tasks_with_a_defined_field(self):
        # survivor NVC08
        eps = [{"task": f"t{i}", "v": 1, "rb": (1 if i < 4 else None)} for i in range(8)]
        self.assertEqual(v3_common.rate_stats(eps, "rb")["T"], 4)
        self.assertEqual(v3_common.rate_stats(eps, "rb")["n"], 4)
        self.assertEqual(v3_common.rate_stats(eps, "v")["T"], 8)

    def test_numbers_keep_six_decimals(self):
        # survivor NVC11: rounding to 2 decimals turned p = 0.0033 into "<0.001"
        N = Numbers(); N.put("x", 0.123456789, ci=(0.1234567, 0.9), n=3, p=0.0033)
        self.assertEqual(N.d["x"]["value"], 0.123457); self.assertEqual(N.d["x"]["ci"][0], 0.123457)
        self.assertEqual(N.d["x"]["p"], 0.0033)


# ---------------------------------------------------------------------------
# analysis/v3_bare_hint.py  (survivors NBH01-NBH12)
# ---------------------------------------------------------------------------
class BareHint(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = self.tmp.name
        self.saved = (bh.OCT, bh.XP, bh.RES, bh.JUL_RAW, bh.OUT, bh.N)
        bh.RES = os.path.join(self.d, "results")
        bh.OCT = os.path.join(bh.RES, "claude_batch_2026-10"); bh.XP = os.path.join(bh.RES, "crossprovider")
        bh.JUL_RAW = os.path.join(bh.RES, "sweep2_raw_v2.jsonl"); bh.OUT = os.path.join(bh.RES, "analysis_v3")
        for p in (bh.OCT, bh.XP, bh.RES, bh.OUT):
            os.makedirs(os.path.join(p, "transcripts2"), exist_ok=True)
        bh.N = Numbers()

    def tearDown(self):
        bh.OCT, bh.XP, bh.RES, bh.JUL_RAW, bh.OUT, bh.N = self.saved
        self.tmp.cleanup()

    def row(self, model, ctx, g, s, v, mit="none", topo="chain4", k=1):
        return {"episode_id": f"{model}__{ctx}__{mit}__g{g}__k{k}__s{s}", "model": model, "resume_context": ctx,
                "mitigation": mit, "key_posture": "model", "interrupt_timing": "after_commit", "topology": topo,
                "graph_id": f"g{g}", "interrupt_node": k, "sample": s, "givr_hit": v, "violations": v}

    def tr(self, d, model, ctx, g, s, refusal=False, lookup=True, mit="none", k=1):
        s2 = [{"role": "assistant", "content": [{"type": "tool_use", "name": "list_effects", "id": "x", "input": {}}]
                if lookup else []}]
        s1 = [{"role": "assistant", "content": [], "stop_reason": "refusal" if refusal else "tool_use"}]
        t = {"model": model, "resume_context": ctx, "mitigation": mit, "graph_id": f"g{g}", "interrupt_node": k, "sample": s,
             "started_utc": "2026-10-27T00:00:00Z", "sessions": [{"turns": s1}, {"turns": s2}]}
        jd(os.path.join(d, "transcripts2", f"{model}__{ctx}__{mit}__g{g}__k{k}__s{s}.json".replace("/", "_")), t)

    def build(self):
        octo, jul = [], []
        for model, sm in bh.MODELS:
            for g in range(16):
                for s in (0, 1):
                    amn = 1 if sm != "sonnet" or g < 4 else 0
                    octo.append(self.row(model, "amnesiac", g, s, amn)); self.tr(bh.OCT, model, "amnesiac", g, s, lookup=False)
                    octo.append(self.row(model, "ledger_readback", g, s, 0)); self.tr(bh.OCT, model, "ledger_readback", g, s)
                    octo.append(self.row(model, "bare_hint", g, s, 0)); self.tr(bh.OCT, model, "bare_hint", g, s)
                    jul.append(self.row(model, "amnesiac", g, s, 1)); self.tr(bh.RES, model, "amnesiac", g, s, lookup=False)
                    # distractors: receipts cell (NBH01) and another topology (NBH02) with the opposite outcome
                    octo.append(self.row(model, "amnesiac", g, s, 0, mit="runtime_receipts"))
                    octo.append(self.row(model, "bare_hint", 100 + g, s, 1, topo="diamond"))
        # one Sonnet bare_hint episode refused before the seam (NBH03, NBH11): excluded, n = 31
        self.tr(bh.OCT, SONNET, "bare_hint", 0, 0, refusal=True)
        jl(os.path.join(bh.OCT, "sweep2_raw.jsonl"), octo); jl(bh.JUL_RAW, jul)
        jl(os.path.join(bh.XP, "sweep2_raw.jsonl"), [])

    def test_same_month_cells_tests_and_pool(self):
        self.build()
        with contextlib.redirect_stdout(io.StringIO()):
            bh.main()
        N = bh.N.d
        self.assertEqual((N["bh.haiku.oct.amnesiac:givr"]["k"], N["bh.haiku.oct.amnesiac:givr"]["n"]), (32, 32))   # NBH01, NBH02
        self.assertEqual((N["bh.sonnet.oct.bare_hint:givr"]["k"], N["bh.sonnet.oct.bare_hint:givr"]["n"]), (0, 31))  # NBH03, NBH11
        self.assertEqual(N["bh.sonnet.oct.bare_hint:givr"]["refusals"], 1)
        self.assertEqual(N["bh.haiku.oct.bare_hint:givr"]["session2_lookups"], "32/32")                           # NBH05
        self.assertEqual(N["bh.haiku.oct.amnesiac:givr"]["session2_lookups"], "0/32")
        self.assertEqual(N["bh.haiku.vs_oct_amnesiac:d"]["value"], -1.0)                                            # NBH12 sign
        self.assertEqual((N["bh.haiku.vs_oct_amnesiac:d"]["plus"], N["bh.haiku.vs_oct_amnesiac:d"]["minus"]), (0, 16))
        self.assertLess(N["bh.haiku.vs_oct_amnesiac:p"]["value"], 0.001)
        self.assertIn("holm_p", N["bh.haiku.vs_oct_amnesiac:p"])                                                   # NBH06
        self.assertNotIn("holm_p", N["bh.haiku.vs_oct_ledger_readback:p"]); self.assertNotIn("holm_p", N["bh.haiku.vs_jul_amnesiac:p"])
        # refusal bounds exist only where a refusal exists, and coding it as a violation is one of the four bounds (NBH04)
        self.assertNotIn("p_bound", N["bh.haiku.vs_oct_amnesiac:p"])
        pb = N["bh.sonnet.vs_oct_amnesiac:p"]["p_bound"]; db = N["bh.sonnet.vs_oct_amnesiac:d"]["d_bound"]
        self.assertLess(db[0], db[1]); self.assertLessEqual(pb[0], pb[1])
        self.assertAlmostEqual(db[1], -4 / 16 + 1 / 32, places=6)   # refusal coded as a violation in the hint arm
        # pooled Claude: G = 3 models x 16 graphs (NBH07)
        e = N["bh.claude_pooled.oct.amnesiac:givr"]
        self.assertEqual((e["k"], e["n"]), (72, 96)); self.assertEqual(e["ci"], [round(c, 6) for c in wilson(72 / 96 * 48, 48)])
        self.assertNotEqual(e["ci"], [round(c, 6) for c in wilson(72 / 96 * 16, 16)])

    def test_v_is_the_episode_duplicate_flag_and_e21_is_inclusive(self):
        # survivors NBH09 (violations > 1), NBH08 (strict <= 0.10)
        rows = [self.row(HAIKU, "bare_hint", g, 0, 1 if g < 3 else 0) for g in range(30)]
        for r in rows:
            r["violations"] = 1 if r["givr_hit"] else 0
        p = jl(os.path.join(bh.OCT, "sweep2_raw.jsonl"), rows)
        eps = bh.load(p, {}, HAIKU, "bare_hint")
        self.assertEqual(sum(e["v"] for e in eps), 3)
        st = bh.stats(eps); self.assertEqual(st["p"], 0.1)
        self.assertTrue(st["p"] <= 0.10)
        self.assertEqual(bh.rates(eps)["g0"], 1.0)
        jul, octo = [], []
        for model, sm in bh.MODELS:
            for g in range(16):
                for s in (0, 1):
                    jul.append(self.row(model, "amnesiac", g, s, 1)); octo.append(self.row(model, "amnesiac", g, s, 1))
                    octo.append(self.row(model, "ledger_readback", g, s, 0))
                    if model == HAIKU:
                        if g < 15:
                            octo.append(self.row(model, "bare_hint", g, s, 1 if g < 3 and s == 0 else 0))   # 3/30 = 0.10
                    else:
                        octo.append(self.row(model, "bare_hint", g, s, 0))
        jl(os.path.join(bh.OCT, "sweep2_raw.jsonl"), octo); jl(bh.JUL_RAW, jul); jl(os.path.join(bh.XP, "sweep2_raw.jsonl"), [])
        with contextlib.redirect_stdout(io.StringIO()):
            bh.main()
        self.assertTrue(bh.N.d["bh.haiku.E2-1.hint_le_0.10"]["value"])

    def test_july_baseline_is_the_v2_rescore(self):
        # survivor NBH10
        self.assertTrue(self.saved[3].endswith("sweep2_raw_v2.jsonl"))


# ---------------------------------------------------------------------------
# analysis/v3_sens_oct.py and v3_sens_oct_fix.py  (survivors NSO01, NSO04, NSOF01, NSOF02)
# ---------------------------------------------------------------------------
def _t1row(model, prompt, mit, fm, task, s, v, src="e1_6_oct"):
    return {"episode_id": f"{model}__{prompt}__{mit}__{fm}__{task}__s{s}", "model": model, "prompt": prompt,
            "mitigation": mit, "failure_mode": fm, "task_id": task, "sample": s, "violations": v,
            "failure_fired": True, "readback_after_failure": False}


class SensOct(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name
        self.saved = {m: (m.RES, m.OUT, m.OCT_RAW, m.E16_RAW, m.OUT_JSON) for m in (so, sof)}
        self.sof_v2 = sof.V2_RAW
        for m in (so, sof):
            m.RES = os.path.join(d, "results"); m.OUT = os.path.join(m.RES, "analysis_v3")
            m.OCT_RAW = os.path.join(m.RES, "tier1", "e1_6_oct", "raw.jsonl")
            m.E16_RAW = os.path.join(m.RES, "tier1", "e1_6", "raw.jsonl")
        so.OUT_JSON = os.path.join(so.OUT, "sens_oct.json"); sof.OUT_JSON = os.path.join(sof.OUT, "sens_oct_fix.json")
        sof.V2_RAW = os.path.join(sof.RES, "sweep_raw_v2.jsonl")
        os.makedirs(so.OUT)
        T = [f"t{i}" for i in range(32)]
        e500, toA, w3, w1 = "error_500_after_commit", "timeout_after_commit", "status_unknown_after_commit", "error_500_body_after_commit"
        octo, e16, v2 = [], [], []
        for m in so.MODELS:
            for i, t in enumerate(T):
                for s in (0, 1):
                    octo.append(_t1row(m, "base", "none", e500, t, s, int(i < 16)))      # Oct 500: 0.50
                    octo.append(_t1row(m, "base", "none", toA, t, s, int(i < 4)))        # Oct timeout: 0.125
                    e16.append(_t1row(m, "base", "none", w3, t, s, 0))                   # W3: 0
                    e16.append(_t1row(m, "base", "none", w1, t, s, int(i < 24)))         # W1: 0.75
                    v2.append(_t1row(m, "base", "none", e500, t, s, int(i < 24)))        # July base 500: 0.75
                    v2.append(_t1row(m, "careful", "none", e500, t, s, 0))               # July careful 500: 0 (must not pool)
                    v2.append(_t1row(m, "base", "none", toA, t, s, int(i < 4)))
        jl(so.OCT_RAW, octo); jl(so.E16_RAW, e16); jl(sof.V2_RAW, v2)

    def tearDown(self):
        for m, v in self.saved.items():
            m.RES, m.OUT, m.OCT_RAW, m.E16_RAW, m.OUT_JSON = v
        sof.V2_RAW = self.sof_v2
        self.tmp.cleanup()

    def test_drift_is_october_minus_july_base_and_w3_is_tested_against_the_500_cell(self):
        with contextlib.redirect_stdout(io.StringIO()):
            so.main()
        s = json.load(open(so.OUT_JSON))
        self.assertAlmostEqual(s["sens.oct.haiku.drift.e500:d"]["value"], 0.5 - 0.75)          # NSO01 sign
        self.assertAlmostEqual(s["sens.oct.haiku.w3_minus_oct_e500:d"]["value"], 0.0 - 0.5)     # NSO04 used the timeout arm
        self.assertAlmostEqual(s["sens.oct.haiku.w1_minus_oct_e500:d"]["value"], 0.75 - 0.5)
        self.assertAlmostEqual(s["sens.oct.haiku.oct_e500_minus_oct_toA:d"]["value"], 0.5 - 0.125)
        self.assertIn("significant", s["sens.oct.haiku.drift.e500:p"]["reading"])
        with contextlib.redirect_stdout(io.StringIO()):
            sof.main()
        f = json.load(open(sof.OUT_JSON))
        self.assertAlmostEqual(f["sens.oct_fix.haiku.drift.e500:d"]["value"], 0.5 - 0.75)      # NSOF01: base only, not base+careful
        self.assertEqual(f["sens.oct_fix.haiku.july.base.none.e500:ivr"]["k"], 48)
        self.assertTrue(f["sens.oct_fix.haiku.check.w3_minus_oct_e500:d"]["matches_sens_oct"])
        # NSOF02: the check must notice a different sens.oct value
        s["sens.oct.haiku.w3_minus_oct_e500:d"]["value"] = -0.4
        json.dump(s, open(so.OUT_JSON, "w"))
        with contextlib.redirect_stdout(io.StringIO()):
            sof.main()
        f = json.load(open(sof.OUT_JSON))
        self.assertFalse(f["sens.oct_fix.haiku.check.w3_minus_oct_e500:d"]["matches_sens_oct"])
        self.assertTrue(f["sens.oct_fix.haiku.check.w1_minus_oct_e500:d"]["matches_sens_oct"])


# ---------------------------------------------------------------------------
# analysis/v3_tier1_prereg.py on the released data  (survivors NTP01-NTP10)
# ---------------------------------------------------------------------------
class TierOnePrereg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.saved = (tp.OUT, tp.N, tp.TESTS, tp.LOG)
        tp.OUT = cls.tmp.name; tp.N = Numbers(); tp.TESTS = []; tp.LOG = []
        tp.SRCFILE.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            tp.main()
        cls.N = dict(tp.N.d); cls.TESTS = list(tp.TESTS)
        cls.cells, _ = v3_load.load_all()

    @classmethod
    def tearDownClass(cls):
        tp.OUT, tp.N, tp.TESTS, tp.LOG = cls.saved
        cls.tmp.cleanup()

    def test_holm_sensitivity_families_are_the_registered_P_tests(self):
        # survivor NTP01: deviation 4 says E1-1 (3 P tests) and E1-6 (4 P tests); the mediator is not a P test
        keys = sorted(k for k in self.N if k.endswith(":holm_sensitivity"))
        self.assertEqual(keys, ["e1_1.P.H2.gemini:holm_sensitivity", "e1_1.P.H2.gpt:holm_sensitivity", "e1_1.P.H2.qwen:holm_sensitivity",
                                "e1_6.P1.haiku:holm_sensitivity", "e1_6.P1.sonnet:holm_sensitivity",
                                "e1_6.P2.haiku:holm_sensitivity", "e1_6.P2.sonnet:holm_sensitivity"])
        ps = [self.N[k.replace(":holm_sensitivity", ":p")]["value"] for k in keys[3:]]
        for k, a in zip(keys[3:], v3_common.holm(ps)):
            self.assertAlmostEqual(self.N[k]["value"], a, places=6)

    def test_e1_4_uses_the_first_sample_of_the_released_tasks_and_an_inclusive_interval(self):
        # survivors NTP03 (both samples: n = 64), NTP02 (strict inequality at the bound)
        for m in ("haiku", "sonnet"):
            for fm in ("toA", "e500"):
                e = self.N[f"e1_4.{m}.{fm}.released_s0.ivr"]; self.assertEqual(e["n"], 32)
                new = self.N[f"e1_4.{m}.{fm}.new.ivr"]["value"]
                lo, hi = e["ci"]
                self.assertEqual(self.N[f"e1_4.{m}.{fm}.P.new_within_released_ci"]["value"], lo <= new <= hi)
        # the released Sonnet timeout cell sits exactly on the lower bound (0/64 inside [0, 0.107]): inclusive
        e = self.N["e1_4.sonnet.toA.P.new_within_released_ci"]
        self.assertEqual(self.N["e1_4.sonnet.toA.new.ivr"]["value"], self.N["e1_4.sonnet.toA.released_s0.ivr"]["ci"][0])
        self.assertTrue(e["value"]); self.assertEqual(e["verdict"], "supported")

    def test_registered_thresholds_and_directions(self):
        # NTP04 (|d| >= 0.2 for E1-1 H2), NTP07 (E1-0 +-0.06), NTP09 (E1-5 stale <= 0.10), NTP10 (sign of H2)
        self.assertEqual(tp.verdict_dir({"p": 0.01, "d": 0.1}, "+", 0.2), "not supported (significant, |d| = 0.100 < 0.2)")
        self.assertEqual(tp.verdict_dir({"p": 0.01, "d": 0.3}, "+", 0.2), "supported")
        self.assertEqual(tp.verdict_dir({"p": 0.01, "d": -0.3}, "+"), "not supported (reversed, significant)")
        self.assertEqual(tp.verdict_dir({"p": 0.2, "d": 0.3}, "+"), "null (p >= 0.05)")
        e = self.N["e1_0.P.diff_rerun_minus_v2"]
        self.assertEqual(e["criterion"], "|diff| <= 0.06"); self.assertIn("+-0.06", PREREG.replace("±", "+-"))
        self.assertEqual(e["verdict"], "supported" if abs(e["value"]) <= 0.06 else "not supported")
        for m in ("haiku", "sonnet"):
            for fm in ("toA", "e500"):
                e = self.N[f"e1_5.{m}.{fm}.S.stale_reexec"]
                self.assertEqual(e["criterion"], "<= 0.10")
                self.assertEqual(e["verdict"], "supported" if e["value"] <= 0.10 else "not supported")
        # H2 d = IVR(500) - IVR(timeout), recomputed from the loaded cells
        for src, m in (("v2", SONNET), ("xp", "openrouter:openai/gpt-5.5-20260423")):
            a = v3_common.task_rates(self.cells[(src, m, "base", "none", "timeout_after_commit")])
            b = v3_common.task_rates(self.cells[(src, m, "base", "none", "error_500_after_commit")])
            ts = sorted(set(a) & set(b))
            d = sum(b[t] - a[t] for t in ts) / len(ts)
            self.assertAlmostEqual(self.N[f"e1_1.P.H2.{v3_common.short_model(m)}:d"]["value"], d, places=6)
            self.assertGreater(d, 0)

    def test_e1_2_secondary_family_is_holm_corrected_over_ten_tests(self):
        # survivor NTP05
        fam = [t for t in self.TESTS if t["test"].startswith("e1_2.S.H")]
        self.assertEqual(len(fam), 10)
        adj = v3_common.holm([t["p"] for t in fam])
        for t, a in zip(fam, adj):
            self.assertAlmostEqual(self.N[t["test"] + ":p"]["holm_p"], a, places=6)
        self.assertTrue(any(abs(self.N[t["test"] + ":p"]["holm_p"] - t["p"]) > 1e-9 for t in fam))

    def test_refusals_are_imputed_in_the_bounds_and_bootstrap_uses_the_registered_seed(self):
        # survivors NTP08 (impute ignored), NTP06 (bootstrap seed)
        eps = [{"task": "t0", "sample": 0, "v": 0, "eid": "a"}]
        out = tp.impute(eps, [("t1", 0, 2), ("t1", 1, 2)], 1)
        self.assertEqual(len(out), 3); self.assertEqual(sum(e["v"] for e in out), 2)
        k = ("e1_0", HAIKU, "base", "idem_keys", "timeout_after_commit")
        by = {}
        for e in self.cells[k]:
            by.setdefault(e["task"], []).append(e["v"])
        ts = sorted(by); rng = random.Random(20261025); vals = []
        for _ in range(10000):
            smp = [ts[rng.randrange(len(ts))] for _ in ts]
            vals.append(sum(sum(by[t]) for t in smp) / sum(len(by[t]) for t in smp))
        vals.sort()
        self.assertEqual(self.N["e1_0.haiku.base.keys.toA:ivr_ci_boot"]["ci"], [round(vals[250], 6), round(vals[9749], 6)])


# ---------------------------------------------------------------------------
# src/tier1.py episode_row  (survivors NT07, NT08)
# ---------------------------------------------------------------------------
class TierOneRows(unittest.TestCase):
    def tr(self, n_eff, n_turns, max_turns=12, fail_on_call=2, last_role="tool_results"):
        t = generate_long_tasks()[1]
        calls = []
        turns = []
        for i in range(n_turns):
            blocks = []
            if i < n_eff:
                tl, a = t.steps[i % len(t.steps)]
                blocks.append({"type": "tool_use", "id": f"tu{i}", "name": tl, "input": dict(a)})
            turns.append({"role": "assistant", "content": blocks})
            turns.append({"role": "tool_results", "content": [{"tool_use_id": f"tu{i}", "content": "ok"}] if blocks else []})
        if last_role == "assistant":
            turns.append({"role": "assistant", "content": []})
        return {"episode_id": "e", "exp": "E1-5", "model": "m", "prompt": "base", "mitigation": "none",
                "failure_mode": "timeout_after_commit", "retry_mode": "agent_retry", "sample": 0,
                "score": {"violations": 0, "achieved": True, "omission": False, "misorder": False},
                "ledger": [], "turns": turns, "fail_on_call": fail_on_call, "max_turns": max_turns}, t

    def test_failure_fired_when_the_kth_valid_call_exists(self):
        tr, t = self.tr(n_eff=2, n_turns=2)
        self.assertTrue(tier1.episode_row(tr, t)["failure_fired"])
        tr, t = self.tr(n_eff=1, n_turns=1)
        self.assertFalse(tier1.episode_row(tr, t)["failure_fired"])

    def test_turn_cap_hit_at_exactly_max_turns(self):
        tr, t = self.tr(n_eff=2, n_turns=3, max_turns=3)
        self.assertTrue(tier1.episode_row(tr, t)["turn_cap_hit"])
        tr, t = self.tr(n_eff=2, n_turns=3, max_turns=3, last_role="assistant")
        self.assertFalse(tier1.episode_row(tr, t)["turn_cap_hit"])
        tr, t = self.tr(n_eff=2, n_turns=2, max_turns=3)
        self.assertFalse(tier1.episode_row(tr, t)["turn_cap_hit"])


# ---------------------------------------------------------------------------
# paper-merged/make_numbers.py: 209 cells, tests, summaries, dups, scripted, power
# ---------------------------------------------------------------------------
def _reset():
    mk.NUMS.clear(); mk.EPI.clear(); mk.EPI249.clear(); mk.PAIRED_MDD.clear(); mk.DUPS.clear(); mk.REPORT.clear()


class MakeNumbers209(unittest.TestCase):
    def rows(self, b="jul", model=HAIKU, prompt="base", mit="none", fm="timeout_after_commit"):
        out = {}
        for i in range(32):
            for s in (0, 1):
                v = 1 if (i < 8 and s == 0) or i < 4 else 0        # 12/64 episodes; 8 of 32 tasks have any
                eid = f"{model}__{prompt}__{mit}__{fm}__task-{i}__s{s}"
                out[eid] = {"episode_id": eid, "model": model, "prompt": prompt, "mitigation": mit, "failure_mode": fm,
                            "task_id": f"task-{i}", "family": "atomic", "sample": s, "violations": v, "achieved": True,
                            "omission": False, "misorder": False}
        return {(b, model, prompt, mit, fm, ()): out}

    def test_task_interval_and_any_sample_rate(self):
        # survivors NMK02 (Wilson(k, T) in place of Wilson(p*T, T)), NMK03 (all samples instead of any)
        _reset()
        mk.emit_209(self.rows(), {})
        N = mk.NUMS
        self.assertEqual(N["jul:haiku:base:none:toA:kn"], "12/64")
        self.assertEqual(N["jul:haiku:base:none:toA:tci"], mk.ci_str(*mk.wilson(12 / 64 * 32, 32)))
        self.assertNotEqual(N["jul:haiku:base:none:toA:tci"], mk.ci_str(*mk.wilson(12, 32)))
        self.assertEqual(N["jul:haiku:base:none:toA:tvr"], mk.f2(8 / 32))
        self.assertEqual(N["jul:haiku:base:none:toA:tvrci"], mk.ci_str(*mk.wilson(8, 32)))

    def test_summaries_count_check_first_cells_without_a_substrate(self):
        # survivors NMK12 (careful/receipts counted as check-first cells), NMK13 (tagged cells in the range)
        _reset()
        cells = {}
        for mit in ("none", "runtime_receipts"):
            cells.update(self.rows(prompt="careful", mit=mit))
        cells.update(self.rows(model=SONNET))
        tagged = self.rows(model=FABLE)
        k = next(iter(tagged)); cells[(k[0], k[1], k[2], k[3], k[4], ("retry_x",))] = tagged[k]
        for r in cells[(k[0], k[1], k[2], k[3], k[4], ("retry_x",))].values():
            r["violations"] = 1
        mk.emit_summaries(cells)
        self.assertEqual(mk.NUMS["careful:cells"], "1")
        self.assertEqual(mk.NUMS["range:toA:nmodels"], "1"); self.assertEqual(mk.NUMS["range:toA:max"], mk.f2(12 / 64))

    def test_readback_is_before_the_next_effectful_call(self):
        # survivor NMK15: list_effects anywhere after the failed call counted (7 printed :rb cells moved)
        def call(name, inp, i):
            return {"type": "tool_use", "id": f"tu{i}", "name": name, "input": inp}
        pay = {"payee": "v", "amount": "10", "memo": "m"}
        t = {"turns": [{"role": "assistant", "content": [call("send_payment", pay, 0)]},
                       {"role": "assistant", "content": [call("send_email", {"to": "a", "subject": "s"}, 1)]},
                       {"role": "assistant", "content": [call("list_effects", {}, 2)]},
                       {"role": "assistant", "content": [{"type": "text", "text": "DONE"}]}]}
        fe = mk.episode_features(t)
        self.assertFalse(fe["readback"]); self.assertFalse(fe["retried"])
        t["turns"][1], t["turns"][2] = t["turns"][2], t["turns"][1]
        self.assertTrue(mk.episode_features(t)["readback"])


class MakeNumbersTests(unittest.TestCase):
    def test_timeout_vs_500_family_holds_ivr_and_readback_and_paired_power_uses_sample_sd(self):
        # survivors NMK06 (Holm family = IVR tests only), NMK19 (population SD)
        _reset()
        mk.EPI["jul:haiku:base:none:toA"] = [(f"t{i}", 1 if i % 4 else 0, 1 if i < 16 else 0) for i in range(32)]
        mk.EPI["jul:haiku:base:none:e500"] = [(f"t{i}", 1 if i % 2 else 0, 1 if i < 14 else 0) for i in range(32)]
        mk.emit_tests()
        N = mk.NUMS
        self.assertEqual(N["test:toA_e500:family"], "2")
        A = {e[0]: e for e in mk.EPI["jul:haiku:base:none:toA"]}; B = {e[0]: e for e in mk.EPI["jul:haiku:base:none:e500"]}
        pairs = {idx: [(A[t][idx], B[t][idx]) for t in sorted(A)] for idx in (1, 2)}     # tasks in sorted order, as the script
        p_ivr = mk.perm_test(pairs[1], "x")[1]; p_rb = mk.perm_test(pairs[2], "x")[1]
        self.assertLess(p_ivr, p_rb)
        self.assertEqual(N["test:haiku:toA_e500:ivr:p"], mk.pfmt(p_ivr)); self.assertEqual(N["test:haiku:toA_e500:rb:p"], mk.pfmt(p_rb))
        self.assertEqual(N["test:haiku:toA_e500:ivr:holm"], mk.pfmt(min(1.0, 2 * p_ivr)))
        self.assertEqual(N["test:haiku:toA_e500:rb:holm"], mk.pfmt(max(min(1.0, 2 * p_ivr), p_rb)))
        self.assertNotEqual(N["test:haiku:toA_e500:ivr:holm"], N["test:haiku:toA_e500:ivr:p"])
        dd = [(1 if i % 2 else 0) - (1 if i % 4 else 0) for i in range(32)]
        mu = sum(dd) / 32; sd = math.sqrt(sum((x - mu) ** 2 for x in dd) / 31)
        self.assertEqual(N["power:paired:max"], mk.f2((1.959964 + 0.841621) * sd / math.sqrt(32)))

    def test_hint_contrast_is_paired_by_graph(self):
        # survivor NMK07: pairing by (graph, node, sample) left no pairs when the arms' cut points differ
        _reset()
        mk.EPI249["oct:haiku:amnesiac:none"] = {(f"g{g}", g % 4, s): 1 for g in range(16) for s in (0, 1)}
        mk.EPI249["oct:haiku:bare_hint:none"] = {(f"g{g}", (g + 1) % 4, s): 0 for g in range(16) for s in (0, 1)}
        mk.NUMS["oct:haiku:amnesiac:none:givr"] = "1.00"; mk.NUMS["oct:haiku:bare_hint:none:givr"] = "0.00"
        mk.NUMS["oct:haiku:amnesiac:none:kn"] = "32/32"; mk.NUMS["oct:haiku:bare_hint:none:kn"] = "0/32"
        mk.emit_tests()
        self.assertEqual(mk.NUMS["test:haiku:hint:G"], "16"); self.assertEqual(mk.NUMS["test:haiku:hint:d"], "-1.00")
        self.assertEqual(mk.NUMS["test:haiku:hint:p"], "<0.001")

    def test_pfmt_and_power_constants(self):
        # survivors NMK17, NMK18
        self.assertEqual(mk.pfmt(0.003), "0.003"); self.assertEqual(mk.pfmt(0.0009), "<0.001"); self.assertEqual(mk.pfmt(0.001), "0.001")
        _reset()
        mk.emit_totals({}, {})
        self.assertEqual(mk.NUMS["power:mdd16"], mk.f2((1.959964 + 0.841621) * math.sqrt(0.5 / 16)))
        self.assertEqual(mk.NUMS["power:mdd16"], "0.50")

    def test_boot_ci_is_the_percentile_interval_with_a_key_seed(self):
        # survivor NMK05
        clusters = [(1, 2), (0, 2), (2, 2), (0, 2), (1, 2), (0, 2), (0, 2), (1, 2)]
        rng = random.Random(int(hashlib.sha256(b"k").hexdigest()[:8], 16)); bs = []
        for _ in range(mk.BOOT_N):
            idx = [rng.randrange(8) for _ in range(8)]
            bs.append(sum(clusters[i][0] for i in idx) / sum(clusters[i][1] for i in idx))
        bs.sort()
        self.assertEqual(mk.boot_ci(clusters, "k"), (bs[250], bs[9749]))


class MakeNumbersDerived(unittest.TestCase):
    def test_analysis_v3_block_totals_and_significance_threshold(self):
        # survivors NMK20 (six experiments), NMK30 (p < 0.10 counted as significant)
        if not os.path.exists(mk.NJ_PATH):
            self.skipTest("analysis_v3/numbers.json missing")
        _reset(); mk.NJ.clear()
        for m, p in (("haiku", "<0.001"), ("sonnet", "0.062"), ("fable", "<0.001"), ("gpt", "0.040"), ("gemini", "0.049"), ("qwen", "0.050")):
            mk.NUMS[f"test:{m}:hint:p"] = p
        mk.emit_analysis_v3()
        nums = json.load(open(mk.NJ_PATH))["numbers"]
        t1 = sum(int(nums[f"hygiene.e1_{i}.rows"]["value"]) for i in range(7))
        self.assertEqual(mk.NUMS["total:tier1"], f"{t1:,}")
        self.assertEqual(mk.NUMS["hint:sig:n"], "4"); self.assertEqual(mk.NUMS["hint:ns:names"], "Sonnet 5 and Qwen3.6-27B")
        self.assertEqual(mk.NUMS["hint:sig:pmaxe"], "$p=0.049$")

    def test_stale_without_duplicate_and_pooled_hint_G(self):
        # survivors NMK11 (or instead of and), NMK21 (G over graphs only)
        _reset()
        rows = []
        for g in range(16):
            for s in (0, 1):
                rows.append({"episode_id": f"m__amnesiac__none__g{g}__k1__s{s}", "model": HAIKU, "resume_context": "amnesiac",
                             "mitigation": "none", "graph_id": f"g{g}", "interrupt_node": 1, "sample": s, "topology": "chain4",
                             "violations": int(g < 6), "givr_hit": int(g < 6), "completion": 1,
                             "order_violation": int(2 <= g < 10)})
        mk.emit_249_cell("oct:haiku:amnesiac:none", rows, "oct", {}, 32)
        self.assertEqual(mk.NUMS["oct:haiku:amnesiac:none:stalekn"], "16/32")
        self.assertEqual(mk.NUMS["oct:haiku:amnesiac:none:stale_nodup"], "8")
        _reset()
        cells = {}
        for model in (HAIKU, SONNET, FABLE):
            cells[("oct", model, "bare_hint", "none")] = {
                f"{model}__bare_hint__none__g{g}__k1__s{s}": {"episode_id": f"{model}__bare_hint__none__g{g}__k1__s{s}", "model": model,
                    "resume_context": "bare_hint", "mitigation": "none", "graph_id": f"g{g}", "interrupt_node": 1, "sample": s,
                    "topology": "chain4", "violations": 0, "givr_hit": 0, "completion": 1, "order_violation": 0}
                for g in range(16) for s in (0, 1)}
        mk.emit_249(cells, {})
        self.assertEqual(mk.NUMS["pool:bare_hint:claude:G"], "48")
        self.assertEqual(mk.NUMS["pool:bare_hint:claude:gci"], mk.ci_str(*mk.wilson(0, 48)))

    def test_double_run_table_reports_the_first_run_values(self):
        # survivor NMK16: "first" was the last row, so the table could never disagree with itself
        _reset()
        key = ("xp", "openrouter:qwen/qwen3.6-27b-20260422", "base", "none", "error_500_after_commit", ())
        cells = {key: {}}
        for i in range(64):
            eid = f"q__base__none__e500__task-{i // 2}__s{i % 2}"
            cells[key][eid] = {"episode_id": eid, "model": key[1], "prompt": "base", "mitigation": "none",
                               "failure_mode": key[4], "task_id": f"task-{i // 2}", "sample": i % 2, "violations": int(i < 30)}
        dup = {}
        for i in range(6):
            eid = f"q__base__none__e500__task-{i // 2}__s{i % 2}"
            first = dict(cells[key][eid]); first["violations"] = 0
            dup[eid] = [first, cells[key][eid]]
        mk.DUPS["xp"] = dup
        mk.emit_dups(cells)
        q = "dup:xp:qwen:base:none:e500"
        self.assertEqual(mk.NUMS[q + ":last"], "30/64"); self.assertEqual(mk.NUMS[q + ":first"], "24/64")
        self.assertEqual(mk.NUMS[q + ":disagree"], "6"); self.assertEqual(mk.NUMS["dup:disagree:total"], "6")

    def test_scripted_pool_is_episode_weighted(self):
        # survivor NMK14 (printed scr2pool:amnesiac:naive:givr moved 0.76 -> 0.58)
        p = os.path.join(mk.RES, "graph_summary.csv")
        if not os.path.exists(p):
            self.skipTest("graph_summary.csv missing")
        _reset()
        with contextlib.redirect_stdout(io.StringIO()):
            mk.emit_scripted()
        g = list(csv.DictReader(open(p)))
        sel = [r for r in g if r["resume_context"] == "amnesiac" and r["mitigation"] == "none" and r["key_posture"] == "naive"]
        n = sum(int(r["n_episodes"]) for r in sel)
        gv = sum(float(r["givr"]) * int(r["n_episodes"]) for r in sel) / n
        self.assertEqual(mk.NUMS["scr2pool:amnesiac:naive:givr"], mk.f2(gv))
        self.assertNotEqual(mk.f2(gv), mk.f2(sum(float(r["givr"]) for r in sel) / len(sel)))

    def test_reporting_constants_match_the_registration(self):
        # equivalent-on-data survivors NMK01 (90% rule) and NMK08 (core batches)
        self.assertEqual(mk.MIN_FRAC, 0.9); self.assertIn("90% of planned n", PREREG.replace("≥", ">=").replace(" ", " "))
        self.assertEqual(mk.min_n(64), 58); self.assertEqual(mk.min_n(32), 29)
        self.assertEqual(mk.CORE, {"jul", "xp", "oct"})



class Sweep2SummaryLastRow(unittest.TestCase):
    def test_summarize_keeps_the_last_row_per_episode(self):
        # iteration-3 note: summarize() kept the FIRST row per episode_id, unlike every other reader (e78e23d)
        with tempfile.TemporaryDirectory() as d:
            raw, out = os.path.join(d, "sweep2_raw.jsonl"), os.path.join(d, "sweep2_summary.csv")
            row = {"episode_id": "e1", "model": "m", "resume_context": "amnesiac", "mitigation": "none",
                   "givr_hit": 0, "order_violation": 0, "completion": 1}
            with open(raw, "w") as f:
                f.write(json.dumps(row) + "\n"); f.write(json.dumps(dict(row, givr_hit=1)) + "\n")
            old = (sweep2.RAW_PATH, sweep2.SUMMARY_PATH)
            try:
                sweep2.RAW_PATH, sweep2.SUMMARY_PATH = raw, out
                with contextlib.redirect_stdout(io.StringIO()):
                    sweep2.summarize()
            finally:
                sweep2.RAW_PATH, sweep2.SUMMARY_PATH = old
            rows = list(csv.DictReader(open(out)))
            self.assertEqual((rows[0]["n"], rows[0]["givr"]), ("1", "1.0"))

if __name__ == "__main__":
    unittest.main(verbosity=1)
