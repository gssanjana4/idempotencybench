"""Tests added by the code-verifier loop (iteration 6): each kills a seeded-random
mutant (random.Random(6), classic operators over the number-producing code) that
changed paper-merged/numbers.tex yet passed every existing suite.
Run from the repo root:  python3 tests_mutants6.py"""
import contextlib, csv, glob, hashlib, importlib.util, io, json, os, re, sys, tempfile, unittest
from collections import defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("analysis", "src", "src2"):
    sys.path.insert(0, os.path.join(ROOT, d))
from v3_common import Numbers                     # noqa: E402
import v3_load                                    # noqa: E402
import v3_tier1_prereg as tp                      # noqa: E402
import v3_iter1_extra as vi                       # noqa: E402
import v3_real_extra as vx                        # noqa: E402
import rescore_c1 as rc                           # noqa: E402

_spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(ROOT, "paper-merged", "make_numbers.py"))
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)

HAIKU, SONNET = "claude-haiku-4-5-20251001", "claude-sonnet-5"
QWEN = "openrouter:qwen/qwen3.6-27b-20260422"


def _reset():
    mk.NUMS.clear(); mk.EPI.clear(); mk.EPI249.clear(); mk.PAIRED_MDD.clear(); mk.DUPS.clear(); mk.REPORT.clear()
    mk.NJ.clear(); mk.FULL209 = ""; mk.DUPROWS = ""; mk.PREREG = ""


def _tu(name, inp, tid="tu1"):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _row(i, violations=0, achieved=True, misorder=False, family="multi_step", omission=False):
    return {"task_id": f"{family}-{i:03d}", "sample": i % 2, "violations": violations, "achieved": achieved,
            "misorder": misorder, "family": family, "omission": omission}


def _cell(n, over=None):
    """n rows; over maps row index -> field overrides, e.g. {3: dict(violations=2)}"""
    rows = {}
    for i in range(n):
        rows[f"e{i}"] = _row(i, **(over or {}).get(i, {}))
    return rows


# ---------------------------------------------------------------------------
# paper-merged/make_numbers.py: unit fixtures
# ---------------------------------------------------------------------------
class Summaries(unittest.TestCase):
    def setUp(self):
        _reset()
        A = _cell(64, {0: dict(violations=1, misorder=True), 1: dict(violations=1, misorder=True),
                       2: dict(violations=2, misorder=True), 3: dict(violations=2)})        # 4 violating, depth 6
        B = _cell(64, {i: dict(achieved=False) for i in range(8)})                          # 56/64 succeed, clean
        C = _cell(64, {0: dict(violations=1), **{i: dict(achieved=False) for i in range(1, 5)}})    # 60/64 succeed
        self.cells = {("jul", HAIKU, "base", "none", "error_500_after_commit", ()): A,
                      ("jul", SONNET, "base", "none", "timeout_after_commit", ()): B,
                      ("xp", QWEN, "base", "none", "error_500_after_commit", ()): C}
        mk.emit_summaries(self.cells)

    def test_misorder_total_counts_only_misordered_rows(self):
        # survivor R6_030: the misorder filter was dropped (every July row counted)
        self.assertEqual(mk.NUMS["jul:misorder_total"], "3")
        self.assertEqual(mk.NUMS["jul:episodes"], "128")

    def test_min_success_is_over_violating_cells_only(self):
        # survivor R6_073: the "any violation" filter was dropped (the clean 0.88 cell entered the minimum)
        self.assertEqual(mk.NUMS["all:cells_violating"], "2")
        self.assertEqual(mk.NUMS["all:succ_min_violating"], "0.94")

    def test_depth_equals_count_cells(self):
        # survivors R6_078 (count filter dropped) and R6_079 (sum + count instead of sum - count)
        self.assertEqual(mk.NUMS["all:depth_eq_cells"], "2")
        self.assertEqual(mk.NUMS["jul:depth_eq_cells"], "1")


class Lists(unittest.TestCase):
    def test_zero_and_nonzero_name_lists_partition_the_models(self):
        # survivor R6_034: the "== 0" filter was dropped, so every model was named as zero
        _reset()
        for q, g in (("oct:haiku:amnesiac:none", "0.00"), ("oct:sonnet:amnesiac:none", "0.50"),
                     ("xp:qwen:amnesiac:none", "0.25"), ("xp:gpt:amnesiac:none", "0.00")):
            mk.put(q + ":givr", g); mk.put(q + ":n", "32"); mk.put(q + ":chk", "32/32"); mk.put(q + ":resumed", "32")
        mk.emit_lists({})
        self.assertEqual(mk.NUMS["list:amnesiac:n"], "4")
        self.assertEqual(mk.NUMS["list:amnesiac:zero"], "Haiku 4.5 and GPT-5.5")
        self.assertEqual(mk.NUMS["list:amnesiac:zero:n"], "2")
        self.assertEqual(mk.NUMS["list:amnesiac:nonzero"], "Sonnet 5 and Qwen3.6-27B")


class Scripted(unittest.TestCase):
    def _run(self, toB_ivrs):
        tmp = tempfile.TemporaryDirectory(); res = tmp.name
        with open(os.path.join(res, "summary_by_failure.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["agent", "retry_mode", "mitigation", "failure_mode", "n_tasks", "ivr", "violation_depth", "success", "omission"])
            for i, v in enumerate(toB_ivrs):
                w.writerow(["naive", f"m{i}", "none", "timeout_before_commit", 320, v, v, 0.9, 0.1])
            w.writerow(["naive", "no_retry", "none", "timeout_after_commit", 320, 0.0, 0.0, 0.875, 0.125])
            w.writerow(["naive", "no_retry", "runtime_receipts", "error_500_after_commit", 320, 0.0, 0.0, 1.0, 0.0])
        with open(os.path.join(res, "graph_summary.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["resume_context", "mitigation", "key_posture", "interrupt_timing", "givr",
                        "order_violation_rate", "mean_violations", "n_episodes", "completion"])
            for ctx in ("amnesiac", "plan_only"):
                for post in ("naive", "stable"):
                    w.writerow([ctx, "none", post, "after_commit", 0.5, 0.1, 0.7, 100, 0.9])
        saved = mk.RES; mk.RES = res; _reset()
        try:
            mk.emit_scripted()
            return dict(mk.NUMS)
        finally:
            mk.RES = saved; tmp.cleanup()

    def test_timeout_before_commit_cells_all_zero_means_ivr_zero(self):
        # survivor R6_043: "== 0" became "== 1" (a 0.0 IVR no longer counted as zero)
        n = self._run([0.0, 0.0])
        self.assertEqual(n["scr1:toB:cells"], "2"); self.assertEqual(n["scr1:toB:allzero"], "yes")
        self.assertEqual(self._run([0.0, 0.5])["scr1:toB:allzero"], "no")
        self.assertEqual(self._run([1.0, 1.0])["scr1:toB:allzero"], "no")


class MisorderTurns(unittest.TestCase):
    def test_parallel_fraction_denominator_is_episodes_with_an_effectful_turn(self):
        # survivor R6_016: the denominator increment became += 0 (72/72 printed as 72/0)
        _reset()
        key = ("jul", HAIKU, "base", "none", "error_500_after_commit", ())
        rows = _cell(64)
        trans = {}
        for i, (eid, r) in enumerate(rows.items()):
            if i < 20:    # both steps in the first effectful turn
                turns = [{"role": "assistant", "content": [_tu("send_email", {}, "a"), _tu("create_ticket", {}, "b")]}]
            elif i < 50:  # one step per turn
                turns = [{"role": "assistant", "content": [{"type": "text", "text": "first"}, _tu("send_email", {}, "a")]},
                         {"role": "tool_results", "content": []},
                         {"role": "assistant", "content": [_tu("create_ticket", {}, "b")]}]
            else:         # never called an effectful tool
                turns = [{"role": "assistant", "content": [_tu("list_effects", {}, "a")]}]
            trans[key[:5] + (r["task_id"], r["sample"])] = {"turns": turns}
        mk.emit_misorder_turns({key: rows}, trans)
        self.assertEqual(mk.NUMS["par:haiku:multistep"], "20/50")


class PermTest(unittest.TestCase):
    def test_sign_flips_are_a_fair_coin(self):
        # survivor R6_019: the flip probability became 0.4 (p-values of the hint contrast changed)
        pairs = [(0.0, 1.0)] * 4            # four tasks, each +1: two-sided exact p = 2 * 0.5**4 = 0.125
        d, p = mk.perm_test(pairs, "t")
        self.assertEqual(d, 1.0)
        self.assertAlmostEqual(p, 0.125, delta=0.012)
        d2, p2 = mk.perm_test([(y, x) for x, y in pairs], "t")   # the mirror image has the same p
        self.assertEqual(d2, -1.0); self.assertAlmostEqual(p, p2, delta=0.004)


class PiiClause(unittest.TestCase):
    def _clause(self, text):
        tmp = tempfile.TemporaryDirectory(); res = os.path.join(tmp.name, "results"); os.makedirs(res)
        open(os.path.join(res, "note.txt"), "w").write(text)
        saved = mk.RES; mk.RES = res; _reset()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mk.emit_pii()
            return dict(mk.NUMS)
        finally:
            mk.RES = saved; tmp.cleanup()

    def test_url_clause_names_the_stripe_base_address_only_when_it_occurs(self):
        # survivor R6_007: the "not" was dropped (clause inverted)
        n = self._clause("payment via https://api.stripe.com/ recorded\n")
        self.assertEqual(n["pii:url_stripe_api"], "1"); self.assertEqual(n["pii:url"], "0")
        self.assertTrue(n["pii:url_clause"].startswith("no URL other than the Stripe API base address"))
        self.assertEqual(self._clause("nothing to see\n")["pii:url_clause"], "no URL")


class AnalysisV3Block(unittest.TestCase):
    """emit_analysis_v3() over a small numbers.json (the real analysis files are not read)."""
    def _run(self, nums, meta=None):
        tmp = tempfile.TemporaryDirectory()
        nj = os.path.join(tmp.name, "numbers.json")
        json.dump({"meta": meta or {}, "numbers": nums}, open(nj, "w"))
        saved = (mk.NJ_PATH, mk.X_PATH, mk.XR_PATH, mk.SENS_FIX_PATH, mk.OCT_RAW)
        mk.NJ_PATH = nj
        mk.X_PATH = mk.XR_PATH = mk.SENS_FIX_PATH = mk.OCT_RAW = os.path.join(tmp.name, "absent")
        _reset()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                mk.emit_analysis_v3()
            return dict(mk.NUMS)
        finally:
            mk.NJ_PATH, mk.X_PATH, mk.XR_PATH, mk.SENS_FIX_PATH, mk.OCT_RAW = saved; tmp.cleanup()

    def test_refusal_bounds_flip_at_the_005_level(self):
        # survivor R6_013: the significance threshold became 0.1
        n = self._run({"e1_1.P.H2.x:p": {"value": 0.04, "p_bound": [0.03, 0.07]},
                       "e1_1.P.H2.y:p": {"value": 0.2, "p_bound": [0.15, 0.3]}})
        self.assertEqual(n["refusal:verdict_flips"], "1")

    def test_stale_inputs_are_the_files_whose_hash_differs(self):
        # survivor R6_014: "!=" became "==" (the fresh files were reported stale)
        h = lambda rel: hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest()[:16]
        n = self._run({}, meta={"input_sha256_16": {"LICENSE": h("LICENSE"), "run_tests.py": h("run_tests.py"),
                                                    "tests_mutants.py": "0" * 16}})
        self.assertEqual(n["nj:meta:stale_inputs"], "1")

    def test_only_drift_keys_need_a_sens_oct_fix_twin(self):
        # survivor R6_025: the drift filter was dropped, so every sens.oct key without a twin was deleted
        n = self._run({"sens.oct.sonnet.base.none.e500:ivr": {"value": 0.53125, "n": 64, "k": 34},
                       "sens.oct.sonnet.drift.a": {"value": 0.1},
                       "sens.oct_fix.sonnet.drift.a": {"value": 0.1},
                       "sens.oct.sonnet.drift.b": {"value": 0.2}})
        self.assertEqual(n["nj:sens.oct.sonnet.base.none.e500:ivr:kn"], "34/64")
        self.assertIn("nj:sens.oct.sonnet.drift.a", n)
        self.assertNotIn("nj:sens.oct.sonnet.drift.b", n)     # disagrees with (missing) twin: dropped

    def test_derived_k_over_n_rounds_to_the_nearest_count(self):
        # survivor R6_039: round() became int() (0.57 * 100 = 56.999... lost its k/n)
        n = self._run({"x.cell:ivr": {"value": 0.57, "n": 100}, "y.cell:ivr": {"value": 0.53125, "n": 64}})
        self.assertEqual(n["nj:x.cell:ivr:kn"], "57/100")
        self.assertEqual(n["nj:y.cell:ivr:kn"], "34/64")

    def test_d1_nonzero_cells_are_those_above_zero(self):
        # survivor R6_074: the "> 0" filter was dropped
        n = self._run({"e1_2.haiku.base.none.none:d1_spontaneous_dup": {"value": 0.0, "n": 64},
                       "e1_2.sonnet.base.none.none:d1_spontaneous_dup": {"value": 0.03125, "n": 64},
                       "e1_2.qwen.careful.none.none:d1_spontaneous_dup": {"value": 0.0, "n": 64}})
        self.assertEqual(n["e12:d1:cells"], "3"); self.assertEqual(n["e12:d1:nonzero_n"], "1")


# ---------------------------------------------------------------------------
# paper-merged/make_numbers.py: one full pass, invariants over the emitted macros
# ---------------------------------------------------------------------------
class EndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.saved = (mk.OUT_TEX, mk.SNAPSHOT, mk.OUT_REP, mk.render_templates)
        mk.OUT_TEX = os.path.join(cls.tmp.name, "numbers.tex"); mk.SNAPSHOT = os.path.join(cls.tmp.name, "numbers.json")
        mk.OUT_REP = os.path.join(cls.tmp.name, "report.txt"); mk.render_templates = lambda: []
        _reset()
        with contextlib.redirect_stdout(io.StringIO()):
            mk.main()
        cls.N = dict(mk.NUMS); cls.PREREG = mk.PREREG; cls.DUPROWS = mk.DUPROWS

    @classmethod
    def tearDownClass(cls):
        mk.OUT_TEX, mk.SNAPSHOT, mk.OUT_REP, mk.render_templates = cls.saved
        cls.tmp.cleanup()

    def test_gave_up_and_retried_are_disjoint_subsets_of_the_cell(self):
        # survivor R6_006: the gaveup filter was dropped (every episode with a transcript counted)
        seen = 0
        for k, v in self.N.items():
            if not k.endswith(":gaveup"):
                continue
            q = k[:-len(":gaveup")]
            if q + ":n" not in self.N or q + ":retried" not in self.N:
                continue
            seen += 1
            self.assertLessEqual(int(v) + int(self.N[q + ":retried"]), int(self.N[q + ":n"]), q)
        self.assertGreater(seen, 10)

    def test_unqualified_aliases_copy_the_single_batch_cell(self):
        # survivor R6_010: the alias key kept the batch prefix (9,226 garbage keys, no alias)
        seen_unq = defaultdict(list)
        for (b, model, prompt, mit, fm, tags) in mk.load_209():
            base = ":".join([mk.short_model(model), prompt, mk.MIT_SHORT.get(mit, mit), mk.FM_SHORT.get(fm, fm)] + list(tags))
            seen_unq[base].append(b + ":" + base)
        checked = 0
        for base, quals in seen_unq.items():
            if len(quals) != 1 or quals[0] + ":ivr" not in self.N:
                continue
            checked += 1
            self.assertEqual(self.N.get(base + ":ivr"), self.N[quals[0] + ":ivr"], base)
            self.assertEqual(self.N.get(base + ":n"), self.N[quals[0] + ":n"], base)
        self.assertGreater(checked, 5)

    def test_check_first_increase_verdict_needs_a_significant_positive_difference(self):
        # survivor R6_011: "holm_p < 0.05 and d > 0" became "or"
        rows = [l for l in self.PREREG.split("\n") if "check-first does not reduce" in l]
        self.assertGreaterEqual(len(rows), 3)
        for l in rows:
            cols = [c.strip() for c in l.split(" & ")]
            d, p, verdict = cols[4], cols[5], cols[6].replace(r"\\", "").strip()
            pos = d.startswith("d +"); sig = p.startswith("<") or float(p.split()[0]) < 0.05
            self.assertEqual(verdict == "supported; increase", pos and sig, l)

    def test_double_run_rows_name_the_resume_context_then_mitigation(self):
        # survivor R6_012: the condition column printed key[3] (mitigation) twice instead of the prompt
        rows = [l for l in self.DUPROWS.split("\n") if l.strip()]
        self.assertGreater(len(rows), 0)
        for l in rows:
            cond = l.split(" & ")[1].split("/")
            self.assertIn(cond[0], {"base", "careful"}, l)
            self.assertIn(cond[1], set(mk.MIT_SHORT.values()), l)

    def test_prereg_commit_time_is_printed_to_the_minute(self):
        # survivor R6_023: the [:16] cut was dropped (seconds and offset leaked into the macro)
        if "prereg:commit_utc" not in self.N:
            self.skipTest("no git history here")
        self.assertRegex(self.N["prereg:commit_utc"], r"^\d{4}-\d\d-\d\d \d\d:\d\d UTC$")
        self.assertRegex(self.N["prereg:first_data_utc"], r"^\d{4}-\d\d-\d\d \d\d:\d\d UTC$")

    def test_stale_events_without_a_duplicate_are_events_minus_cooccurrences(self):
        # survivor R6_046: "ev - co" became "ev + co"
        bs = [k.split(":")[1] for k in self.N if re.fullmatch(r"stale:\w+:events", k)]
        self.assertIn("all", bs)
        for b in bs:
            self.assertEqual(int(self.N[f"stale:{b}:without_dup"]),
                             int(self.N[f"stale:{b}:events"]) - int(self.N[f"stale:{b}:cooccur"]), b)

    def test_design_topologies_is_read_from_the_graph_module(self):
        # survivor R6_050: "tp and put(...)" became "tp or put(...)" (the macro was never written)
        src = open(os.path.join(ROOT, "src2", "graph.py")).read()
        m = re.search(r"^TOPOLOGIES\s*=\s*\(([^)]*)\)", src, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(self.N.get("design:topologies"), str(len(re.findall(r'"\w+"', m.group(1)))))

    def test_sampling_dates_drop_the_year(self):
        # survivor R6_061: dates[0][5:] became dates[0][:] (2026-10-25 instead of 10-25)
        keys = [k for k in self.N if re.fullmatch(r"samp:\w+:\w+:dates", k)]
        self.assertGreater(len(keys), 2)
        for k in keys:
            self.assertNotRegex(self.N[k], r"\d{4}-\d\d-\d\d", k)
            self.assertRegex(self.N[k], r"^\d\d-\d\d", k)

    def test_truncated_graph_episodes_are_a_strict_subset(self):
        # survivor R6_065: the truncated filter was dropped (trunc == n in every 249 cell)
        quals = [k[:-len(":givr")] for k in self.N if k.endswith(":givr") and k[:-len(":givr")] + ":trunc" in self.N]
        self.assertGreater(len(quals), 5)
        tr = sum(int(self.N[q + ":trunc"]) for q in quals); n = sum(int(self.N[q + ":n"]) for q in quals)
        for q in quals:
            self.assertLessEqual(int(self.N[q + ":trunc"]), int(self.N[q + ":n"]), q)
        self.assertLess(tr, n)


# ---------------------------------------------------------------------------
# analysis/v3_tier1_prereg.py
# ---------------------------------------------------------------------------
class PreregHelpers(unittest.TestCase):
    def test_verdict_reversed_needs_significance(self):
        # survivor R6_094: "sig and not good" became "sig or not good"
        self.assertEqual(tp.verdict_dir({"p": 0.5, "d": -0.1}, "+"), "null (p >= 0.05)")
        self.assertEqual(tp.verdict_dir({"p": 0.01, "d": -0.1}, "+"), "not supported (reversed, significant)")
        self.assertEqual(tp.verdict_dir({"p": 0.01, "d": 0.1}, "+"), "supported")
        self.assertEqual(tp.verdict_dir({"p": 0.5, "d": 0.1}, "-"), "null (p >= 0.05)")

    def test_bootstrap_interval_is_the_central_95_percent(self):
        # survivor R6_084: the upper quantile became 0.95 (a 92.5% interval)
        eps = [{"task": f"t{i}", "v": int(i < 25)} for i in range(50)]
        lo, hi = tp.boot(eps)
        self.assertEqual((lo, hi), (0.36, 0.64))


class PreregMain(unittest.TestCase):
    """One run of the pre-registered analysis with its outputs redirected; invariants over its store."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.saved = (tp.N, tp.TESTS, tp.LOG, tp.OUT, dict(tp.SRCFILE))
        tp.N = Numbers(); tp.TESTS = []; tp.LOG = []; tp.OUT = cls.tmp.name
        with contextlib.redirect_stdout(io.StringIO()):
            tp.main()
        cls.D = tp.N.d

    @classmethod
    def tearDownClass(cls):
        tp.N, tp.TESTS, tp.LOG, tp.OUT = cls.saved[:4]; tp.SRCFILE.clear(); tp.SRCFILE.update(cls.saved[4])
        cls.tmp.cleanup()

    def test_keyed_and_reused_episodes_are_the_retries_with_the_same_key(self):
        # survivor R6_083: "key_first and retried and same_key" became "or"
        keys = [k for k in self.D if re.fullmatch(r"e1_1\.S\.H4\.\w+\.\w+", k)]
        self.assertGreaterEqual(len(keys), 4)
        for k in keys:
            e = self.D[k]
            self.assertLessEqual(e["n_keyed_and_reused"], e["retry_key_categories"].get("same_key", 0), k)
            self.assertLessEqual(e["n_keyed_and_reused"], round(e["key_attach_first"] * e["n"]) + 1e-9, k)

    def test_key_category_names_lose_their_d2_prefix(self):
        # survivor R6_087: k[3:] became k[:] (nj:...:d2_key_category:same_key went missing)
        keys = [k for k in self.D if k.endswith(":d2_key_category")]
        self.assertGreaterEqual(len(keys), 6)
        for k in keys:
            cats = set(self.D[k]["value"])
            self.assertFalse([c for c in cats if c.startswith("d2_")], (k, cats))
            self.assertTrue(cats & {"same_key", "new_key", "dropped_key", "added_key", "keyless", "not_redone", "none"}, (k, cats))

    def test_ivr_by_readback_status_partitions_the_cell(self):
        # survivor R6_090: "not e['rb']" lost its not (IVR|no read-back printed the read-back episodes)
        keys = [k for k in self.D if k.endswith(":ivr_given_readback")]
        self.assertGreaterEqual(len(keys), 4)
        for k in keys:
            e = self.D[k]; rb = self.D[k.replace(":ivr_given_readback", ":readback")]
            ky, ny = map(int, e["value"].split("/")); kn, nn = map(int, e["ivr_given_no_readback"].split("/"))
            self.assertEqual(ny, rb["k"], k)
            self.assertEqual(ny + nn, rb["n"], k)
            self.assertLessEqual(ky, ny); self.assertLessEqual(kn, nn)


# ---------------------------------------------------------------------------
# analysis/v3_iter1_extra.py, v3_real_extra.py, v3_load.py; src/rescore_c1.py
# ---------------------------------------------------------------------------
def _pay(key, tid, amount="10.00"):
    return _tu("send_payment", {"payee": "acme", "amount": amount, "memo": "inv-1", "idempotency_key": key}, tid)


class RetryText(unittest.TestCase):
    def test_a_read_back_before_the_first_effectful_call_is_not_the_first_call(self):
        # survivor R6_127: the effectful-tool filter was dropped (list_effects counted as the first call)
        t = {"turns": [
            {"role": "assistant", "content": [_tu("list_effects", {}, "r0")]},
            {"role": "tool_results", "content": []},
            {"role": "assistant", "content": [_pay("k1", "a")]},
            {"role": "tool_results", "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "The call failed; retrying with the same key."}, _pay("k1", "b")]}]}
        self.assertEqual(vi.retry_text(t), (True, False, True))

    def test_text_after_the_retry_is_not_between_the_calls(self):
        # survivor R6_131: the slice lost its upper bound (text after the retry counted)
        t = {"turns": [
            {"role": "assistant", "content": [_pay("k1", "a")]},
            {"role": "tool_results", "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "Retrying."}, _pay("k2", "b")]},
            {"role": "tool_results", "content": []},
            {"role": "assistant", "content": [{"type": "text", "text": "Done; the idempotency key was new."}]}]}
        self.assertEqual(vi.retry_text(t), (True, False, False))
        t["turns"][2]["content"][0]["text"] = "Retrying with the same idempotency key."
        t["turns"][2]["content"][1] = _pay("k1", "b")
        self.assertEqual(vi.retry_text(t), (True, True, True))


class RealHygiene(unittest.TestCase):
    def test_hygiene_value_is_the_episode_count(self):
        # survivor R6_125: the value assignment became setdefault (stayed None)
        tmp = tempfile.TemporaryDirectory(); p = os.path.join(tmp.name, "raw.jsonl")
        with open(p, "w") as fh:
            for i in range(5):
                fh.write(json.dumps({"episode_id": f"e{i}", "kind": "retry" if i < 3 else "restart", "failure_fired": i < 2,
                                     "stripe_errors": 0, "refusal": False, "truncated": False, "turn_cap_hit": False}) + "\n")
        saved = (vx.REAL_RAW, vx.N); vx.REAL_RAW = p; vx.N = Numbers()
        try:
            vx.hygiene()
            e = vx.N.d["xr.real.hygiene"]
            self.assertEqual(e["value"], 5); self.assertEqual(e["n"], 5); self.assertEqual(e["failure_fired"], "2/3")
        finally:
            vx.REAL_RAW, vx.N = saved; tmp.cleanup()


class LoadV1(unittest.TestCase):
    def test_v1_cell_is_filtered_on_all_four_factors(self):
        # survivor R6_144: the cell filter was dropped (every July row returned)
        tmp = tempfile.TemporaryDirectory(); p = os.path.join(tmp.name, "sweep_raw.jsonl")
        base = {"prompt": "base", "mitigation": "none", "failure_mode": "error_500_after_commit", "task_id": "t", "sample": 0}
        rows = [dict(base, episode_id="e1", model=HAIKU), dict(base, episode_id="e2", model=SONNET),
                dict(base, episode_id="e3", model=HAIKU, prompt="careful"), dict(base, episode_id="e4", model=HAIKU, mitigation="idem_keys"),
                dict(base, episode_id="e5", model=HAIKU, failure_mode="timeout_after_commit")]
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        saved = v3_load.V1_RAW; v3_load.V1_RAW = p
        try:
            got = v3_load.load_v1_cell(HAIKU, "base", "none", "error_500_after_commit")
            self.assertEqual([r["episode_id"] for r in got], ["e1"])
        finally:
            v3_load.V1_RAW = saved; tmp.cleanup()


class StrictReplay(unittest.TestCase):
    def test_strict_replay_passes_the_idempotency_key_to_the_world(self):
        # survivor R6_143: 'args.pop("idempotency_key", "") or ""' became 'and ""' (keys never reached the world)
        tmp = tempfile.TemporaryDirectory(); res = tmp.name
        d = os.path.join(res, "sweep_transcripts"); os.makedirs(d)
        def write(fname, key2):
            t = {"model": HAIKU, "task_id": "payments-0001", "turns": [
                {"role": "assistant", "content": [_pay("k1", "a")]},
                {"role": "tool_results", "content": []},
                {"role": "assistant", "content": [_pay(key2, "b")]}]}
            json.dump(t, open(os.path.join(d, fname), "w"))
        write("claude-haiku-4-5-20251001_base_idem_keys_error_500_after_commit__s0__payments-0001.json", "k1")
        write("claude-haiku-4-5-20251001_base_idem_keys_error_500_after_commit__s1__payments-0001.json", "k2")
        saved = v3_load.RES; v3_load.RES = res
        try:
            out = v3_load._strict_v2()
            same = [v for k, v in out.items() if k.endswith("__s0")]; new = [v for k, v in out.items() if k.endswith("__s1")]
            self.assertEqual((same, new), ([0], [1]))
        finally:
            v3_load.RES = saved; tmp.cleanup()


class RescoreValues(unittest.TestCase):
    def test_p1_values_include_the_tool_inputs(self):
        # survivor R6_175: 'b.get("input") or {}' became 'and {}' (inputs ignored, only goals remained)
        t = {"turns": [{"role": "assistant", "content": [_pay("k1", "a", amount="$1,000.00"), _tu("list_effects", {}, "b")]}],
             "score": {"goals": [{"canon_args": {"payee": "acme", "amount": "1000", "memo": "inv-1"}}]}}
        vals = rc.values_p1(t)
        self.assertEqual(vals, ["acme", "$1,000.00", "inv-1", "acme", "1000", "inv-1"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
