"""Tests added by the code-verifier loop (iteration 5): each kills a mutant that survived
every existing suite (94 new mutants, written only in number-producing functions that no
earlier round had mutated -- the coverage map over paper-merged/make_numbers.py and
analysis/*.py -- plus untouched scoring lines of src/world.py, src2/resume.py,
src2/sweep2.py and src/rescore_c1.py). Run from the repo root:  python3 tests_mutants5.py"""
import contextlib, csv, glob, importlib.util, io, json, os, re, shutil, sys, tempfile, unittest
from collections import defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("analysis", "src", "src2"):
    sys.path.insert(0, os.path.join(ROOT, d))
import v3_common                                   # noqa: E402
from v3_common import Numbers, wilson, perm_test, task_rates   # noqa: E402
import v3_load                                     # noqa: E402
import v3_tier1_prereg as tp                       # noqa: E402
import v3_e1_6_belief as vb                        # noqa: E402
import v3_matcher_audit_all as vm                  # noqa: E402
import v3_real_extra as vx                         # noqa: E402
import rescore_c1 as rc                            # noqa: E402

_spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(ROOT, "paper-merged", "make_numbers.py"))
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)

HAIKU, SONNET = "claude-haiku-4-5-20251001", "claude-sonnet-5"


def _reset():
    mk.NUMS.clear(); mk.EPI.clear(); mk.EPI249.clear(); mk.PAIRED_MDD.clear(); mk.DUPS.clear(); mk.REPORT.clear()
    mk.NJ.clear(); mk.FULL209 = ""


def _tu(name, inp, tid="tu1"):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


# ---------------------------------------------------------------------------
# paper-merged/make_numbers.py: helpers no earlier mutant had touched
# ---------------------------------------------------------------------------
class Helpers(unittest.TestCase):
    def test_tex_id_is_typewriter_with_specials_outside_detokenize(self):
        # survivor N507: the \texttt wrapper was dropped (15 printed id macros changed)
        self.assertEqual(mk.tex_id("a#b"), r"\texttt{\detokenize{a}\#\detokenize{b}}")
        self.assertEqual(mk.tex_id("qwen/qwen3.6-27b"), r"\texttt{\detokenize{qwen/qwen3.6-27b}}")

    def test_temp_label_distinguishes_default_unverified_and_zero(self):
        # survivors N517 (None -> "0"), N518 ("unverified" -> "0"): Table 6 temperature column
        self.assertEqual(mk.temp_label(["0.0", "0.0"]), "0")
        self.assertEqual(mk.temp_label(["0"]), "0")
        self.assertEqual(mk.temp_label(["None", "None"]), "default")
        self.assertEqual(mk.temp_label(["unverified"]), "0 sent, unverified")
        self.assertIsNone(mk.temp_label(["absent"]))
        self.assertEqual(mk.temp_label(["0.0", "None"]), "mixed")

    def test_retry_text_sentence_is_the_one_naming_the_key(self):
        # survivor N513: the first sentence was taken instead of the key-naming one
        a = {"to": "x@example.com", "subject": "Hi", "idempotency_key": "k1"}
        t = {"turns": [
            {"role": "assistant", "content": [_tu("send_email", a, "t0")]},
            {"role": "tool_results", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "ERROR", "is_error": True}]},
            {"role": "assistant", "content": [{"type": "text", "text": "The call failed. I will retry with the same idempotency key to avoid a duplicate."},
                                              _tu("send_email", a, "t1")]},
            {"role": "tool_results", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}]}
        txt, sent, reused = mk.retry_text(t)
        self.assertTrue(txt.startswith("The call failed."))
        self.assertEqual(sent, "I will retry with the same idempotency key to avoid a duplicate.")
        self.assertTrue(reused)
        # survivor N512: "reused" needs the SAME key on the retry
        b = dict(a, idempotency_key="k2")
        t["turns"][2]["content"][1] = _tu("send_email", b, "t1")
        self.assertFalse(mk.retry_text(t)[2])
        t["turns"][2]["content"][0]["text"] = "The call failed. Retrying now."
        self.assertEqual(mk.retry_text(t)[1], "")

    def test_fmt_rounds_half_up_signs_differences_and_formats_bounds(self):
        # survivors N527 (half-even), N528 (unsigned d), N529 (p_bound not a p), N530 (no thousands), N531 (CI swapped)
        self.assertEqual(mk._fmt("e1.x:ivr", "value", 0.625), "0.63")
        self.assertEqual(mk._fmt("e1.x:ivr", "value", 0.125), "0.13")
        self.assertEqual(mk._fmt("e1.x:d", "value", -0.125), "$-$0.13")
        self.assertEqual(mk._fmt("e1.x:d", "value", 0.3), "+0.30")
        self.assertEqual(mk._fmt("e1.x:d", "value", 0.0), "0.00")
        self.assertEqual(mk._fmt("e1.x", "p_bound", [0.0004, 0.03]), "[<0.001, 0.030]")
        self.assertEqual(mk._fmt("e1.x", "d_bound", [0.1, 0.2]), "[0.10, 0.20]")
        self.assertEqual(mk._fmt("e1.x", "ci", [0.4, 0.6]), "[0.40, 0.60]")
        self.assertEqual(mk._fmt("e1.x", "n", 14833), "14,833")
        self.assertEqual(mk._fmt("e1.x", "n", 9999), "9999")
        self.assertEqual(mk._fmt("e1.x", "value", None), "provisional")

    def test_check_analysis_v3_compares_k_over_n_and_maps_e1_2_to_at_least_once(self):
        # survivors N535 (never a mismatch), N536 (E1-2 key without the :at_least_once tag)
        _reset()
        mk.NJ.update({"e1_2.haiku.base.none.none:ivr": {"k": 56, "n": 64, "value": 0.875},
                      "e1_3.haiku.base.keys.e500:ivr": {"k": 2, "n": 64, "value": 0.03}})
        mk.put("e12:haiku:base:none:none:at_least_once:kn", "56/64")
        mk.put("e13:haiku:base:keys:e500:kn", "3/64")
        mk.check_analysis_v3()
        self.assertEqual(mk.NUMS["nj:check:cells"], "2")
        self.assertEqual(mk.NUMS["nj:check:mismatch"], "1")
        self.assertTrue(any("e1_3.haiku.base.keys.e500:ivr" in r for r in mk.REPORT))


class TranscriptDirs(unittest.TestCase):
    def test_smoke_runs_are_not_released_transcripts(self):
        # survivor N511: results/**/smoke/sweep_transcripts entered the sampling section (samp:crossprovider-smoke:*)
        if not glob.glob(os.path.join(ROOT, "results", "**", "smoke", "sweep_transcripts", "*.json"), recursive=True):
            self.skipTest("no smoke transcripts here (the anonymized artifact does not ship them)")
        batches = {k[0] for k in mk.transcripts_209()}
        self.assertFalse([b for b in batches if "smoke" in b], batches)
        self.assertTrue({"jul", "xp"} <= batches)
        self.assertFalse([k[0] for k in mk.transcripts_249() if "smoke" in k[0]])


class FullTable(unittest.TestCase):
    def test_full209_rows_are_core_cells_in_column_order(self):
        # survivors N519 (Tier-1 cells added to the appendix table), N520 (CI columns swapped)
        _reset()
        vals = {"ivr": "0.69", "kn": "44/64", "ci": "[0.57, 0.79]", "tci": "[0.51, 0.82]", "boot": "[0.53, 0.84]",
                "tvr": "0.69", "tvrci": "[0.51, 0.82]", "succ": "1.00"}
        for q in ("jul:haiku:base:none:e500", "e12:haiku:base:none:none:at_least_once", "xp:qwen:base:none:toA"):
            for k, v in vals.items():
                mk.put(f"{q}:{k}", v)
        rows64 = {f"e{i}": {"sample": i % 2} for i in range(64)}
        cells = {("jul", HAIKU, "base", "none", "error_500_after_commit", ()): rows64,
                 ("e12", HAIKU, "base", "none", "none", ("at_least_once",)): rows64,
                 ("xp", "openrouter:qwen/qwen3.6-27b-20260422", "base", "none", "timeout_after_commit", ()): rows64,
                 ("xp", "openrouter:openai/gpt-5.5-20260423", "base", "none", "timeout_after_commit", ()): dict(list(rows64.items())[:40])}
        mk.write_full209(cells)
        lines = mk.FULL209.split("\n")
        self.assertEqual(mk.NUMS["full209:rows"], "2")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("Haiku 4.5 & base/none/error\\_500\\_after\\_commit & 0.69 & 44/64 & [0.57, 0.79] & [0.51, 0.82] & [0.53, 0.84] & 0.69 [0.51, 0.82] & 1.00"), lines[0])
        self.assertTrue(lines[1].startswith("Qwen3.6-27B & base/none/timeout"), lines[1])
        self.assertFalse(any("at\\_least\\_once" in l for l in lines))


class Design(unittest.TestCase):
    """emit_design() / check_samples() against the checked-in Tier-0 files and the sources."""
    @classmethod
    def setUpClass(cls):
        _reset()
        mk.emit_design()
        cls.N = dict(mk.NUMS)

    def test_haiku_agreement_is_the_worst_cell(self):
        # survivor N521: max instead of min
        rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "results", "analysis_v2", "e0_1_corrected_table.csv"))) if "haiku" in r["model"]]
        worst = min(int(r["verdict_agree_tasks"]) for r in rows)
        self.assertLess(worst, max(int(r["verdict_agree_tasks"]) for r in rows))
        self.assertEqual(self.N["haiku:agree_min"], f"{worst} of {rows[0]['n_tasks']}")

    def test_hand_check_controls_exclude_normalization_only_pairs(self):
        # survivor N522
        rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "analysis_v2", "e0_3_handcheck.csv"))))
        ctl = sum(1 for r in rows if r["auto_class"] != "normalization_only")
        self.assertLess(ctl, len(rows))
        self.assertEqual(self.N["match:hand:controls"], str(ctl))

    def test_design_constants_come_from_the_sources(self):
        # survivors N523 (graphs = number of topologies), N524 (domains + 1), N525 (temperature "0.0")
        mix = json.loads(re.search(r"^MIX\s*=\s*(\{[^}]*\})", open(os.path.join(ROOT, "src2", "graph.py")).read(), re.M).group(1).replace("'", '"'))
        self.assertEqual(self.N["design:graphs"], str(sum(mix.values())))
        self.assertNotEqual(sum(mix.values()), len(mix))
        import world
        self.assertEqual(self.N["design:domains"], str(len(world.EFFECTFUL_TOOLS)))
        self.assertEqual(self.N["design:domains"], "5")
        self.assertEqual(self.N["design:temp_req"], "0")
        self.assertEqual(self.N["design:maxturns1"], str(__import__("model_agent").MAX_TURNS))

    def test_samples_per_task(self):
        # survivor N526
        _reset()
        mk.check_samples({"c": {"a": {"sample": 0}, "b": {"sample": 1}}})
        self.assertEqual(mk.NUMS["design:samples"], "2")
        self.assertEqual(mk.REPORT, [])
        mk.check_samples({"c": {"a": {"sample": 0}}})
        self.assertEqual(mk.NUMS["design:samples"], "1")
        self.assertTrue(mk.REPORT and mk.REPORT[0].startswith("WARNING"))


class EndToEnd(unittest.TestCase):
    """One full make_numbers pass with its outputs redirected (nothing in the repo is written)."""
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.saved = (mk.OUT_TEX, mk.SNAPSHOT, mk.OUT_REP, mk.render_templates)
        mk.OUT_TEX = os.path.join(cls.tmp.name, "numbers.tex"); mk.SNAPSHOT = os.path.join(cls.tmp.name, "numbers.json")
        mk.OUT_REP = os.path.join(cls.tmp.name, "report.txt"); mk.render_templates = lambda: []
        _reset()
        with contextlib.redirect_stdout(io.StringIO()):
            mk.main()
        cls.N = dict(mk.NUMS)

    @classmethod
    def tearDownClass(cls):
        mk.OUT_TEX, mk.SNAPSHOT, mk.OUT_REP, mk.render_templates = cls.saved
        cls.tmp.cleanup()

    def test_word_forms_cover_zero_to_ten(self):
        # survivors N537 ("10" has no word form), N538 ("0" -> "0" instead of "zero")
        # the ":n" denominators and ":pct" values are derived after the word forms, so they have none
        late = lambda k: k.endswith((":w", ":pct")) or re.fullmatch(r"samp:\w+:\w+:ident:n", k)
        tens = [k for k, v in self.N.items() if v == "10" and not late(k)]
        zeros = [k for k, v in self.N.items() if v == "0" and not late(k)]
        self.assertTrue(tens and zeros)
        for k in tens:
            self.assertEqual(self.N.get(k + ":w"), "ten", k)
        for k in zeros:
            self.assertEqual(self.N.get(k + ":w"), "zero", k)
        self.assertEqual(self.N["design:samples:w"], "two")
        self.assertEqual(self.N["design:domains:w"], "five")
        self.assertNotIn("design:graphs:w", self.N)

    def test_c1_rescore_macros_match_the_analysis_outputs(self):
        # the re-score changed three October Haiku scores and no released one (main.tex ASSERTs 6798 episodes)
        c1 = json.load(open(os.path.join(ROOT, "results", "analysis_v3", "rescore_c1.json")))
        self.assertEqual(self.N["nj:c1.rescore.score_changed"], str(c1["score_changed"]))
        self.assertEqual(self.N["nj:c1.audit.episodes"], str(c1["episodes"]))
        self.assertEqual(self.N["nj:c1.audit.episodes"], self.N["total:all"].replace(",", ""))
        self.assertEqual(self.N["nj:check:mismatch"], "0")


# ---------------------------------------------------------------------------
# analysis/v3_common.py: the Numbers store (survivors N542, N544)
# ---------------------------------------------------------------------------
class NumbersStore(unittest.TestCase):
    def test_cell_block_primary_interval_is_the_task_level_one(self):
        N = Numbers()
        st = {"p": 0.5, "ci_task": (0.3, 0.7), "ci_ep": (0.4, 0.6), "n": 64, "k": 32, "T": 32}
        N.cell("c", st, "src")
        self.assertEqual(N.d["c:ivr"]["ci"], [0.3, 0.7]); self.assertEqual(N.d["c:ivr"]["ci_method"], "wilson95_task(p*T,T)")
        self.assertEqual(N.d["c:ivr_ci_ep"]["ci"], [0.4, 0.6]); self.assertEqual(N.d["c:ivr_ci_ep"]["ci_method"], "wilson95_episode")
        self.assertEqual(N.d["c:ivr"]["k"], 32); self.assertEqual(N.d["c:ivr"]["T"], 32)

    def test_test_block_keeps_sign_and_plus_minus_counts(self):
        N = Numbers()
        N.test("t", {"d": -0.25, "p": 0.04, "T": 32, "plus": 3, "minus": 11}, "src", holm_p=0.08, verdict="v")
        self.assertEqual(N.d["t:d"]["value"], -0.25); self.assertEqual(N.d["t:d"]["plus"], 3); self.assertEqual(N.d["t:d"]["minus"], 11)
        self.assertEqual(N.d["t:p"]["holm_p"], 0.08); self.assertEqual(N.d["t:p"]["verdict"], "v")


# ---------------------------------------------------------------------------
# analysis/v3_tier1_prereg.py: cell_block / contrast / put_test on synthetic cells
# ---------------------------------------------------------------------------
def _ep(task, s, v, **kw):
    e = {"task": task, "sample": s, "v": v, "eid": f"{task}:{s}"}
    e.update(kw); return e


class PreregBlocks(unittest.TestCase):
    def setUp(self):
        self.saved = (tp.N, tp.TESTS, dict(tp.SRCFILE))
        tp.N = Numbers(); tp.TESTS = []; tp.SRCFILE.update({"e1_3": "results/tier1/e1_3/raw.jsonl", "xp": "results/crossprovider/sweep_raw.jsonl"})

    def tearDown(self):
        tp.N, tp.TESTS = self.saved[0], self.saved[1]; tp.SRCFILE.clear(); tp.SRCFILE.update(self.saved[2])

    def test_cell_block_rates_use_their_own_denominators(self):
        # survivors N546 (strict over all episodes), N547 (read-back over all), N549 (task-any = last sample)
        k = ("e1_3", HAIKU, "base", "idem_keys", "error_500_after_commit")
        eps = [_ep("t1", 0, 1, strict=1, rb=1), _ep("t1", 1, 0, strict=0, rb=0), _ep("t2", 0, 1, strict=None, rb=None),
               _ep("t2", 1, 1), _ep("t3", 0, 0, rb=1), _ep("t3", 1, 0, rb=0)]
        _, st, row = tp.cell_block({k: eps}, {}, k)
        N = tp.N.d; nm = "e1_3.haiku.base.keys.e500"
        self.assertEqual(N[nm + ":ivr"]["value"], 0.5); self.assertEqual(N[nm + ":ivr"]["T"], 3)
        self.assertEqual(N[nm + ":ivr_strict"]["value"], 0.5); self.assertEqual(N[nm + ":ivr_strict"]["n"], 2)
        self.assertEqual(N[nm + ":readback_after_failure"]["value"], 0.5); self.assertEqual(N[nm + ":readback_after_failure"]["n"], 4)
        self.assertAlmostEqual(N[nm + ":ivr_task_any"]["value"], 2 / 3, places=5); self.assertEqual(N[nm + ":ivr_task_any"]["n"], 3)
        self.assertEqual(row["ivr_task_any"], "0.6667")
        self.assertNotIn(nm + ":refusals", N)

    def test_cell_block_refusal_bounds_and_bootstrap(self):
        # survivors N548 (bounds swapped), N550 (bootstrap over the imputed episodes)
        k = ("xp", "openrouter:google/gemini-4-argon-20261013", "base", "none", "error_500_after_commit")
        eps = [_ep(f"t{i}", s, 1 if i % 2 else 0) for i in range(6) for s in (0, 1)]
        refs = {k: [("t6", 0, "blocked"), ("t6", 1, "blocked")]}
        _, st, row = tp.cell_block({k: eps}, refs, k)
        N = tp.N.d; nm = "xp.gemini.base.none.e500"
        self.assertEqual(N[nm + ":refusals"]["value"], 2); self.assertEqual(N[nm + ":refusals"]["n"], 14)
        lo, hi = N[nm + ":ivr_bound"]["value"]
        self.assertAlmostEqual(lo, 6 / 14); self.assertAlmostEqual(hi, 8 / 14)
        self.assertEqual(N[nm + ":ivr_ci_boot"]["ci"], [round(x, 6) for x in tp.boot(eps)])
        self.assertNotEqual(N[nm + ":ivr_ci_boot"]["ci"], [round(x, 6) for x in tp.boot(tp.impute(eps, refs[k], 1))])
        self.assertEqual(row["ivr_if_refusal_no"], "0.4286"); self.assertEqual(row["ivr_if_refusal_yes"], "0.5714")

    def test_contrast_is_y_minus_x_with_bounds_over_four_codings(self):
        # survivors N551 (x, y swapped), N552 (two codings), N553 (max, min), N554 (holm_p dropped), N555 (p_bound := d_bound)
        kx = ("xp", "openrouter:google/gemini-4-argon-20261013", "base", "none", "timeout_after_commit")
        ky = ("xp", "openrouter:google/gemini-4-argon-20261013", "base", "none", "error_500_after_commit")
        ex = [_ep(f"t{i}", s, 1 if i < 2 else 0) for i in range(8) for s in (0, 1)]
        ey = [_ep(f"t{i}", s, 1 if i < 7 else 0) for i in range(8) for s in (0, 1)]
        refs = {kx: [("t8", 0, "b"), ("t8", 1, "b")], ky: [("t8", 0, "b"), ("t8", 1, "b"), ("t9", 0, "b")]}
        t = tp.contrast({kx: ex, ky: ey}, refs, kx, ky, "e1_1.P.H2.gemini", "500 vs timeout", "P")
        self.assertAlmostEqual(t["d"], (7 - 2) / 8); self.assertEqual(t["T"], 8)
        self.assertEqual((t["refusals_x"], t["refusals_y"]), (2, 3))
        bnds = [perm_test(task_rates(tp.impute(ex, refs[kx], vx)), task_rates(tp.impute(ey, refs[ky], vy)))
                for vx, vy in ((1, 0), (0, 1), (0, 0), (1, 1))]
        self.assertEqual(t["d_bound"], [min(b["d"] for b in bnds), max(b["d"] for b in bnds)])
        self.assertEqual(t["p_bound"], [min(b["p"] for b in bnds), max(b["p"] for b in bnds)])
        self.assertLess(t["d_bound"][0], t["d_bound"][1])
        self.assertNotEqual(t["d_bound"][0], min(b["d"] for b in bnds[2:]))   # the (1,0) coding is the minimum
        tp.put_test(t, verdict="supported", holm_p=0.02)
        N = tp.N.d
        self.assertEqual(N["e1_1.P.H2.gemini:p"]["holm_p"], 0.02); self.assertEqual(N["e1_1.P.H2.gemini:p"]["verdict"], "supported")
        self.assertEqual(N["e1_1.P.H2.gemini:p_bound"]["value"], t["p_bound"]); self.assertEqual(N["e1_1.P.H2.gemini:d_bound"]["value"], t["d_bound"])
        self.assertNotEqual(N["e1_1.P.H2.gemini:p_bound"]["value"], N["e1_1.P.H2.gemini:d_bound"]["value"])
        self.assertEqual(N["e1_1.P.H2.gemini:d"]["value"], 0.625)


# ---------------------------------------------------------------------------
# analysis/v3_e1_6_belief.py (never mutated before; 7 printed macros)
# ---------------------------------------------------------------------------
def _belief_transcript(text, first_error=True):
    args = {"payee": "p", "amount": "1", "memo": "m", "idempotency_key": "k"}
    turns = [{"role": "assistant", "content": [_tu("send_payment", args, "a")]},
             {"role": "tool_results", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "ok" if not first_error else "ERROR", "is_error": first_error}]}]
    if not first_error:
        turns += [{"role": "assistant", "content": [{"type": "text", "text": "First step done. Now the second."}, _tu("send_email", {"to": "x", "subject": "s"}, "b")]},
                  {"role": "tool_results", "content": [{"type": "tool_result", "tool_use_id": "b", "content": "ERROR", "is_error": True}]}]
    content = ([{"type": "text", "text": text}] if text else []) + [_tu("list_effects", {}, "c")]
    turns += [{"role": "assistant", "content": content},
              {"role": "tool_results", "content": [{"type": "tool_result", "tool_use_id": "c", "content": "(no effects recorded)"}]},
              {"role": "assistant", "content": [{"type": "text", "text": "The payment may have been processed; checking the ledger showed nothing, so I retried."}]}]
    return {"turns": turns}


class BeliefCoder(unittest.TestCase):
    def test_window_is_the_first_assistant_turn_after_the_injected_error(self):
        # crashers N557 (first tool result of any kind), N558 (every later assistant text)
        txt, nxt = vb.window(_belief_transcript("I will retry.", first_error=False))
        self.assertEqual(txt, "I will retry."); self.assertEqual(nxt, "list_effects")
        txt, nxt = vb.window(_belief_transcript("", first_error=True))
        self.assertEqual(txt, ""); self.assertEqual(nxt, "list_effects")
        self.assertEqual(vb.window({"turns": [{"role": "assistant", "content": []}]}), (None, None))

    def test_codes_in_rule_order_with_the_clause_filter(self):
        # crashers N563 (R_UNCERTAIN not checked first), N564 (no clause split)
        self.assertEqual(vb.code("The payment was processed but may have been duplicated; let me check the effect log.")[0], "possibly_committed")
        self.assertEqual(vb.code("The payment was processed, error notwithstanding.")[0], "committed")
        self.assertEqual(vb.code("The email was sent successfully, but the ticket creation encountered an internal server error")[0], "uncoded")
        self.assertEqual(vb.code("The payment failed. I will retry.")[:2], ("not_committed", "explicit"))
        self.assertEqual(vb.code("I will retry.")[:2], ("not_committed", "implied_by_retry"))
        self.assertEqual(vb.code("Please try again later.")[0], "handoff")
        self.assertEqual(vb.code("")[0], "no_text"); self.assertEqual(vb.code(None)[0], "no_failure")


class BeliefMain(unittest.TestCase):
    """main() on a synthetic eight-cell data set: the denominators of the printed k/n."""
    TEXTS = {"haiku": {"error_500_after_commit": ["The payment may have been processed; let me check.", "The payment was processed, error notwithstanding.",
                                                  "The payment failed; I will retry.", ""],
                       "timeout_after_commit": ["I will retry.", "", "", ""],
                       "error_500_body_after_commit": ["Let me check the effect log.", "I will retry.", "", ""],
                       "status_unknown_after_commit": ["Let me check whether it went through.", "Let me verify the ledger.", "I will retry.", ""]},
             "sonnet": {fm: ["", "", "", ""] for fm in vb.FM_ORDER}}

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = cls.tmp.name
        cls.saved = (vb.OUT, vb.episodes)
        vb.OUT = os.path.join(d, "out"); os.makedirs(vb.OUT)
        eps = []
        for m, model in (("haiku", HAIKU), ("sonnet", SONNET)):
            for fm in vb.FM_ORDER:
                for i, text in enumerate(cls.TEXTS[m][fm]):
                    task, s = f"payments-{i // 2}", i % 2        # two samples per task
                    p = os.path.join(d, f"{m}_{fm}_{i}.json"); json.dump(_belief_transcript(text), open(p, "w"))
                    eps.append((p, {"episode_id": f"{model}__{fm}__{task}__s{s}", "model": model, "failure_mode": fm, "task_id": task,
                                    "sample": s, "violations": 1 if i == 0 else 0, "readback_after_failure": True}, "e1_6"))
        vb.episodes = lambda: eps
        with contextlib.redirect_stdout(io.StringIO()):
            vb.main()
        cls.N = json.load(open(os.path.join(vb.OUT, "e1_6_belief.json")))
        cls.rows = list(csv.DictReader(open(os.path.join(vb.OUT, "e1_6_belief_coded.csv"))))

    @classmethod
    def tearDownClass(cls):
        vb.OUT, vb.episodes = cls.saved; cls.tmp.cleanup()

    def test_entertains_commit_is_over_episodes_with_text_and_tasks_with_text(self):
        # survivors N559 (committed only), N560 (no_text in the denominator), N561 (T = episodes), N562 (rate over all episodes)
        e = self.N["e1_6.belief.haiku.e500.entertains_commit"]
        self.assertEqual((e["k"], e["n"], e["T"]), (2, 3, 2)); self.assertAlmostEqual(e["value"], 2 / 3, places=5)
        self.assertEqual(e["ci"], [round(x, 6) for x in wilson(2 / 3 * 2, 2)])
        self.assertEqual(e["no_text"], 1); self.assertEqual(e["codes"], {"possibly_committed": 1, "committed": 1, "not_committed": 1, "no_text": 1})
        e = self.N["e1_6.belief.haiku.w3.entertains_commit"]
        self.assertEqual((e["k"], e["n"], e["T"]), (2, 3, 2))
        e = self.N["e1_6.belief.haiku.toA.entertains_commit"]
        self.assertEqual((e["k"], e["n"], e["T"]), (0, 1, 1))
        self.assertIsNone(self.N["e1_6.belief.sonnet.e500.entertains_commit"]["value"])

    def test_counts_of_the_sample_and_the_silent_model(self):
        # survivors N565 (Sonnet's no_text counted on Haiku), N566 (hand-check n = the extra-60 batch)
        self.assertEqual(self.N["e1_6.belief.no_text.sonnet"]["value"], 16); self.assertEqual(self.N["e1_6.belief.no_text.sonnet"]["n"], 16)
        self.assertEqual(self.N["e1_6.belief.n_episodes"]["value"], 32)
        self.assertEqual(self.N["e1_6.belief.handcheck.n"]["value"], 32)      # fewer rows than 40: the whole set is batch A
        hc = list(csv.DictReader(open(os.path.join(vb.OUT, "e1_6_handcheck.csv"))))
        self.assertEqual(len(hc), 32)
        extra = os.path.join(vb.OUT, "e1_6_handcheck_extra60.csv")
        self.assertTrue(not os.path.exists(extra) or len(list(csv.DictReader(open(extra)))) == 0)
        self.assertEqual(sorted(r["code"] for r in self.rows if r["model"] == SONNET), ["no_text"] * 16)


# ---------------------------------------------------------------------------
# analysis/v3_matcher_audit_all.py (never mutated before; c1.* macros, ASSERT 6798)
# ---------------------------------------------------------------------------
class MatcherAuditAll(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name
        self.saved = (vm.RES, vm.OUT, vm.DIRS)
        vm.RES = os.path.join(d, "results"); vm.OUT = os.path.join(vm.RES, "analysis_v3"); vm.DIRS = [("x", "tr")]
        os.makedirs(os.path.join(vm.RES, "tr")); os.makedirs(vm.OUT)
        ev = lambda start, tid: _tu("create_event", {"calendar_id": "c", "title": "Sync", "start": start}, tid)
        t1 = {"turns": [{"role": "assistant", "content": [ev("2026-08-01T10:00:00", "a"), _tu("send_email", {"to": "x", "subject": "Sync"}, "b"),
                                                          ev("2026-08-01T10:00", "c")]}]}
        t2 = {"sessions": [{"turns": [{"role": "assistant", "content": [ev("2026-08-01T10:00", "a")]}]},
                           {"turns": [{"role": "assistant", "content": [ev("2026-08-01T10:00", "b"), ev("2026-08-01T11:00", "c")]}]}]}
        json.dump(t1, open(os.path.join(vm.RES, "tr", "t1.json"), "w")); json.dump(t2, open(os.path.join(vm.RES, "tr", "t2.json"), "w"))
        json.dump({"episodes": 2, "score_changed": 1, "rows_changed": 2, "episodes_with_value_merge": 1},
                  open(os.path.join(vm.OUT, "rescore_c1.json"), "w"))

    def tearDown(self):
        vm.RES, vm.OUT, vm.DIRS = self.saved; self.tmp.cleanup()

    def test_pairs_are_same_tool_and_the_before_fix_class_uses_the_old_normalizer(self):
        # survivors N568 (cross-tool pairs counted), N569 (before-fix counts = fixed counts), N571 (episodes + 1), N570
        with contextlib.redirect_stdout(io.StringIO()):
            vm.main()
        N = json.load(open(os.path.join(vm.OUT, "matcher_audit_all.json")))
        self.assertEqual(N["c1.audit.episodes"]["value"], 2); self.assertEqual(N["c1.audit.effcalls"]["value"], 6)
        self.assertEqual(N["c1.audit.pairs"]["value"], 4)                       # t1: 1 same-tool pair; t2: 3
        self.assertEqual(N["c1.audit.normalization_only"]["value"], 1); self.assertEqual(N["c1.audit.normalization_only"]["before_fix"], 0)
        self.assertEqual(N["c1.audit.near_miss"]["value"], 2); self.assertEqual(N["c1.audit.near_miss"]["before_fix"], 3)
        self.assertEqual(N["c1.audit.raw_identical"]["value"], 1); self.assertEqual(N["c1.audit.pairs_moved_by_fix"]["value"], 1)
        self.assertEqual(N["c1.audit.near_miss_before_fix"]["value"], 3)
        self.assertEqual(N["c1.rescore.score_changed"]["value"], 1); self.assertEqual(N["c1.rescore.episodes"]["value"], 2)
        pairs = list(csv.DictReader(open(os.path.join(vm.OUT, "matcher_audit_all_pairs.csv"))))
        self.assertEqual(sorted(p["cls"] for p in pairs), ["near_miss", "near_miss", "normalization_only"])


# ---------------------------------------------------------------------------
# analysis/v3_real_extra.py cost()  (survivor N574)
# ---------------------------------------------------------------------------
class RealExtraCost(unittest.TestCase):
    def test_cost_sums_both_providers(self):
        tmp = tempfile.TemporaryDirectory(); p = os.path.join(tmp.name, "raw.jsonl")
        with open(p, "w") as f:
            for i, (m, c) in enumerate(((HAIKU, 1.25), ("openrouter:qwen/qwen3.6-27b-20260422", 0.5), (SONNET, 0.75), (HAIKU, 99.0))):
                f.write(json.dumps({"episode_id": f"e{i if i < 3 else 0}", "model": m, "cost_usd": c}) + "\n")   # last row of e0 wins
        saved = (vx.REAL_RAW, vx.N); vx.REAL_RAW = p; vx.N = Numbers()
        try:
            vx.cost()
            e = vx.N.d["xr.cost.r1"]
        finally:
            vx.REAL_RAW, vx.N = saved; tmp.cleanup()
        self.assertEqual((e["value"], e["anthropic"], e["openrouter"], e["n"]), (100.25, 99.75, 0.5, 3))


# ---------------------------------------------------------------------------
# src/rescore_c1.py on the E1-3 run alone (survivors N591, N593)
# ---------------------------------------------------------------------------
class RescoreC1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(); d = cls.tmp.name
        res = os.path.join(d, "results"); os.makedirs(os.path.join(res, "crossprovider")); os.makedirs(os.path.join(res, "claude_batch_2026-10"))
        for p in ("sweep_raw_v2.jsonl", "sweep2_raw_v2.jsonl", "crossprovider/sweep_raw.jsonl", "crossprovider/sweep2_raw.jsonl",
                  "claude_batch_2026-10/sweep2_raw.jsonl"):
            open(os.path.join(res, p), "w").close()
        shutil.copytree(os.path.join(ROOT, "results", "tier1", "e1_3"), os.path.join(res, "tier1", "e1_3"))
        for f in glob.glob(os.path.join(res, "tier1", "e1_3", "raw_v2.jsonl")):
            os.remove(f)
        os.makedirs(os.path.join(d, "configs"))
        cfg = json.load(open(os.path.join(ROOT, "configs", "e1_3_keys_under_500.json")))
        cfg["out_dir"] = os.path.join(res, "tier1", "e1_3"); cfg["tier1_root"] = os.path.join(res, "tier1")
        json.dump(cfg, open(os.path.join(d, "configs", "e1_3_keys_under_500.json"), "w"))
        cls.saved = (rc.HERE, rc.RES, rc.OUT)
        rc.HERE, rc.RES, rc.OUT = d, res, os.path.join(res, "analysis_v3")
        with contextlib.redirect_stdout(io.StringIO()):
            rc.main()
        cls.res = res
        cls.summary = json.load(open(os.path.join(rc.OUT, "rescore_c1.json")))
        cls.diffs = list(csv.DictReader(open(os.path.join(rc.OUT, "rescore_c1.csv"))))

    @classmethod
    def tearDownClass(cls):
        rc.HERE, rc.RES, rc.OUT = cls.saved; rc.set_norm(rc.NEW_NORM); cls.tmp.cleanup()

    def test_the_fixed_normalizer_changes_one_e1_3_score(self):
        # survivor N591: the "new" replay ran under the old normalizer, so nothing changed
        self.assertEqual(self.summary["score_changed"], 1); self.assertEqual(self.summary["rows_changed"], 1)
        self.assertEqual(self.summary["by_source"]["e1_3"]["episodes"], 128)
        d = self.diffs[0]
        self.assertEqual((d["old_violations"], d["new_violations"]), ("0", "1"))
        self.assertIn("calendar-98fe897062__s1", d["episode_id"])
        self.assertTrue(os.path.exists(os.path.join(self.res, "tier1", "e1_3", "raw_v2.jsonl")))

    def test_tier1_rows_are_recomputed_field_by_field(self):
        # survivor N593: only the score fields were overwritten; episode_row's derived fields must follow
        ch = set(self.diffs[0]["fields_changed"].split())
        self.assertIn("violations", ch); self.assertIn("retried", ch)
        self.assertTrue(ch & {"retry_key_category", "stale_reexec", "key_attached_first"}, ch)
        new = json.loads(self.diffs[0]["new_values"]); old = json.loads(self.diffs[0]["old_values"])
        self.assertTrue(new["retried"]); self.assertFalse(old["retried"])
        v2 = [json.loads(l) for l in open(os.path.join(self.res, "tier1", "e1_3", "raw_v2.jsonl")) if l.strip()]
        row = next(r for r in v2 if r["episode_id"] == self.diffs[0]["episode_id"])
        self.assertEqual(row["violations"], 1); self.assertTrue(row["retried"]); self.assertEqual(row.get("cost_usd"), None if "cost_usd" not in row else row["cost_usd"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
