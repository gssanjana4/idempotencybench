"""Tests added by the code-verifier loop (iteration 4): each kills a mutant that survived
every existing suite (72 new mutants over paper-merged/make_numbers.py sections never
mutated before and the code changed since c7373b4 -- tex_text quote mapping,
untracked_results(), the PII sweep -- plus analysis/v3_load.py, src/tier1.py,
src_real/stripe_client.py, src_real/run_real.py and src/providers.py). Run from the
repo root:  python3 tests_mutants4.py"""
import contextlib, csv, glob, importlib.util, io, json, math, os, re, shutil, subprocess, sys, tempfile, unittest
import urllib.error
ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ("analysis", "src", "src2", "src_real"):
    sys.path.insert(0, os.path.join(ROOT, d))
import v3_load                                     # noqa: E402
import tier1, sweep, providers, run_real           # noqa: E402
import stripe_client                               # noqa: E402
from stripe_client import (StripeClient, UnsafeKeyError, StripeInfraError, StripeAPIError,   # noqa: E402
                           assert_test_key, _check_live)

_spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(ROOT, "paper-merged", "make_numbers.py"))
mk = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(mk)

HAIKU, SONNET, FABLE = "claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-fable-5"
GPT, GEMINI, QWEN = "openrouter:openai/gpt-5.5-20260423", "openrouter:google/gemini-4-argon-20261013", "openrouter:qwen/qwen3.6-27b-20260422"
FAKE_KEY = "sk_test_FAKEKEY0123456789secretvalue"


def _reset():
    mk.NUMS.clear(); mk.EPI.clear(); mk.EPI249.clear(); mk.PAIRED_MDD.clear(); mk.DUPS.clear(); mk.REPORT.clear()


def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                          cwd=cwd, capture_output=True, text=True, check=True)


def _tool_use(name, inp, tid="tu1"):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _turns(*calls, text="done"):
    """One assistant turn per call (tool_use) followed by its ok result, then a closing text."""
    out = []
    for i, (name, inp) in enumerate(calls):
        out.append({"role": "assistant", "content": [_tool_use(name, inp, f"tu{i}")]})
        out.append({"role": "tool_results", "content": [{"type": "tool_result", "tool_use_id": f"tu{i}", "content": "ok"}]})
    out.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
    return out


# ---------------------------------------------------------------------------
# paper-merged/make_numbers.py: code changed since c7373b4
# ---------------------------------------------------------------------------
class TexText(unittest.TestCase):
    def test_opening_straight_quote_becomes_a_latex_opening_quote(self):
        # survivors M401 (every ' became `), M402 (no mapping), M403 (only after a space)
        self.assertEqual(mk.tex_text("subject 'Receipt: x' now"), "subject `Receipt: x' now")
        self.assertEqual(mk.tex_text("'Follow-up: y' on the calendar"), "`Follow-up: y' on the calendar")
        self.assertEqual(mk.tex_text("don't"), "don't")                      # apostrophe inside a word
        self.assertEqual(mk.tex_text("('a')"), "(`a')")                      # opening after "("
        self.assertEqual(mk.tex_text("x'"), "x'")                            # closing only
        self.assertEqual(mk.tex_text("a_b & 'c'"), r"a\_b \& `c'")


class UntrackedResults(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = os.path.join(self.tmp, "repo")
        os.makedirs(os.path.join(self.repo, "results", "analysis_v3"))
        _git(self.tmp, "init", "-q", self.repo)
        with open(os.path.join(self.repo, ".gitignore"), "w") as f:
            f.write("*.log\n")
        with open(os.path.join(self.repo, "results", "analysis_v3", "tracked.csv"), "w") as f:
            f.write("a,b\n1,2\n")
        _git(self.repo, "add", "-A"); _git(self.repo, "commit", "-q", "-m", "init")
        with open(os.path.join(self.repo, "results", "analysis_v3", "e1_6_handcheck_model_coder.csv"), "w") as f:
            f.write("a,b\n3,4\n")
        with open(os.path.join(self.repo, "results", "analysis_v3", "ignored.log"), "w") as f:
            f.write("x\n")
        self.saved = mk.RES

    def tearDown(self):
        mk.RES = self.saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_the_untracked_files_by_real_path(self):
        # survivors M405 (joined to RES, not the toplevel), M407 (paths relative to RES),
        # M409 (ignored files counted as untracked), M406 (no realpath through a symlink)
        res = os.path.join(self.repo, "results")
        mk.RES = res
        got = mk.untracked_results()
        unt = os.path.realpath(os.path.join(res, "analysis_v3", "e1_6_handcheck_model_coder.csv"))
        self.assertEqual(got, {unt})
        link = os.path.join(self.tmp, "link")
        os.symlink(self.repo, link)
        mk.RES = os.path.join(link, "results")
        self.assertEqual(mk.untracked_results(), {unt})

    def test_no_git_means_nothing_is_skipped(self):
        # survivor M404 (raised instead of returning the empty set): the artifact has no .git
        plain = os.path.join(self.tmp, "plain", "results")
        os.makedirs(plain)
        mk.RES = plain
        self.assertEqual(mk.untracked_results(), set())

    def test_pii_sweep_skips_untracked_files(self):
        # survivor M408: the sweep counted the untracked hand-check CSV
        mk.RES = os.path.join(self.repo, "results")
        _reset()
        with contextlib.redirect_stdout(io.StringIO()):
            mk.emit_pii()
        # tracked.csv and the git-ignored ignored.log are swept; the untracked hand-check CSV is not
        self.assertEqual(mk.NUMS["pii:files"], "2")
        self.assertEqual(mk.NUMS["pii:total"], "0")


# ---------------------------------------------------------------------------
# paper-merged/make_numbers.py: sections never mutated before
# ---------------------------------------------------------------------------
class MakeNumbersStats(unittest.TestCase):
    def test_holm_is_monotone_step_down(self):
        # survivors M410 (no running max), M411 (rank off by one)
        self.assertEqual(mk.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        self.assertEqual(mk.holm([0.2, 0.01]), [0.2, 0.02])
        self.assertEqual(mk.holm([0.5]), [0.5])

    def test_task_rates_average_the_samples(self):
        # survivor M412 (max over samples; moved power:paired:min/max)
        epi = [("t1", 1, 0), ("t1", 0, 1), ("t2", 0, None), ("t2", 0, 0)]
        self.assertEqual(mk.task_rates(epi, 1), {"t1": 0.5, "t2": 0.0})
        self.assertEqual(mk.task_rates(epi, 2), {"t1": 0.5, "t2": 0.0})

    def test_paired_mdd_uses_the_task_mean_difference(self):
        # survivor M412 end to end: EPI -> emit_tests -> power:paired
        _reset()
        qa, qe = "jul:haiku:base:none:toA", "jul:haiku:base:none:e500"
        mk.EPI[qa] = [(f"t{i}", 1 if i < 4 else 0, 0) for i in range(8) for _ in (0, 1)]
        mk.EPI[qe] = [(f"t{i}", 1 if (i < 6 and s == 0) else 0, 0) for i in range(8) for s in (0, 1)]
        with contextlib.redirect_stdout(io.StringIO()):
            mk.emit_tests()
        dd = [(0.5 if i < 6 else 0.0) - (1.0 if i < 4 else 0.0) for i in range(8)]
        mu = sum(dd) / 8; sd = math.sqrt(sum((x - mu) ** 2 for x in dd) / 7)
        self.assertAlmostEqual(mk.PAIRED_MDD["haiku"], (1.959964 + 0.841621) * sd / math.sqrt(8))
        self.assertEqual(mk.NUMS["power:paired:min"], mk.f2(mk.PAIRED_MDD["haiku"]))


class MakeNumbersContrasts(unittest.TestCase):
    # per-model (toA ivr, e500 ivr, toA read-back rate, e500 read-back rate)
    CELLS = {"haiku": (0.9, 0.7, 0.1, 0.0), "sonnet": (0.0, 0.7, 0.9, 0.3), "fable": (0.0, 0.3, 0.8, 0.2),
             "gpt": (0.3, 0.1, 0.2, 0.6), "gemini": (0.1, 0.1, 0.9, 0.9), "qwen": (0.5, 0.4, 0.0, 0.0)}

    def fill(self):
        _reset()
        for m, (ta, ea, tr, er) in self.CELLS.items():
            b = "jul" if m in ("haiku", "sonnet", "fable") else "xp"
            mk.NUMS[f"{b}:{m}:base:none:toA:ivr"] = mk.f2(ta); mk.NUMS[f"{b}:{m}:base:none:e500:ivr"] = mk.f2(ea)
            mk.NUMS[f"{b}:{m}:base:none:toA:rbrate"] = mk.f2(tr); mk.NUMS[f"{b}:{m}:base:none:e500:rbrate"] = mk.f2(er)

    def test_checkers_are_defined_by_the_timeout_read_back(self):
        # survivors M413 (e500 read-back used), M414 (higher/lower swapped), M415 (largest rise)
        self.fill()
        mk.emit_contrasts()
        N = mk.NUMS
        self.assertEqual(N["checkers:names"], "Sonnet 5, Fable 5 and Gemini")
        self.assertEqual(N["noncheckers:names"], "Haiku 4.5, GPT-5.5 and Qwen3.6-27B")
        self.assertEqual(N["contrast:e500_higher:names"], "Sonnet 5 and Fable 5")
        self.assertEqual(N["contrast:e500_lower:names"], "Haiku 4.5, GPT-5.5 and Qwen3.6-27B")
        self.assertEqual(N["contrast:e500_higher:n"], "2"); self.assertEqual(N["contrast:e500_lower:n:w"], "three")
        self.assertEqual(N["checkers:smallest_rise"], "Gemini")
        self.assertEqual(N["checkers:rb_toA:min"], "0.80"); self.assertEqual(N["noncheckers:rb_max"], "0.60")


class MakeNumbersDerived(unittest.TestCase):
    def test_hint_p_range_and_alpha(self):
        # survivors M416 (pmax took the smallest p), M417 (alpha 0.10)
        _reset()
        mk.NUMS["checkers:names"] = "Sonnet 5 and Fable 5"; mk.NUMS["noncheckers:names"] = "Haiku 4.5"
        for m, hp, ap in (("haiku", "<0.001", "0.001"), ("sonnet", "0.034", "0.070"), ("fable", "0.062", "0.002")):
            mk.NUMS[f"test:{m}:hint:p"] = hp
            mk.NUMS[f"oct:{m}:amnesiac:none:givr"] = "1.00"; mk.NUMS[f"oct:{m}:bare_hint:none:givr"] = "0.00"
            mk.NUMS[f"oct:{m}:amnesiac:none:kn"] = "32/32"; mk.NUMS[f"oct:{m}:bare_hint:none:kn"] = "0/32"
            mk.NUMS[f"test:{m}:toA_e500:ivr:holm"] = ap; mk.NUMS[f"test:{m}:toA_e500:ivr:p"] = ap
            mk.NUMS[f"test:{m}:toA_e500:rb:holm"] = "0.001"; mk.NUMS[f"test:{m}:toA_e500:rb:p"] = "0.001"
        mk.emit_derived()
        N = mk.NUMS
        self.assertEqual(N["sm:all:n"], "3"); self.assertEqual(N["sm:all:amn:kn"], "96/96")
        self.assertEqual(N["sm:all:pmaxe"], "$p=0.062$")
        self.assertEqual(N["sm:all:sig"], "no")                 # 0.062 is not significant at .05
        self.assertEqual(N["sm:cl:hint_lower_sig"], "no")
        self.assertEqual(N["test:checkers:ivr_holm_sig"], "no")  # sonnet holm 0.070
        self.assertEqual(N["test:noncheckers:ivr_raw_sig"], "yes")
        self.assertEqual(N["test:haiku:hint:pe"], "$p<0.001$"); self.assertEqual(N["test:sonnet:hint:pe"], "$p=0.034$")


class MakeNumbersLists(unittest.TestCase):
    def test_primary_cell_is_the_same_month_batch_at_full_n(self):
        # survivors M418 (July preferred), M419 (any n accepted)
        _reset()
        def cell(q, givr, n, chk="0/0"):
            mk.NUMS[q + ":givr"] = givr; mk.NUMS[q + ":n"] = str(n); mk.NUMS[q + ":chk"] = chk
        cell("oct:haiku:amnesiac:none", "1.00", 32); cell("jul:haiku:amnesiac:none", "0.50", 32)
        cell("oct:sonnet:amnesiac:none", "0.00", 10); cell("jul:sonnet:amnesiac:none", "0.75", 32)   # October cell incomplete
        cell("xp:gpt:amnesiac:none", "0.25", 32); cell("jul:gpt:amnesiac:none", "1.00", 32)
        mk.emit_lists({})
        N = mk.NUMS
        self.assertEqual(N["best:haiku:amnesiac"], "oct:haiku:amnesiac:none")
        self.assertEqual(N["best:sonnet:amnesiac"], "jul:sonnet:amnesiac:none")
        self.assertEqual(N["best:gpt:amnesiac"], "xp:gpt:amnesiac:none")
        self.assertEqual(N["list:amnesiac:n"], "3"); self.assertEqual(N["list:amnesiac:max"], "1.00")
        self.assertEqual(N["list:amnesiac:min"], "0.25"); self.assertEqual(N["list:amnesiac:one"], "Haiku 4.5")


class MakeNumbersBaselines(unittest.TestCase):
    def test_pooled_october_amnesiac_interval_is_over_model_graph_pairs(self):
        # survivor M420 (G = distinct graph ids, shared across the three models)
        _reset()
        cells = {}
        for m, hits in ((HAIKU, 32), (SONNET, 8), (FABLE, 16)):
            rows = {}
            for g in range(16):
                for s in (0, 1):
                    i = g * 2 + s
                    rows[f"{m}-{g}-{s}"] = {"model": m, "graph_id": f"chain4-{g:02d}", "givr_hit": 1 if i < hits else 0}
            cells[("oct", m, "amnesiac", "none")] = rows
            for ctx in ("amnesiac", "ledger_readback"):
                mk.NUMS[f"oct:{mk.short_model(m)}:{ctx}:none:givr"] = "0.50"
        mk.emit_baselines(cells)
        self.assertEqual(mk.NUMS["base:oct:complete"], "yes")
        self.assertEqual(mk.NUMS["pool:amnesiac:oct:kn"], "56/96")
        self.assertEqual(mk.NUMS["pool:amnesiac:oct:gci"], mk.ci_str(*mk.wilson(56 / 96 * 48, 48)))
        self.assertNotEqual(mk.NUMS["pool:amnesiac:oct:gci"], mk.ci_str(*mk.wilson(56 / 96 * 16, 16)))


class MakeNumbersTranscriptSections(unittest.TestCase):
    def t209(self, calls, temperature=0.0, b="jul", model=HAIKU, task="t1", s=0, served=None, started="2026-10-25T10:00:00Z"):
        t = {"model": model, "task_id": task, "sample": s, "turns": _turns(*calls), "temperature": temperature,
             "started_utc": started, "model_served": served or model}
        return (b, model, "base", "none", "timeout_after_commit", task, s), t

    def test_identical_call_pairs_compare_the_idempotency_key_too(self):
        # survivor M421
        email = {"to": "a@example.com", "subject": "s"}
        k0, t0 = self.t209([("send_email", dict(email, idempotency_key="k-a"))], s=0)
        k1, t1 = self.t209([("send_email", dict(email, idempotency_key="k-b"))], s=1)
        k2, t2 = self.t209([("send_email", dict(email, idempotency_key="k-a"))], task="t2", s=0)
        k3, t3 = self.t209([("send_email", dict(email, idempotency_key="k-a"))], task="t2", s=1)
        _reset()
        mk.emit_ident_calls({k0: t0, k1: t1, k2: t2, k3: t3})
        self.assertEqual(mk.NUMS["samp:209:haiku:ident_calls"], "1/2")

    def test_temperature_is_read_per_batch(self):
        # survivor M422 (every batch folded into the July 209 run)
        _, tj0 = self.t209([("send_email", {"to": "a@example.com", "subject": "s"})], temperature=0.0, s=0)
        _, tj1 = self.t209([("send_email", {"to": "a@example.com", "subject": "s"})], temperature=0.0, s=1)
        _, to0 = self.t209([("send_email", {"to": "a@example.com", "subject": "s"})], temperature=None, b="oct", s=0)
        trans = {("jul", HAIKU, "base", "none", "toA", "t1", 0): tj0, ("jul", HAIKU, "base", "none", "toA", "t1", 1): tj1,
                 ("oct", HAIKU, "base", "none", "toA", "t1", 0): to0}
        _reset()
        with contextlib.redirect_stdout(io.StringIO()):
            mk.emit_temps(trans, {})
        self.assertEqual(mk.NUMS["samp:209:haiku:temp"], "0"); self.assertEqual(mk.NUMS["samp:209:haiku:temp_n"], "2")
        self.assertEqual(mk.NUMS["samp:oct:haiku:temp"], "default"); self.assertEqual(mk.NUMS["samp:oct:haiku:temp_n"], "1")
        self.assertEqual(mk.NUMS["haiku:temp:changed"], "yes")

    def test_batch_sampling_identical_pairs_compare_tool_calls(self):
        # survivors M427 (tool calls ignored in the signature), M434 (count inverted)
        pay = {"payee": "v", "amount": "1.00", "memo": "m"}
        same0 = self.t209([("send_payment", pay)], b="xp", model=QWEN, task="p1", s=0, served="qwen/x @A")
        same1 = self.t209([("send_payment", pay)], b="xp", model=QWEN, task="p1", s=1, served="qwen/x @A")
        diff0 = self.t209([("send_payment", pay)], b="xp", model=QWEN, task="p2", s=0, served="qwen/x @B")
        diff1 = self.t209([("send_payment", dict(pay, memo="other"))], b="xp", model=QWEN, task="p2", s=1, served="qwen/x @B")
        same2 = self.t209([("send_payment", pay)], b="xp", model=QWEN, task="p3", s=0, served="qwen/x @A")
        same3 = self.t209([("send_payment", pay)], b="xp", model=QWEN, task="p3", s=1, served="qwen/x @A")
        trans = dict([same0, same1, diff0, diff1, same2, same3])
        _reset()
        with contextlib.redirect_stdout(io.StringIO()):
            mk.emit_batch_sampling(trans, {})
        N = mk.NUMS
        self.assertEqual(N["samp:xp:qwen:ident"], "2/3")
        self.assertEqual(N["samp:xp:qwen:episodes"], "6"); self.assertEqual(N["samp:xp:qwen:served_counts"], "A 4, B 2")

    def test_totals_count_both_papers_and_truncations(self):
        # survivors M424 (core total without the plan episodes), M433 (truncations of the 209 rows only), M425
        _reset()
        c209 = {("jul", HAIKU, "base", "none", "toA", ()): {f"e{i}": {"sample": i % 2, "truncated": i == 0, "invalid_calls": 0} for i in range(64)},
                ("e16", HAIKU, "base", "none", "x", ()): {f"x{i}": {"sample": 0, "truncated": False, "invalid_calls": 1} for i in range(10)},
                ("jul", SONNET, "base", "none", "toA", ()): {f"s{i}": {"sample": 0, "truncated": False, "invalid_calls": 0} for i in range(5)}}
        c249 = {("oct", HAIKU, "amnesiac", "none"): {f"g{i}": {"sample": 0, "truncated": i < 3, "invalid_calls": 2} for i in range(32)}}
        mk.emit_totals(c209, c249)
        N = mk.NUMS
        self.assertEqual(N["total:p1"], "79"); self.assertEqual(N["total:p2"], "32"); self.assertEqual(N["total:all"], "111")
        self.assertEqual(N["total:core"], "101")
        self.assertEqual(N["models:n"], "1")                      # Sonnet's 5-episode cell does not count
        self.assertEqual(N["total:oct:trunc"], "3"); self.assertEqual(N["total:oct:invalid"], "64")
        self.assertEqual(N["total:jul:trunc"], "1"); self.assertEqual(N["total:e16:invalid"], "10")


class MakeNumbersFileSections(unittest.TestCase):
    """Sections that copy values out of released files: the macro must equal the file."""

    def test_v1_macros_copy_the_july_summary_columns(self):
        # survivor M426 (ivr and success swapped)
        p = os.path.join(mk.RES, "sweep_summary.csv")
        if not os.path.exists(p):
            self.skipTest("sweep_summary.csv missing")
        _reset(); mk.emit_v1()
        rows = list(csv.DictReader(open(p)))
        self.assertTrue(rows)
        for r in rows:
            q = ":".join(["v1", mk.short_model(r["model"]), r["prompt"], mk.MIT_SHORT[r["mitigation"]], mk.FM_SHORT[r["failure_mode"]]])
            self.assertEqual(mk.NUMS[q + ":ivr"], mk.f2(float(r["ivr"])), q)
            self.assertEqual(mk.NUMS[q + ":succ"], mk.f2(float(r["success"])), q)
        self.assertTrue(any(mk.f2(float(r["ivr"])) != mk.f2(float(r["success"])) for r in rows))

    def test_matcher_audit_macros_copy_the_summary(self):
        # survivor M428 (raw-identical and normalization-only swapped)
        p = os.path.join(mk.RES, "analysis_v2", "e0_3_summary.json")
        if not os.path.exists(p):
            self.skipTest("e0_3_summary.json missing")
        s3 = json.load(open(p))
        _reset()
        with contextlib.redirect_stdout(io.StringIO()):
            mk.emit_tier0()
        self.assertEqual(mk.NUMS["match:raw"], str(s3["raw_identical"])); self.assertEqual(mk.NUMS["match:normonly"], str(s3["normalization_only"]))
        self.assertEqual(mk.NUMS["match:pairs"], str(s3["pairs"])); self.assertNotEqual(s3["raw_identical"], s3["normalization_only"])
        pairs = os.path.join(mk.RES, "analysis_v2", "e0_3_pairs.csv")
        if os.path.exists(pairs):
            cls = [r["cls"] for r in csv.DictReader(open(pairs))]
            self.assertEqual(mk.NUMS["match:raw"], str(cls.count("raw_identical")))
            self.assertEqual(mk.NUMS["match:normonly"], str(cls.count("normalization_only")))

    def test_hand_check_false_positives_are_the_N_verdicts(self):
        # survivor M429 (equivalent on the shipped file, which has no verdicts yet)
        src = os.path.join(mk.RES, "analysis_v2")
        if not os.path.isdir(src):
            self.skipTest("analysis_v2 missing")
        tmp = tempfile.mkdtemp(); saved = mk.RES
        try:
            shutil.copytree(src, os.path.join(tmp, "analysis_v2"))
            hc = os.path.join(tmp, "analysis_v2", "e0_3_handcheck.csv")
            rows = list(csv.DictReader(open(hc))); fields = rows[0].keys()
            verdicts = ["Y", "N", "y", "n ", "Y"] + [""] * (len(rows) - 5)
            for r, v in zip(rows, verdicts):
                r["your_verdict_same_action_Y_N"] = v
            with open(hc, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(fields)); w.writeheader(); w.writerows(rows)
            mk.RES = tmp; _reset()
            with contextlib.redirect_stdout(io.StringIO()):
                mk.emit_tier0()
            self.assertEqual(mk.NUMS["match:hand:n"], "5"); self.assertEqual(mk.NUMS["match:hand:fp"], "2")
            # the clause reports the normalization-only pairs, the controls clause the rest; together 3 "same"
            m1 = re.search(r"judged (\d+) of these (\d+) pairs to be the same action", mk.NUMS["match:hand:clause"])
            m2 = re.search(r"of which it judged (\d+) the same action", mk.NUMS["match:hand:controls:clause"])
            self.assertTrue(m1 and m2, (mk.NUMS["match:hand:clause"], mk.NUMS["match:hand:controls:clause"]))
            self.assertEqual(int(m1.group(1)) + int(m2.group(1)), 3)
            self.assertEqual(m1.group(1), mk.NUMS["match:hand:norm:same"])
        finally:
            mk.RES = saved; shutil.rmtree(tmp, ignore_errors=True)

    def test_figure_macros_copy_the_transcripts(self):
        # survivors M430 (committed = k, not k + 1), M431 (duplicates hard-coded 0)
        fs = os.path.join(mk.RES, "crossprovider", "transcripts2", mk.FIGSEAM["am"])
        f1 = os.path.join(mk.RES, "sweep_transcripts", mk.FIG1["e5"])
        if not (os.path.exists(fs) and os.path.exists(f1)):
            self.skipTest("figure transcripts missing")
        _reset(); mk.emit_figseam(); mk.emit_fig1()
        t = json.load(open(fs))
        self.assertEqual(mk.NUMS["figs:am:k"], str(t["interrupt_node"])); self.assertEqual(mk.NUMS["figs:am:committed"], str(t["interrupt_node"] + 1))
        self.assertEqual(mk.NUMS["figs:am:dups"], str(t["score"]["violations"]))
        t = json.load(open(f1))
        self.assertEqual(mk.NUMS["fig1:e5:dups"], str(t["score"]["violations"])); self.assertGreater(t["score"]["violations"], 0)

    def test_finish_reasons_count_tool_calling_turns_only(self):
        # survivor M423 (equivalent on the shipped transcripts): a text-only "length" turn is not counted
        tmp = tempfile.mkdtemp(); saved = mk.RES
        try:
            d = os.path.join(tmp, "crossprovider", "sweep_transcripts"); os.makedirs(d)
            os.makedirs(os.path.join(tmp, "crossprovider", "transcripts2")); os.makedirs(os.path.join(tmp, "tier1", "e1_1", "transcripts"))
            turns = _turns(("send_payment", {"payee": "v", "amount": "1", "memo": "m"}))
            turns[0]["provider_meta"] = {"native_finish_reason": "length"}          # tool call cut at the limit: counted
            turns[-1]["provider_meta"] = {"native_finish_reason": "length"}         # text only: not counted
            turns[-1]["stop_reason"] = "error"
            json.dump({"model": QWEN, "turns": turns}, open(os.path.join(d, "a.json"), "w"))
            mk.RES = tmp; _reset(); mk.emit_finish_reasons()
            self.assertEqual((mk.NUMS["fin:xp:qwen:length"], mk.NUMS["fin:xp:qwen:error"], mk.NUMS["fin:t1:qwen:length"]), ("1", "0", "0"))
        finally:
            mk.RES = saved; shutil.rmtree(tmp, ignore_errors=True)

    def test_launch_times_are_the_first_two_starts(self):
        # survivor M432 (second launch dropped)
        tmp = tempfile.mkdtemp(); saved = mk.RES
        try:
            os.makedirs(os.path.join(tmp, "crossprovider"))
            with open(os.path.join(tmp, "crossprovider", "run.log"), "w") as f:
                f.write("=== start launch Oct 25 12:48:47 UTC 2026\nstuff\n=== start fixes Oct 25 13:59:44 UTC 2026\n"
                        "=== start more Oct 26 01:02:03 UTC 2026\n")
            mk.RES = tmp; _reset(); mk.emit_launches()
            self.assertEqual(mk.NUMS["dup:xp:launch1"], "12:48 UTC"); self.assertEqual(mk.NUMS["dup:xp:launch2"], "13:59 UTC")
            self.assertEqual(mk.NUMS["dup:xp:date"], "25 October 2026")
        finally:
            mk.RES = saved; shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# analysis/v3_load.py: strict replay, cross-provider features, refusal mapping
# ---------------------------------------------------------------------------
class V3LoadStrict(unittest.TestCase):
    def _write(self, d, model, fm, task, s, inputs, mit="none"):
        turns = _turns(*[("send_email", i) for i in inputs])
        # the failed first call: make its result an error, as the harness records it
        turns[1]["content"][0].update(content="ERROR: request timed out", is_error=True)
        t = {"model": model, "task_id": task, "sample": s, "turns": turns, "score": {"violations": 0}}
        name = f"{model}_base_{mit}_{fm}__s{s}__{task}.json"
        json.dump(t, open(os.path.join(d, "sweep_transcripts", name), "w"))
        return sweep.episode_id(model, "base", mit, fm, task, s)

    def test_strict_flag_is_the_raw_string_duplicate_after_the_replayed_failure(self):
        # survivors V401 (normalized values), V402 (failure on the second call), V407 (non-Claude
        # transcripts replayed), V408 (needs three copies)
        tmp = tempfile.mkdtemp(); saved = v3_load.RES
        try:
            os.makedirs(os.path.join(tmp, "sweep_transcripts"))
            a = {"to": "a@example.com", "subject": "Weekly digest"}
            e_norm = self._write(tmp, HAIKU, "timeout_after_commit", "email-1", 0, [a, {"to": "A@example.com", "subject": "weekly  digest"}])
            e_raw = self._write(tmp, HAIKU, "timeout_after_commit", "email-2", 0, [a, dict(a)])
            e_tob = self._write(tmp, HAIKU, "timeout_before_commit", "email-3", 0, [a, dict(a)])
            e_or = self._write(tmp, "openrouter:qwen_x", "timeout_after_commit", "email-4", 0, [a, dict(a)])
            b = {"to": "b@example.com", "subject": "Other"}
            e_abb = self._write(tmp, HAIKU, "timeout_before_commit", "email-5", 1, [a, b, dict(b)])
            v3_load.RES = tmp
            out = v3_load._strict_v2()
            self.assertEqual(out[e_norm], 0)      # normalized matcher would say duplicate; raw strings differ
            self.assertEqual(out[e_raw], 1)       # two raw-identical commits (the failed call committed)
            self.assertEqual(out[e_tob], 0)       # nothing committed before the timeout: one commit
            self.assertNotIn(e_or, out)           # Claude transcripts only (the released July cells)
            self.assertEqual(out[e_abb], 1)       # the FIRST call is the failed one: b, b commit (V402 would commit a, b)
        finally:
            v3_load.RES = saved; shutil.rmtree(tmp, ignore_errors=True)

    def test_released_cells_carry_the_strict_flag_and_cross_provider_features(self):
        # survivors V410 (strict never attached), V411 (imported features never merged), V404
        if not os.path.exists(v3_load.V2_RAW):
            self.skipTest("released rows missing")
        cells, meta = v3_load.load_all()
        v2 = [e for k, eps in cells.items() if k[0] == "v2" for e in eps]
        self.assertTrue(v2); self.assertTrue(all(e["strict"] in (0, 1) for e in v2))
        self.assertTrue(any(e["strict"] == 1 for e in v2))
        imp = {}
        for p in glob.glob(os.path.join(v3_load.T1, "e1_*", "imported.jsonl")):
            for r in v3_load.read_jsonl(p):
                imp[(r["source"], r["episode_id"])] = r
        xp = [e for k, eps in cells.items() if k[0] == "xp" for e in eps]
        self.assertTrue(xp)
        for e in xp:
            f = imp.get(("crossprovider", e["eid"]))
            if f is None:
                self.assertIsNone(e["rb"]); continue
            exp = None if f.get("failure_fired") is False else (None if f.get("readback_after_failure") is None else int(bool(f["readback_after_failure"])))
            self.assertEqual(e["rb"], exp, e["eid"]); self.assertEqual(e["retried"], f.get("retried"))
        self.assertTrue(any(e["rb"] == 1 for e in xp))
        for e in v2:
            if imp.get(("released_v2", e["eid"])) is None:
                self.assertIsNone(e["rb"])

    def test_refusal_cells_follow_the_runner_and_the_block_message(self):
        # survivors V405 (any blocked attempt), V406 (E1-1/E1-2 swapped)
        tmp = tempfile.mkdtemp(); saved = v3_load.LOGS
        try:
            e1 = f"{GEMINI}__base__idem_keys__error_500_after_commit__email-1__s0"
            e2 = f"{GEMINI}__base__none__none__email-2__s1"
            e3 = f"{GEMINI}__careful__none__none__email-3__s0"
            lines = [f"!! [1/9] {e1} failed, will retry on next run: RuntimeError: {v3_load.BLOCK}",
                     f"!! [2/9] {e2} failed, will retry on next run: RuntimeError: {v3_load.BLOCK}",
                     f"!! [3/9] {e2} failed, will retry on next run: RuntimeError: HTTP 502",
                     f"!! episode {e3} failed, will retry on next run: RuntimeError: {v3_load.BLOCK}"]
            p = os.path.join(tmp, "run.log")
            open(p, "w").write("\n".join(lines) + "\n"); open(os.path.join(tmp, "xp.log"), "w").write("")
            v3_load.LOGS = {"tier1": p, "crossprovider": os.path.join(tmp, "xp.log")}
            out, detail = v3_load.refusals({})
            self.assertEqual(dict(out), {("e1_1", GEMINI, "base", "idem_keys", "error_500_after_commit"): [("email-1", 0, 1)],
                                         ("e1_2", GEMINI, "careful", "none", "none"): [("email-3", 0, 1)]})
            d2 = [d for d in detail if d["episode_id"] == e2][0]
            self.assertEqual((d2["attempts"], d2["all_attempts_provider_block"], d2["src"]), (2, False, "e1_2"))
        finally:
            v3_load.LOGS = saved; shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# src/tier1.py runner: token caps
# ---------------------------------------------------------------------------
class Tier1Runner(unittest.TestCase):
    def test_max_tokens_matches_model_prefixes(self):
        # survivor T403 (exact match only): the OpenRouter cap applies to every openrouter: model
        cfg = {"max_tokens": {"default": 1024, "openrouter:": 4096, HAIKU: 512}}
        self.assertEqual(tier1.max_tokens_for(cfg, QWEN), 4096)
        self.assertEqual(tier1.max_tokens_for(cfg, HAIKU), 512)
        self.assertEqual(tier1.max_tokens_for(cfg, SONNET), 1024)


# ---------------------------------------------------------------------------
# src_real: key safety, Stripe transport rules, config guards
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, status, body, headers=None):
        self.status = status; self._body = json.dumps(body).encode(); self.headers = headers or {}
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _http_error(code, body, headers=None):
    return urllib.error.HTTPError("https://api.stripe.test/x", code, "err", headers or {}, io.BytesIO(json.dumps(body).encode()))


class _Script:
    """Scripted urlopen: each entry is a _Resp or an exception to raise."""
    def __init__(self, items):
        self.items = list(items); self.requests = []
    def __call__(self, req, timeout=None):
        self.requests.append(req)
        x = self.items.pop(0)
        if isinstance(x, BaseException):
            raise x
        return x


class StripeClientRules(unittest.TestCase):
    def client(self, items, retries=3):
        c = StripeClient(key=FAKE_KEY, base="https://api.stripe.test", max_retries=retries, sleep=lambda s: None, min_interval=0)
        self.script = _Script(items); self.saved = stripe_client.urllib.request.urlopen
        stripe_client.urllib.request.urlopen = self.script
        return c

    def tearDown(self):
        if hasattr(self, "saved"):
            stripe_client.urllib.request.urlopen = self.saved

    def test_key_must_start_with_the_test_prefix(self):
        # survivor R401 (prefix anywhere in the key)
        with self.assertRaises(UnsafeKeyError):
            assert_test_key("rk_live_sk_test_abcdef")
        with self.assertRaises(UnsafeKeyError):
            assert_test_key("xsk_test_abcdef")
        assert_test_key("sk_test_abcdef")

    def test_live_mode_is_detected_in_nested_objects(self):
        # survivor R403 (top level only): the list API wraps objects in "data"
        with self.assertRaises(UnsafeKeyError):
            _check_live({"object": "list", "livemode": False, "data": [{"id": "pi_1", "livemode": True}]})
        with self.assertRaises(UnsafeKeyError):
            _check_live({"a": {"b": {"livemode": True}}})
        _check_live({"object": "list", "data": [{"livemode": False}]})

    def test_401_is_an_unsafe_key(self):
        # survivor R405 (treated as an ordinary API error)
        c = self.client([_http_error(401, {"error": {"type": "invalid_request_error", "message": "bad key"}})])
        with self.assertRaises(UnsafeKeyError):
            c.request("GET", "/v1/customers")

    def test_409_is_retried_with_the_same_key_and_param_mismatch_is_not(self):
        # survivors R406 (409 not retried), R414 (idempotency_error with different parameters retried)
        ok = _Resp(200, {"id": "pi_1", "livemode": False})
        c = self.client([_http_error(409, {"error": {"type": "lock_timeout", "message": "concurrent"}}), ok])
        st, body, _ = c.request("POST", "/v1/payment_intents", {"amount": 1}, idem_key="k1")
        self.assertEqual((st, body["id"], c.stats["retries_409"]), (200, "pi_1", 1))
        self.assertEqual({r.get_header("Idempotency-key") for r in self.script.requests}, {"k1"})
        c = self.client([_http_error(409, {"error": {"type": "idempotency_error",
                                                        "message": "Keys for idempotent requests can only be used with the same parameters"}}), ok])
        with self.assertRaises(StripeAPIError) as cm:
            c.request("POST", "/v1/payment_intents", {"amount": 2}, idem_key="k1")
        self.assertEqual((cm.exception.status, c.stats["retries_409"], len(self.script.requests)), (400, 0, 1))

    def test_5xx_is_not_retried(self):
        # survivor R407: v1 saves a 500 under the key, a retry replays it
        c = self.client([_http_error(500, {"error": {"message": "boom"}}), _Resp(200, {"id": "x"})])
        with self.assertRaises(StripeInfraError):
            c.request("POST", "/v1/customers", {"name": "n"}, idem_key="k2")
        self.assertEqual(len(self.script.requests), 1)

    def test_redact_hides_the_key(self):
        # survivor R413
        c = self.client([])
        self.assertEqual(c.redact(f"Authorization: Bearer {FAKE_KEY} failed"), "Authorization: Bearer sk_test_*** failed")
        self.assertNotIn(FAKE_KEY, c.redact(RuntimeError(FAKE_KEY)))


class RunRealConfig(unittest.TestCase):
    def cfg(self, **over):
        c = {"exp": "R1", "models": [HAIKU], "cells": [["base", "none", "error_500_after_commit"]], "out_dir": "results/real_stripe"}
        c.update(over)
        d = tempfile.mkdtemp(); p = os.path.join(d, "c.json"); json.dump(c, open(p, "w"))
        return p

    def test_cells_are_restricted_to_the_registered_slice(self):
        # survivor R409 (assertion removed)
        run_real.load_config(self.cfg())
        for bad in (["careful", "none", "error_500_after_commit"], ["base", "runtime_receipts", "error_500_after_commit"],
                    ["base", "none", "not_a_mode"]):
            with self.assertRaises(AssertionError, msg=bad):
                run_real.load_config(self.cfg(cells=[bad]))
        with self.assertRaises(AssertionError):
            run_real.load_config(self.cfg(restart_cells=["plan_only"]))

    def test_out_dir_must_not_be_the_released_results(self):
        # survivor R410 (guard removed)
        with self.assertRaises(ValueError):
            run_real.load_config(self.cfg(out_dir="results"))
        with self.assertRaises(ValueError):
            run_real.load_config(self.cfg(out_dir=os.path.join(run_real.HERE, "results")))


# ---------------------------------------------------------------------------
# src/providers.py: OpenRouter transport and bookkeeping (no network)
# ---------------------------------------------------------------------------
class OpenRouterUnit(unittest.TestCase):
    def setUp(self):
        self.saved = (providers.urllib.request.urlopen, providers.time.sleep)
        self.sleeps = []
        providers.time.sleep = lambda s: self.sleeps.append(s)
        self.prov = providers.OpenRouterProvider(api_key="or-test-key", max_retries=3)

    def tearDown(self):
        providers.urllib.request.urlopen, providers.time.sleep = self.saved

    def _ok(self, **over):
        d = {"model": "qwen/qwen3.6-27b", "provider": "Alibaba",
             "choices": [{"message": {"role": "assistant", "content": "hi", "tool_calls": []}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 11, "completion_tokens": 7}}
        d.update(over)
        return _Resp(200, d)

    def test_retries_only_transient_codes_with_doubling_backoff(self):
        # survivors P403 (every 4xx retried), P404 (no backoff)
        s = _Script([_http_error(429, {"error": "slow"}), _http_error(503, {"error": "down"}), self._ok()])
        providers.urllib.request.urlopen = s
        c = self.prov.complete(QWEN, "sys", [], [], 64, 0.0)
        self.assertEqual((len(s.requests), self.sleeps), (3, [2.0, 4.0]))
        self.assertEqual(c.usage, {"input": 11, "output": 7})                       # survivor P408
        self.assertEqual(c.model_served, "qwen/qwen3.6-27b @Alibaba")               # survivor P406
        self.assertEqual(c.provider_meta["upstream_provider"], "Alibaba")
        s = _Script([_http_error(400, {"error": {"message": "bad request"}}), self._ok()])
        providers.urllib.request.urlopen = s
        with self.assertRaises(RuntimeError):
            self.prov.complete(QWEN, "sys", [], [], 64, 0.0)
        self.assertEqual(len(s.requests), 1)

    def test_unsupported_temperature_is_remembered_per_model(self):
        # survivor P405
        s = _Script([_http_error(400, {"error": {"message": "temperature is not supported for this model"}}), self._ok(), self._ok()])
        providers.urllib.request.urlopen = s
        c1 = self.prov.complete(QWEN, "sys", [], [], 64, 0.0)
        c2 = self.prov.complete(QWEN, "sys", [], [], 64, 0.0)
        self.assertEqual(len(s.requests), 3)
        bodies = [json.loads(r.data) for r in s.requests]
        self.assertIn("temperature", bodies[0]); self.assertNotIn("temperature", bodies[1]); self.assertNotIn("temperature", bodies[2])
        self.assertIsNone(c1.temperature_effective); self.assertIsNone(c2.temperature_effective)
        self.assertEqual(c2.provider_meta["temperature_sent"], None)



class UntrackedOnlyInOwnRepo(unittest.TestCase):
    def test_artifact_inside_an_unrelated_git_repo_counts_every_file(self):
        # iteration-4 robustness bug: an artifact unzipped inside another git repository made
        # untracked_results() list every results file, so the PII sweep counted 0 files
        import subprocess
        spec = importlib.util.spec_from_file_location("mk_iso", os.path.join(ROOT, "paper-merged", "make_numbers.py"))
        with tempfile.TemporaryDirectory() as outer:
            subprocess.run(["git", "init", "-q", outer], check=True)
            art = os.path.join(outer, "artifact"); os.makedirs(os.path.join(art, "results"))
            open(os.path.join(art, "results", "a.jsonl"), "w").write("{}\n")
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            m.ROOT, m.RES = art, os.path.join(art, "results")
            self.assertEqual(m.untracked_results(), set())

if __name__ == "__main__":
    unittest.main(verbosity=1)
