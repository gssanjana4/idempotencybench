"""Independent recomputation check of every number the paper prints.

Runs independent/check.py (a second, stdlib-only implementation that
recomputes the printed macros from the raw files in results/) and fails if
any recomputed macro disagrees with paper-merged/numbers.tex.

    python3 tests_independent.py           # from the repo root

Skips with a message when results/ is incomplete (e.g. a partial checkout).
"""
import json, os, subprocess, sys, unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
CHECK = os.path.join(ROOT, "independent", "check.py")
REPORT = os.path.join(ROOT, "independent", "report.json")


class IndependentRecomputation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.proc = subprocess.run([sys.executable, "-I", CHECK, "--root", ROOT, "--report", REPORT],
                                  capture_output=True, text=True, timeout=600)
        if cls.proc.returncode == 2:
            raise unittest.SkipTest("independent check skipped: " + cls.proc.stdout.strip())
        cls.report = json.load(open(REPORT)) if os.path.exists(REPORT) else None

    def test_frozen_printed_keys_match_main_tex(self):
        # the artifact (no main.tex) checks the keys in independent/printed_keys.json
        tex = os.path.join(ROOT, "paper-merged", "main.tex")
        if not os.path.exists(tex):
            self.skipTest("no main.tex here (anonymized artifact)")
        sys.path.insert(0, os.path.dirname(CHECK))
        import check
        self.assertEqual(sorted(check.used_keys(tex)), json.load(open(check.FROZEN_KEYS)),
                         "run: python3 independent/check.py --freeze-keys")

    def test_check_ran(self):
        self.assertIsNotNone(self.report, "check.py wrote no report:\n" + self.proc.stdout + self.proc.stderr)
        self.assertIn(self.proc.returncode, (0, 1), self.proc.stderr)

    def test_no_mismatch(self):
        bad = [m for m in self.report["mismatches"] if not m.get("explained")]
        msg = "\n".join("%s: printed %r, recomputed %r [%s]" % (m["key"], m["printed"], m["mine"], m["family"]) for m in bad)
        self.assertEqual(bad, [], "independently recomputed macros disagree with numbers.tex:\n" + msg)

    def test_known_differences_still_reproduce(self):
        # a fixed pipeline retires its entry in check.KNOWN_DIFFERENCES
        self.assertEqual(self.report["known_differences_no_longer_reproducing"], [])

    def test_numeric_coverage(self):
        cov = self.report["numeric"]
        self.assertGreaterEqual(cov["coverage"], 0.97, "independent coverage of numeric macros fell below 97%%: %s" % cov)

    def test_every_printed_macro_defined(self):
        self.assertEqual(self.report["not_covered"].get("undefined in numbers.tex", []), [])

    def test_hygiene(self):
        h = self.report["hygiene"]
        gt = h.get("stripe_ground_truth_recomputed_vs_recorded", {})
        self.assertEqual(gt.get("mismatch", 0), 0, gt)
        led = h.get("ledger_vs_row_violations_209", {})
        # the only cell whose stored (pre-fix) ledger may disagree with its re-scored row
        self.assertLessEqual(set(led.get("cells_with_stale_ledger", {})), {"jul:haiku:base:keys:toA"}, led)
        r2 = h.get("r2_transcripts_vs_rows", {})
        # R2 rows must agree with their transcripts (ledger duplicates, read-back, session-2 lookups)
        for k in ("violation_mismatch", "readback_mismatch", "session2_lookup_mismatch"):
            self.assertEqual(r2.get(k, 0), 0, r2)
        r3 = h.get("r3_transcripts_vs_rows", {})
        # R3 likewise; and the refusal rows are exactly the run.log "REFUSED by the provider" lines
        for k in ("violation_mismatch", "readback_mismatch", "session2_lookup_mismatch", "refusals_rows_vs_runlog_differ"):
            self.assertEqual(r3.get(k, 0), 0, r3)
        r4 = h.get("r4_transcripts_vs_rows", {})
        # R4: omission, misorder, success and duplication re-derived from each transcript's ledger
        # and goals, and read-back from its turns, must equal the row; every row has its transcript
        for k in ("omission_mismatch", "misorder_mismatch", "success_mismatch", "violation_mismatch",
                  "readback_mismatch", "rows_without_transcript", "refusals_rows_vs_runlog_differ"):
            self.assertEqual(r4.get(k, 0), 0, r4)
        self.assertEqual(r4.get("transcripts"), r4.get("rows"), r4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
