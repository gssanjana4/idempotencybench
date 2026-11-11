"""Regression tests for the analysis layer. Run from repo root: python3 tests_analysis.py"""
import json, os, re, sys, tempfile, unittest
from collections import defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.path.insert(0, os.path.join(ROOT, "src"))
import v3_common, v3_sens_oct

def _row(eid, model, prompt, mit, fm, task, s, v):
    return {"episode_id": eid, "model": model, "prompt": prompt, "mitigation": mit, "failure_mode": fm,
            "task_id": task, "sample": s, "violations": v, "achieved": True, "omission": False, "misorder": False}

class CellSelection(unittest.TestCase):
    def test_sens_oct_cells_from_never_pools_prompts(self):
        rows = [_row(f"m__base__none__e500__t{i}__s0", "m", "base", "none", "error_500_after_commit", f"t{i}", 0, 1) for i in range(4)]
        rows += [_row(f"m__careful__none__e500__t{i}__s0", "m", "careful", "none", "error_500_after_commit", f"t{i}", 0, 0) for i in range(4)]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            for r in rows: fh.write(json.dumps(r) + "\n")
        eps = v3_sens_oct.cells_from(fh.name, "v2")[("m", "none", "error_500_after_commit")]
        self.assertEqual(len(eps), 4); self.assertTrue(all(e["prompt"] == "base" for e in eps))
        self.assertEqual(sum(e["v"] for e in eps), 4)  # 4/4, not 4/8

    def test_dedupe_last_row_wins_and_first_is_sensitivity(self):
        a = _row("e1", "m", "base", "none", "none", "t", 0, 0); b = dict(a, violations=2)
        self.assertEqual(v3_common.dedupe([a, b])["e1"]["violations"], 2)
        self.assertEqual(v3_common.dedupe([a, b], "first")["e1"]["violations"], 0)

    def test_every_printed_cell_matches_raw_rows_filtered_on_all_factors(self):
        RES = os.path.join(ROOT, "results")
        FM = {"none": "none", "timeout_after_commit": "toA", "timeout_before_commit": "toB", "error_500_after_commit": "e500",
              "error_500_body_after_commit": "e500body", "status_unknown_after_commit": "unknown"}
        FMJ = dict(FM, error_500_body_after_commit="w1", status_unknown_after_commit="w3")
        MIT = {"none": "none", "idem_keys": "keys", "runtime_receipts": "rcpt"}
        srcs = [("jul", "v2", os.path.join(RES, "sweep_raw_v2.jsonl")), ("xp", "xp", os.path.join(RES, "crossprovider", "sweep_raw.jsonl"))]
        tex_b = {"e1_0": "e10", "e1_1": "e13", "e1_2": "e12", "e1_3": "e13", "e1_4": "e14", "e1_5": "e15", "e1_6": "e16"}
        for x in sorted(tex_b):
            p = os.path.join(RES, "tier1", x, "raw.jsonl")
            if os.path.exists(p): srcs.append((tex_b[x], x, p))
        # check what the paper prints: the C1 re-scored rows (raw_v2.jsonl) where they exist
        srcs = [(tb, nb, v3_common.prefer_v2(p)) for tb, nb, p in srcs]
        kn_tex, kn_nj = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for tb, nb, p in srcs:
            for r in v3_common.dedupe(v3_common.read_jsonl(p)).values():
                m = v3_common.short_model(r["model"]); tag = ":at_least_once" if r.get("retry_mode") == "at_least_once" else ""
                k1 = f"{tb}:{m}:{r['prompt']}:{MIT[r['mitigation']]}:{FM[r['failure_mode']]}{tag}"
                k2 = f"{nb}.{m}.{r['prompt']}.{MIT[r['mitigation']]}.{FMJ[r['failure_mode']]}"
                for d, k in ((kn_tex, k1), (kn_nj, k2)):
                    d[k][1] += 1; d[k][0] += int(r["violations"] > 0)
        tex = dict(re.findall(r"\\defnum\{([^}]*)\}\{([^}]*)\}", open(os.path.join(ROOT, "paper-merged", "numbers.tex")).read()))
        checked = 0
        for key, val in tex.items():
            if re.match(r"^(jul|xp|e1[0-6]):([a-z]+):(base|careful):(none|keys|rcpt):([a-z0-9]+)(:at_least_once)?:kn$", key) and not val.startswith("\\PENDING"):
                self.assertEqual(val, "%d/%d" % tuple(kn_tex[key[:-3]]), key); checked += 1
        nj = json.load(open(os.path.join(RES, "analysis_v3", "numbers.json")))["numbers"]
        for key, e in nj.items():
            if re.match(r"^(v2|xp|e1_[0-6])\.([a-z]+)\.(base|careful)\.(none|keys|rcpt)\.([a-z0-9]+):ivr$", key) and "k" in e:
                self.assertEqual((e["k"], e["n"]), tuple(kn_nj[key[:-4]]), key); checked += 1
        self.assertGreater(checked, 80)

class Tests(unittest.TestCase):
    def test_perm_test_pairs_on_shared_tasks_two_sided_and_symmetric(self):
        x = {f"t{i}": 0.0 for i in range(20)}; y = {f"t{i}": 1.0 for i in range(20)}; y["extra"] = 1.0
        r = v3_common.perm_test(x, y)
        self.assertEqual(r["T"], 20); self.assertEqual(r["d"], 1.0); self.assertLess(r["p"], 0.001)
        r2 = v3_common.perm_test(y, x); self.assertEqual(r2["d"], -1.0); self.assertEqual(r2["p"], r["p"])
        same = v3_common.perm_test(x, dict(x)); self.assertEqual(same["p"], 1.0); self.assertEqual(same["d"], 0.0)
        r5 = v3_common.perm_test({f"t{i}": 0.0 for i in range(5)}, {f"t{i}": 1.0 for i in range(5)})
        self.assertAlmostEqual(r5["p"], 2 / 32, delta=0.01)  # exact two-sided sign-flip

    def test_holm_and_task_level_wilson(self):
        self.assertEqual(v3_common.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
        st = v3_common.rate_stats([{"task": f"t{i // 2}", "v": i % 2} for i in range(64)])
        self.assertEqual((st["k"], st["n"], st["T"]), (32, 64, 32))
        self.assertEqual(tuple(st["ci_task"]), v3_common.wilson(16, 32))

if __name__ == "__main__":
    unittest.main(verbosity=2)
