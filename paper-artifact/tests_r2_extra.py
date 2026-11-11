"""Extra tests for analysis/r2_wording.py and the R2 runner, written by the
R2 code verifier to kill mutants that tests_r2.py lets through (each test
names the mutant it kills). Synthetic raw.jsonl files play the data; no
network.  Style as tests_r2.py.

  python3 tests_r2_extra.py
"""
import contextlib, io, json, os, random, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("src", "src2", "analysis"):
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import tier1, sweep2                                       # noqa: E402
import r2_wording                                          # noqa: E402
from v3_common import wilson, holm, PERM_N, PERM_SEED      # noqa: E402

CFG = os.path.join(HERE, "configs", "r2_wording.json")
HAIKU, SONNET, FABLE = "claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-fable-5"
GRAPHS = [(g.graph_id, k) for g, k in sweep2.benchmark_graphs()]          # 16 graphs
TASKS = [f"task-{i:02d}" for i in range(32)]
E500, TOA, W3 = "error_500_after_commit", "timeout_after_commit", "status_unknown_after_commit"
P1, P2, P3 = ("status_unknown_p1_after_commit", "status_unknown_p2_after_commit",
              "status_unknown_p3_after_commit")


def restart_row(model, ctx, gid, k, s, hit):
    return {"kind": "restart", "exp": "R2", "model": model, "resume_context": ctx, "mitigation": "none",
            "graph_id": gid, "topology": gid.split("-")[0], "interrupt_node": k, "sample": s,
            "violations": int(hit), "givr_hit": int(hit), "completion": 1, "order_violation": 0,
            "truncated": False, "session2_lookup": not hit, "refusal": False, "cost_usd": 0.01,
            "episode_id": sweep2.episode_id(model, ctx, "none", gid, k, s)}


def retry_row(model, fm, task, s, hit):
    return {"exp": "R2", "model": model, "prompt": "base", "mitigation": "none", "failure_mode": fm,
            "task_id": task, "sample": s, "violations": int(hit), "truncated": False, "failure_fired": True,
            "readback_after_failure": not hit, "retried": bool(hit), "cost_usd": 0.001,
            "episode_id": f"{model}__base__none__{fm}__{task}__s{s}"}


def part_a(model, ctx, hits, n=32):
    """32 rows (16 graphs x 2) unless n < 32; hits = per-graph gIVR (list of 16)."""
    rows = []
    for gi, (gid, k) in enumerate(GRAPHS):
        for s in range(2):
            rows.append(restart_row(model, ctx, gid, k, s, hits[gi]))
    return rows[:n]


def part_b(model, fm, hits, n=64):
    rows = []
    for ti, t in enumerate(TASKS):
        for s in range(2):
            rows.append(retry_row(model, fm, t, s, hits[ti]))
    return rows[:n]


def run_analysis(rows, transcripts=()):
    """Write rows (and optional transcripts) into a temp dir and run the
    analysis on them; returns the numbers dict."""
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "transcripts"))
    with open(os.path.join(d, "raw.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    for i, t in enumerate(transcripts):
        json.dump(t, open(os.path.join(d, "transcripts", f"t{i}.json"), "w"))
    out = os.path.join(d, "an")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = r2_wording.main(["--dir", d, "--out", out, "--config", CFG])
    assert rc == 0
    return json.load(open(os.path.join(out, "r2_wording.json")))["numbers"]


def haiku_a(p1=0, p2=0, p3=0, n1=32):
    """Haiku Part A: amnesiac all 1, bare_hint all 0, paraphrases as given (per-graph constant)."""
    rows = part_a(HAIKU, "amnesiac", [1] * 16) + part_a(HAIKU, "bare_hint", [0] * 16)
    rows += part_a(HAIKU, "hint_p1", [p1] * 16, n1) + part_a(HAIKU, "hint_p2", [p2] * 16)
    rows += part_a(HAIKU, "hint_p3", [p3] * 16)
    return rows


# ---------------------------------------------------------------------------
# 1. The permutation test and Holm (M02, M03, M10, M11)
# ---------------------------------------------------------------------------

class TestPermutationScheme(unittest.TestCase):
    def test_p_is_hits_plus_one_over_n_plus_one_and_one_sided(self):
        # 16 graphs, every per-graph difference -1: no permutation other than the
        # identity reaches the observed sum, so the one-sided p is exactly 1/(N+1)
        # (hit/N would print 0, M02; two-sided counts the mirror image too, M10).
        x = {f"g{i}": 1.0 for i in range(16)}
        y = {f"g{i}": 0.0 for i in range(16)}
        r = r2_wording.perm_test_one_sided(x, y)
        self.assertEqual(r["p"], 1 / (PERM_N + 1))
        self.assertEqual((r["d"], r["plus"], r["minus"], r["T"]), (-1.0, 0, 16, 16))
        # smallest reachable p is 1/(N+1): never 0
        self.assertGreaterEqual(r2_wording.perm_test_one_sided(x, y)["p"], 1 / (PERM_N + 1))

    def test_one_sided_p_follows_the_registered_seed_and_flip_order(self):
        # A mixed case (the Haiku unknown_p2 pattern: 5 up, 23 down, 4 ties)
        # reproduced with an independent sign-flip loop on the registered seed;
        # any other seed or flip order changes the count (M11).
        x = {f"t{i:02d}": 1.0 for i in range(32)}
        y = {f"t{i:02d}": (0.0 if i < 23 else 1.0) for i in range(32)}
        for i in range(23, 28):
            x[f"t{i:02d}"], y[f"t{i:02d}"] = 0.0, 1.0        # 5 tasks go the other way
        units = sorted(x)
        diffs = [y[u] - x[u] for u in units]
        obs = sum(diffs)
        rng = random.Random(PERM_SEED)
        hit = 0
        for _ in range(PERM_N):
            s = sum(dv if rng.random() < 0.5 else -dv for dv in diffs)
            if s <= obs + 1e-12:
                hit += 1
        expect = (hit + 1) / (PERM_N + 1)
        self.assertLess(0.0, expect); self.assertLess(expect, 0.05)           # a sensitive case
        self.assertEqual(r2_wording.perm_test_one_sided(x, y)["p"], expect)
        self.assertEqual(r2_wording.PERM_SEED, 20261025)
        other = random.Random(PERM_SEED + 1); hit2 = 0
        for _ in range(PERM_N):
            s = sum(dv if other.random() < 0.5 else -dv for dv in diffs)
            if s <= obs + 1e-12:
                hit2 += 1
        self.assertNotEqual(hit, hit2)                       # the seed matters for this case

    def test_holm_is_step_down_and_applied_to_the_primary_tests(self):
        self.assertEqual(holm([0.01, 0.02, 0.03]), [0.03, 0.04, 0.04])
        self.assertEqual(holm([0.04, 0.001, 0.02]), [0.04, 0.003, 0.04])
        self.assertEqual(holm([0.5, 0.5, 0.5]), [1.0, 1.0, 1.0])
        n = run_analysis(haiku_a())
        for p in ("hint_p1", "hint_p2", "hint_p3"):
            e = n[f"r2.A.haiku.{p}_vs_amnesiac:p"]
            self.assertAlmostEqual(e["value"], 1 / (PERM_N + 1), places=9)      # stored to 6 significant digits
            self.assertAlmostEqual(e["holm_p"], 3 / (PERM_N + 1), places=9)     # 3 equal p's: x3 (M03)
            self.assertIn("one-sided", e["stat"])


# ---------------------------------------------------------------------------
# 2. Verdict rule, 90% rule, incomplete cells (M04, M05, M14, M15)
# ---------------------------------------------------------------------------

class TestVerdictAndCompleteness(unittest.TestCase):
    def test_supported_needs_at_least_two_of_three_paraphrases(self):
        n = run_analysis(haiku_a(p1=0, p2=1, p3=1))          # only hint_p1 lowers gIVR
        self.assertEqual(n["r2.A.haiku.H1:supported"]["value"], False)
        self.assertEqual(n["r2.A.haiku.H1:supported"]["n_significant"], 1)
        n = run_analysis(haiku_a(p1=0, p2=0, p3=1))
        self.assertEqual(n["r2.A.haiku.H1:supported"]["value"], True)
        self.assertEqual(n["r2.A.haiku.H1:supported"]["n_significant"], 2)

    def test_cell_below_90_percent_gets_no_test_and_no_verdict(self):
        # hint_p1 at 23/32 (72%, the real Fable 5 cell), hint_p2 / hint_p3 missing
        rows = part_a(FABLE, "amnesiac", [1] * 16) + part_a(FABLE, "bare_hint", [0] * 16)
        rows += part_a(FABLE, "hint_p1", [0] * 16, n=23)
        n = run_analysis(rows)
        c = n["r2.A.fable.hint_p1:givr"]
        self.assertEqual((c["n"], c["planned"]), (23, 32))
        self.assertTrue(c["status"].startswith("incomplete"))
        for k in n:                                           # no contrast touches the incomplete cell (M05, M15)
            self.assertNotIn("hint_p1_vs", k, k)
        self.assertEqual(n["r2.A.fable.H1:supported"]["value"], "not evaluable")   # not False (M14)
        self.assertEqual(n["r2.A.fable.H1:supported"]["n"], 0)
        self.assertNotIn("r2.A.fable.hint_p2:givr_ci_ep", n)
        # one complete paraphrase cell (hint_p1 32/32) and two missing: still no verdict (M14)
        rows = part_a(FABLE, "amnesiac", [1] * 16) + part_a(FABLE, "hint_p1", [0] * 16)
        n = run_analysis(rows)
        self.assertNotIn("r2.A.fable.hint_p1_vs_amnesiac:p", n)       # no Holm-adjusted primary p is emitted
        self.assertEqual(n["r2.A.fable.H1:supported"]["value"], "not evaluable")
        self.assertEqual(n["r2.A.fable.H1:supported"]["n"], 1)

    def test_90_percent_boundary_counts_refusals_toward_planned(self):
        # 29/32 = 90.6% complete; 28/32 = 87.5% incomplete
        n = run_analysis(haiku_a(n1=29))
        self.assertNotIn("status", n["r2.A.haiku.hint_p1:givr"])
        self.assertIn("r2.A.haiku.hint_p1_vs_amnesiac:p", n)
        n = run_analysis(haiku_a(n1=28))
        self.assertTrue(n["r2.A.haiku.hint_p1:givr"]["status"].startswith("incomplete: 28/32"))
        self.assertNotIn("r2.A.haiku.hint_p1_vs_amnesiac:p", n)
        # 56 scored + 8 refusals = 64 planned: complete (refusals count toward n, M17);
        # the refusals are excluded from the rate (M08)
        rows = part_b(HAIKU, E500, [1] * 32) + part_b(HAIKU, P1, [0] * 32)
        rows += part_b(HAIKU, P2, [0] * 32) + part_b(HAIKU, P3, [0] * 32)
        refused = [r["episode_id"] for r in rows if r["failure_mode"] == P1][:8]
        for r in rows:
            if r["episode_id"] in refused:
                r["violations"] = 1                         # would count as a hit if not excluded
        tr = [{"episode_id": e, "turns": [{"role": "assistant", "stop_reason": "refusal", "content": []}]}
              for e in refused]
        n = run_analysis(rows, tr)
        c = n["r2.B.haiku.unknown_p1:ivr"]
        self.assertEqual((c["n"], c["refusals"], c["k"], c["value"]), (56, 8, 0, 0.0))
        self.assertNotIn("status", c)
        self.assertIn("r2.B.haiku.unknown_p1_vs_e500:p", n)
        self.assertEqual(n["r2.total:refusals"]["value"], 8)
        # a restart row flagged refusal is excluded the same way
        rows = haiku_a()
        for r in rows[-4:]:
            r["refusal"] = True; r["givr_hit"] = 1
        n = run_analysis(rows)
        c = n["r2.A.haiku.hint_p3:givr"]
        self.assertEqual((c["n"], c["refusals"], c["k"]), (28, 4, 0))


# ---------------------------------------------------------------------------
# 3. Intervals (M06, M09)
# ---------------------------------------------------------------------------

class TestIntervals(unittest.TestCase):
    def test_primary_interval_is_unit_level_wilson(self):
        hits = [1] * 10 + [0] * 22                           # 20/64 episodes over 32 tasks
        rows = part_b(HAIKU, E500, hits) + part_b(HAIKU, TOA, hits)
        n = run_analysis(rows)
        c = n["r2.B.haiku.e500:ivr"]
        self.assertEqual((c["k"], c["n"], c["T"]), (20, 64, 32))
        lo, hi = wilson(20 / 64 * 32, 32)
        self.assertAlmostEqual(c["ci"][0], lo, places=5); self.assertAlmostEqual(c["ci"][1], hi, places=5)
        elo, ehi = wilson(20, 64)
        e = n["r2.B.haiku.e500:ivr_ci_ep"]
        self.assertAlmostEqual(e["ci"][0], elo, places=5); self.assertAlmostEqual(e["ci"][1], ehi, places=5)
        self.assertLess(c["ci"][0], e["ci"][0]); self.assertGreater(c["ci"][1], e["ci"][1])   # unit CI is wider
        self.assertEqual(c["ci_method"], "wilson95_task(p*T,T)")

    def test_spread_interval_is_the_bootstrap_percentile(self):
        # W3 hits on 7 of 32 tasks (both samples), the three paraphrases never:
        # spread = 14/64 = 0.21875; under a task bootstrap the spread is 2j/64 with
        # j ~ Binomial(32, 7/32), whose 2.5th and 97.5th percentiles are 3 and 12
        # (P(j<=2) = 0.018, P(j<=3) = 0.059; P(j<=11) = 0.968, P(j<=12) = 0.987):
        # bounds 0.09375 and 0.375. The minimum over 10,000 draws is 0 or 1 (M09).
        hits = [1] * 7 + [0] * 25
        rows = part_b(SONNET, E500, [1] * 32) + part_b(SONNET, TOA, [0] * 32) + part_b(SONNET, W3, hits)
        rows += part_b(SONNET, P1, [0] * 32) + part_b(SONNET, P2, [0] * 32) + part_b(SONNET, P3, [0] * 32)
        n = run_analysis(rows)
        s = n["r2.B.sonnet.spread:maxmin"]
        self.assertEqual(s["value"], 0.21875)
        self.assertEqual(s["ci"][0], 0.09375)
        self.assertEqual(s["ci"][1], 0.375)
        self.assertEqual((s["max_wording"], s["min_wording"]), ("w3", "unknown_p1"))


# ---------------------------------------------------------------------------
# 4. Runner: the restart refusal flag (M28)
# ---------------------------------------------------------------------------

class TestRestartRow(unittest.TestCase):
    def test_refusal_flag_from_a_refusal_stop_in_either_session(self):
        cfg = {"exp": "R2"}
        row = {"graph_id": "chain4-x", "topology": "chain4", "violations": 0, "givr_hit": 0,
               "completion": 1, "order_violation": 0, "episode_id": "e", "model": HAIKU, "sample": 0}
        def tr(stop2):
            return {"model": HAIKU, "provider": "anthropic", "model_served": HAIKU, "temperature": None,
                    "sessions": [{"n": 1, "interrupted": True, "turns": [
                                      {"role": "assistant", "content": [], "stop_reason": "tool_use"},
                                      {"role": "interrupt"}],
                                  "meta": {"usage": {"input": 10, "output": 1}, "truncated": False}},
                                 {"n": 2, "turns": [{"role": "assistant", "content": [], "stop_reason": stop2}],
                                  "meta": {"usage": {"input": 10, "output": 1}, "truncated": False}}]}
        self.assertFalse(tier1.restart_row(cfg, row, tr("end_turn"))["refusal"])
        self.assertTrue(tier1.restart_row(cfg, row, tr("refusal"))["refusal"])



class RegisteredReportingRules(unittest.TestCase):
    """Two registration rules the first R2 analysis missed (found by the R2 code verifier)."""
    def setUp(self):
        import json
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "analysis_v3", "r2_wording.json")
        if not os.path.exists(p):
            self.skipTest("no R2 analysis output here")
        d = json.load(open(p)); self.n = d.get("numbers", d)

    def test_secondary_tests_carry_holm_within_their_block(self):
        sec = [k for k, v in self.n.items() if k.endswith(":p") and isinstance(v, dict)
               and str(v.get("status", "")).startswith("secondary")]
        self.assertTrue(sec)
        self.assertEqual([k for k in sec if self.n[k].get("holm_p") is None], [])

    def test_no_spread_over_an_incomplete_wording_cell(self):
        for k, v in self.n.items():
            if k.endswith("spread:maxmin") and isinstance(v.get("value"), float):
                part, model = k.split(".")[1:3]
                inc = [c for c, w in self.n.items() if c.startswith(f"r2.{part}.{model}.") and isinstance(w, dict)
                       and str(w.get("status") or "").startswith("incomplete")]
                self.assertEqual(inc, [], k)

if __name__ == "__main__":
    unittest.main(verbosity=1)
