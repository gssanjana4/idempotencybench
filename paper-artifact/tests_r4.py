"""Tests for R4 (outcome unknown BEFORE commit): the new World failure modes
(error_500_before_commit, status_unknown_before_commit), configs/r4_precommit.json,
src/r4_estimate.py, run_r4.sh (static checks), analysis/r4_precommit.py, the
make_numbers / independent-check skip of results/r4_precommit/, and the
artifact build list. No network: the scripted fakes of tests_r2.py /
tests_tier1.py play the subject.

  python3 tests_r4.py
"""
import contextlib, csv, importlib.util, io, json, os, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("src", "src2", "analysis"):
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import tests_tier1 as tt                                   # noqa: E402
import tests_r2 as t2                                      # noqa: E402
import tier1, providers, model_agent, r4_estimate, rescore  # noqa: E402
import world                                               # noqa: E402
from world import (World, ToolTimeout, ToolServerError, FAILURE_MODES, SIGNAL_VARIANTS,  # noqa: E402
                   WORDING_VARIANTS, PRECOMMIT_VARIANTS)
from tasks import generate_tasks                           # noqa: E402

CFG = os.path.join(HERE, "configs", "r4_precommit.json")
CFG3 = os.path.join(HERE, "configs", "r3_wording.json")
QWEN = "openrouter:qwen/qwen3.6-27b-20260422"
HAIKU, SONNET = "claude-haiku-4-5-20251001", "claude-sonnet-5"
MODELS = [QWEN, HAIKU, SONNET]
TO, E500, UNK = "timeout_before_commit", "error_500_before_commit", "status_unknown_before_commit"
FMS = [TO, E500, UNK]
TEXTS = {TO: "ERROR: request timed out", E500: "ERROR: internal server error",
         UNK: "ERROR: The request may have been processed; its outcome is unknown."}
AFTER = {TO: "timeout_after_commit", E500: "error_500_after_commit", UNK: "status_unknown_after_commit"}
TASKS = generate_tasks(4)


def r4_cfg(d, **over):
    c = json.load(open(CFG))
    c.update(over)
    c["out_dir"] = os.path.join(d, "r4_precommit")
    c["tier1_root"] = c["out_dir"]
    p = os.path.join(d, "r4_precommit.json")
    json.dump(c, open(p, "w"))
    return tier1.load_config(p)


class SkipOnUnknown(tt.Policy):
    """A model that treats an outcome-unknown error as 'probably done' and
    does NOT retry (the under-execution R4 tests for); other errors are
    retried blindly."""
    def next_blocks(self, messages):
        errs = [str(b["content"]) for m in messages[1:] for b in m["content"]
                if t2._b(b, "type") == "tool_result" and b.get("is_error")]
        self.retry = not any("unknown" in e for e in errs)
        return super().next_blocks(messages)


class RecFake(t2.SafeFake):
    def __init__(self, *a, refuse_instructions=(), **kw):
        super().__init__(*a, **kw)
        self.max_tokens = []
        self.refuse = set(refuse_instructions)

    def complete(self, model, system, tools, messages, max_tokens, temperature):
        with self.lock:
            self.max_tokens.append((model, max_tokens))
        if messages[0]["content"] in self.refuse:
            raise RuntimeError('OpenRouter response without choices: {"error": {"message": '
                               '"Upstream blocked the response: OTHER", "code": 502}}')
        return super().complete(model, system, tools, messages, max_tokens, temperature)


# ---------------------------------------------------------------------------
# 1. The new failure modes: same call, same text and class, nothing committed
# ---------------------------------------------------------------------------

class TestPrecommitModes(unittest.TestCase):
    def test_existing_modes_untouched_and_new_ones_kept_apart(self):
        self.assertEqual(FAILURE_MODES, ("none", "timeout_after_commit", "timeout_before_commit",
                                         "error_500_after_commit"))
        self.assertEqual(set(SIGNAL_VARIANTS), {"error_500_body_after_commit", "error_503_retry_after_after_commit",
                                                "status_unknown_after_commit", "timeout_504_after_commit"})
        self.assertEqual(set(WORDING_VARIANTS), {"status_unknown_p1_after_commit", "status_unknown_p2_after_commit",
                                                 "status_unknown_p3_after_commit"})
        self.assertEqual(set(PRECOMMIT_VARIANTS), {E500, UNK})
        for d in (FAILURE_MODES, SIGNAL_VARIANTS, WORDING_VARIANTS):
            self.assertFalse(set(PRECOMMIT_VARIANTS) & set(d))
        # the W3 text is the same object as R2/R3's, byte for byte
        self.assertIs(PRECOMMIT_VARIANTS[UNK], SIGNAL_VARIANTS["status_unknown_after_commit"])
        self.assertEqual(PRECOMMIT_VARIANTS[UNK], ("timeout", t2.UNKNOWN_TEXTS["status_unknown_after_commit"][7:]))

    def test_nothing_commits_and_the_text_and_class_match_the_after_commit_cell(self):
        for task in TASKS:
            for k in (1, 2):
                if k > len(task.steps):
                    continue
                for fm in FMS:
                    w = World(failure_mode=fm, fail_on_call=k, seed=7)
                    wa = World(failure_mode=AFTER[fm], fail_on_call=k, seed=7)
                    for i in range(k - 1):                       # earlier calls commit normally
                        w.call(*task.steps[i], "", task.task_id); wa.call(*task.steps[i], "", task.task_id)
                    n0 = len(w.ledger.effects)
                    with self.assertRaises((ToolTimeout, ToolServerError)) as cm:
                        w.call(*task.steps[k - 1], "", task.task_id)
                    with self.assertRaises((ToolTimeout, ToolServerError)) as ca:
                        wa.call(*task.steps[k - 1], "", task.task_id)
                    self.assertIs(type(cm.exception), type(ca.exception), fm)       # same class
                    self.assertEqual(str(cm.exception), str(ca.exception), fm)      # same text
                    self.assertEqual("ERROR: " + str(cm.exception), TEXTS[fm])
                    self.assertEqual(len(w.ledger.effects), n0, fm)                 # NOTHING committed
                    self.assertEqual(len(wa.ledger.effects), n0 + 1, fm)            # the after-commit cell commits
                    self.assertFalse(w._failure_armed)                              # fired once, on call k
                    self.assertTrue(w.call(*task.steps[k - 1], "", task.task_id).startswith("ok-"))
                    self.assertEqual(len(w.ledger.effects), n0 + 1)                 # the retry commits once
                    self.assertEqual(w.ledger.violations(task.task_id), 0)
        self.assertIs(type(self._exc(E500)), ToolServerError)
        self.assertIs(type(self._exc(UNK)), ToolTimeout)
        self.assertIs(type(self._exc(TO)), ToolTimeout)

    def _exc(self, fm):
        t = TASKS[0]
        try:
            World(failure_mode=fm).call(*t.steps[0], "", t.task_id)
        except Exception as e:
            return e

    def test_invalid_call_does_not_consume_the_failure(self):
        t = TASKS[0]
        for fm in (E500, UNK):
            w = World(failure_mode=fm)
            with self.assertRaises(world.ToolCallError):
                w.call(t.steps[0][0], {}, "", t.task_id)
            self.assertTrue(w._failure_armed)
            with self.assertRaises((ToolTimeout, ToolServerError)):
                w.call(*t.steps[0], "", t.task_id)
            self.assertEqual(w.ledger.effects, [])

    def test_through_the_model_loop_omission_vs_retry(self):
        task = TASKS[0]
        for fm in FMS:
            for pol, want_om in ((tt.Policy(), False), (SkipOnUnknown(), fm == UNK)):
                w = World(failure_mode=fm, seed=3)
                tr = model_agent.run_model_episode(tt.FakeProvider(pol), w, task, model_agent.BASE_SYSTEM, model=QWEN)
                first = next(r for t in tr["turns"] if t["role"] == "tool_results" for r in t["content"])
                self.assertEqual((first["content"], first["is_error"]), (TEXTS[fm], True))
                out = w.ledger.outcome(task.task_id, task.goals, task.ordered)
                self.assertEqual(out["omission"], want_om, (fm, type(pol).__name__))
                self.assertEqual(w.ledger.violations(task.task_id), 0)

    def test_rescore_parses_an_r4_transcript_name(self):
        for fm in FMS:
            m = rescore.FNAME.match(f"{HAIKU}_base_none_{fm}__s1__{TASKS[3].task_id}.json")
            self.assertIsNotNone(m, fm)
            self.assertEqual((m["fm"], m["s"], m["task"]), (fm, "1", TASKS[3].task_id))
        m = rescore.FNAME.match(f"{HAIKU}_base_none_status_unknown_after_commit__s0__{TASKS[0].task_id}.json")
        self.assertEqual(m["fm"], "status_unknown_after_commit")


# ---------------------------------------------------------------------------
# 2. Grid, models, caps
# ---------------------------------------------------------------------------

class TestGrid(unittest.TestCase):
    def test_576_episodes_three_cells_same_32_tasks_as_r3(self):
        c = tier1.load_config(CFG)
        eps = tier1.plan(c)
        self.assertEqual(len(eps), 576); self.assertEqual(len({tier1.eid_of(e) for e in eps}), 576)
        cells = {}
        for e in eps:
            cells.setdefault(e[:4], []).append(e)
        self.assertEqual(set(cells), {(m, "base", "none", fm) for m in MODELS for fm in FMS})
        for k, v in cells.items():
            self.assertEqual(len(v), 64, k); self.assertEqual(len({e[4].task_id for e in v}), 32)
            self.assertEqual(sorted({e[5] for e in v}), [0, 1])
        c3 = tier1.load_config(CFG3)
        r3_tasks = {e[4].task_id for e in tier1.plan(c3) if not tier1.is_restart(e)}
        self.assertEqual({e[4].task_id for e in eps}, r3_tasks)
        self.assertEqual({getattr(e[4], "fail_on_call", 1) for e in eps},
                         {getattr(e[4], "fail_on_call", 1) for e in tier1.plan(c3) if not tier1.is_restart(e)})

    def test_models_settings_and_caps(self):
        c = tier1.load_config(CFG)
        c3 = json.load(open(CFG3))
        self.assertIn(QWEN, c3["groups"][0]["models"])                      # the R3 served id
        self.assertIn(f'"{QWEN}"', open(os.path.join(HERE, "run_crossprovider.sh")).read())
        self.assertNotIn(r4_estimate.GPT, c["models"]); self.assertNotIn(r4_estimate.GEMINI, c["models"])
        self.assertEqual(tier1.max_tokens_for(c, QWEN), 4096)
        self.assertEqual((tier1.max_tokens_for(c, HAIKU), tier1.max_tokens_for(c, SONNET)), (1024, 1024))
        self.assertEqual((c["exp_budget_usd"], c["exp_or_budget_usd"]), (8, 4))
        self.assertTrue(c["prompt_cache"]); self.assertTrue(c["refusal_rows"])
        self.assertEqual((c["max_turns"], c["samples"], c["workers"], c["plan_order"]), (8, 2, 4, "groups"))
        self.assertTrue(c["out_dir"].endswith(os.path.join("results", "r4_precommit")))
        self.assertEqual(c["tier1_root"], c["out_dir"])                    # spend counted in R4's own dir
        self.assertEqual(c["exp"], "R4")

    def test_r3_config_unchanged(self):
        c = json.load(open(CFG3))
        self.assertEqual((c["exp_budget_usd"], c["exp_or_budget_usd"]), (25, 15))
        self.assertEqual(len(c["groups"]), 5)


# ---------------------------------------------------------------------------
# 3. Runner: phases, full run, caps, resume, refusals, analysis on the run
# ---------------------------------------------------------------------------

class TestRunner(tt._Tier1Case):
    def setUp(self):
        super().setUp()
        self.saved_r4 = (tier1.WORKERS, tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S, providers.PROMPT_CACHE,
                         tier1.SKIP_MODELS, dict(os.environ))
        tier1.WORKERS = 0; tier1.DEADLINE_S = 0.0; tier1.DEADLINE_MARGIN_S = 0.0
        for k in ("BENCH_EXP_BUDGET_USD", "BENCH_EXP_OR_BUDGET_USD"):
            os.environ.pop(k, None)

    def tearDown(self):
        (tier1.WORKERS, tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S, providers.PROMPT_CACHE,
         tier1.SKIP_MODELS, env) = self.saved_r4
        os.environ.clear(); os.environ.update(env)
        super().tearDown()

    def run_quiet(self, cfg):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc, rows, summ = self.run_cfg(cfg)
        return rc, rows, summ, out.getvalue()

    def analyse(self, c):
        import r4_precommit
        an = os.path.join(self.d, "an")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(r4_precommit.main(["--dir", c["out_dir"], "--out", an, "--config", CFG]), 0)
        return json.load(open(os.path.join(an, "r4_precommit.json")))

    def test_full_run_retrying_model_nothing_left_undone_and_h1_supported(self):
        c = r4_cfg(self.d, workers=4, prompt_cache=False)
        self.use(lambda m: RecFake(t2.ErrPolicy(), cost=0.001 if m.startswith("openrouter:") else None))
        rc, rows, summ, log = self.run_quiet(c)
        self.assertEqual(rc, 0)
        self.assertIn("4 episodes in flight at a time; 2 phases by max_tokens [4096, 1024]", log)
        self.assertEqual(len(rows), 576)
        for m, fake in self.providers.items():
            self.assertEqual({mt for _, mt in fake.max_tokens}, {4096 if m == QWEN else 1024}, m)
        for r in rows:
            self.assertTrue(r["failure_fired"]); self.assertFalse(r["omission"]); self.assertTrue(r["achieved"])
            self.assertEqual(r["violations"], 0)
            self.assertTrue(r["retried"])                                     # the failed action re-issued
            self.assertEqual(r["readback_after_failure"], r["failure_mode"] == UNK)
        # every transcript: the failed call's effect is not in the ledger before the retry
        for fn in os.listdir(c["transcript_dir"]):
            t = json.load(open(os.path.join(c["transcript_dir"], fn)))
            sids = [(e["tool"], json.dumps(e["args"], sort_keys=True)) for e in t["ledger"]]
            self.assertEqual(len(sids), len(set(sids)))
        self.assertEqual(len(summ), 9)
        self.assertEqual({s["omission"] for s in summ}, {"0.0"})
        j = self.analyse(c)
        self.assertEqual(j["meta"]["status"], "complete")
        n = j["numbers"]
        for sm in ("qwen", "haiku", "sonnet"):
            h = n[f"r4.{sm}.H1:supported"]
            self.assertEqual(h["value"], "supported", sm); self.assertEqual(h["d"], 0.0)
            self.assertLess(h["p"], 0.001); self.assertEqual(h["margin"], 0.1)
            self.assertEqual(n[f"r4.{sm}.omission.w3B_vs_toB:p"]["holm_p"], h["p"])   # family of one
            self.assertEqual(n[f"r4.{sm}.w3B:readback_after_failure"]["value"], 1.0)
            self.assertEqual(n[f"r4.{sm}.toB:readback_after_failure"]["value"], 0.0)
            self.assertEqual(n[f"r4.{sm}.e500B:omission"]["value"], 0.0)
            self.assertEqual(n[f"r4.{sm}.e500B:omission"]["T"], 32)
            for k in ("omission.e500B_vs_toB", "omission.w3B_vs_e500B", "ivr.w3B_vs_toB", "ivr.e500B_vs_toB",
                      "readback_after_failure.w3B_vs_toB", "readback_after_failure.e500B_vs_toB"):
                self.assertIn(f"r4.{sm}.{k}:p", n)
                self.assertEqual(n[f"r4.{sm}.{k}:p"]["status"], "secondary")
            self.assertEqual(n[f"r4.{sm}.readback_after_failure.w3B_vs_toB:d"]["value"], 1.0)
        self.assertEqual(n["r4.total:episodes"]["value"], 576)

    def test_a_model_that_skips_on_unknown_fails_h1(self):
        c = r4_cfg(self.d, workers=4, prompt_cache=False)
        self.use(lambda m: RecFake(SkipOnUnknown(), cost=0.001 if m.startswith("openrouter:") else None))
        rc, rows, _, _ = self.run_quiet(c)
        self.assertEqual((rc, len(rows)), (0, 576))
        for r in rows:
            self.assertEqual(r["omission"], r["failure_mode"] == UNK, r["episode_id"])
        n = self.analyse(c)["numbers"]
        for sm in ("qwen", "haiku", "sonnet"):
            self.assertEqual(n[f"r4.{sm}.H1:supported"]["value"], "not supported")
            self.assertEqual(n[f"r4.{sm}.H1:supported"]["d"], 1.0)
            self.assertEqual(n[f"r4.{sm}.w3B:omission"]["value"], 1.0)
            self.assertEqual(n[f"r4.{sm}.e500B:omission"]["value"], 0.0)
            self.assertEqual(n[f"r4.{sm}.omission.w3B_vs_e500B:d"]["value"], 1.0)

    def test_registered_caps_stop_each_provider(self):
        # OpenRouter (Qwen first): $0.10 per turn x 3 turns = $0.30; an episode starts only while
        # spent + max(max_ep, 0.5) <= cap: cap $1.00 -> 0, 0.30 run, 0.60 + 0.5 > 1: stop = 2 episodes.
        # Anthropic (config cap $0.50): Haiku $1/$5 per MTok, (1000, 100) x 3 turns = $0.0045;
        # 0 + 0.5 <= 0.5 run, then stop: 1 episode, and Sonnet is skipped with it.
        c = r4_cfg(self.d, workers=1, prompt_cache=False, exp_or_budget_usd=1.0, exp_budget_usd=0.5)
        self.use(lambda m: RecFake(tt.Policy(), cost=0.10 if m.startswith("openrouter:") else None))
        rc, rows, _, log = self.run_quiet(c)
        self.assertIn("R4 openrouter budget $1.00 would be exceeded", log)
        self.assertIn("R4 anthropic budget $0.50 would be exceeded", log)
        self.assertEqual(sorted(r["model"] for r in rows), [HAIKU, QWEN, QWEN])
        self.assertEqual(rc, 2)
        self.assertEqual(len(os.listdir(c["transcript_dir"])), 3)
        # the real config caps are read as registered
        self.assertEqual(json.load(open(CFG))["exp_budget_usd"], 8)
        # rerun with larger caps: done episodes skipped, no duplicates
        c = r4_cfg(self.d, workers=1, prompt_cache=False, exp_or_budget_usd=1000, exp_budget_usd=1000)
        self.tier1.LIMIT = 4
        rc, rows, _, log = self.run_quiet(c)
        self.assertIn("3 done, 573 remaining", log)
        self.assertEqual(len(rows), 7); self.assertEqual(len({r["episode_id"] for r in rows}), 7)

    def test_resume_dedupe(self):
        c = r4_cfg(self.d, workers=2, prompt_cache=False)
        self.use(lambda m: RecFake(tt.Policy(), cost=0.001 if m.startswith("openrouter:") else None))
        self.tier1.LIMIT = 5
        rc, rows, _, _ = self.run_quiet(c)
        done = len(rows); self.assertIn(done, (5, 6))
        self.tier1.LIMIT = 3
        rc, rows, _, log = self.run_quiet(c)
        self.assertIn(f"{done} done, {576 - done} remaining", log)
        self.assertEqual(len({r["episode_id"] for r in rows}), len(rows))
        with open(c["raw_path"], "a") as f:
            f.write(json.dumps(rows[0]) + "\n")
        self.assertEqual(len(tier1.read_rows(c["raw_path"])), len(rows))

    def test_refusal_rows_count_toward_n_and_are_excluded(self):
        c = r4_cfg(self.d, workers=4, prompt_cache=False)
        blocked = TASKS[5]
        self.use(lambda m: RecFake(t2.ErrPolicy(), cost=0.001 if m.startswith("openrouter:") else None,
                                   refuse_instructions=[blocked.instruction] if m == QWEN else []))
        rc, rows, summ, log = self.run_quiet(c)
        self.assertEqual((rc, len(rows)), (0, 576))
        ref = [r for r in rows if r.get("refusal")]
        self.assertEqual(len(ref), 6)                                       # 3 cells x 2 samples
        self.assertNotIn("below 64 episodes", log)
        n = self.analyse(c)["numbers"]
        self.assertEqual(n["r4.total:refusals"]["value"], 6)
        self.assertEqual(n["r4.qwen.w3B:omission"]["n"], 62); self.assertEqual(n["r4.qwen.w3B:omission"]["T"], 31)
        self.assertEqual(n["r4.qwen.w3B:omission"]["refusals"], 2)
        self.assertNotIn("status", n["r4.qwen.w3B:omission"])
        self.assertEqual(n["r4.qwen.H1:supported"]["value"], "supported")
        self.assertEqual(n["r4.qwen.H1:supported"]["n"], 31)


# ---------------------------------------------------------------------------
# 4. Analysis: no data, 90% rule, the non-inferiority test
# ---------------------------------------------------------------------------

def row(model, fm, task, s, om, viol=0, rb=False):
    return {"episode_id": f"{model}__base__none__{fm}__{task.task_id}__s{s}", "model": model, "prompt": "base",
            "mitigation": "none", "failure_mode": fm, "task_id": task.task_id, "sample": s,
            "omission": bool(om), "achieved": not om, "misorder": False, "violations": viol,
            "failure_fired": True, "readback_after_failure": rb, "retried": not om, "truncated": False,
            "model_served": model, "cost_usd": 0.001}


def cell(model, fm, om_by_task, n=64):
    return [row(model, fm, t, s, om_by_task[i][s] if isinstance(om_by_task[i], (list, tuple)) else om_by_task[i])
            for i, t in enumerate(TASKS) for s in range(2)][:n]


def run_an(rows, cfg=CFG):
    import r4_precommit
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "transcripts"))
    with open(os.path.join(d, "raw.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with contextlib.redirect_stdout(io.StringIO()):
        assert r4_precommit.main(["--dir", d, "--out", os.path.join(d, "an"), "--config", cfg]) == 0
    return json.load(open(os.path.join(d, "an", "r4_precommit.json")))


def one_model_cfg(model):
    c = json.load(open(CFG)); c["groups"][0]["models"] = [model]
    p = os.path.join(tempfile.mkdtemp(), "cfg.json"); json.dump(c, open(p, "w"))
    return p


class TestAnalysis(unittest.TestCase):
    def test_no_data_writes_status_and_exits_zero(self):
        import r4_precommit
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = r4_precommit.main(["--dir", os.path.join(d, "none"), "--out", d])
            self.assertEqual(rc, 0)
            j = json.load(open(os.path.join(d, "r4_precommit.json")))
            self.assertEqual(j["meta"]["status"], "no data"); self.assertEqual(j["numbers"], {})
            self.assertEqual(j["meta"]["planned_episodes"], 576); self.assertEqual(j["meta"]["margin"], 0.1)
            self.assertEqual(j["meta"]["preregistration"], "analysis/preregistration.md (section R4)")
        self.assertEqual(r4_precommit.DIR, os.path.join(HERE, "results", "r4_precommit"))   # default output stays inside it

    def test_non_inferiority_margin(self):
        p = one_model_cfg(HAIKU)
        z = [0] * 32
        def verdict(unk):
            rows = cell(HAIKU, TO, z) + cell(HAIKU, E500, z) + cell(HAIKU, UNK, unk)
            return run_an(rows, p)["numbers"]["r4.haiku.H1:supported"]
        h = verdict(z)
        self.assertEqual((h["value"], h["d"]), ("supported", 0.0)); self.assertLess(h["p"], 0.001)
        self.assertEqual(h["ub95_boot"], 0.0)
        # 3 tasks with one of two samples omitted: d = 0.047 < 0.10 -> supported
        h = verdict([(1, 0)] * 3 + [0] * 29)
        self.assertAlmostEqual(h["d"], 3 * 0.5 / 32); self.assertEqual(h["value"], "supported")
        self.assertLess(h["ub95_boot"], 0.10)
        # 4 tasks fully omitted: d = 0.125 >= 0.10 -> not supported
        h = verdict([1] * 4 + [0] * 28)
        self.assertAlmostEqual(h["d"], 0.125); self.assertEqual(h["value"], "not supported")
        self.assertGreater(h["p"], 0.5)
        # the margin is on the DIFFERENCE: a baseline that also omits cancels out
        rows = cell(HAIKU, TO, [1] * 8 + [0] * 24) + cell(HAIKU, E500, z) + cell(HAIKU, UNK, [1] * 8 + [0] * 24)
        h = run_an(rows, p)["numbers"]["r4.haiku.H1:supported"]
        self.assertEqual((h["value"], h["d"]), ("supported", 0.0))

    def test_incomplete_cell_is_not_evaluable(self):
        p = one_model_cfg(SONNET)
        z = [0] * 32
        rows = cell(SONNET, TO, z, n=57) + cell(SONNET, E500, z) + cell(SONNET, UNK, z)   # 57 < 0.9 x 64
        n = run_an(rows, p)["numbers"]
        self.assertEqual(n["r4.sonnet.H1:supported"]["value"], "not evaluable")
        self.assertIn("incomplete", n["r4.sonnet.toB:omission"]["status"])
        self.assertNotIn("r4.sonnet.omission.w3B_vs_toB:p", n)
        self.assertNotIn("r4.sonnet.omission.e500B_vs_toB:p", n)
        self.assertIn("r4.sonnet.omission.w3B_vs_e500B:p", n)              # secondary on complete cells
        rows = cell(SONNET, TO, z, n=58) + cell(SONNET, E500, z) + cell(SONNET, UNK, z)   # 58 >= 57.6
        self.assertEqual(run_an(rows, p)["numbers"]["r4.sonnet.H1:supported"]["value"], "supported")
        n = run_an(cell(SONNET, TO, z), p)["numbers"]                                     # UNK never ran
        self.assertEqual(n["r4.sonnet.H1:supported"]["value"], "not evaluable")
        self.assertEqual(n["r4.sonnet.w3B:omission"]["value"], None)

    def test_omission_is_the_scorer_field(self):
        import r4_precommit
        src = open(os.path.join(HERE, "analysis", "r4_precommit.py")).read()
        self.assertIn('"om": int(bool(r.get("omission")))', src)
        self.assertEqual(r4_precommit.DELTA, 0.10)
        t = next(t for t in TASKS if t.family == "multi_step")
        w = World(failure_mode=UNK)
        with self.assertRaises(ToolTimeout):
            w.call(*t.steps[0], "", t.task_id)
        for st in t.steps[1:]:
            w.call(*st, "", t.task_id)
        out = w.ledger.outcome(t.task_id, t.goals, t.ordered)
        self.assertTrue(out["omission"]); self.assertFalse(out["success"])


# ---------------------------------------------------------------------------
# 5. Estimate, launcher, make_numbers / independent skip, artifact list, registration
# ---------------------------------------------------------------------------

class TestEstimate(unittest.TestCase):
    def test_optional_rule(self):
        cfg = json.load(open(CFG))
        def fake(tot, orc):
            return lambda c, ms: {"cost": tot, "openrouter": {"cost": orc}}
        self.assertTrue(r4_estimate.optional_rule(cfg, fake(7.9, 3.9))["add_optional"])
        self.assertFalse(r4_estimate.optional_rule(cfg, fake(8.1, 3.0))["add_optional"])   # total > $8
        self.assertFalse(r4_estimate.optional_rule(cfg, fake(7.8, 5.2))["add_optional"])   # OpenRouter > $4 cap

    def test_real_estimate_fits_the_caps_and_leaves_gpt_gemini_out(self):
        if not os.path.isdir(os.path.join(HERE, "results", "crossprovider", "sweep_transcripts")) or \
                not os.path.isdir(os.path.join(HERE, "results", "r2_wording", "transcripts")):
            self.skipTest("October / R2 transcripts not present")
        cfg = json.load(open(CFG))
        est = r4_estimate.estimate_r4(cfg)
        t = est["totals"]
        self.assertEqual((t["anthropic"]["episodes"], t["openrouter"]["episodes"]), (384, 192))
        self.assertEqual(t["anthropic"]["unpriced"] + t["openrouter"]["unpriced"], 0)
        self.assertLessEqual(t["anthropic"]["worst"], 8.0); self.assertLessEqual(t["openrouter"]["worst"], 4.0)
        opt = r4_estimate.optional_rule(cfg)
        self.assertFalse(opt["add_optional"]); self.assertGreater(opt["or_with"], 4.0)
        self.assertGreater(t["minutes"], 0)


class TestLauncherAndSkips(unittest.TestCase):
    def test_run_r4_sh_static(self):
        p = os.path.join(HERE, "run_r4.sh")
        s = open(p).read()
        self.assertEqual(subprocess.run(["bash", "-n", p]).returncode, 0)
        self.assertIn("source .env", s)
        for k in ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"):
            self.assertNotIn(f"echo ${k}", s); self.assertNotIn(f"echo ${{{k}", s); self.assertNotIn(f"${k}\"", s)
            self.assertIn(f"{k} missing", s)
        for flag in ("--yes", "--estimate", "--status", "--or-only", "--workers", "--help"):
            self.assertIn(flag, s)
        self.assertIn("--check-keys", s); self.assertIn("BENCH_CHECK_SKIP_ANTHROPIC=1", s)
        self.assertIn('BENCH_TIER1_SKIP_PROVIDERS="anthropic"', s)
        self.assertIn("unset BENCH_EXP_BUDGET_USD BENCH_EXP_OR_BUDGET_USD", s)
        self.assertIn("src/r4_estimate.py --optional", s); self.assertIn("analysis/r4_precommit.py", s)
        self.assertIn('[ "$MODE" = estimate ] && exit 0', s)
        self.assertLess(s.index('[ "$MODE" = estimate ] && exit 0'), s.index("--check-keys"))   # --estimate: no call
        for bad in ("results/r3_wording", "results/crossprovider", "results/tier1", "paper-merged"):
            self.assertNotIn(bad, s)
        r = subprocess.run(["bash", p, "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0); self.assertIn("run_r4.sh --yes", r.stdout)

    def test_make_numbers_skips_r4_precommit(self):
        spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(HERE, "paper-merged", "make_numbers.py"))
        mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
        with tempfile.TemporaryDirectory() as d:
            res = os.path.join(d, "results")
            for sub in ("r4_precommit", "some_batch"):
                os.makedirs(os.path.join(res, sub))
                for name in ("sweep_raw.jsonl", "raw.jsonl"):
                    with open(os.path.join(res, sub, name), "w") as f:
                        f.write(json.dumps({"episode_id": "x", "model": "m", "failure_mode": "none"}) + "\n")
            open(os.path.join(res, "sweep_raw_v2.jsonl"), "w").write("")
            saved = mk.RES; mk.RES = res
            try:
                got = [os.path.relpath(p, res) for p in mk.raw_files("sweep_raw_v2.jsonl", "sweep_raw*.jsonl")]
            finally:
                mk.RES = saved
            self.assertEqual(got, ["sweep_raw_v2.jsonl", os.path.join("some_batch", "sweep_raw.jsonl")])
        # R4 is wired into the paper: make_numbers.py merges the R4
        # analysis like R3's and sweeps results/r4_precommit/ for PII like R3;
        # its raw file stays out of the main-grid scan above
        src = open(os.path.join(HERE, "paper-merged", "make_numbers.py")).read()
        self.assertIn('os.path.join(RES, "r4_precommit", "analysis", "r4_precommit.json")', src)
        self.assertNotIn('x == "r4_precommit")]', src)

    def test_artifact_build_lists_r4_like_r3(self):
        p = os.path.join(HERE, "make_anon_release_merged.sh")
        if not os.path.exists(p):
            self.skipTest("build script not shipped in the artifact")
        s = open(p).read()
        rs = next(l for l in s.splitlines() if "tests_r3.py" in l and "LICENSE" in l)
        self.assertIn("tests_r4.py", rs); self.assertIn("run_r4.sh", rs)
        suites = next(l for l in s.splitlines() if l.startswith("SUITES="))
        self.assertIn("tests_r4.py", suites)

    def test_registration_section_is_last_and_heading_exact(self):
        s = open(os.path.join(HERE, "analysis", "preregistration.md")).read()
        h = "## R4. Outcome unknown before commit (registered 2026-11-01, before any R4 data)"
        self.assertEqual(s.count(h), 1)
        self.assertGreater(s.index(h), s.index("\n16. E1-6 S (stated commit belief)"))
        tail = s[s.index(h):]
        # later registrations (R5, 2026-11-02) are appended after R4; check only the R4 section
        tail = tail.split("\n## R5. ", 1)[0]
        import re
        # the registered body (up to the first dated deviation heading appended
        # after the data, e.g. deviation 17) has no numbered line, so it adds no deviation
        body = tail.split("\n### 20", 1)[0]
        self.assertEqual(re.findall(r"^\d+\.\s", body, re.M), [])
        self.assertNotIn("\n## ", tail[len(h):])                            # nothing after it
        for k in ("0.10", "timeout_before_commit", "error_500_before_commit", "status_unknown_before_commit",
                  QWEN, HAIKU, SONNET, "The request may have been processed; its outcome is unknown."):
            self.assertIn(k, tail)


if __name__ == "__main__":
    unittest.main(verbosity=1)
