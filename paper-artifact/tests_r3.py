"""Tests for R3 (cross-provider wording): configs/r3_wording.json, the R3
additions to src/tier1.py (group-major plan order, max_tokens phases,
refusal rows, exp-cap override), src/r3_estimate.py (estimate and the
registered cut order), run_r3.sh (static checks), analysis/r3_wording.py,
and the make_numbers raw-file glob. No network: the scripted fakes of
tests_r2.py / tests_tier1.py play the subject.

  python3 tests_r3.py
"""
import contextlib, copy, csv, importlib.util, io, json, os, re, subprocess, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("src", "src2", "analysis"):
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import tests_tier1 as tt                                   # noqa: E402
import tests_r2 as t2                                      # noqa: E402
import tier1, sweep2, providers, model_agent, r3_estimate  # noqa: E402
from world import World, ToolTimeout, WORDING_VARIANTS     # noqa: E402
from tasks import generate_tasks                           # noqa: E402

CFG = os.path.join(HERE, "configs", "r3_wording.json")
CFG2 = os.path.join(HERE, "configs", "r2_wording.json")
GPT = "openrouter:openai/gpt-5.5-20260423"
GEMINI = "openrouter:google/gemini-4-argon-20261013"
QWEN = "openrouter:qwen/qwen3.6-27b-20260422"
FABLE = "claude-fable-5"
OR_MODELS = [GPT, GEMINI, QWEN]
B_FMS = ["timeout_after_commit", "error_500_after_commit", "status_unknown_after_commit",
         "status_unknown_p1_after_commit", "status_unknown_p3_after_commit"]
A_CTX = ["amnesiac", "bare_hint", "hint_p1", "hint_p3"]
BLOCK = ('OpenRouter response without choices: {"id": "gen-1", "error": {"message": '
         '"Gemini blocked the response: OTHER", "code": 502, "metadata": {"error_type": "provider_unavailable"}}}')


class RecFake(t2.SafeFake):
    """tests_r2.SafeFake that also records (model, max_tokens) per call and
    can refuse (raise the October Gemini block) on given task instructions."""
    def __init__(self, *a, refuse_instructions=(), **kw):
        super().__init__(*a, **kw)
        self.max_tokens = []
        self.refuse = set(refuse_instructions)

    def complete(self, model, system, tools, messages, max_tokens, temperature):
        with self.lock:
            self.max_tokens.append((model, max_tokens))
        if messages[0]["content"] in self.refuse:
            raise RuntimeError(BLOCK)
        return super().complete(model, system, tools, messages, max_tokens, temperature)


def r3_cfg(d, **over):
    c = json.load(open(CFG))
    c.update(over)
    c["out_dir"] = os.path.join(d, "r3_wording")
    c["tier1_root"] = c["out_dir"]
    p = os.path.join(d, "r3_wording.json")
    json.dump(c, open(p, "w"))
    return tier1.load_config(p)


def split(cfg):
    out = {"A": [], "B": []}
    for e in tier1.plan(cfg):
        out["A" if tier1.is_restart(e) else "B"].append(e)
    return out


# ---------------------------------------------------------------------------
# 1. Grid, models, settings, strings
# ---------------------------------------------------------------------------

class TestGrid(unittest.TestCase):
    def test_counts_1280_plus_384_and_cells(self):
        c = tier1.load_config(CFG)
        parts = split(c)
        self.assertEqual((len(parts["B"]), len(parts["A"])), (1280, 384))
        self.assertEqual(len(tier1.plan(c)), 1664)
        self.assertEqual(len({tier1.eid_of(e) for e in tier1.plan(c)}), 1664)
        cells = {}
        for e in parts["B"]:
            cells.setdefault((e[0], e[1], e[2], e[3]), []).append(e)
        self.assertEqual(set(cells), {(m, "base", "none", fm) for m in OR_MODELS + [FABLE] for fm in B_FMS})
        for k, eps in cells.items():
            self.assertEqual(len(eps), 64, k)
            self.assertEqual(len({e[4].task_id for e in eps}), 32, k)
        acells = {}
        for e in parts["A"]:
            acells.setdefault((e[0], e[1], e[2]), []).append(e)
        self.assertEqual(set(acells), {(m, ctx, "none") for m in OR_MODELS for ctx in A_CTX})
        for k, eps in acells.items():
            self.assertEqual(len(eps), 32, k)
            self.assertEqual(len({e[3].graph_id for e in eps}), 16, k)
            self.assertEqual({e[3].topology for e in eps}, {"chain4", "fanout"})
        # the R3-A graphs and interrupt points are the October ones
        self.assertEqual({(e[3].graph_id, e[4]) for e in parts["A"]},
                         {(g.graph_id, k) for g, k in sweep2.benchmark_graphs()})

    def test_october_model_ids_and_provider_settings(self):
        c = tier1.load_config(CFG)
        xp = open(os.path.join(HERE, "run_crossprovider.sh")).read()
        for m in OR_MODELS:
            self.assertIn(f'"{m}"', xp)                                  # exact October ids
            self.assertEqual(tier1.max_tokens_for(c, m), 4096)           # BENCH_MAX_TOKENS=4096 in October
            self.assertEqual(tier1.provider_of(m), "openrouter")
        self.assertEqual(tier1.max_tokens_for(c, FABLE), 1024)
        self.assertEqual(model_agent.TEMPERATURE, 0.0)
        self.assertEqual((c["max_turns"], c["samples"], c["restart_samples"]), (8, 2, 2))
        self.assertTrue(c["prompt_cache"]); self.assertTrue(c["refusal_rows"])
        self.assertEqual(c["plan_order"], "groups")
        self.assertEqual((c["exp_budget_usd"], c["exp_or_budget_usd"]), (25, 15))
        self.assertEqual(c["workers"], 4)
        self.assertTrue(c["out_dir"].endswith(os.path.join("results", "r3_wording")))
        self.assertEqual(c["exp"], "R3")
        # Gemini is in R3-A because its October amnesiac gIVR was >= 0.5 (numbers.tex)
        tex = os.path.join(HERE, "paper-merged", "numbers.tex")
        if os.path.exists(tex):
            s = open(tex).read()
            for sm in ("gpt", "qwen", "gemini"):
                v = float(re.search(r"\\defnum\{xp:%s:amnesiac:none:givr\}\{([0-9.]+)\}" % sm, s).group(1))
                self.assertGreaterEqual(v, 0.5, sm)

    def test_plan_order_is_the_cut_order(self):
        c = tier1.load_config(CFG)
        eps = tier1.plan(c)
        labels = [g["label"] for g in c["groups"]]
        self.assertEqual(labels, ["R3-B core", "R3-A core", "R3-A Gemini (cut 3)", "R3-A hint_p3 (cut 2)",
                                  "R3-B unknown_p3 (cut 1)"])
        def grp(e):
            if tier1.is_restart(e):
                if e[1] == "hint_p3":
                    return 3
                return 2 if e[0] == GEMINI else 1
            return 4 if e[3] == "status_unknown_p3_after_commit" else 0
        seq = [grp(e) for e in eps]
        self.assertEqual(seq, sorted(seq))                   # group-major: later groups are cut first
        self.assertEqual(seq[0], 0); self.assertEqual(seq[-1], 4)
        # within the first group the OpenRouter models come first, Fable last
        first = [e[0] for e in eps[:1024]]
        self.assertEqual(first[-1], FABLE); self.assertEqual(first[0], GPT)
        # the default order (kind-major) is unchanged for R2
        c2 = tier1.load_config(CFG2)
        self.assertEqual(c2["plan_order"], "kind")
        e2 = tier1.plan(c2)
        self.assertFalse(tier1.is_restart(e2[767])); self.assertTrue(tier1.is_restart(e2[768]))

    def test_strings_delivered_verbatim_to_an_openrouter_model(self):
        task = generate_tasks(4)[0]
        for fm in B_FMS:
            w = World(failure_mode=fm, seed=1)
            tr = model_agent.run_model_episode(tt.FakeProvider(tt.Policy()), w, task,
                                               model_agent.BASE_SYSTEM, model=GPT)
            first = next(r for t in tr["turns"] if t["role"] == "tool_results" for r in t["content"])
            want = {"timeout_after_commit": "ERROR: request timed out",
                    "error_500_after_commit": "ERROR: internal server error"}.get(fm) or t2.UNKNOWN_TEXTS[fm]
            self.assertEqual(first["content"], want)
            self.assertIs(first["is_error"], True)
            self.assertEqual(len(w.ledger.for_task(task.task_id)) >= 1, True)   # committed before the error
        self.assertEqual(WORDING_VARIANTS["status_unknown_p3_after_commit"],
                         ("timeout", "Request status unknown; it might have succeeded."))
        for g, k in sweep2.benchmark_graphs():
            for ctx in A_CTX[1:]:
                self.assertEqual(sweep2.resume_user_msg(g, ctx, {0}), f"{g.instruction}\n\n{t2.HINT_TEXTS[ctx]}")
            self.assertEqual(sweep2.resume_user_msg(g, "amnesiac", {0}), g.instruction)
        # the same OpenAI-format conversion the October runs used carries the error text as a tool message
        msgs = providers.messages_to_openai("sys", [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "c1", "name": "send_email", "input": {}}]},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "is_error": True,
                                          "content": t2.UNKNOWN_TEXTS["status_unknown_p3_after_commit"]}]}])
        self.assertEqual(msgs[-1], {"role": "tool", "tool_call_id": "c1",
                                    "content": "ERROR: Request status unknown; it might have succeeded."})

    def test_r2_config_unchanged(self):
        c = json.load(open(CFG2))
        self.assertEqual(c["exp_budget_usd"], 45); self.assertEqual(len(c["groups"]), 3)
        self.assertNotIn("plan_order", c); self.assertNotIn("refusal_rows", c)


# ---------------------------------------------------------------------------
# 2. Runner: phases, parallel full run, budget, resume, refusals
# ---------------------------------------------------------------------------

class TestRunner(tt._Tier1Case):
    def setUp(self):
        super().setUp()
        self.saved_r3 = (tier1.WORKERS, tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S, providers.PROMPT_CACHE,
                         sweep2.MAX_TOKENS, dict(os.environ))
        tier1.WORKERS = 0; tier1.DEADLINE_S = 0.0; tier1.DEADLINE_MARGIN_S = 0.0
        for k in ("BENCH_EXP_BUDGET_USD", "BENCH_EXP_OR_BUDGET_USD"):
            os.environ.pop(k, None)

    def tearDown(self):
        (tier1.WORKERS, tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S, providers.PROMPT_CACHE,
         sweep2.MAX_TOKENS, env) = self.saved_r3
        os.environ.clear(); os.environ.update(env)
        super().tearDown()

    def run_quiet(self, cfg):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc, rows, summ = self.run_cfg(cfg)
        return rc, rows, summ, out.getvalue()

    def test_full_parallel_run_phases_rows_and_analysis(self):
        c = r3_cfg(self.d, workers=4, prompt_cache=False)
        self.use(lambda m: RecFake(t2.ErrPolicy(), cost=0.001 if m.startswith("openrouter:") else None))
        rc, rows, summ, log = self.run_quiet(c)
        self.assertEqual(rc, 0)
        self.assertIn("4 episodes in flight at a time; 4 phases by max_tokens [4096, 1024, 4096, 1024]", log)
        self.assertEqual(len(rows), 1664)
        self.assertEqual({tier1.eid_of(e) for e in tier1.plan(c)}, {r["episode_id"] for r in rows})
        restart = [r for r in rows if r.get("kind") == "restart"]
        retry = [r for r in rows if r.get("kind") != "restart"]
        self.assertEqual((len(restart), len(retry)), (384, 1280))
        # max_tokens: 4096 for every OpenRouter call (retry AND restart sessions), 1024 for Fable
        for m, fake in self.providers.items():
            want = 4096 if m.startswith("openrouter:") else 1024
            self.assertTrue(fake.max_tokens, m)
            self.assertEqual({mt for _, mt in fake.max_tokens}, {want}, m)
        self.assertEqual({r["max_tokens"] for r in restart}, {4096})
        mts = {}
        for fn in os.listdir(c["transcript_dir"]):        # retry transcripts record max_tokens per episode
            t = json.load(open(os.path.join(c["transcript_dir"], fn)))
            if "turns" in t:
                mts.setdefault(t["model"], set()).add(t["max_tokens"])
        self.assertEqual(mts, {m: {4096} for m in OR_MODELS} | {FABLE: {1024}})
        for r in restart:
            self.assertEqual(r["givr_hit"], 1 if r["resume_context"] == "amnesiac" else 0)
            self.assertEqual(r["session2_lookup"], r["resume_context"] != "amnesiac")
            self.assertIn(r["model"], OR_MODELS)
            self.assertEqual(r["model_served"], r["model"] + "-served")
        for r in retry:
            self.assertTrue(r["failure_fired"])
            unknown = r["failure_mode"].startswith("status_unknown")
            self.assertEqual(r["violations"], 0 if unknown else 1, r["failure_mode"])
            self.assertEqual(r["readback_after_failure"], unknown)
            self.assertNotIn("refusal", r)
        # OpenRouter spend = logged usage.cost; Fable = list price x tokens
        orc = sum(r["cost_usd"] for r in rows if r["model"] != FABLE)
        self.assertGreater(orc, 0); self.assertNotIn("refusals", summ[0])
        self.assertEqual(len(summ), 20)                                   # 4 models x 5 conditions
        rs = list(csv.DictReader(open(os.path.join(c["out_dir"], "summary_restart.csv"))))
        self.assertEqual(len(rs), 12)                                     # 3 models x 4 contexts
        # analysis
        import r3_wording
        with contextlib.redirect_stdout(io.StringIO()):
            r3_wording.main(["--dir", c["out_dir"], "--out", os.path.join(self.d, "an"), "--config", CFG])
        nums = json.load(open(os.path.join(self.d, "an", "r3_wording.json")))
        self.assertEqual(nums["meta"]["status"], "complete")
        n = nums["numbers"]
        for sm in ("gpt", "gemini", "qwen", "fable"):
            self.assertTrue(n[f"r3.B.{sm}.H2:supported"]["value"], sm)
            self.assertEqual(n[f"r3.B.{sm}.H2:supported"]["wordings"], ["w3", "unknown_p1", "unknown_p3"])
            self.assertEqual(n[f"r3.B.{sm}.e500:ivr"]["value"], 1.0)
            self.assertEqual(n[f"r3.B.{sm}.e500:ivr"]["T"], 32)
            self.assertEqual(n[f"r3.B.{sm}.e500:ivr"]["refusals"], 0)
            for w in ("w3", "unknown_p1", "unknown_p3"):
                self.assertLess(n[f"r3.B.{sm}.{w}_vs_e500:p"]["holm_p"], 0.05)
                self.assertEqual(n[f"r3.B.{sm}.{w}_vs_e500:d"]["value"], -1.0)
                self.assertIn(f"r3.B.{sm}.{w}_vs_toA:p", n)                 # secondary
                self.assertEqual(n[f"r3.B.{sm}.{w}:readback_after_failure"]["value"], 1.0)
            self.assertIn(f"r3.B.{sm}.e500_vs_toA:p", n)                     # 500 vs timeout, same batch
            self.assertEqual(n[f"r3.B.{sm}.toA:readback_after_failure"]["value"], 0.0)
        for sm in ("gpt", "gemini", "qwen"):
            self.assertTrue(n[f"r3.A.{sm}.H1:supported"]["value"], sm)
            self.assertEqual(n[f"r3.A.{sm}.H1:supported"]["wordings"], ["bare_hint", "hint_p1", "hint_p3"])
            self.assertEqual(n[f"r3.A.{sm}.amnesiac:givr"]["value"], 1.0)
            self.assertEqual(n[f"r3.A.{sm}.amnesiac:givr"]["T"], 16)
            for w in ("bare_hint", "hint_p1", "hint_p3"):
                self.assertEqual(n[f"r3.A.{sm}.{w}_vs_amnesiac:d"]["value"], -1.0)
            self.assertEqual(n[f"r3.A.{sm}.bare_hint:session2_lookup"]["value"], 1.0)
            self.assertIn(f"r3.A.{sm}.hint_p3_vs_bare_hint:p", n)
        self.assertNotIn("r3.A.fable.H1:supported", n)
        self.assertEqual(n["r3.total:episodes"]["value"], 1664)
        self.assertEqual(n["r3.total:refusals"]["value"], 0)
        self.assertEqual(set(n["r3.total:model_served"]["value"]), {m + "-served" for m in OR_MODELS + [FABLE]})
        self.assertEqual(n["r3.B.gpt.e500:ivr"]["model_served"], [GPT + "-served"])

    def test_openrouter_cap_stops_openrouter_only_and_env_overrides_the_anthropic_cap(self):
        # OpenRouter: $0.10 per turn, 3 turns per retry episode = $0.30; an episode starts only while
        # spent + max(max_ep, 0.5) <= cap: cap $1.00 -> 0 (ok), 0.30 (ok), 0.60 (1.10 > 1: stop) = 2 episodes.
        # Fable (Anthropic) continues under its own cap, overridden to $0.50 by the env: at $10/$50 per
        # MTok, (1000 in, 100 out) x 3 turns = $0.045 per episode; 0 (ok), 0.045 (0.545 > 0.5: stop) = 1.
        c = r3_cfg(self.d, workers=1, prompt_cache=False, exp_or_budget_usd=1.0)
        os.environ["BENCH_EXP_BUDGET_USD"] = "0.5"
        self.use(lambda m: RecFake(tt.Policy(), cost=0.10 if m.startswith("openrouter:") else None))
        rc, rows, _, log = self.run_quiet(c)
        self.assertIn("openrouter budget $1.00 would be exceeded", log)
        self.assertIn("R3 anthropic budget $0.50 would be exceeded", log)
        orr = [r for r in rows if r["model"] != FABLE]
        self.assertEqual(len(orr), 2)
        self.assertAlmostEqual(sum(r["cost_usd"] for r in orr), 0.60, places=6)
        fab = [r for r in rows if r["model"] == FABLE]
        self.assertEqual(len(fab), 1)
        self.assertIn(rc, (2, 3))                                   # both providers stopped by their caps
        self.assertIn("'openrouter': 'exp_budget'", log); self.assertIn("'anthropic': 'exp_budget'", log)
        self.assertEqual(len(os.listdir(c["transcript_dir"])), 3)   # nothing half-written
        # rerun with the caps lifted: the 3 done episodes are skipped, no duplicates
        os.environ["BENCH_EXP_BUDGET_USD"] = "1000"
        c = r3_cfg(self.d, workers=1, prompt_cache=False, exp_or_budget_usd=1000)
        self.tier1.LIMIT = 6
        rc, rows, _, log = self.run_quiet(c)
        self.assertIn("3 done, 1661 remaining", log)
        self.assertEqual(len(rows), 9); self.assertEqual(len({r["episode_id"] for r in rows}), 9)

    def test_resume_dedupe_across_phases(self):
        c = r3_cfg(self.d, workers=2, prompt_cache=False)
        self.use(lambda m: RecFake(tt.Policy(), cost=0.001 if m.startswith("openrouter:") else None))
        self.tier1.LIMIT = 5
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual(rc, 0); self.assertIn(len(rows), (5, 6))
        done = len(rows)
        self.tier1.LIMIT = 3
        rc, rows, _, log = self.run_quiet(c)
        self.assertIn(f"{done} done, {1664 - done} remaining", log)
        self.assertEqual(len({r["episode_id"] for r in rows}), len(rows))
        self.assertGreaterEqual(len(rows), done + 3)
        # a duplicated row (overlapping run) counts once
        with open(c["raw_path"], "a") as f:
            f.write(json.dumps(rows[0]) + "\n")
        self.assertEqual(len(tier1.read_rows(c["raw_path"])), len(rows))

    def test_provider_block_becomes_a_refusal_row_and_the_run_completes(self):
        c = r3_cfg(self.d, workers=4, prompt_cache=False)
        blocked = generate_tasks(4)[5]                                   # one task refused by "Gemini"
        self.use(lambda m: RecFake(t2.ErrPolicy(), cost=0.001 if m.startswith("openrouter:") else None,
                                   refuse_instructions=[blocked.instruction] if m == GEMINI else []))
        rc, rows, summ, log = self.run_quiet(c)
        self.assertEqual(rc, 0)                                          # not a failure: the run completes
        self.assertEqual(len(rows), 1664)
        ref = [r for r in rows if r.get("refusal")]
        self.assertEqual(len(ref), 10)                                   # 5 conditions x 2 samples
        self.assertTrue(all(r["model"] == GEMINI and r["task_id"] == blocked.task_id for r in ref))
        self.assertTrue(all("blocked the response" in r["refusal_error"] for r in ref))
        self.assertTrue(all(r["cost_usd"] == 0 and r["violations"] == 0 for r in ref))
        self.assertIn("REFUSED by the provider", log); self.assertIn("10 provider refusals", log)
        self.assertNotIn("failed, will retry", log)
        self.assertEqual(len(os.listdir(c["transcript_dir"])), 1654)      # no transcript for a refusal
        # summary: refusals are their own column and excluded from the cell's n and rate
        g = [s for s in summ if s["model"] == GEMINI]
        self.assertEqual({s["refusals"] for s in g}, {"10" if False else "2"})
        self.assertEqual({s["n_episodes"] for s in g}, {"62"})
        self.assertEqual({s["refusals"] for s in summ if s["model"] != GEMINI}, {"0"})
        self.assertNotIn("below 64 episodes", log)                       # refusals count toward the planned n
        # a rerun does not retry the refused episodes
        rc, rows2, _, log = self.run_quiet(c)
        self.assertIn("1664 done, 0 remaining", log); self.assertEqual(len(rows2), 1664)
        # analysis: excluded from every estimate and test, counted per cell; 90% rule counts them as planned
        import r3_wording
        with contextlib.redirect_stdout(io.StringIO()):
            r3_wording.main(["--dir", c["out_dir"], "--out", os.path.join(self.d, "an"), "--config", CFG])
        n = json.load(open(os.path.join(self.d, "an", "r3_wording.json")))["numbers"]
        self.assertEqual(n["r3.total:refusals"]["value"], 10)
        self.assertEqual(n["r3.total:refusals"]["by_cell"]["gemini.e500"], 2)
        self.assertEqual(n["r3.B.gemini.e500:ivr"]["n"], 62); self.assertEqual(n["r3.B.gemini.e500:ivr"]["T"], 31)
        self.assertEqual(n["r3.B.gemini.e500:ivr"]["refusals"], 2)
        self.assertNotIn("status", n["r3.B.gemini.e500:ivr"])
        self.assertTrue(n["r3.B.gemini.H2:supported"]["value"])
        self.assertEqual(n["r3.B.gemini.w3_vs_e500:d"]["n"], 31)        # paired over the 31 scored tasks
        self.assertTrue(os.path.exists(os.path.join(self.d, "an", "r3_wording_refusals.csv")))

    def test_provider_block_in_a_restart_session_is_a_refusal_row_too(self):
        cfg = json.load(open(CFG))
        cfg["groups"] = [g for g in cfg["groups"] if g["label"] == "R3-A Gemini (cut 3)"]
        c = r3_cfg(self.d, workers=2, prompt_cache=False, groups=cfg["groups"])
        g0, _ = sweep2.benchmark_graphs()[0]
        self.use(lambda m: RecFake(t2.ErrPolicy(), cost=0.001, refuse_instructions=[g0.instruction]))
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual((rc, len(rows)), (0, 96))
        ref = [r for r in rows if r.get("refusal")]
        # session 1 of every context starts with the bare instruction: 3 contexts in this group x 2 samples
        self.assertEqual(len(ref), 6)
        self.assertTrue(all(r["kind"] == "restart" and r["graph_id"] == g0.graph_id and r["givr_hit"] == 0 for r in ref))
        rs = list(csv.DictReader(open(os.path.join(c["out_dir"], "summary_restart.csv"))))
        self.assertEqual({(r["resume_context"], r["refusals"], r["n_episodes"], r["n_graphs"]) for r in rs},
                         {(ctx, "2", "30", "15") for ctx in A_CTX[:3]})
        import r3_wording
        with contextlib.redirect_stdout(io.StringIO()):
            r3_wording.main(["--dir", c["out_dir"], "--out", os.path.join(self.d, "an"), "--config", CFG])
        n = json.load(open(os.path.join(self.d, "an", "r3_wording.json")))["numbers"]
        self.assertEqual(n["r3.A.gemini.amnesiac:givr"]["refusals"], 2)
        self.assertEqual(n["r3.A.gemini.amnesiac:givr"]["T"], 15)
        self.assertEqual(n["r3.A.gemini.H1:supported"]["value"], "not evaluable")   # hint_p3 planned, not run
        self.assertNotIn("r3.A.gemini.bare_hint_vs_amnesiac:d", n)        # no primary test without a verdict
        self.assertEqual(n["r3.A.gemini.hint_p1_vs_bare_hint:d"]["n"], 15)   # secondary: 15 graphs in both arms

    def test_refusal_rows_off_by_default_keeps_the_old_failure_path(self):
        c = r3_cfg(self.d, workers=1, prompt_cache=False, refusal_rows=False)
        blocked = generate_tasks(4)[0]
        self.use(lambda m: RecFake(tt.Policy(), cost=0.001, refuse_instructions=[blocked.instruction]))
        self.tier1.LIMIT = 2
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual(rc, 1); self.assertIn("failed, will retry on next run", log)
        self.assertFalse(any(r.get("refusal") for r in rows))


# ---------------------------------------------------------------------------
# 3. Estimate and the registered cut order
# ---------------------------------------------------------------------------

class TestEstimateAndCuts(unittest.TestCase):
    def test_cut_order_until_the_estimate_fits(self):
        cfg = json.load(open(CFG))
        def est(c):      # $1 per planned OpenRouter episode cell unit: 1344 at full design
            return sum(n for _, _, m, _, n in r3_estimate._cells(c) if m.startswith("openrouter:")) / 100.0
        self.assertAlmostEqual(est(cfg), 13.44)
        c0, cuts, e = r3_estimate.apply_cuts(cfg, 15.0, est)
        self.assertEqual(cuts, []); self.assertEqual(c0, cfg)
        c1, cuts, e = r3_estimate.apply_cuts(cfg, 12.0, est)
        self.assertEqual(cuts, ["drop status_unknown_p3 from R3-B"])
        self.assertAlmostEqual(e, 13.44 - 1.92); self.assertEqual(len(c1["groups"]), 4)
        self.assertFalse(any(x[2] == "status_unknown_p3_after_commit" for g in c1["groups"] for x in g.get("cells", [])))
        c2, cuts, e = r3_estimate.apply_cuts(cfg, 11.0, est)
        self.assertEqual(cuts, ["drop status_unknown_p3 from R3-B", "drop hint_p3 from R3-A"])
        self.assertAlmostEqual(e, 13.44 - 1.92 - 0.96)
        self.assertFalse(any(x[0] == "hint_p3" for g in c2["groups"] for x in g.get("restart_cells", [])))
        c3, cuts, e = r3_estimate.apply_cuts(cfg, 10.0, est)
        self.assertEqual(cuts, ["drop status_unknown_p3 from R3-B", "drop hint_p3 from R3-A", "drop Gemini from R3-A"])
        self.assertAlmostEqual(e, 13.44 - 1.92 - 0.96 - 0.96)
        self.assertFalse(any(GEMINI in g["models"] for g in c3["groups"] if g.get("restart_cells")))
        self.assertTrue(any(GEMINI in g["models"] for g in c3["groups"] if g.get("cells")))    # Gemini stays in R3-B
        self.assertEqual(json.load(open(CFG)), cfg)                   # the file is never modified
        # the cut config still loads and plans
        p = os.path.join(tempfile.mkdtemp(), "cut.json"); json.dump(c3, open(p, "w"))
        self.assertEqual(len(tier1.plan(tier1.load_config(p))), 1664 - 256 - 96 - 96)

    def test_real_estimate_fits_the_caps_without_cuts(self):
        if not os.path.isdir(os.path.join(HERE, "results", "crossprovider", "sweep_transcripts")):
            self.skipTest("October transcripts not present")
        cfg = json.load(open(CFG))
        est = r3_estimate.estimate_r3(cfg)
        self.assertEqual(set(est["prices"]), set(OR_MODELS))
        t = est["totals"]
        self.assertEqual(t["openrouter"]["unpriced"], 0); self.assertEqual(t["anthropic"]["unpriced"], 0)
        self.assertEqual((t["openrouter"]["episodes"], t["anthropic"]["episodes"]), (1344, 320))
        self.assertLessEqual(t["openrouter"]["cost"], 15.0)
        _, cuts, _ = r3_estimate.apply_cuts(cfg, 15.0)
        self.assertEqual(cuts, [])
        r2c = r3_estimate.estimate_r2_completion()
        self.assertEqual(r2c["planned"], 1136)
        self.assertLessEqual(r2c["cost"] + t["anthropic"]["cost"], 25.0)
        gpt = est["prices"][GPT]
        self.assertAlmostEqual(gpt[0], 5.0, places=1); self.assertAlmostEqual(gpt[1], 30.0, places=1)


# ---------------------------------------------------------------------------
# 4. Analysis without data; launcher; make_numbers glob
# ---------------------------------------------------------------------------

class TestAnalysisNoData(unittest.TestCase):
    def test_no_data_writes_status_and_exits_zero(self):
        import r3_wording
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = r3_wording.main(["--dir", os.path.join(d, "none"), "--out", d])
            self.assertEqual(rc, 0)
            j = json.load(open(os.path.join(d, "r3_wording.json")))
            self.assertEqual(j["meta"]["status"], "no data"); self.assertEqual(j["numbers"], {})
            self.assertEqual(j["meta"]["planned_episodes"], 1664)
            self.assertEqual(j["meta"]["preregistration"], "analysis/preregistration.md (sections R2 / R3)")

    def test_incomplete_cell_gets_no_test_and_dropped_p3_means_both(self):
        """A model whose p3 cell was cut (2 wordings planned) needs both significant; an
        incomplete baseline makes the verdict not evaluable."""
        import r3_wording
        from tests_r2_extra import part_b, run_analysis
        cfg = json.load(open(CFG))
        cfg["groups"] = [g for g in cfg["groups"] if g["label"] != "R3-B unknown_p3 (cut 1)"]
        cfg["groups"][0]["models"] = [GPT]
        for g in cfg["groups"][1:]:
            g["models"] = []
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "cfg.json"); json.dump(cfg, open(p, "w"))
            rows = (part_b(GPT, "timeout_after_commit", [1] * 32) + part_b(GPT, "error_500_after_commit", [1] * 32)
                    + part_b(GPT, "status_unknown_after_commit", [0] * 32)
                    + part_b(GPT, "status_unknown_p1_after_commit", [0] * 32))
            raw = os.path.join(d, "r3", "raw.jsonl"); os.makedirs(os.path.dirname(raw))
            with open(raw, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                r3_wording.main(["--dir", os.path.join(d, "r3"), "--out", d, "--config", p])
            n = json.load(open(os.path.join(d, "r3_wording.json")))["numbers"]
            h = n["r3.B.gpt.H2:supported"]
            self.assertTrue(h["value"]); self.assertEqual(h["wordings"], ["w3", "unknown_p1"])
            self.assertIn(">= 2 of 2 wordings", h["criterion"])
            # now the 500 cell is below 90%: no primary test, verdict not evaluable
            with open(raw, "w") as f:
                for r in rows:
                    if not (r["failure_mode"] == "error_500_after_commit" and r["task_id"] >= "task-04"):
                        f.write(json.dumps(r) + "\n")
            with contextlib.redirect_stdout(io.StringIO()):
                r3_wording.main(["--dir", os.path.join(d, "r3"), "--out", d, "--config", p])
            n = json.load(open(os.path.join(d, "r3_wording.json")))["numbers"]
            self.assertEqual(n["r3.B.gpt.H2:supported"]["value"], "not evaluable")
            self.assertIn("incomplete", n["r3.B.gpt.e500:ivr"]["status"])
            self.assertNotIn("r3.B.gpt.w3_vs_e500:p", n)
            self.assertIn("r3.B.gpt.w3_vs_toA:p", n)                    # the secondary vs timeout still runs


class TestLauncherAndGlob(unittest.TestCase):
    def test_run_r3_sh_static(self):
        s = open(os.path.join(HERE, "run_r3.sh")).read()
        self.assertEqual(subprocess.run(["bash", "-n", os.path.join(HERE, "run_r3.sh")]).returncode, 0)
        self.assertIn("source .env", s); self.assertNotIn("echo $ANTHROPIC_API_KEY", s)
        self.assertNotIn("echo $OPENROUTER_API_KEY", s)
        self.assertIn("ANTHROPIC_API_KEY missing", s); self.assertIn("OPENROUTER_API_KEY missing", s)
        self.assertIn("bash run_r2.sh --yes", s)                       # Part 1 = R2 completion as registered
        self.assertIn("--check-keys", s)
        self.assertIn("r3_estimate.py --cuts", s)
        self.assertIn('BENCH_EXP_BUDGET_USD="$CAP1"', s)               # shared Anthropic cap, part 1
        self.assertIn('BENCH_EXP_BUDGET_USD="$FABLE_CAP"', s)          # shared Anthropic cap, Fable R3-B
        self.assertIn("analysis/r3_wording.py", s)
        self.assertIn("--yes", s)
        for bad in ("results/claude_batch", "results/crossprovider", "results/tier1", "paper-merged"):
            self.assertNotIn(bad, s)
        # --help prints the usage, no API call, no .env needed
        r = subprocess.run(["bash", os.path.join(HERE, "run_r3.sh"), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0); self.assertIn("run_r3.sh --yes", r.stdout)

    def test_make_numbers_raw_glob_skips_r3_wording(self):
        spec = importlib.util.spec_from_file_location("make_numbers", os.path.join(HERE, "paper-merged", "make_numbers.py"))
        mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
        with tempfile.TemporaryDirectory() as d:
            res = os.path.join(d, "results")
            for sub in ("r3_wording", "r2_wording", "some_batch"):
                os.makedirs(os.path.join(res, sub))
                for name in ("sweep_raw.jsonl", "sweep2_raw.jsonl", "raw.jsonl"):
                    with open(os.path.join(res, sub, name), "w") as f:
                        f.write(json.dumps({"episode_id": "x", "model": "m", "failure_mode": "none"}
                                           if "sweep2" not in name else
                                           {"episode_id": "x", "model": "m", "resume_context": "amnesiac"}) + "\n")
            open(os.path.join(res, "sweep_raw_v2.jsonl"), "w").write("")
            open(os.path.join(res, "sweep2_raw_v2.jsonl"), "w").write("")
            saved = mk.RES; mk.RES = res
            try:
                f209 = [os.path.relpath(p, res) for p in mk.raw_files("sweep_raw_v2.jsonl", "sweep_raw*.jsonl")]
                f249 = [os.path.relpath(p, res) for p in mk.raw_files("sweep2_raw_v2.jsonl", "sweep2_raw*.jsonl")]
            finally:
                mk.RES = saved
            self.assertEqual(f209, ["sweep_raw_v2.jsonl", os.path.join("some_batch", "sweep_raw.jsonl")])
            self.assertEqual(f249, ["sweep2_raw_v2.jsonl", os.path.join("some_batch", "sweep2_raw.jsonl")])


if __name__ == "__main__":
    unittest.main(verbosity=1)
