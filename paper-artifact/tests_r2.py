"""Tests for R2 (wording robustness): configs/r2_wording.json, the restart
cells in src/tier1.py, the W3 paraphrases in src/world.py, the bare_hint
paraphrases in src2/sweep2.py, Anthropic prompt caching in src/providers.py,
the budget / deadline / parallel runner, and analysis/r2_wording.py.
No network: scripted fake models play the subject (style of tests_tier1.py).

  python3 tests_r2.py
"""
import contextlib, copy, io, json, os, re, sys, tempfile, threading, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("src", "src2", "analysis"):
    sys.path.insert(0, os.path.join(HERE, d))
sys.path.insert(0, HERE)

import tests_tier1 as tt                                   # noqa: E402
import tier1, sweep, sweep2, providers, model_agent, resume   # noqa: E402
from providers import Completion, strip_cache_control      # noqa: E402
from world import (World, FAILURE_MODES, SIGNAL_VARIANTS, WORDING_VARIANTS,  # noqa: E402
                   ToolTimeout, ToolServerError, EFFECTFUL_TOOLS)
from tasks import generate_tasks                           # noqa: E402

CFG = os.path.join(HERE, "configs", "r2_wording.json")
CFG_OR = os.path.join(HERE, "configs", "r2_wording_or.json")
HAIKU, SONNET, FABLE = "claude-haiku-4-5-20251001", "claude-sonnet-5", "claude-fable-5"

# The registered strings (analysis/prereg_r2_draft.md), as the model sees them.
UNKNOWN_TEXTS = {
    "status_unknown_after_commit": "ERROR: The request may have been processed; its outcome is unknown.",
    "status_unknown_p1_after_commit": "ERROR: Connection lost after the request was sent. It is not known whether it was completed.",
    "status_unknown_p2_after_commit": "ERROR: The server did not confirm the result. The action may or may not have taken effect.",
    "status_unknown_p3_after_commit": "ERROR: Request status unknown; it might have succeeded.",
}
HINT_TEXTS = {
    "bare_hint": "NOTE: a previous session was working on this task and was interrupted. Finish the task.",
    "hint_p1": "NOTE: this task was started earlier in another session, which stopped before it finished. Complete the task.",
    "hint_p2": "This is a resumed task: an earlier attempt was cut off partway through. Please finish it.",
    "hint_p3": "Heads up: someone already began working on this task before being interrupted. Carry on and finish it.",
}
PART_B_FMS = ["timeout_after_commit", "error_500_after_commit"] + list(UNKNOWN_TEXTS)
GRAPHS = {g.instruction: (g, k) for g, k in sweep2.benchmark_graphs()}


def _b(x, k, default=None):
    return x.get(k, default) if isinstance(x, dict) else getattr(x, k, default)


class GraphPolicy:
    """Scripted model for restart episodes: executes the graph's nodes in
    order, one per turn, skipping nodes already acknowledged in THIS
    conversation. With a continuity note in the user message (anything after
    the instruction) it first calls list_effects and skips what is listed; an
    amnesiac session 2 therefore redoes the seam node (gIVR 1), a hinted one
    does not (gIVR 0)."""
    def __init__(self, readback_on_hint=True):
        self.readback_on_hint = readback_on_hint
        self.lock = threading.Lock()
        self.n = 0

    def _id(self):
        with self.lock:
            self.n += 1
            return f"g{self.n}"

    def next_blocks(self, messages):
        user0 = messages[0]["content"]
        if isinstance(user0, list):
            user0 = "".join(_b(b, "text", "") for b in user0)
        instr, _, note = user0.partition("\n\n")
        g, _ = GRAPHS[instr]
        calls, results = [], {}
        for m in messages[1:]:
            for b in m["content"]:
                if _b(b, "type") == "tool_use":
                    calls.append(b)
                elif _b(b, "type") == "tool_result":
                    results[b["tool_use_id"]] = b
        rb = None
        done = set()
        for c in calls:
            r = results.get(_b(c, "id"), {})
            if _b(c, "name") == "list_effects":
                rb = str(r.get("content", ""))
            elif not r.get("is_error") and r:
                done.add(_b(c, "name") + json.dumps(_b(c, "input"), sort_keys=True))
        if note and self.readback_on_hint and not calls:
            return [{"type": "tool_use", "id": self._id(), "name": "list_effects", "input": {}}]
        for node in g.nodes:
            if node.tool + json.dumps(node.args, sort_keys=True) in done:
                continue
            first = str(node.goal.canon_args[list(node.goal.canon_args)[0]])
            if rb and node.tool in rb and first in rb:
                continue
            return [{"type": "tool_use", "id": self._id(), "name": node.tool, "input": dict(node.args)}]
        return [{"type": "text", "text": "DONE"}]


class ErrPolicy(tt.Policy):
    """tests_tier1.Policy that reads back before retrying ONLY when the error
    text says the outcome is unknown (W3 and its paraphrases); a bare 500 /
    timeout is retried blindly. Makes R2-H2 true for the fake model."""
    UNKNOWN = ("unknown", "not known", "may or may not", "might have", "may have been")

    def next_blocks(self, messages):
        errs = [str(b["content"]) for m in messages[1:] for b in m["content"]
                if _b(b, "type") == "tool_result" and b.get("is_error")]
        self.rb_err = any(any(u in e for u in self.UNKNOWN) for e in errs)
        return super().next_blocks(messages)


class SafeFake(tt.FakeProvider):
    """tests_tier1.FakeProvider with a lock (parallel runs) that also records
    the system prompt of every call; routes graph instructions to a
    GraphPolicy and task instructions to the task policy."""
    def __init__(self, task_policy, graph_policy=None, **kw):
        super().__init__(task_policy, **kw)
        self.graph_policy = graph_policy or GraphPolicy()
        self.lock = threading.Lock()
        self.systems = []

    def complete(self, model, system, tools, messages, max_tokens, temperature):
        with self.lock:
            self.systems.append(system)
            user0 = messages[0]["content"]
            pol = self.graph_policy if user0.partition("\n\n")[0] in GRAPHS else self.policy
            self.calls += 1
            if self.fail:
                raise RuntimeError(self.fail)
            blocks = pol.next_blocks(copy.deepcopy(messages))
            meta = {"cost": self.cost} if self.cost is not None else {}
            return Completion(blocks=blocks, provider=self.name, model_requested=model,
                              model_served=model + "-served", temperature_requested=temperature,
                              temperature_effective=None,
                              usage={"input": self.usage[0], "output": self.usage[1]},
                              timestamp_utc="2026-10-31T00:00:00Z",
                              stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
                              provider_meta=meta)


def r2_cfg(d, name="r2_wording", **over):
    c = json.load(open(os.path.join(HERE, "configs", name + ".json")))
    c.update(over)
    c["out_dir"] = os.path.join(d, "r2_wording")
    c["tier1_root"] = c["out_dir"]
    p = os.path.join(d, name + ".json")
    json.dump(c, open(p, "w"))
    return tier1.load_config(p)


def eps_by_part(cfg):
    out = {"A": [], "B": [], "C": []}
    for e in tier1.plan(cfg):
        if not tier1.is_restart(e):
            out["B"].append(e)
        elif e[1] == "plan_only":
            out["C"].append(e)
        else:
            out["A"].append(e)
    return out


# ---------------------------------------------------------------------------
# 1. The W3 paraphrases (Part B)
# ---------------------------------------------------------------------------

class TestUnknownParaphrases(unittest.TestCase):
    def test_delivered_verbatim_in_the_failed_tool_result_with_w3_error_flag(self):
        task = generate_tasks(4)[0]
        for fm, text in UNKNOWN_TEXTS.items():
            w = World(failure_mode=fm, seed=1)
            with self.assertRaises(ToolTimeout) as cm:           # timeout class, exactly like W3
                w.call(*task.steps[0], "", task.task_id)
            self.assertIs(type(cm.exception), ToolTimeout)
            self.assertEqual("ERROR: " + str(cm.exception), text)
            self.assertEqual(len(w.ledger.for_task(task.task_id)), 1)   # committed
            # through the real model loop: the first effectful call's result
            w = World(failure_mode=fm, seed=1)
            tr = model_agent.run_model_episode(tt.FakeProvider(tt.Policy()), w, task,
                                               model_agent.BASE_SYSTEM, model=HAIKU)
            first = next(r for t in tr["turns"] if t["role"] == "tool_results" for r in t["content"])
            self.assertEqual(first["content"], text)
            self.assertIs(first["is_error"], True)
            self.assertEqual(w.ledger.violations(task.task_id), 1)    # the blind retry duplicates

    def test_committed_state_identical_across_the_six_conditions(self):
        task = next(t for t in generate_tasks(4) if t.family == "multi_step")
        ledgers, contents = {}, {}
        for fm in PART_B_FMS:
            w = World(failure_mode=fm, seed=12345)
            tr = model_agent.run_model_episode(tt.FakeProvider(tt.Policy()), w, task,
                                               model_agent.BASE_SYSTEM, model=HAIKU)
            self.assertFalse(w._failure_armed)                       # fired on call 1
            ledgers[fm] = [(e.tool, json.dumps(e.args, sort_keys=True), e.key, e.seq)
                           for e in w.ledger.for_task(task.task_id)]
            contents[fm] = [r["content"] for t in tr["turns"] if t["role"] == "tool_results"
                            for r in t["content"]]
        ref = ledgers[PART_B_FMS[0]]
        for fm in PART_B_FMS:
            self.assertEqual(ledgers[fm], ref, fm)                   # same effects, same order, same acks
            # only the first (failed) result differs between conditions
            self.assertEqual(contents[fm][1:], contents[PART_B_FMS[0]][1:], fm)
        self.assertEqual(len({c[0] for c in contents.values()}), 6)

    def test_registered_modes_unchanged(self):
        self.assertEqual(FAILURE_MODES, ("none", "timeout_after_commit", "timeout_before_commit",
                                         "error_500_after_commit"))
        self.assertEqual(set(SIGNAL_VARIANTS), {"error_500_body_after_commit", "error_503_retry_after_after_commit",
                                                "status_unknown_after_commit", "timeout_504_after_commit"})
        self.assertEqual(SIGNAL_VARIANTS["status_unknown_after_commit"],
                         ("timeout", "The request may have been processed; its outcome is unknown."))
        self.assertEqual(set(WORDING_VARIANTS), set(PART_B_FMS[3:]))
        self.assertTrue(all(k == "timeout" for k, _ in WORDING_VARIANTS.values()))
        import rescore
        for fm in WORDING_VARIANTS:
            self.assertEqual(rescore.FNAME.match(f"claude-sonnet-5_base_none_{fm}__s1__email-abc.json")["fm"], fm)


# ---------------------------------------------------------------------------
# 2. The bare_hint paraphrases (Part A)
# ---------------------------------------------------------------------------

class TestHintParaphrases(unittest.TestCase):
    def test_exact_text_appended_like_bare_hint_and_independent_of_progress(self):
        for g, k in sweep2.benchmark_graphs():
            for ctx, text in HINT_TEXTS.items():
                self.assertEqual(sweep2.resume_user_msg(g, ctx, {0}), f"{g.instruction}\n\n{text}")
                self.assertEqual(sweep2.resume_user_msg(g, ctx, set()), sweep2.resume_user_msg(g, ctx, {0, 1, 2}))
            self.assertEqual(sweep2.resume_user_msg(g, "amnesiac", {0}), g.instruction)
            self.assertEqual(sweep2.resume_user_msg(g, "plan_only", {0, 2}),
                             f"{g.instruction}\n\nNOTE: a previous session was working on this task and was "
                             f"interrupted. Its saved plan notes say it believed it had completed: step 1, step 3. "
                             f"It had no record of anything else. Finish the task.")
            self.assertEqual(sweep2.resume_user_msg(g, "ledger_readback", {0}),
                             f"{g.instruction}\n\nNOTE: a previous session was working on this task and was "
                             f"interrupted. Use list_effects to see exactly which actions already happened, then "
                             f"finish the task without repeating any of them.")

    def test_no_progress_information_and_no_readback_instruction(self):
        for ctx in ("hint_p1", "hint_p2", "hint_p3"):
            t = sweep2.HINT_PARAPHRASES[ctx].lower()
            self.assertEqual(t, HINT_TEXTS[ctx].lower())
            self.assertFalse(re.search(r"\d", t), ctx)                       # no step numbers
            for banned in ("step", "list_effects", "effects", "log", "record", "check", "completed:",
                           "already done", "already happened", "repeat"):
                self.assertNotIn(banned, t, (ctx, banned))

    def test_system_prompt_is_the_plain_base_system(self):
        g, k = sweep2.benchmark_graphs()[0]
        with tempfile.TemporaryDirectory() as d:
            for ctx in ("hint_p1", "hint_p2", "hint_p3", "bare_hint", "amnesiac"):
                prov = SafeFake(tt.Policy())
                row = sweep2.run_episode(prov, HAIKU, ctx, "none", g, k, 0, transcript_dir=d)
                self.assertEqual(set(prov.systems), {sweep2.BASE_SYSTEM}, ctx)
                self.assertEqual(row["givr_hit"], 1 if ctx == "amnesiac" else 0, ctx)
                self.assertTrue(os.path.exists(os.path.join(d, row["episode_id"] + ".json")))
            prov = SafeFake(tt.Policy())
            sweep2.run_episode(prov, HAIKU, "ledger_readback", "none", g, k, 0, transcript_dir=d)
            self.assertIn(sweep2.BASE_SYSTEM + sweep2.READBACK_HINT, prov.systems)
        self.assertEqual(sweep2.CELLS[:4], (("amnesiac", "none"), ("plan_only", "none"),
                                            ("ledger_readback", "none"), ("amnesiac", "runtime_receipts")))


# ---------------------------------------------------------------------------
# 3. Cell grid
# ---------------------------------------------------------------------------

class TestGrid(unittest.TestCase):
    def test_counts_320_768_48(self):
        cfg = tier1.load_config(CFG)
        parts = eps_by_part(cfg)
        self.assertEqual({p: len(v) for p, v in parts.items()}, {"A": 320, "B": 768, "C": 48})
        eps = tier1.plan(cfg)
        self.assertEqual(len(eps), 1136)
        self.assertEqual(len({tier1.eid_of(e) for e in eps}), 1136)
        # Part A: Haiku + Fable x 5 contexts x 16 graphs x 2
        self.assertEqual({e[0] for e in parts["A"]}, {HAIKU, FABLE})
        self.assertEqual({e[1] for e in parts["A"]}, {"amnesiac", "bare_hint", "hint_p1", "hint_p2", "hint_p3"})
        for m in (HAIKU, FABLE):
            for ctx in ("amnesiac", "bare_hint", "hint_p1", "hint_p2", "hint_p3"):
                cell = [e for e in parts["A"] if e[0] == m and e[1] == ctx]
                self.assertEqual(len(cell), 32)
                self.assertEqual(sorted((e[3].graph_id, e[4]) for e in cell[::2]),
                                 sorted((g.graph_id, k) for g, k in sweep2.benchmark_graphs()))
        # Part B: Haiku + Sonnet x 6 conditions x 32 tasks x 2, base prompt, no mitigation
        self.assertEqual({e[0] for e in parts["B"]}, {HAIKU, SONNET})
        self.assertEqual({(e[1], e[2]) for e in parts["B"]}, {("base", "none")})
        self.assertEqual({e[3] for e in parts["B"]}, set(PART_B_FMS))
        for m in (HAIKU, SONNET):
            for fm in PART_B_FMS:
                self.assertEqual(sum(1 for e in parts["B"] if e[0] == m and e[3] == fm), 64)
        self.assertEqual({e[4].task_id for e in parts["B"]}, {t.task_id for t in generate_tasks(4)})
        # Part C: 3 Claude models x the 8 chain4 graphs x 2 = 16 per model
        self.assertEqual({e[0] for e in parts["C"]}, {HAIKU, SONNET, FABLE})
        for m in (HAIKU, SONNET, FABLE):
            cell = [e for e in parts["C"] if e[0] == m]
            self.assertEqual(len(cell), 16)
            self.assertTrue(all(e[3].topology == "chain4" and e[2] == "none" for e in cell))
        self.assertTrue(cfg["prompt_cache"]); self.assertEqual(cfg["exp_budget_usd"], 45)
        self.assertEqual(cfg["exp_or_budget_usd"], 0)
        self.assertTrue(all(tier1.provider_of(m) == "anthropic" for m in cfg["models"]))
        self.assertIn(os.path.join("results", "r2_wording"), cfg["out_dir"])

    def test_part_c_matches_the_july_chain_plan_only_cells_exactly(self):
        jul = os.path.join(HERE, "results", "sweep2_raw.jsonl")
        if not os.path.exists(jul):
            self.skipTest("released results not present")
        want = {json.loads(l)["episode_id"] for l in open(jul) if l.strip()}
        want = {e for e in want if "__plan_only__none__chain4-" in e}
        cfg = tier1.load_config(CFG)
        got = {tier1.eid_of(e) for e in eps_by_part(cfg)["C"]}
        self.assertEqual(got, want)                     # same graphs, k, samples, id format
        self.assertEqual(len(got), 48)

    def test_retry_episodes_come_first_and_smoke_is_one_part_b_haiku_episode(self):
        cfg = tier1.load_config(CFG)
        eps = tier1.plan(cfg)
        first_restart = next(i for i, e in enumerate(eps) if tier1.is_restart(e))
        self.assertTrue(all(not tier1.is_restart(e) for e in eps[:first_restart]))
        self.assertTrue(all(tier1.is_restart(e) for e in eps[first_restart:]))
        e0 = tier1.plan(cfg, [HAIKU])[0]
        self.assertEqual((e0[0], e0[1], e0[2]), (HAIKU, "base", "none"))

    def test_optional_openrouter_config_is_separate(self):
        cfg = tier1.load_config(CFG_OR)
        parts = eps_by_part(cfg)
        self.assertEqual({p: len(v) for p, v in parts.items()}, {"A": 320, "B": 384, "C": 0})
        self.assertTrue(all(tier1.provider_of(m) == "openrouter" for m in cfg["models"]))
        self.assertEqual(cfg["exp_or_budget_usd"], 40)
        self.assertFalse(cfg["prompt_cache"])
        self.assertNotEqual(cfg["out_dir"], tier1.load_config(CFG)["out_dir"])

    def test_existing_configs_unchanged(self):
        for name, n in tt.TestConfigsAndPlans.EXPECT.items():
            c = tier1.load_config(os.path.join(HERE, "configs", name + ".json"))
            self.assertEqual(len(tier1.plan(c)), n, name)
            self.assertEqual(c["restart_cells"], []); self.assertEqual(c["groups"], [])
            self.assertFalse(c["prompt_cache"]); self.assertEqual(c["workers"], 1)
            self.assertTrue(all(len(e) == 6 for e in tier1.plan(c)))


# ---------------------------------------------------------------------------
# 4. Runner: fake end-to-end, budget stop, resume / dedupe, parallel, deadline
# ---------------------------------------------------------------------------

class TestRunner(tt._Tier1Case):
    def setUp(self):
        super().setUp()
        self.saved_r2 = (tier1.WORKERS, tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S, providers.PROMPT_CACHE)
        tier1.WORKERS = 0; tier1.DEADLINE_S = 0.0; tier1.DEADLINE_MARGIN_S = 0.0

    def tearDown(self):
        (tier1.WORKERS, tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S, providers.PROMPT_CACHE) = self.saved_r2
        super().tearDown()

    def run_quiet(self, cfg):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc, rows, summ = self.run_cfg(cfg)
        return rc, rows, summ, out.getvalue()

    def test_full_fake_run_rows_summaries_and_analysis(self):
        c = r2_cfg(self.d, workers=1, prompt_cache=False)
        self.use(lambda m: SafeFake(ErrPolicy()))
        rc, rows, summ, log = self.run_quiet(c)
        self.assertEqual(rc, 0)
        self.assertEqual(len(rows), 1136)
        self.assertEqual(len({r["episode_id"] for r in rows}), 1136)
        self.assertEqual(len(os.listdir(c["transcript_dir"])), 1136)
        restart = [r for r in rows if r.get("kind") == "restart"]
        retry = [r for r in rows if r.get("kind") != "restart"]
        self.assertEqual((len(restart), len(retry)), (368, 768))
        for r in restart:                     # schema the analyses and the paper loader need
            for k in ("graph_id", "topology", "resume_context", "mitigation", "interrupt_node", "givr_hit",
                      "violations", "order_violation", "completion", "task_id", "family", "prompt",
                      "failure_mode", "achieved", "session2_lookup", "refusal", "cost_usd", "model_served",
                      "temperature", "usage_input", "usage_output", "started_utc", "finished_utc", "exp"):
                self.assertIn(k, r)
            self.assertEqual((r["prompt"], r["failure_mode"], r["task_id"]), ("base", "none", r["graph_id"]))
            self.assertTrue(r["interrupted"]); self.assertEqual(r["n_sessions"], 2)
            self.assertEqual(r["givr_hit"], 1 if r["resume_context"] == "amnesiac" else 0)
            self.assertEqual(r["session2_lookup"], r["resume_context"] != "amnesiac")
            self.assertEqual(r["interrupt_timing"], "after_commit")
        for r in retry:
            self.assertTrue(r["failure_fired"]); self.assertEqual(r["fail_on_call"], 1)
            unknown = r["failure_mode"].startswith("status_unknown")
            self.assertEqual(r["violations"], 0 if unknown else 1, r["failure_mode"])
            self.assertEqual(r["readback_after_failure"], unknown)
            self.assertNotIn("resume_context", r)
        # summaries: retry cells in summary.csv, restart cells in summary_restart.csv
        self.assertEqual(len(summ), 12)
        import csv
        rs = list(csv.DictReader(open(os.path.join(c["out_dir"], "summary_restart.csv"))))
        self.assertEqual(len(rs), 13)                                     # 10 Part A + 3 Part C cells
        amn = next(r for r in rs if r["model"] == HAIKU and r["resume_context"] == "amnesiac")
        self.assertEqual((amn["n_episodes"], amn["n_graphs"], amn["givr"]), ("32", "16", "1.0"))
        # the first line of raw.jsonl is a Part-B row (the paper's loader classifies a file by its first row)
        first = json.loads(open(c["raw_path"]).readline())
        self.assertNotIn("resume_context", first); self.assertIn("failure_mode", first)
        # analysis on this run
        import r2_wording
        with contextlib.redirect_stdout(io.StringIO()):
            r2_wording.main(["--dir", c["out_dir"], "--out", os.path.join(self.d, "an"), "--config", CFG])
        nums = json.load(open(os.path.join(self.d, "an", "r2_wording.json")))
        self.assertEqual(nums["meta"]["status"], "complete")
        n = nums["numbers"]
        for sm in ("haiku", "fable"):
            self.assertEqual(n[f"r2.A.{sm}.amnesiac:givr"]["value"], 1.0)
            self.assertEqual(n[f"r2.A.{sm}.amnesiac:givr"]["T"], 16)
            self.assertTrue(n[f"r2.A.{sm}.H1:supported"]["value"])
            for p in ("hint_p1", "hint_p2", "hint_p3"):
                self.assertLess(n[f"r2.A.{sm}.{p}_vs_amnesiac:p"]["holm_p"], 0.05)
                self.assertEqual(n[f"r2.A.{sm}.{p}_vs_amnesiac:d"]["value"], -1.0)
            self.assertEqual(n[f"r2.A.{sm}.spread:maxmin"]["value"], 0.0)
        for sm in ("haiku", "sonnet"):
            self.assertTrue(n[f"r2.B.{sm}.H2:supported"]["value"])
            self.assertEqual(n[f"r2.B.{sm}.e500:ivr"]["value"], 1.0)
            self.assertEqual(n[f"r2.B.{sm}.e500:ivr"]["T"], 32)
            self.assertEqual(n[f"r2.B.{sm}.unknown_p1:readback_after_failure"]["value"], 1.0)
            self.assertIn(f"r2.B.{sm}.w3_vs_toA:p", n)
        for sm in ("haiku", "sonnet", "fable"):
            self.assertEqual(n[f"r2.C.{sm}.plan_only:givr"]["T"], 8)
        self.assertEqual(n["r2.total:episodes"]["value"], 1136)
        self.assertEqual(n["r2.total:refusals"]["value"], 0)

    def test_budget_stop_is_clean_and_the_rerun_resumes_without_duplicates(self):
        # Haiku $1/$5 per MTok; 3 turns (call, retry, DONE) x (100k in, 10k out)
        # = $0.45 per episode. Cap $1.00: an episode starts only while
        # spent + max(max_ep, 0.5) <= cap -> 0 (ok), 0.45 (ok), 0.90 (1.40 > 1:
        # stop) => exactly 2 episodes, exit 2, nothing half-written.
        c = r2_cfg(self.d, workers=1, prompt_cache=False, exp_budget_usd=1.0)
        self.use(lambda m: SafeFake(tt.Policy(), usage=(100_000, 10_000)))
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual(rc, 2)
        self.assertEqual(len(rows), 2)
        self.assertIn("anthropic budget $1.00 would be exceeded", log)
        self.assertIn("stopped: {'anthropic': 'exp_budget'}", log)
        self.assertAlmostEqual(sum(r["cost_usd"] for r in rows), 0.90, places=6)
        self.assertAlmostEqual(tier1.spent_in_exp(c)["anthropic"], 0.90, places=6)   # logged spend = transcripts
        self.assertEqual(len(os.listdir(c["transcript_dir"])), 2)
        # rerun with a higher cap: the 2 done episodes are skipped, nothing is duplicated
        c = r2_cfg(self.d, workers=1, prompt_cache=False, exp_budget_usd=2.0)
        self.tier1.LIMIT = 10
        rc, rows, _, log = self.run_quiet(c)
        self.assertIn("2 done", log)
        self.assertEqual(len(rows), 4)             # 0.90 -> 1.35 (1.85 <= 2 ok) -> 1.80 (2.30 > 2 stop)
        self.assertEqual(len({r["episode_id"] for r in rows}), 4)
        self.assertEqual(rc, 2)

    def test_resume_dedupe_and_limit(self):
        c = r2_cfg(self.d, workers=1, prompt_cache=False)
        self.use(lambda m: SafeFake(tt.Policy()))
        self.tier1.LIMIT = 5
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual((rc, len(rows)), (0, 5))
        before = self.providers[HAIKU].calls
        self.tier1.LIMIT = 7
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual(len(rows), 12)
        self.assertEqual(len({r["episode_id"] for r in rows}), 12)
        self.assertIn("5 done, 1131 remaining", log)
        self.assertGreater(self.providers[HAIKU].calls, before)
        # a row written twice (an overlapping run) counts once
        with open(c["raw_path"], "a") as f:
            f.write(json.dumps(rows[0]) + "\n")
        self.assertEqual(len(tier1.read_rows(c["raw_path"])), 12)
        self.assertEqual(len(tier1.done_ids(c)), 12)

    def test_smoke_is_exactly_one_haiku_part_b_episode(self):
        c = r2_cfg(self.d, workers=4, prompt_cache=False)
        self.use(lambda m: SafeFake(tt.Policy()))
        self.tier1.LIMIT = 1
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                rc = self.tier1.launch(c, [HAIKU])
            finally:
                tt.release_locks()
        rows = tier1.read_rows(c["raw_path"])
        self.assertEqual((rc, len(rows)), (0, 1))
        self.assertEqual(rows[0]["model"], HAIKU); self.assertNotIn("resume_context", rows[0])
        self.assertIn("BENCH_LIMIT=1 reached", out.getvalue())

    def test_parallel_workers_give_the_same_rows(self):
        c = r2_cfg(self.d, workers=4, prompt_cache=False)
        self.use(lambda m: SafeFake(ErrPolicy()))
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual(rc, 0)
        self.assertIn("4 episodes in flight", log)
        self.assertEqual(len(rows), 1136)
        self.assertEqual(len({r["episode_id"] for r in rows}), 1136)
        self.assertEqual({tier1.eid_of(e) for e in tier1.plan(c)}, {r["episode_id"] for r in rows})
        for r in rows:
            if r.get("kind") == "restart":
                self.assertEqual(r["givr_hit"], 1 if r["resume_context"] == "amnesiac" else 0)
            else:
                self.assertEqual(r["violations"], 0 if r["failure_mode"].startswith("status_unknown") else 1)

    def test_deadline_with_margin_stops_before_starting_more(self):
        c = r2_cfg(self.d, workers=2, prompt_cache=False)
        self.use(lambda m: SafeFake(tt.Policy()))
        tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S = 100.0, 100.0        # elapsed + margin > deadline at once
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual((rc, len(rows)), (0, 0))
        self.assertIn("deadline 100s reached (margin 100s)", log)
        tier1.DEADLINE_S, tier1.DEADLINE_MARGIN_S = 1000.0, 0.0
        self.tier1.LIMIT = 3
        rc, rows, _, log = self.run_quiet(c)
        self.assertEqual(rc, 0)
        self.assertIn(len(rows), (3, 4))           # with 2 in flight the limit may be passed by one
        self.assertNotIn("deadline", log)


# ---------------------------------------------------------------------------
# 5. Anthropic prompt caching
# ---------------------------------------------------------------------------

class _Blk:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeAnthropic:
    """anthropic.Anthropic stand-in: records every messages.create(**kwargs)
    and answers with a Policy; usage carries cache fields."""
    def __init__(self, policy, reject_cache=False):
        self.policy, self.reject_cache, self.kwargs = policy, reject_cache, []
        self.messages = self
        self.n = 0

    def create(self, **kw):
        self.kwargs.append(copy.deepcopy(kw))
        body = json.dumps(kw, default=lambda o: o.__dict__, sort_keys=True)
        if self.reject_cache and "cache_control" in body:
            import anthropic, types
            raise anthropic.BadRequestError("cache_control is not supported",
                                            response=types.SimpleNamespace(status_code=400, headers={}, request=None),
                                            body=None)
        msgs = []
        for m in kw["messages"]:
            c = m["content"]
            if isinstance(c, list) and c and not isinstance(c[0], dict):      # our own _Blk history
                c = [b.__dict__ for b in c]
            if isinstance(c, list):
                c = [{k: v for k, v in b.items() if k != "cache_control"} for b in c]
                if len(c) == 1 and c[0].get("type") == "text":
                    c = c[0]["text"]
            msgs.append({"role": m["role"], "content": c})
        blocks = self.policy.next_blocks(msgs)
        self.n += 1
        content = [_Blk(type=b["type"], text=b.get("text"), id=b.get("id"), name=b.get("name"),
                        input=b.get("input")) for b in blocks]
        tool = any(b["type"] == "tool_use" for b in blocks)
        return _Blk(content=content, model=kw["model"] + "-served", stop_reason="tool_use" if tool else "end_turn",
                    usage=_Blk(input_tokens=100 * self.n, output_tokens=10, cache_creation_input_tokens=500,
                               cache_read_input_tokens=50 * self.n))


class TestPromptCaching(unittest.TestCase):
    def setUp(self):
        self.saved = providers.PROMPT_CACHE

    def tearDown(self):
        providers.PROMPT_CACHE = self.saved

    def _episode(self, cache, reject=False, model=HAIKU):
        providers.PROMPT_CACHE = cache
        fake = FakeAnthropic(tt.Policy(), reject_cache=reject)
        prov = providers.AnthropicProvider(client=fake)
        task = next(t for t in generate_tasks(4) if t.family == "multi_step")
        w = World(failure_mode="timeout_after_commit", seed=3)
        tr = model_agent.run_model_episode(prov, w, task, model_agent.BASE_SYSTEM, model=model)
        return fake, prov, tr

    def test_off_by_default_and_body_unchanged(self):
        self.assertEqual(os.environ.get("BENCH_PROMPT_CACHE", ""), "")
        fake, _, tr = self._episode(False)
        for kw in fake.kwargs:
            self.assertIsInstance(kw["system"], str)
            self.assertNotIn("cache_control", json.dumps(kw, default=lambda o: o.__dict__))
        self.assertEqual(set(tr["usage"]), {"input", "output"})       # no cache fields when off

    def test_cached_body_differs_only_in_cache_control(self):
        plain, _, tr0 = self._episode(False)
        fake, _, tr = self._episode(True)
        self.assertEqual(len(fake.kwargs), len(plain.kwargs))
        self.assertGreaterEqual(len(fake.kwargs), 3)
        for kw_c, kw_p in zip(fake.kwargs, plain.kwargs):
            body_c = json.loads(json.dumps(kw_c, default=lambda o: o.__dict__, sort_keys=True))
            body_p = json.loads(json.dumps(kw_p, default=lambda o: o.__dict__, sort_keys=True))
            self.assertEqual(strip_cache_control(body_c), strip_cache_control(body_p))
            self.assertEqual(strip_cache_control(body_p), body_p)      # the plain body has none
            # exactly three breakpoints: system block, last tool, last block of the last user turn
            self.assertEqual(json.dumps(body_c).count('"cache_control"'), 3)
            self.assertEqual(body_c["system"][-1]["cache_control"], {"type": "ephemeral"})
            self.assertEqual(body_c["tools"][-1]["cache_control"], {"type": "ephemeral"})
            self.assertTrue(all("cache_control" not in t for t in body_c["tools"][:-1]))
            last = body_c["messages"][-1]
            self.assertEqual(last["role"], "user")
            self.assertEqual(last["content"][-1]["cache_control"], {"type": "ephemeral"})
            for m in body_c["messages"][:-1]:
                self.assertNotIn("cache_control", json.dumps(m))
        # cache tokens logged per call and summed per episode
        self.assertEqual(set(tr["usage"]), {"input", "output", "cache_write", "cache_read"})
        n = len(fake.kwargs)
        self.assertEqual(tr["usage"]["cache_write"], 500 * n)
        self.assertEqual(tr["usage"]["cache_read"], 50 * n * (n + 1) // 2)
        for turn in tr["turns"]:
            if turn["role"] == "assistant":
                self.assertEqual(set(turn["provider_meta"]), {"cache_write", "cache_read"})

    def test_history_is_not_mutated_by_the_cached_request(self):
        system, tools = "sys", [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}]
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": [{"type": "text", "text": "x"}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "1", "content": "ok"}]}]
        snap = copy.deepcopy(msgs)
        s2, t2, m2 = providers.cached_request(system, tools, msgs)
        self.assertEqual(msgs, snap); self.assertEqual(tools, [{"name": "a", "input_schema": {}}, {"name": "b", "input_schema": {}}])
        self.assertEqual(s2, [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}])
        self.assertEqual(t2[-1]["cache_control"], {"type": "ephemeral"}); self.assertNotIn("cache_control", t2[0])
        self.assertEqual(m2[-1]["content"][-1]["cache_control"], {"type": "ephemeral"})
        self.assertEqual(m2[0], msgs[0]); self.assertEqual(m2[1], msgs[1])
        self.assertEqual(strip_cache_control({"system": s2, "tools": t2, "messages": m2}),
                         strip_cache_control({"system": system, "tools": tools, "messages": msgs}))

    def test_cache_tokens_are_priced_1_25x_and_0_1x(self):
        t = {"model": HAIKU, "usage": {"input": 1000, "output": 100, "cache_write": 2000, "cache_read": 5000}}
        self.assertAlmostEqual(sweep.episode_cost(t), (1000 * 1 + 100 * 5 + 2000 * 1.25 + 5000 * 0.1) / 1e6)
        self.assertEqual(sweep.CACHE_WRITE_MULT, 1.25); self.assertEqual(sweep.CACHE_READ_MULT, 0.1)
        # restart transcripts (per-session usage) and transcripts without cache fields
        t2 = {"model": SONNET, "sessions": [{"meta": {"usage": {"input": 1000, "output": 0, "cache_read": 10000}}},
                                            {"meta": {"usage": {"input": 0, "output": 100, "cache_write": 1000}}}]}
        self.assertAlmostEqual(sweep.episode_cost(t2), (1000 * 2 + 100 * 10 + 10000 * 0.2 + 1000 * 2.5) / 1e6)
        self.assertAlmostEqual(sweep.episode_cost({"model": HAIKU, "usage": {"input": 10**6, "output": 10**5}}), 1.5)

    def test_rejected_cache_control_falls_back_without_caching_and_logs_once(self):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            self.skipTest("anthropic SDK not installed")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            fake, prov, tr = self._episode(True, reject=True)
        self.assertTrue(prov._cache_unsupported)
        self.assertEqual(out.getvalue().count("prompt caching not accepted"), 1)
        bodies = [json.dumps(k, default=lambda o: o.__dict__) for k in fake.kwargs]
        self.assertIn("cache_control", bodies[0])                        # tried once
        self.assertTrue(all("cache_control" not in b for b in bodies[1:]))   # then never again
        self.assertEqual(set(tr["usage"]), {"input", "output"})
        self.assertGreaterEqual(len(tr["turns"]), 3)                     # the episode still ran

    def test_tier1_turns_caching_on_from_the_config(self):
        import tempfile as tf
        with tf.TemporaryDirectory() as d:
            c = r2_cfg(d, workers=1)
            self.assertTrue(c["prompt_cache"])
            saved = (tier1._client_for, tier1._anthropic_client, tier1.LIMIT)
            fake = FakeAnthropic(tt.Policy())
            prov = providers.AnthropicProvider(client=fake)
            try:
                tier1._anthropic_client = lambda models: None
                tier1._client_for = lambda m, _c: prov
                tier1.LIMIT = 1
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = tier1.launch(c, [HAIKU])
            finally:
                tier1._client_for, tier1._anthropic_client, tier1.LIMIT = saved
                tt.release_locks()
                model_agent.MAX_TURNS, model_agent.MAX_TOKENS = 8, int(os.environ.get("BENCH_MAX_TOKENS", "1024"))
            self.assertEqual(rc, 0)
            self.assertTrue(providers.PROMPT_CACHE)
            row = tier1.read_rows(c["raw_path"])[0]
            self.assertIn("usage_cache_read", row); self.assertIn("usage_cache_write", row)
            t = json.load(open(os.path.join(c["transcript_dir"], os.listdir(c["transcript_dir"])[0])))
            self.assertIn("cache_read", t["usage"])
            self.assertEqual(row["cost_usd"], round(sweep.episode_cost(t), 6))
            self.assertIn("cache_control", json.dumps(fake.kwargs[0], default=lambda o: o.__dict__))


# ---------------------------------------------------------------------------
# 6. Analysis: no data
# ---------------------------------------------------------------------------

class TestAnalysisNoData(unittest.TestCase):
    def test_no_data_writes_status_and_exits_zero(self):
        import r2_wording
        with tempfile.TemporaryDirectory() as d:
            with contextlib.redirect_stdout(io.StringIO()):
                rc = r2_wording.main(["--dir", os.path.join(d, "none"), "--out", d, "--config", CFG])
            self.assertEqual(rc, 0)
            j = json.load(open(os.path.join(d, "r2_wording.json")))
            self.assertEqual(j["meta"]["status"], "no data")
            self.assertEqual(j["numbers"], {})
            self.assertEqual(j["meta"]["planned_episodes"], 1136)
            self.assertEqual(j["meta"]["perm_seed"], 20261025)

    def test_one_sided_test_matches_the_common_protocol_scheme(self):
        import r2_wording
        from v3_common import perm_test
        x = {f"g{i}": 1.0 for i in range(16)}
        y = {f"g{i}": (0.0 if i < 12 else 1.0) for i in range(16)}
        one, two = r2_wording.perm_test_one_sided(x, y), perm_test(x, y)
        self.assertEqual((one["T"], one["d"]), (16, -0.75))
        self.assertLess(one["p"], two["p"])                 # one-sided in the stated direction
        self.assertLess(one["p"], 0.05)
        self.assertEqual(r2_wording.perm_test_one_sided(x, x)["p"], 1.0)   # no difference: p = 1
        rev = r2_wording.perm_test_one_sided(y, x)          # wrong direction: p ~ 1
        self.assertGreater(rev["p"], 0.95)


if __name__ == "__main__":
    unittest.main(verbosity=1)
