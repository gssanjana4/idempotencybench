"""Tests for the real-API validation slice (src_real/, configs/real_stripe.json,
analysis/real_stripe.py). No network: an in-process fake Stripe server
(src_real/fake_stripe.py) and a scripted fake model; the code path from the
model loop through the HTTP client is the real one.

  python3 tests_real.py
"""
import copy, csv, glob, hashlib, io, json, os, re, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
for p in ("src", "src2", "src_real"):
    sys.path.insert(0, os.path.join(HERE, p))

from providers import Completion, OpenRouterProvider  # noqa: E402
from world import World, ToolTimeout, ToolServerError, ToolCallError, Goal  # noqa: E402
import stripe_client  # noqa: E402
from stripe_client import StripeClient, UnsafeKeyError, StripeInfraError, StripeAPIError  # noqa: E402
from stripe_world import StripeWorld, stripe_ground_truth, amount_to_cents, stripe_key_for  # noqa: E402
from fake_stripe import FakeStripe  # noqa: E402
import run_real  # noqa: E402
import tasks_real  # noqa: E402

FAKE_KEY = "sk_test_FAKEKEY0123456789secretvalue"
PAY_RE = re.compile(r"[Pp]ay (\S+) \$([\d.]+) for ([\w-]+)")


def _jload(p):
    with open(p) as f:
        return json.load(f)


def _jlines(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def _csv(p):
    with open(p) as f:
        return list(csv.DictReader(f))


def release_locks():
    import runlock
    while runlock._HELD:
        runlock._HELD.pop().close()


def _b(x, k, default=None):
    return x.get(k, default) if isinstance(x, dict) else getattr(x, k, default)


class PayPolicy:
    """Scripted model for payment tasks/plans. Pays each 'Pay X $A for M' in
    order, one call per turn. key: none | stable | fresh. On an error it
    optionally reads back first (readback_on_error) and retries once unless
    the read-back lists the payment. hint_readback: when the user message
    carries a NOTE (bare_hint), start with list_effects."""

    def __init__(self, key="none", readback_on_error=False, retry=True, hint_readback=False,
                 amount_fmt=None):
        self.key, self.rb_err, self.retry = key, readback_on_error, retry
        self.hint_rb, self.amount_fmt = hint_readback, amount_fmt
        self.n = 0

    def _use(self, name, inp):
        self.n += 1
        return {"type": "tool_use", "id": f"tu{self.n}", "name": name, "input": inp}

    def next_blocks(self, messages):
        user = messages[0]["content"]
        pays = PAY_RE.findall(user)
        calls, results = [], {}
        for m in messages[1:]:
            for b in m["content"]:
                if _b(b, "type") == "tool_use":
                    calls.append(b)
                elif _b(b, "type") == "tool_result":
                    results[b["tool_use_id"]] = b
        if self.hint_rb and "NOTE:" in user and not calls:
            return [self._use("list_effects", {})]
        rb_text, done, errs, last_err_then_rb = None, set(), {}, False
        for c in calls:
            r = results.get(_b(c, "id"), {})
            if _b(c, "name") == "list_effects":
                rb_text = str(r.get("content", ""))
                last_err_then_rb = True
                continue
            inp = dict(_b(c, "input"))
            ident = (inp["payee"], inp["memo"])
            if r.get("is_error"):
                errs[ident] = errs.get(ident, 0) + 1
                last_err_then_rb = False
            else:
                done.add(ident)
        for payee, amount, memo in pays:
            ident = (payee, memo)
            if ident in done:
                continue
            if rb_text and f"payee={payee}" in rb_text and f"memo={memo}" in rb_text:
                continue
            ne = errs.get(ident, 0)
            if ne:
                if self.rb_err and not last_err_then_rb:
                    return [self._use("list_effects", {})]
                if not self.retry or ne >= 2:
                    continue
            a = {"payee": payee, "amount": self.amount_fmt(amount) if self.amount_fmt else amount,
                 "memo": memo}
            if self.key == "stable":
                a["idempotency_key"] = "k-" + hashlib.sha256(f"{payee}{amount}{memo}".encode()).hexdigest()[:12]
            elif self.key == "fresh":
                a["idempotency_key"] = f"fresh-{self.n}"
            return [self._use("send_payment", a)]
        return [{"type": "text", "text": "DONE"}]


class FakeProvider:
    def __init__(self, policy, name="fake", usage=(1000, 100), cost=None, fail=None,
                 fail_after=None):
        self.policy, self.name, self.usage, self.cost = policy, name, usage, cost
        self.fail, self.fail_after = fail, fail_after
        self.calls = 0

    def complete(self, model, system, tools, messages, max_tokens, temperature):
        self.calls += 1
        self.last_tools = [t["name"] for t in tools]
        if self.fail and (self.fail_after is None or self.calls > self.fail_after):
            raise RuntimeError(self.fail)
        blocks = self.policy.next_blocks(copy.deepcopy(messages))
        meta = {"cost": self.cost} if self.cost is not None else {}
        return Completion(blocks=blocks, provider=self.name, model_requested=model,
                          model_served=model + "-served", temperature_requested=temperature,
                          temperature_effective=temperature,
                          usage={"input": self.usage[0], "output": self.usage[1]},
                          timestamp_utc="2026-10-26T00:00:00Z",
                          stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks)
                          else "end_turn", provider_meta=meta)
    history_assistant = staticmethod(OpenRouterProvider.history_assistant)
    history_tool_results = staticmethod(OpenRouterProvider.history_tool_results)


def client_for(fake, **kw):
    c = StripeClient(key=FAKE_KEY, base=fake.base, min_interval=0, sleep=lambda s: None, **kw)
    return c


class _Case(unittest.TestCase):
    """run_real.launch against the fake Stripe with a fake model."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.fake = FakeStripe().__enter__()
        self.saved = (run_real._client_for, run_real._anthropic_client, run_real._stripe_client,
                      run_real.LIMIT, run_real.BUDGET_USD, run_real.OR_BUDGET_USD,
                      os.environ.get("STRIPE_SECRET_KEY"))
        os.environ["STRIPE_SECRET_KEY"] = FAKE_KEY
        run_real._anthropic_client = lambda models: None
        run_real._stripe_client = lambda cfg: client_for(self.fake, api_version=cfg["stripe_api_version"])
        run_real.LIMIT = 0
        run_real.BUDGET_USD = run_real.OR_BUDGET_USD = 0.0
        self.providers = {}

    def tearDown(self):
        (run_real._client_for, run_real._anthropic_client, run_real._stripe_client,
         run_real.LIMIT, run_real.BUDGET_USD, run_real.OR_BUDGET_USD, k) = self.saved
        if k is None:
            os.environ.pop("STRIPE_SECRET_KEY", None)
        else:
            os.environ["STRIPE_SECRET_KEY"] = k
        import providers
        providers.RECORD_COST = False
        self.fake.__exit__()
        self.tmp.cleanup()

    def use(self, factory):
        def cf(model, _c):
            if model not in self.providers:
                self.providers[model] = factory(model)
            return self.providers[model]
        run_real._client_for = cf

    def cfg(self, **over):
        c = _jload((os.path.join(HERE, "configs", "real_stripe.json")))
        c.update({"models": ["claude-haiku-4-5-20251001"], "n_tasks": 2, "samples": 1,
                  "out_dir": os.path.join(self.d, "real_stripe")})
        c.update(over)
        p = os.path.join(self.d, "cfg.json")
        with open(p, "w") as f:
            json.dump(c, f)
        return run_real.load_config(p)

    def run_cfg(self, cfg, models=None):
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                rc = run_real.launch(cfg, models)
        finally:
            release_locks()
        self.out = buf.getvalue()
        import tier1
        return rc, {r["episode_id"]: r for r in tier1.read_rows(cfg["raw_path"])}

    def one_cell(self, cell, policy, restart=False):
        if restart:
            c = self.cfg(cells=[], restart_cells=[cell])
        else:
            c = self.cfg(cells=[cell], restart_cells=[])
        self.use(lambda m: FakeProvider(policy))
        rc, rows = self.run_cfg(c)
        self.assertEqual(rc, 0, self.out)
        self.assertEqual(len(rows), 2)
        return c, list(rows.values())


# ---------------------------------------------------------------------------
# key safety
# ---------------------------------------------------------------------------

class TestKeySafety(unittest.TestCase):
    def test_client_refuses_non_test_keys(self):
        for k in ("sk_live_abc", "rk_test_abc", "rk_live_abc", "", "sk_test_", None, "pk_test_x"):
            with self.subTest(k=k):
                with self.assertRaises(UnsafeKeyError):
                    StripeClient(key=k if k is not None else 123)

    def test_env_key_checked_when_not_passed(self):
        old = os.environ.get("STRIPE_SECRET_KEY")
        try:
            os.environ["STRIPE_SECRET_KEY"] = "sk_live_123"
            with self.assertRaises(UnsafeKeyError):
                StripeClient()
        finally:
            if old is None:
                os.environ.pop("STRIPE_SECRET_KEY", None)
            else:
                os.environ["STRIPE_SECRET_KEY"] = old

    def test_cli_aborts_with_exit_5_and_never_prints_key(self):
        for k in ("sk_live_SUPERSECRET999", "rk_test_SUPERSECRET999", ""):
            env = dict(os.environ, STRIPE_SECRET_KEY=k)
            for cmd in ("--launch", "--stripe-check", "--verify-search"):
                p = subprocess.run([sys.executable, os.path.join(HERE, "src_real", "run_real.py"),
                                    "--config", os.path.join(HERE, "configs", "real_stripe.json"), cmd],
                                   env=env, capture_output=True, text=True, timeout=60)
                self.assertEqual(p.returncode, 5, (k, cmd, p.stdout, p.stderr))
                self.assertNotIn("SUPERSECRET", p.stdout + p.stderr)
                self.assertIn("sk_test_", p.stdout)

    def test_repr_and_errors_redact_key(self):
        c = StripeClient(key=FAKE_KEY, base="http://127.0.0.1:9", max_retries=0,
                         sleep=lambda s: None, timeout=1)
        self.assertNotIn(FAKE_KEY, repr(c))
        with self.assertRaises(StripeInfraError) as cm:
            c.request("GET", "/v1/payment_intents", {"customer": FAKE_KEY})
        self.assertNotIn(FAKE_KEY, str(cm.exception))

    def test_live_mode_object_aborts(self):
        with FakeStripe() as fake:
            fake.livemode = True
            c = client_for(fake)
            with self.assertRaises(UnsafeKeyError):
                c.create_customer("a1", {})

    def test_launch_stops_on_live_object(self):
        case = _Case("run")
        case.setUp()
        try:
            case.fake.livemode = True
            c = case.cfg()
            case.use(lambda m: FakeProvider(PayPolicy()))
            rc, rows = case.run_cfg(c)
            self.assertEqual((rc, len(rows)), (5, 0))
        finally:
            case.tearDown()


# ---------------------------------------------------------------------------
# fake Stripe + client: idempotency semantics, backoff, transport retries
# ---------------------------------------------------------------------------

def _params(cus, amount=43700, att="a1", **md):
    return {"amount": amount, "currency": "usd", "confirm": True, "payment_method": "pm_card_visa",
            "customer": cus, "automatic_payment_methods": {"enabled": True, "allow_redirects": "never"},
            "metadata": dict({"attempt_id": att, "payee": "vendor-01", "memo": "m1"}, **md)}


class TestFakeStripeAndClient(unittest.TestCase):
    def setUp(self):
        self.fake = FakeStripe().__enter__()
        self.c = client_for(self.fake)
        self.cus = self.c.create_customer("a1", {"attempt_id": "a1"})["id"]

    def tearDown(self):
        self.fake.__exit__()

    def test_same_key_is_replayed_not_charged_twice(self):
        p1, r1 = self.c.create_payment_intent(_params(self.cus), "K1")
        p2, r2 = self.c.create_payment_intent(_params(self.cus), "K1")
        self.assertEqual((p1["id"], r1, r2), (p2["id"], False, True))
        self.assertEqual(len(self.fake.committed("a1")), 1)
        self.assertEqual(p1["status"], "succeeded")

    def test_same_key_other_params_refused(self):
        self.c.create_payment_intent(_params(self.cus), "K1")
        with self.assertRaises(StripeAPIError) as cm:
            self.c.create_payment_intent(_params(self.cus, amount=100), "K1")
        self.assertEqual(cm.exception.type, "idempotency_error")
        self.assertEqual(len(self.fake.committed("a1")), 1)

    def test_no_key_twice_charges_twice(self):
        self.c.create_payment_intent(_params(self.cus), "K1")
        self.c.create_payment_intent(_params(self.cus), "K2")
        self.assertEqual(len(self.fake.committed("a1")), 2)

    def test_post_requires_key(self):
        with self.assertRaises(ValueError):
            self.c.request("POST", "/v1/payment_intents", _params(self.cus))

    def test_429_backoff_then_single_charge(self):
        self.fake.rate_limit_next = 3
        self.c.create_payment_intent(_params(self.cus), "K1")
        self.assertEqual(self.c.stats["retries_429"], 3)
        self.assertEqual(len(self.fake.committed("a1")), 1)

    def test_dropped_connection_after_commit_is_replayed_not_recharged(self):
        self.fake.drop_after_commit_next = 1
        p, replayed = self.c.create_payment_intent(_params(self.cus), "K1")
        self.assertTrue(replayed)
        self.assertGreaterEqual(self.c.stats["retries_transport"], 1)
        self.assertEqual(len(self.fake.committed("a1")), 1)

    def test_stripe_500_is_infra_error(self):
        self.fake.error_next = 1
        with self.assertRaises(StripeInfraError):
            self.c.create_payment_intent(_params(self.cus), "K1")

    def test_list_paginates_and_is_newest_first(self):
        for i in range(105):
            self.c.create_payment_intent(_params(self.cus, memo=f"m{i}"), f"K{i}")
        got = self.c.list_payment_intents(self.cus)
        self.assertEqual(len(got), 105)
        self.assertEqual(got[0]["metadata"]["memo"], "m104")

    def test_search_by_metadata(self):
        self.c.create_payment_intent(_params(self.cus, att="zz"), "K1")
        self.assertEqual(len(self.c.search_payment_intents('metadata["attempt_id"]:"zz"')), 1)

    def test_amount_parsing(self):
        self.assertEqual(amount_to_cents("437.00"), 43700)
        self.assertEqual(amount_to_cents("$1,200.5"), 120050)
        self.assertEqual(amount_to_cents(437), 43700)
        for bad in ("abc", "0.10", "1.234", "-5"):
            with self.assertRaises(ToolCallError):
                amount_to_cents(bad)

    def test_key_namespacing(self):
        a, b = stripe_key_for("att1", "k"), stripe_key_for("att2", "k")
        self.assertNotEqual(a, b)
        self.assertEqual(a, stripe_key_for("att1", "k"))
        self.assertLessEqual(len(stripe_key_for("att1", "x" * 400)), 255)


# ---------------------------------------------------------------------------
# StripeWorld: failure injection, key forwarding, ground truth
# ---------------------------------------------------------------------------

class TestStripeWorld(unittest.TestCase):
    def setUp(self):
        self.fake = FakeStripe().__enter__()
        self.c = client_for(self.fake)
        self.cus = self.c.create_customer("att", {})["id"]

    def tearDown(self):
        self.fake.__exit__()

    def w(self, mit="none", fm="none"):
        return StripeWorld(self.c, episode_id="ep", attempt_id="att", customer=self.cus,
                           mitigation=mit, failure_mode=fm)

    A = {"payee": "vendor-01", "amount": "437.00", "memo": "inv-1"}
    G = [Goal("send_payment", dict(A))]

    def test_after_commit_modes_charge_then_fail(self):
        for fm, exc, text in (("timeout_after_commit", ToolTimeout, "request timed out"),
                              ("error_500_after_commit", ToolServerError, "internal server error"),
                              ("status_unknown_after_commit", ToolTimeout, "outcome is unknown")):
            with self.subTest(fm=fm):
                cus = self.c.create_customer("att-" + fm, {})["id"]
                w = StripeWorld(self.c, episode_id="ep", attempt_id="att-" + fm, customer=cus,
                                failure_mode=fm)
                with self.assertRaises(exc) as cm:
                    w.call("send_payment", dict(self.A), "", "t")
                self.assertIn(text, str(cm.exception))
                self.assertEqual(len(self.fake.committed("att-" + fm)), 1)   # charge exists
                self.assertTrue(w.call("send_payment", dict(self.A), "", "t").startswith("ok-pi_"))
                gt = stripe_ground_truth(self.c, cus, "att-" + fm, self.G)
                self.assertEqual((gt["violations"], gt["per_goal"]), (1, [2]))
                self.assertEqual(w.ledger.violations("t"), 1)

    def test_before_commit_sends_nothing(self):
        w = self.w(fm="timeout_before_commit")
        n0 = len(self.fake.requests)
        with self.assertRaises(ToolTimeout):
            w.call("send_payment", dict(self.A), "", "t")
        self.assertEqual(len(self.fake.requests), n0)
        self.assertEqual(len(self.fake.committed("att")), 0)
        w.call("send_payment", dict(self.A), "", "t")
        gt = stripe_ground_truth(self.c, self.cus, "att", self.G)
        self.assertEqual((gt["violations"], gt["achieved"]), (0, True))

    def test_idem_keys_forwarded_stripe_dedups(self):
        w = self.w(mit="idem_keys", fm="error_500_after_commit")
        with self.assertRaises(ToolServerError):
            w.call("send_payment", dict(self.A), "my-key", "t")
        out = w.call("send_payment", dict(self.A), "my-key", "t")
        self.assertTrue(out.startswith("ok-"))
        self.assertTrue(w.events[-1]["replayed"])
        self.assertEqual(len(self.fake.committed("att")), 1)      # Stripe suppressed it
        self.assertEqual(stripe_ground_truth(self.c, self.cus, "att", self.G)["violations"], 0)
        self.assertEqual(w.ledger.violations("t"), 0)               # simulator agrees

    def test_idem_keys_new_key_charges_twice(self):
        w = self.w(mit="idem_keys")
        w.call("send_payment", dict(self.A), "k1", "t")
        w.call("send_payment", dict(self.A), "k2", "t")
        self.assertEqual(len(self.fake.committed("att")), 2)

    def test_none_mode_does_not_forward_keys(self):
        w = self.w(mit="none")
        w.call("send_payment", dict(self.A), "same", "t")
        w.call("send_payment", dict(self.A), "same", "t")
        self.assertEqual(len(self.fake.committed("att")), 2)
        self.assertFalse(any(e.get("forwarded_key") for e in w.events))
        self.assertEqual(w.ledger.violations("t"), 1)

    def test_key_reuse_with_other_args_is_a_real_error(self):
        w = self.w(mit="idem_keys")
        w.call("send_payment", dict(self.A), "k", "t")
        with self.assertRaises(ToolServerError) as cm:
            w.call("send_payment", dict(self.A, amount="438.00"), "k", "t")
        self.assertIn("idempotency_error", str(cm.exception))
        self.assertEqual(len(self.fake.committed("att")), 1)

    def test_invalid_calls_do_not_consume_failure_or_send(self):
        w = self.w(fm="timeout_after_commit")
        n0 = len(self.fake.requests)
        for tool, args in (("send_email", {"to": "a", "subject": "b"}),
                           ("send_payment", {"payee": "v", "amount": "abc", "memo": "m"}),
                           ("send_payment", {"payee": "v", "memo": "m"})):
            with self.assertRaises(ToolCallError):
                w.call(tool, args, "", "t")
        self.assertEqual(len(self.fake.requests), n0)
        with self.assertRaises(ToolTimeout):
            w.call("send_payment", dict(self.A), "", "t")

    def test_list_effects_is_real_readback(self):
        w = self.w()
        self.assertEqual(w.call("list_effects", {}, "", "t"), "(no effects recorded)")
        w.call("send_payment", dict(self.A, amount="$437"), "", "t")
        self.assertEqual(w.call("list_effects", {}, "", "t"),
                         "send_payment(payee=vendor-01, amount=437.00, memo=inv-1)")

    def test_ground_truth_counts_only_this_attempt_and_normalizes(self):
        w = self.w()
        w.call("send_payment", dict(self.A), "", "t")
        w.call("send_payment", dict(self.A, amount="437", payee="Vendor-01 "), "", "t")
        other = StripeWorld(self.c, episode_id="ep", attempt_id="OTHER", customer=self.cus)
        other.call("send_payment", dict(self.A), "", "t")
        other.call("send_payment", dict(self.A), "", "t")
        gt = stripe_ground_truth(self.c, self.cus, "att", self.G)
        self.assertEqual((gt["n_committed"], gt["violations"], gt["per_goal"]), (2, 1, [2]))
        self.assertTrue(gt["ivr_hit"])

    def test_order_violation_from_stripe_order(self):
        B = {"payee": "vendor-02", "amount": "100.00", "memo": "inv-2"}
        w = self.w()
        for a in (B, self.A):
            w.call("send_payment", dict(a), "", "t")
        gt = stripe_ground_truth(self.c, self.cus, "att", self.G + [Goal("send_payment", dict(B))],
                                 ordered=True)
        self.assertTrue(gt["order_violation"])


# ---------------------------------------------------------------------------
# end to end through run_real.launch (real code path, fake Stripe + model)
# ---------------------------------------------------------------------------

class TestEndToEnd(_Case):
    def check_agree(self, c, rows):
        self.assertTrue(all(r["agree"] and r["agree_exact"] for r in rows))
        gts = _jlines(c["gt_path"])
        self.assertEqual(len(gts), len(rows))

    def test_timeout_after_commit_naive_retry_duplicates(self):
        c, rows = self.one_cell(["base", "none", "timeout_after_commit"], PayPolicy())
        self.assertTrue(all(r["ivr_stripe"] and r["violations_stripe"] == 1 for r in rows))
        self.assertTrue(all(r["retried"] and r["failure_fired"] for r in rows))
        self.assertEqual(len(self.fake.committed()), 4)
        self.check_agree(c, rows)
        self.assertEqual(self.providers["claude-haiku-4-5-20251001"].last_tools,
                         ["send_payment", "list_effects"])

    def test_error_500_after_commit(self):
        c, rows = self.one_cell(["base", "none", "error_500_after_commit"], PayPolicy())
        self.assertTrue(all(r["ivr_stripe"] for r in rows))
        self.check_agree(c, rows)

    def test_status_unknown_with_readback_no_duplicate(self):
        c, rows = self.one_cell(["base", "none", "status_unknown_after_commit"],
                                PayPolicy(readback_on_error=True))
        self.assertTrue(all(not r["ivr_stripe"] and r["readback_after_failure"] for r in rows))
        self.assertTrue(all(r["achieved_stripe"] for r in rows))
        self.check_agree(c, rows)

    def test_idem_keys_stable_key_stripe_suppresses_second_charge(self):
        c, rows = self.one_cell(["base", "idem_keys", "error_500_after_commit"],
                                PayPolicy(key="stable"))
        self.assertTrue(all(r["retried"] and r["retry_key_category"] == "same_key" for r in rows))
        self.assertTrue(all(not r["ivr_stripe"] and r["replayed"] == 1 for r in rows))
        self.assertTrue(all(r["forwarded_keys"] == 2 for r in rows))
        self.assertEqual(len(self.fake.committed()), 2)       # one per episode
        self.check_agree(c, rows)

    def test_idem_keys_keyless_model_still_duplicates(self):
        c, rows = self.one_cell(["base", "idem_keys", "error_500_after_commit"], PayPolicy())
        self.assertTrue(all(r["ivr_stripe"] and r["forwarded_keys"] == 0 for r in rows))
        self.check_agree(c, rows)

    def test_before_commit_timeout_retry_is_safe(self):
        c, rows = self.one_cell(["base", "none", "timeout_before_commit"], PayPolicy())
        self.assertTrue(all(not r["ivr_stripe"] and r["achieved_stripe"] for r in rows))
        self.check_agree(c, rows)

    def test_restart_amnesiac_pays_A_twice(self):
        c, rows = self.one_cell("amnesiac", PayPolicy(hint_readback=True), restart=True)
        for r in rows:
            self.assertTrue(r["interrupted"])
            self.assertTrue(r["ivr_stripe"])                 # A paid in both sessions
            self.assertEqual(r["violations_stripe"], 1)
            self.assertTrue(r["achieved_stripe"])
            self.assertFalse(r["s2_lookup"])
        self.check_agree(c, rows)

    def test_restart_bare_hint_wording_and_readback(self):
        import sweep2
        c, rows = self.one_cell("bare_hint", PayPolicy(hint_readback=True), restart=True)
        for r in rows:
            self.assertFalse(r["ivr_stripe"])
            self.assertTrue(r["s2_lookup_first"] and r["achieved_stripe"])
            tr = _jload((os.path.join(c["transcript_dir"], r["episode_id"].replace("/", "_") + ".json")))
            g = tasks_real.payment_plan(int(r["task_id"][-2:]), r["episode_id"])
            self.assertEqual(tr["sessions"][1]["user_msg"], sweep2.resume_user_msg(g, "bare_hint", set()))
            self.assertIn("NOTE: a previous session was working on this task and was interrupted. "
                          "Finish the task.", tr["sessions"][1]["user_msg"])
        self.check_agree(c, rows)

    def test_full_grid_plan_and_summary(self):
        c = self.cfg(models=["claude-haiku-4-5-20251001", "openrouter:qwen/qwen3.6-27b-20260422"])
        self.use(lambda m: FakeProvider(PayPolicy(), cost=0.001))
        rc, rows = self.run_cfg(c)
        self.assertEqual(rc, 0, self.out)
        self.assertEqual(len(rows), 2 * (4 + 2) * 2)
        summ = _csv(c["summary_path"])
        self.assertEqual(len(summ), 12)
        self.assertTrue(all(int(s["agree"]) == int(s["n_episodes"]) for s in summ))
        full = run_real.load_config(os.path.join(HERE, "configs", "real_stripe.json"))
        self.assertEqual(len(run_real.plan(full)), 576)
        self.assertEqual(len({run_real.eid_of(e) for e in run_real.plan(full)}), 576)

    def test_unique_reference_per_episode(self):
        c = self.cfg(n_tasks=2, samples=2)
        self.use(lambda m: FakeProvider(PayPolicy()))
        self.run_cfg(c)
        memos = [p["metadata"]["memo"] for p in self.fake.pis]
        per_attempt = {}
        for p in self.fake.pis:
            per_attempt.setdefault(p["metadata"]["attempt_id"], set()).add(p["metadata"]["memo"])
        flat = [m for s in per_attempt.values() for m in s]
        self.assertEqual(len(flat), len(set(flat)))          # no memo shared across episodes
        self.assertTrue(all(p["metadata"]["ibench"] == "benchmark-real-v1" for p in self.fake.pis))
        self.assertTrue(all("episode_id" in p["metadata"] and "task_id" in p["metadata"]
                            and "reference" in p["metadata"] for p in self.fake.pis))

    def test_key_never_written_to_outputs(self):
        c = self.cfg()
        self.use(lambda m: FakeProvider(PayPolicy()))
        self.run_cfg(c)
        blob = ""
        for f in glob.glob(os.path.join(c["out_dir"], "**", "*"), recursive=True):
            if os.path.isfile(f):
                with open(f, errors="replace") as fh:
                    blob += fh.read()
        self.assertGreater(len(blob), 1000)
        self.assertNotIn(FAKE_KEY, blob)
        self.assertNotIn("FAKEKEY", self.out)


class TestResumeBudgetFailures(_Case):
    def test_resume_no_duplicates(self):
        c = self.cfg()
        self.use(lambda m: FakeProvider(PayPolicy()))
        run_real.LIMIT = 5
        rc, rows = self.run_cfg(c)
        self.assertEqual(len(rows), 5)
        run_real.LIMIT = 0
        rc, rows = self.run_cfg(c)
        self.assertEqual((rc, len(rows)), (0, 12))
        rc, rows = self.run_cfg(c)
        self.assertEqual((rc, len(rows)), (0, 12))
        self.assertIn("0 remaining", self.out)

    def test_crashed_attempt_not_counted_on_rerun(self):
        c = self.cfg(cells=[["base", "none", "timeout_after_commit"]], restart_cells=[], n_tasks=1)
        # model crashes after the first (committed) payment: no row, orphan charge
        self.use(lambda m: FakeProvider(PayPolicy(), fail="connection reset", fail_after=1))
        rc, rows = self.run_cfg(c)
        self.assertEqual((rc, len(rows)), (1, 0))
        self.assertEqual(len(self.fake.committed()), 1)
        self.providers.clear()
        self.use(lambda m: FakeProvider(PayPolicy()))
        rc, rows = self.run_cfg(c)
        self.assertEqual((rc, len(rows)), (0, 1))
        r = list(rows.values())[0]
        self.assertEqual(r["n_committed_stripe"], 2)   # this attempt: original + retry
        self.assertEqual(len(self.fake.committed()), 3)  # + the orphan, never counted

    def test_budget_stop_then_resume(self):
        # Haiku $1/$5: 3 turns x (100k in, 10k out) = $0.45 per episode
        c = self.cfg(cells=[["base", "none", "timeout_after_commit"]], restart_cells=[], n_tasks=4)
        self.use(lambda m: FakeProvider(PayPolicy(), usage=(100_000, 10_000)))
        run_real.BUDGET_USD = 1.5
        rc, rows = self.run_cfg(c)
        self.assertEqual((rc, len(rows)), (2, 3))
        run_real.BUDGET_USD = 0.0
        rc, rows = self.run_cfg(c)
        self.assertEqual((rc, len(rows)), (0, 4))

    def test_openrouter_402_stops_only_openrouter(self):
        c = self.cfg(models=["openrouter:qwen/qwen3.6-27b-20260422", "claude-haiku-4-5-20251001"],
                     cells=[["base", "none", "timeout_after_commit"]], restart_cells=[])
        self.use(lambda m: FakeProvider(PayPolicy(), fail="OpenRouter HTTP 402: limit")
                 if m.startswith("openrouter:") else FakeProvider(PayPolicy()))
        rc, rows = self.run_cfg(c)
        self.assertEqual(rc, 3)
        self.assertEqual({r["model"] for r in rows.values()}, {"claude-haiku-4-5-20251001"})

    def test_stripe_down_stops_with_6(self):
        c = self.cfg()
        self.use(lambda m: FakeProvider(PayPolicy()))
        os.environ["BENCH_REAL_STRIPE_SLEEP"] = "0"
        self.fake.error_next = 100
        try:
            rc, rows = self.run_cfg(c)
        finally:
            os.environ.pop("BENCH_REAL_STRIPE_SLEEP", None)
        self.assertEqual((rc, len(rows)), (6, 0))

    def test_per_model_lock(self):
        import runlock
        c = self.cfg()
        self.assertTrue(runlock.acquire(c["out_dir"], "real", c["models"]))
        try:
            self.use(lambda m: FakeProvider(PayPolicy()))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = run_real.launch(c)
            self.assertEqual(rc, 0)
            self.assertIn("already running", buf.getvalue())
            self.assertFalse(os.path.exists(c["raw_path"]))
        finally:
            release_locks()


class TestChecksAndAnalysis(_Case):
    def test_stripe_check_round_trip(self):
        c = self.cfg()
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_real.stripe_check(c)
        self.assertEqual(rc, 0, buf.getvalue())
        res = _jload((c["check_path"]))
        self.assertTrue(res["ok"] and res["second_replayed"] and res["replay_same_id"])
        self.assertTrue(res["key_reuse_other_params"].startswith("refused"))

    def test_verify_search(self):
        c = self.cfg()
        self.use(lambda m: FakeProvider(PayPolicy()))
        self.run_cfg(c)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = run_real.verify_search(c, min_age_s=0)
        self.assertEqual(rc, 0, buf.getvalue())
        chk = _jlines(c["search_path"])
        self.assertEqual(len(chk), 12)
        self.assertTrue(all(x["agree"] for x in chk))

    def test_analysis_numbers(self):
        c = self.cfg()
        self.use(lambda m: FakeProvider(PayPolicy(readback_on_error=True)))
        self.run_cfg(c)
        sys.path.insert(0, os.path.join(HERE, "analysis"))
        import real_stripe
        sim = {"v2.haiku.base.none.toA:ivr": {"value": 0.9, "ci": [0.75, 0.97]},
               "e1_6.haiku.base.none.w3:ivr": {"value": 0.06, "ci": [0.0, 0.2]}}
        out = os.path.join(self.d, "real_stripe.json")
        nums = real_stripe.build(c, sim_numbers=sim, out_path=out)
        k = "real.haiku.base.none.toA:ivr"
        self.assertIn(k, nums)
        self.assertEqual(nums[k]["value"], 0.0)            # read-back policy never duplicates
        self.assertEqual(nums[k]["sim_key"], "v2.haiku.base.none.toA:ivr")
        self.assertFalse(nums[k]["within_sim_ci"])
        self.assertAlmostEqual(nums[k]["d"], -0.9)
        self.assertEqual(nums[k]["T"], 2)
        self.assertEqual(nums["real.haiku.base.none.w3:ivr"]["within_sim_ci"], True)
        self.assertIsNone(nums["real.haiku.base.keys.e500:ivr"]["within_sim_ci"])  # no sim number given
        self.assertIn("real.haiku.restart.bare_hint:givr", nums)
        self.assertEqual(nums["real.agreement.ledger_vs_stripe"]["value"], 1.0)
        self.assertTrue(_jload((out)))


if __name__ == "__main__":
    unittest.main(verbosity=1)
