"""Tests for the provider layer and the provider-agnostic episode loop.
No network: uses a scripted fake provider."""
import json, sys, unittest
sys.path.insert(0, "src")
from providers import (Completion, tools_to_openai, messages_to_openai,
                       blocks_from_openai_message, OpenRouterProvider)
from model_agent import run_model_episode, _tool_schemas, BASE_SYSTEM
from world import World
from tasks import generate_tasks


class FakeProvider:
    """Replays a script of block-lists; records what it was sent.
    A script entry may be (blocks, stop_reason) to set the stop reason."""
    name = "fake"
    def __init__(self, script):
        self.script, self.sent = list(script), []
    def complete(self, model, system, tools, messages, max_tokens, temperature):
        self.sent.append(json.loads(json.dumps(messages)))
        blocks = self.script.pop(0) if self.script else [{"type": "text", "text": "DONE"}]
        stop = None
        if isinstance(blocks, tuple):
            blocks, stop = blocks
        kw = {"stop_reason": stop} if stop else {}
        return Completion(blocks=blocks, provider=self.name, model_requested=model,
                          model_served=model + "-2026-01-01", temperature_requested=temperature,
                          temperature_effective=None, usage={"input": 10, "output": 5},
                          timestamp_utc="t", **kw)
    history_assistant = staticmethod(OpenRouterProvider.history_assistant)
    history_tool_results = staticmethod(OpenRouterProvider.history_tool_results)


class TestConversions(unittest.TestCase):
    def test_tool_schema_roundtrip(self):
        t = tools_to_openai(_tool_schemas())
        self.assertTrue(all(x["type"] == "function" for x in t))
        names = {x["function"]["name"] for x in t}
        self.assertIn("list_effects", names)
        self.assertIn("parameters", t[0]["function"])

    def test_messages_to_openai_tool_cycle(self):
        hist = [{"role": "user", "content": "do it"},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "tool_use", "id": "c1", "name": "send_email",
                     "input": {"to": "a@example.com"}}]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "c1", "content": "ok-1"}]}]
        out = messages_to_openai("SYS", hist)
        self.assertEqual(out[0], {"role": "system", "content": "SYS"})
        self.assertEqual(out[2]["tool_calls"][0]["function"]["name"], "send_email")
        self.assertEqual(json.loads(out[2]["tool_calls"][0]["function"]["arguments"]),
                         {"to": "a@example.com"})
        self.assertEqual(out[3], {"role": "tool", "tool_call_id": "c1", "content": "ok-1"})

    def test_blocks_from_openai_bad_json_args_do_not_crash(self):
        msg = {"content": None, "tool_calls": [{"id": "x", "type": "function",
               "function": {"name": "file_ticket", "arguments": "{not json"}}]}
        b = blocks_from_openai_message(msg)
        self.assertEqual(b[0]["type"], "tool_use")
        self.assertIn("_unparsed_arguments", b[0]["input"])


class TestEpisodeLoopProviderAgnostic(unittest.TestCase):
    def test_duplicate_retry_scored_as_violation_via_fake_provider(self):
        task = next(t for t in generate_tasks() if t.family == "atomic")
        tool, args = task.tool, dict(task.args)
        # Script: call effectful tool, see error, retry identically, DONE.
        script = [[{"type": "tool_use", "id": "a", "name": tool, "input": args}],
                  [{"type": "tool_use", "id": "b", "name": tool, "input": args}],
                  [{"type": "text", "text": "DONE"}]]
        prov = FakeProvider(script)
        world = World(failure_mode="error_500_after_commit", mitigation="none")
        tr = run_model_episode(prov, world, task, BASE_SYSTEM, model="fake-model")
        self.assertEqual(tr["provider"], "fake")
        self.assertEqual(tr["model_served"], "fake-model-2026-01-01")
        self.assertIn("started_utc", tr)
        self.assertEqual(tr["usage"], {"input": 30, "output": 15})
        # second call saw the error from the first in history
        self.assertEqual(prov.sent[1][-1]["content"][0]["type"], "tool_result")
        self.assertTrue(prov.sent[1][-1]["content"][0].get("is_error"))
        self.assertGreaterEqual(world.ledger.violations(task.task_id), 1)



class TestResumeSessionLoop(unittest.TestCase):
    def test_interrupted_session_returns_meta_and_cuts_after_commit(self):
        sys.path.insert(0, "src2")
        import sweep2
        g, k = sweep2.benchmark_graphs()[0]
        tool, args = g.nodes[k].tool, dict(g.nodes[k].args)
        prov = FakeProvider([[{"type": "tool_use", "id": "a", "name": tool, "input": args}]])
        world = World(mitigation="none", failure_mode="none")
        turns, interrupted, acked, meta = sweep2.run_model_session(
            prov, "fake-model", world, g, sweep2.BASE_SYSTEM, g.instruction,
            interrupt_node=k)
        self.assertTrue(interrupted)
        self.assertEqual(turns[-1]["role"], "interrupt")
        self.assertEqual(meta["provider"], "fake")
        self.assertEqual(meta["usage"], {"input": 10, "output": 5})


class _FakeOpenRouter(OpenRouterProvider):
    """OpenRouterProvider with _post replaced by a canned response queue."""
    def __init__(self, responses):
        super().__init__(api_key="test-key")
        self.responses, self.payloads = list(responses), []
    def _post(self, payload):
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)


def _or_response(message, finish="stop", **top):
    return {"model": "org/m-2026", "choices": [{"message": message,
            "finish_reason": finish, "native_finish_reason": finish.upper()}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2}, **top}


class TestRound2ProviderFixes(unittest.TestCase):
    def test_unknown_tool_returns_error_and_episode_continues(self):
        # audit F5: world.call raised ValueError -> whole sweep crashed
        task = next(t for t in generate_tasks() if t.family == "atomic")
        script = [[{"type": "tool_use", "id": "a", "name": "file_ticket",
                    "input": {"title": "x"}}],
                  [{"type": "tool_use", "id": "b", "name": task.tool,
                    "input": dict(task.args)}],
                  [{"type": "text", "text": "DONE"}]]
        world = World(failure_mode="none", mitigation="none")
        tr = run_model_episode(FakeProvider(script), world, task, BASE_SYSTEM,
                               model="fake-model")
        res = tr["turns"][1]["content"][0]
        self.assertTrue(res["is_error"])
        self.assertIn("unknown tool", res["content"])
        self.assertEqual(tr["invalid_calls"][0]["name"], "file_ticket")
        self.assertEqual(len(world.ledger.for_task(task.task_id)), 1)
        self.assertEqual(world.ledger.violations(task.task_id), 0)

    def test_malformed_args_error_not_committed_not_duplicate(self):
        # audit F5: two unparsable calls committed and scored as a duplicate
        task = next(t for t in generate_tasks() if t.tool == "send_email")
        bad = {"content": None, "tool_calls": [{"id": "x", "type": "function",
               "function": {"name": "send_email", "arguments": '{"to": "a@b'}}]}
        partial = {"content": None, "tool_calls": [{"id": "y", "type": "function",
                   "function": {"name": "send_email", "arguments": {"to": "a@b.com"}}}]}
        script = [blocks_from_openai_message(bad), blocks_from_openai_message(partial),
                  [{"type": "text", "text": "DONE"}]]
        world = World(failure_mode="none", mitigation="none")
        tr = run_model_episode(FakeProvider(script), world, task, BASE_SYSTEM,
                               model="fake-model")
        self.assertEqual(world.ledger.for_task(task.task_id), [])
        self.assertEqual(world.ledger.violations(task.task_id), 0)
        self.assertEqual(len(tr["invalid_calls"]), 2)

    def test_dict_arguments_accepted(self):
        msg = {"content": None, "tool_calls": [{"id": "x", "type": "function",
               "function": {"name": "send_email",
                            "arguments": {"to": "a@b.com", "subject": "s"}}}]}
        self.assertEqual(blocks_from_openai_message(msg)[0]["input"],
                         {"to": "a@b.com", "subject": "s"})

    def test_max_tokens_stop_without_tool_call_is_flagged_truncated(self):
        # audit F7: a cut-off reply was indistinguishable from DONE
        task = next(t for t in generate_tasks() if t.family == "atomic")
        world = World(failure_mode="none", mitigation="none")
        tr = run_model_episode(FakeProvider([([], "max_tokens")]), world, task,
                               BASE_SYSTEM, model="fake-model")
        self.assertTrue(tr["truncated"])
        self.assertEqual(tr["turns"][-1]["stop_reason"], "max_tokens")
        world = World(failure_mode="none", mitigation="none")
        tr = run_model_episode(FakeProvider([([{"type": "text", "text": "DONE"}],
                                              "end_turn")]),
                               world, task, BASE_SYSTEM, model="fake-model")
        self.assertFalse(tr["truncated"])

    def test_openrouter_records_finish_reason(self):
        prov = _FakeOpenRouter([_or_response({"content": ""}, finish="length")])
        c = prov.complete("openrouter:org/m", "S", [], [{"role": "user", "content": "hi"}],
                          16, 0.0)
        self.assertEqual(c.stop_reason, "length")
        self.assertEqual(c.provider_meta["native_finish_reason"], "LENGTH")

    def test_openrouter_temperature_is_unverified_not_claimed(self):
        # audit F8: OpenRouter drops unsupported params silently; sending
        # T=0 must not be recorded as effective T=0.
        prov = _FakeOpenRouter([_or_response({"content": "DONE"})])
        c = prov.complete("openrouter:org/m", "S", [], [{"role": "user", "content": "hi"}],
                          16, 0.0)
        self.assertEqual(prov.payloads[0]["temperature"], 0.0)
        self.assertEqual(c.temperature_effective, "unverified")
        self.assertEqual(c.provider_meta["temperature_sent"], 0.0)

    def test_openrouter_reasoning_details_sent_back_next_turn(self):
        # audit F9: reasoning_details (Gemini thought signatures) were dropped
        rd = [{"type": "reasoning.encrypted", "data": "SIG", "id": "r1",
               "format": "google-gemini-v1", "index": 0}]
        call = {"content": None, "reasoning_details": rd, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "list_effects", "arguments": "{}"}}]}
        prov = _FakeOpenRouter([_or_response(call, finish="tool_calls"),
                                _or_response({"content": "DONE"})])
        task = next(t for t in generate_tasks() if t.family == "atomic")
        run_model_episode(prov, World(), task, BASE_SYSTEM, model="openrouter:org/m")
        second = prov.payloads[1]["messages"]
        asst = [m for m in second if m["role"] == "assistant"][0]
        self.assertEqual(asst["reasoning_details"], rd)
        self.assertEqual(asst["tool_calls"][0]["id"], "c1")

class TestAnthropicNewSdkTemperature(unittest.TestCase):
    """New anthropic SDKs drop `temperature` from messages.create's signature.
    The provider must still send it (raw body) and keep the old fallback."""

    def _client(self, rejects_temperature):
        import anthropic
        calls = []

        class TempRejected(anthropic.BadRequestError):  # no HTTP objects needed
            def __init__(self):
                Exception.__init__(self, "temperature is deprecated for this model")

        class Msgs:
            def create(self, *, model, max_tokens, system, tools, messages,
                       extra_body=None):          # no `temperature` kwarg
                calls.append(dict(extra_body or {}))
                if rejects_temperature and extra_body and "temperature" in extra_body:
                    raise TempRejected()
                class Block:  # minimal text block
                    type = "text"; text = "DONE"
                class Usage:
                    input_tokens = 3; output_tokens = 1
                class Resp:
                    content = [Block()]; usage = Usage()
                    stop_reason = "end_turn"
                r = Resp(); r.model = model
                return r

        class Client:
            messages = Msgs()
        return Client(), calls

    def test_temperature_sent_in_raw_body_when_sdk_lacks_kwarg(self):
        from providers import AnthropicProvider
        client, calls = self._client(rejects_temperature=False)
        c = AnthropicProvider(client).complete("claude-haiku-4-5-20251001", "s", [],
                                               [{"role": "user", "content": "x"}], 8, 0.0)
        self.assertEqual(calls[-1].get("temperature"), 0.0)
        self.assertEqual(c.temperature_effective, 0.0)

    def test_model_rejecting_temperature_falls_back_to_default(self):
        from providers import AnthropicProvider
        client, calls = self._client(rejects_temperature=True)
        c = AnthropicProvider(client).complete("claude-fable-5", "s", [],
                                               [{"role": "user", "content": "x"}], 8, 0.0)
        self.assertNotIn("temperature", calls[-1])
        self.assertIsNone(c.temperature_effective)


if __name__ == "__main__":
    unittest.main(verbosity=1)
