"""Session-resume (TaskGraph) test suite. Stdlib only: python3 run_tests2.py
Additive to the single-session suite: run_tests.py must stay green and untouched."""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, os.path.join(_HERE, "src2"))

from graph import (TaskGraph, generate_graphs, one_node_graph,  # noqa: E402
                   MIX, TOPOLOGIES)
from tasks import generate_tasks  # noqa: E402
from world import normalize_value  # noqa: E402
from session import Interrupt  # noqa: E402
from resume import run_graph_episode, RESUME_CONTEXTS  # noqa: E402


class TestGraphGeneration(unittest.TestCase):
    def test_graph_count_and_unique(self):
        gs = generate_graphs()
        self.assertEqual(len(gs), sum(MIX.values()))  # 200
        self.assertEqual(len({g.graph_id for g in gs}), len(gs))
        from collections import Counter
        by_topo = Counter(g.topology for g in gs)
        self.assertEqual(dict(by_topo), MIX)

    def test_edges_are_valid_dag_in_topo_order(self):
        for g in generate_graphs():
            n = len(g.nodes)
            for (i, j) in g.edges:
                self.assertTrue(0 <= i < n and 0 <= j < n, g.graph_id)
                self.assertLess(i, j, (g.graph_id, "nodes must be topo-ordered"))

    def test_deterministic(self):
        a = [g.graph_id for g in generate_graphs()]
        b = [g.graph_id for g in generate_graphs()]
        self.assertEqual(a, b)

    def test_every_canonical_value_stated_in_instruction(self):
        # Same guard as single-session (pilot audit 2026-07-11): a model agent only
        # sees the instruction; every canonical goal value must appear in it.
        for g in generate_graphs():
            ni = normalize_value(g.instruction)
            for node in g.nodes:
                for k, v in node.goal.canon_args.items():
                    self.assertIn(normalize_value(v), ni,
                                  (g.graph_id, node.node_id, k, v))

    def test_one_node_graph_wraps_single_session_task(self):
        t = generate_tasks(per_domain=1)[0]
        g = one_node_graph(t)
        self.assertEqual(len(g.nodes), 1)
        self.assertEqual(g.edges, [])
        self.assertEqual(g.nodes[0].tool, t.steps[0][0])
        self.assertEqual(g.nodes[0].goal.canon_args, t.goals[0].canon_args)


def _single_session_cell(agent, retry_mode, mitigation):
    """Read the frozen single-session summary.csv (crux mode) as regression truth."""
    import csv
    path = os.path.join(_HERE, "results", "summary.csv")
    for r in csv.DictReader(open(path)):
        if (r["agent"], r["retry_mode"], r["mitigation"]) == \
                (agent, retry_mode, mitigation):
            return float(r["ivr"])
    raise KeyError((agent, retry_mode, mitigation))


class TestContinuityRegression(unittest.TestCase):
    """THE GATE (design notes §9.2): a |V|=1 graph under interruption+resume
    must reproduce the single-session whole_run cells from results/summary.csv —
    the proof that gIVR restricted to one node IS the single-session IVR."""

    def _one_node_givr(self, context, mitigation, posture):
        hits = 0
        tasks = [t for t in generate_tasks(per_domain=1) if t.family == "atomic"]
        for t in tasks:
            g = one_node_graph(t)
            row = run_graph_episode(g, context, mitigation, posture,
                                    interrupt=Interrupt(0, "after_commit"))
            hits += row["givr_hit"]
            self.assertEqual(row["completion"], 1, g.graph_id)
        return hits / len(tasks)

    def test_amnesiac_naive_matches_single_session_naive_whole_run(self):
        self.assertEqual(self._one_node_givr("amnesiac", "none", "naive"),
                         _single_session_cell("naive", "whole_run", "none"))  # 1.0

    def test_ledger_readback_matches_single_session_careful_whole_run(self):
        self.assertEqual(self._one_node_givr("ledger_readback", "none", "naive"),
                         _single_session_cell("careful", "whole_run", "none"))  # 0.0

    def test_amnesiac_stable_idem_keys_matches_single_session(self):
        self.assertEqual(self._one_node_givr("amnesiac", "idem_keys", "stable"),
                         _single_session_cell("stable_key", "whole_run", "idem_keys"))  # 0.0

    def test_amnesiac_naive_idem_keys_fresh_keys_still_defeat(self):
        self.assertEqual(self._one_node_givr("amnesiac", "idem_keys", "naive"),
                         _single_session_cell("naive", "whole_run", "idem_keys"))  # 1.0

    def test_checkpoint_robust_one_node(self):
        self.assertEqual(self._one_node_givr("checkpoint", "none", "naive"), 0.0)


class TestGraphEpisodes(unittest.TestCase):
    def _graph(self, topo):
        return next(g for g in generate_graphs() if g.topology == topo)

    def test_no_interrupt_is_clean_everywhere(self):
        for topo in TOPOLOGIES:
            for ctx in RESUME_CONTEXTS:
                row = run_graph_episode(self._graph(topo), ctx, "none", "naive")
                self.assertEqual((row["violations"], row["completion"],
                                  row["order_violation"]), (0, 1, 0),
                                 (topo, ctx))

    def test_amnesiac_duplicates_grow_with_interrupt_depth(self):
        # H1: later interrupts expose more committed nodes to re-execution
        g = self._graph("chain")  # 3 nodes
        v = [run_graph_episode(g, "amnesiac", "none", "naive",
                               interrupt=Interrupt(k, "after_commit"))["violations"]
             for k in range(3)]
        self.assertEqual(v, [1, 2, 3])

    def test_plan_only_duplicates_exactly_the_seam_node(self):
        g = self._graph("chain")
        row = run_graph_episode(g, "plan_only", "none", "naive",
                                interrupt=Interrupt(1, "after_commit"))
        self.assertEqual(row["violations"], 1)   # only node 1 redone
        self.assertEqual(row["completion"], 1)
        row = run_graph_episode(g, "plan_only", "none", "naive",
                                interrupt=Interrupt(1, "before_commit"))
        self.assertEqual(row["violations"], 0)   # nothing committed at seam

    def test_stale_parent_reexecution_is_an_order_violation(self):
        # H4 mechanism: amnesiac resume re-commits the parent AFTER the
        # child's original commit (re-derived payment after receipt sent)
        g = self._graph("chain")
        row = run_graph_episode(g, "amnesiac", "none", "naive",
                                interrupt=Interrupt(1, "after_commit"))
        self.assertEqual(row["order_violation"], 1)

    def test_checkpoint_clean_at_every_interrupt_point(self):
        # H3: substrate checkpoint flat across interruption points
        for topo in TOPOLOGIES:
            g = self._graph(topo)
            for k in range(len(g.nodes)):
                for timing in ("before_commit", "after_commit"):
                    row = run_graph_episode(g, "checkpoint", "none", "naive",
                                            interrupt=Interrupt(k, timing))
                    self.assertEqual((row["violations"], row["completion"],
                                      row["order_violation"]), (0, 1, 0),
                                     (topo, k, timing))


class TestModelSessionAcknowledgement(unittest.TestCase):
    # Round-2 audit F6: plan_only notes must list only nodes whose tool
    # result the model actually received. A node committed earlier in the
    # same turn as the seam node never had its result delivered.
    def test_same_turn_commits_before_seam_are_not_acknowledged(self):
        import sweep2
        sys.path.insert(0, _HERE)
        from tests_providers import FakeProvider
        from world import World
        g, k = next((g, k) for g, k in sweep2.benchmark_graphs()
                    if g.topology == "fanout" and k == 2)
        uses = [{"type": "tool_use", "id": f"c{i}", "name": g.nodes[i].tool,
                 "input": dict(g.nodes[i].args)} for i in range(3)]
        # turn 1: node 0 alone (acknowledged); turn 2: nodes 1 and 2 in one
        # turn, the seam (2) commits mid-turn -> node 1's result is never sent
        prov = FakeProvider([uses[:1], uses[1:3]])
        turns, interrupted, acked, _ = sweep2.run_model_session(
            prov, "fake-model", World(), g, sweep2.BASE_SYSTEM, g.instruction,
            interrupt_node=k)
        self.assertTrue(interrupted)
        self.assertEqual(sorted(acked), [0])

    def test_completed_turn_is_acknowledged(self):
        import sweep2
        sys.path.insert(0, _HERE)
        from tests_providers import FakeProvider
        from world import World
        g, k = next((g, k) for g, k in sweep2.benchmark_graphs()
                    if g.topology == "fanout" and k == 2)
        uses = [{"type": "tool_use", "id": f"c{i}", "name": g.nodes[i].tool,
                 "input": dict(g.nodes[i].args)} for i in range(3)]
        prov = FakeProvider([uses[:2], uses[2:3]])
        _, interrupted, acked, _ = sweep2.run_model_session(
            prov, "fake-model", World(), g, sweep2.BASE_SYSTEM, g.instruction,
            interrupt_node=k)
        self.assertTrue(interrupted)
        self.assertEqual(sorted(acked), [0, 1])

    def test_invalid_call_in_session_is_error_not_crash(self):
        import sweep2
        sys.path.insert(0, _HERE)
        from tests_providers import FakeProvider
        from world import World
        g, k = sweep2.benchmark_graphs()[0]
        prov = FakeProvider([[{"type": "tool_use", "id": "x", "name": "nope",
                               "input": {}}]])
        w = World()
        turns, interrupted, acked, meta = sweep2.run_model_session(
            prov, "fake-model", w, g, sweep2.BASE_SYSTEM, g.instruction,
            interrupt_node=k)
        self.assertFalse(interrupted)
        self.assertTrue(turns[1]["content"][0]["is_error"])
        self.assertEqual(meta["invalid_calls"][0]["name"], "nope")
        self.assertEqual(w.ledger.for_task(g.graph_id), [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
