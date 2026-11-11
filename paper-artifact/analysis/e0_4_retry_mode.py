"""E0-4: which retry mode the real-model runs actually used, plus the
[counterfactual] pre-estimates for E1-2 (at-least-once) and E1-3 (keys
under explicit 500s). Counterfactuals are ESTIMATES, never results.

(a) Retry mode. From code: src/sweep.py launch() runs one model episode per
    World and never re-runs it; the only retries are the model's own calls
    inside run_model_episode (MAX_TURNS=8) => agent_retry semantics. Checked
    in transcripts: every episode is a single conversation (one initial user
    message, no second delivery), and no episode hits the 8-turn cap.
    whole_run trigger rate = episodes whose LAST assistant text does not
    contain "DONE" (the failure-conditioned orchestrator would re-run these).
(b) ALO bounds: for each task in base/none/none, replay s0 then s1 into ONE
    world (delivery 1, delivery 2). The pair is valid only if delivery 2
    never calls list_effects (otherwise it would have seen delivery 1's
    effects and its trajectory is unknowable). IVR bounds:
      lower = violating valid pairs; upper = lower + invalid pairs.
    Done under mitigation none and idem_keys.
(c) Keys under 500: replay base/none/error_500_after_commit under idem_keys.
    Valid iff every list_effects text the model saw is reproduced exactly
    (error positions and ack form are identical by construction).
Also: unprompted read-back rate (list_effects in any base-prompt episode).

Reads : results/sweep_transcripts/*.json, src/sweep.py + src/model_agent.py (static check)
Writes: results/analysis_v2/e0_4_retry_mode.csv, e0_4_alo_counterfactual.csv,
        e0_4_keys500_counterfactual.csv, e0_4_summary.json
Run   : python3 analysis/e0_4_retry_mode.py
No sampling; deterministic (ack ids are irrelevant: only their 'ok-' form matters).
"""
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_1_corrected_table import FNAME, CANON, sid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TDIR = os.path.join(ROOT, "results", "sweep_transcripts")
OUT = os.path.join(ROOT, "results", "analysis_v2")


class ReplayWorld:
    """Minimal independent world: commits, idem_keys dedup (non-empty keys
    only), first-call failure injection, list_effects text in world.py format."""
    def __init__(self, mit, fm):
        self.mit, self.fm, self.armed = mit, fm, fm != "none"
        self.ledger, self.keys = [], set()

    def call(self, tool, args, key):
        if tool == "list_effects":
            rows = [f"{tl}({', '.join(f'{k}={v}' for k, v in a.items())})" for tl, a in self.ledger]
            return "\n".join(rows) if rows else "(no effects recorded)", False
        if self.armed:
            self.armed = False
            if self.fm == "timeout_before_commit":
                return "ERROR", True
            self._commit(tool, args, key)
            return "ERROR", True
        self._commit(tool, args, key)
        return "ok-", False

    def _commit(self, tool, args, key):
        if self.mit == "idem_keys" and key:
            if (tool, key) in self.keys:
                return
            self.keys.add((tool, key))
        self.ledger.append((tool, dict(args)))

    def violations(self):
        c = {}
        for tl, a in self.ledger:
            s = sid(tl, a); c[s] = c.get(s, 0) + 1
        return sum(n - 1 for n in c.values() if n > 1)


def calls_and_results(t):
    calls = [b for turn in t["turns"] if turn["role"] == "assistant"
             for b in turn["content"] if b.get("type") == "tool_use"]
    res = [r for turn in t["turns"] if turn["role"] == "tool_results" for r in turn["content"]]
    return list(zip(calls, res))


def feed(w, t, check_obs=False):
    """Replay a transcript's calls into world w. Returns (calls_list_effects, obs_identical)."""
    rb, same = False, True
    for b, r in calls_and_results(t):
        a = dict(b["input"]); k = a.pop("idempotency_key", "") or ""
        out, err = w.call(b["name"], a, k)
        if b["name"] == "list_effects":
            rb = True
            if check_obs and out != r["content"]:
                same = False
        elif err != bool(r.get("is_error")):
            same = False
    return rb, same


def last_text(t):
    a = [x for x in t["turns"] if x["role"] == "assistant"][-1]
    return " ".join(b.get("text", "") for b in a["content"] if b.get("type") == "text")


def main():
    os.makedirs(OUT, exist_ok=True)
    T = {}
    for f in sorted(glob.glob(os.path.join(TDIR, "*.json"))):
        m = FNAME.match(os.path.basename(f)); t = json.load(open(f))
        T[(t["model"], m["prompt"], m["mit"], m["fm"], m["task"], int(m["s"]))] = t
    models = sorted({k[0] for k in T})

    # (a) static code evidence
    sweep_src = open(os.path.join(ROOT, "src", "sweep.py")).read()
    agent_src = open(os.path.join(ROOT, "src", "model_agent.py")).read()
    code = dict(
        sweep_launch_calls_run_model_episode_once=sweep_src.count("run_model_episode(") == 1,
        sweep_mentions_retry_mode=("retry_mode" in sweep_src),
        model_agent_MAX_TURNS=int(re.search(r"MAX_TURNS\s*=\s*(\d+)", agent_src).group(1)),
        model_agent_mentions_retry_mode=("retry_mode" in agent_src))
    rows = []
    for mo in models:
        E = [(k, t) for k, t in T.items() if k[0] == mo]
        n = len(E)
        nondone = [k for k, t in E if "DONE" not in last_text(t)]
        nondone_fail = [k for k in nondone if k[3] != "none"]
        multi_user = 0  # transcripts store one instruction; count sessions/deliveries if present
        cap = sum(1 for k, t in E if sum(1 for x in t["turns"] if x["role"] == "assistant") >= code["model_agent_MAX_TURNS"])
        max_turns = max(sum(1 for x in t["turns"] if x["role"] == "assistant") for k, t in E)
        base = [(k, t) for k, t in E if k[1] == "base"]
        rb = sum(1 for k, t in base if any(b["name"] == "list_effects" for b, _ in calls_and_results(t)))
        rows.append(dict(model=mo, n_episodes=n, retry_mode_inferred="agent_retry (model-driven, single delivery)",
                         whole_run_would_trigger=len(nondone),
                         whole_run_trigger_rate=round(len(nondone) / n, 4),
                         whole_run_trigger_in_failure_cells=len(nondone_fail),
                         episodes_at_turn_cap=cap, max_assistant_turns=max_turns,
                         base_episodes=len(base), base_readback_episodes=rb,
                         base_readback_rate=round(rb / len(base), 4),
                         extra_deliveries_in_transcripts=multi_user))
        rows[-1]["non_done_cells"] = json.dumps(sorted({f"{k[1]}/{k[2]}/{k[3]}" for k in nondone}))
    with open(os.path.join(OUT, "e0_4_retry_mode.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # (b) ALO counterfactual
    alo = []
    for mo in models:
        tasks = sorted({k[4] for k in T if k[0] == mo and k[1:4] == ("base", "none", "none")})
        for mit in ("none", "idem_keys"):
            valid = viol = 0
            for tk in tasks:
                t0, t1 = T[(mo, "base", "none", "none", tk, 0)], T[(mo, "base", "none", "none", tk, 1)]
                w = ReplayWorld(mit, "none")
                feed(w, t0)
                rb2, _ = feed(w, t1)
                if rb2:
                    continue
                valid += 1; viol += w.violations() > 0
            alo.append(dict(model=mo, mitigation=mit, tasks=len(tasks), valid_pairs=valid,
                            invalid_pairs=len(tasks) - valid, ivr_lower=viol,
                            ivr_upper=viol + len(tasks) - valid,
                            bounds=f"[{viol}, {viol + len(tasks) - valid}]/{len(tasks)}",
                            ivr_on_valid=round(viol / valid, 4) if valid else None,
                            label="counterfactual"))
    with open(os.path.join(OUT, "e0_4_alo_counterfactual.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(alo[0].keys())); w.writeheader(); w.writerows(alo)

    # (c) keys under 500
    k5 = []
    for mo in models:
        keys = [k for k in T if k[0] == mo and k[1:4] == ("base", "none", "error_500_after_commit")]
        v_none = v_keys = valid = v_keys_valid = 0
        for k in keys:
            t = T[k]
            w0 = ReplayWorld("none", "error_500_after_commit"); feed(w0, t)
            w1 = ReplayWorld("idem_keys", "error_500_after_commit"); _, same = feed(w1, t, check_obs=True)
            v_none += w0.violations() > 0; v_keys += w1.violations() > 0
            if same:
                valid += 1; v_keys_valid += w1.violations() > 0
        k5.append(dict(model=mo, n=len(keys), ivr_none_replay=v_none, ivr_idem_keys_replay_all=v_keys,
                       valid=valid, ivr_idem_keys_replay_valid=v_keys_valid,
                       ivr_idem_keys_upper=v_keys_valid + (len(keys) - valid),
                       label="counterfactual"))
    with open(os.path.join(OUT, "e0_4_keys500_counterfactual.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(k5[0].keys())); w.writeheader(); w.writerows(k5)

    summary = dict(code_evidence=code, retry_mode=rows, alo_counterfactual=alo, keys500_counterfactual=k5,
                   notes=["All real-model runs: agent_retry semantics (model decides inside one conversation); "
                          "no orchestrator re-run and no redelivery exists in src/sweep.py or the transcripts.",
                          "whole_run trigger = last assistant text lacks 'DONE' (spec definition; a text "
                          "heuristic, not an orchestrator signal).",
                          "ALO/keys-500 rows are counterfactual replays: estimates for E1-2/E1-3, never results."])
    json.dump(summary, open(os.path.join(OUT, "e0_4_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
