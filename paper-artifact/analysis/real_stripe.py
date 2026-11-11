"""Real-API validation slice (Stripe test mode): numbers + comparison with the
simulator. Pre-registered in analysis/preregistration.md, section R1.

  python3 analysis/real_stripe.py        # no network, stdlib only

Reads (read-only): results/real_stripe/{raw.jsonl, stripe_ground_truth.jsonl,
stripe_check.json, search_check.jsonl}, configs/real_stripe.json, and the
simulator cell numbers from results/analysis_v3/{tier1_prereg,bare_hint}.json
(their "source" raw files for the descriptive payments-only subset).
Writes results/analysis_v3/real_stripe.json, merged into numbers.json by
analysis/v3_numbers.py (optional part, present only once the run exists).

Keys (values raw, proportions in [0, 1]):
  real.<m>.base.<none|keys>.<toA|e500|w3>:ivr   IVR from Stripe's record,
      ci = task-level Wilson(p*T, T) (primary, as for the simulator cells);
      sim_key / sim_value / sim_ci = the registered simulator cell;
      within_sim_ci = real point estimate inside the simulator's CI (R1-H1);
      d = real - simulator; sim_payments_subset = same simulator cell
      restricted to its payment tasks (descriptive)
  real.<m>.restart.<amnesiac|bare_hint>:givr    same, for the 2-payment plan
  ...:ivr_ledger / :agree                       the simulator's verdict on the
      same call sequence (shadow ledger) and its agreement with Stripe
  real.agreement.ledger_vs_stripe               all episodes (R1-H2)
  real.search_check.agree                       Search API vs list API (R1-S)
  real.stripe_check.ok, real.n_episodes, real.H1.within
"""
import csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src_real"))
import sweep  # noqa: E402

OUT = os.path.join(ROOT, "results", "analysis_v3")
FM = {"timeout_after_commit": "toA", "error_500_after_commit": "e500",
      "status_unknown_after_commit": "w3", "timeout_before_commit": "toB"}
MIT = {"none": "none", "idem_keys": "keys"}
SHORT = (("haiku", "haiku"), ("sonnet", "sonnet"), ("qwen", "qwen"), ("gpt", "gpt"),
         ("gemini", "gemini"), ("fable", "fable"))
MIN_FRAC = 0.9   # preregistration "Exclusions": report a cell only at >= 90% of planned n


def short(m):
    return next((s for k, s in SHORT if k in m), m.replace("/", "_"))


def read_jsonl(p):
    out = []
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def last_by_id(rows):
    d = {}
    for r in rows:
        d[r["episode_id"]] = r
    return list(d.values())


def sim_numbers_default():
    nums = {}
    for p in ("tier1_prereg.json", "bare_hint.json"):
        f = os.path.join(OUT, p)
        if os.path.exists(f):
            with open(f) as fh:
                nums.update(json.load(fh))
    if not nums and os.path.exists(os.path.join(OUT, "numbers.json")):
        with open(os.path.join(OUT, "numbers.json")) as fh:
            nums = json.load(fh)["numbers"]
    return nums


def payments_subset(sim, model, prompt, mit, fm):
    """Descriptive: the simulator cell restricted to its payment tasks."""
    rows = []
    for src in sim.get("source") or []:
        rows += [r for r in read_jsonl(os.path.join(ROOT, src))
                 if r.get("model") == model and r.get("prompt") == prompt
                 and r.get("mitigation") == mit and r.get("failure_mode") == fm
                 and str(r.get("task_id", "")).startswith("payments-")]
    rows = last_by_id(rows)
    if not rows:
        return None
    k = sum(1 for r in rows if (r.get("violations") or 0) > 0)
    lo, hi = sweep.wilson_ci(k, len(rows))
    return {"value": k / len(rows), "k": k, "n": len(rows), "ci_ep": [lo, hi],
            "T": len({r["task_id"] for r in rows})}


def cell_stats(eps):
    sc = [r for r in eps if not r.get("refusal")]
    n = len(sc)
    k = sum(1 for r in sc if r["ivr_stripe"])
    T = len({r["task_id"] for r in sc})
    p = k / n if n else None
    ci = list(sweep.wilson_ci(p * T, T)) if n else [0.0, 1.0]
    ci_ep = list(sweep.wilson_ci(k, n)) if n else [0.0, 1.0]
    led = sum(1 for r in sc if r["ivr_ledger"])
    fired = [r for r in sc if r.get("failure_fired")]
    return {"n": n, "k": k, "T": T, "p": p, "ci": ci, "ci_ep": ci_ep,
            "refusals": len(eps) - n, "k_ledger": led,
            "agree": sum(1 for r in sc if r["agree"]),
            "agree_exact": sum(1 for r in sc if r["agree_exact"]),
            "readback": (sum(1 for r in fired if r.get("readback_after_failure")) / len(fired))
            if fired else None,
            "s2_lookup": [sum(1 for r in sc if r.get("s2_lookup")),
                          sum(1 for r in sc if r.get("s2_lookup") is not None)],
            "interrupted": sum(1 for r in sc if r.get("interrupted")),
            "replayed": sum(r.get("replayed") or 0 for r in sc),
            "forwarded_keys": sum(r.get("forwarded_keys") or 0 for r in sc),
            "stripe_errors": sum(r.get("stripe_errors") or 0 for r in sc),
            "cost_usd": round(sum(r.get("cost_usd") or 0 for r in eps), 4)}


def build(cfg, sim_numbers=None, out_path=None):
    sim_numbers = sim_numbers_default() if sim_numbers is None else sim_numbers
    rows = last_by_id(read_jsonl(cfg["raw_path"]))
    src = [os.path.relpath(cfg["raw_path"], ROOT), os.path.relpath(cfg["gt_path"], ROOT)]
    expect = int(cfg.get("expect_per_cell") or cfg["n_tasks"] * cfg["samples"])
    N = {}
    cells = {}
    for r in rows:
        cond = (r["prompt"], r["mitigation"], r["failure_mode"]) if r["kind"] == "retry" \
            else ("restart", r["resume_context"])
        cells.setdefault((r["model"], r["kind"], cond), []).append(r)
    within = []
    for (model, kind, cond), eps in sorted(cells.items()):
        st = cell_stats(eps)
        if kind == "retry":
            p, mit, fm = cond
            key = f"real.{short(model)}.{p}.{MIT[mit]}.{FM[fm]}:ivr"
            ckey = f"{p}.{mit}.{fm}"
        else:
            key = f"real.{short(model)}.restart.{cond[1]}:givr"
            ckey = f"restart.{cond[1]}"
        skey = ((cfg.get("sim_compare") or {}).get(model) or {}).get(ckey)
        sim = sim_numbers.get(skey) if skey else None
        e = {"value": st["p"], "ci": st["ci"], "ci_method": "wilson95_task(p*T,T)",
             "ci_ep": st["ci_ep"], "n": st["n"], "k": st["k"], "T": st["T"],
             "refusals": st["refusals"], "source": src, "model": model,
             "status": "complete" if st["n"] >= expect else
             ("reported" if st["n"] >= MIN_FRAC * expect else f"PENDING ({st['n']}/{expect})"),
             "sim_key": skey, "sim_value": sim.get("value") if sim else None,
             "sim_ci": sim.get("ci") if sim else None,
             "within_sim_ci": (sim["ci"][0] <= st["p"] <= sim["ci"][1])
             if sim and sim.get("ci") and st["p"] is not None else None,
             "d": (st["p"] - sim["value"]) if sim and st["p"] is not None
             and sim.get("value") is not None else None,
             "ci_overlap": (st["ci"][0] <= sim["ci"][1] and sim["ci"][0] <= st["ci"][1])
             if sim and sim.get("ci") else None,
             "criterion": "R1-H1: real IVR inside the simulator's task-level 95% CI"}
        if kind == "retry" and sim:
            e["sim_payments_subset"] = payments_subset(sim, model, *cond)
        N[key] = e
        base = key.rsplit(":", 1)[0]
        N[base + (":ivr_ledger" if kind == "retry" else ":givr_ledger")] = {
            "value": st["k_ledger"] / st["n"] if st["n"] else None, "n": st["n"],
            "k": st["k_ledger"], "source": src,
            "note": "simulator verdict (shadow World) on the same call sequence"}
        N[base + ":agree"] = {"value": st["agree"] / st["n"] if st["n"] else None,
                              "k": st["agree"], "n": st["n"], "k_exact": st["agree_exact"],
                              "source": src}
        if kind == "retry":
            N[base + ":readback_after_failure"] = {"value": st["readback"], "n": st["n"], "source": src}
            N[base + ":stripe_replayed"] = {"value": st["replayed"], "forwarded_keys": st["forwarded_keys"],
                                            "stripe_errors": st["stripe_errors"], "source": src}
        else:
            N[base + ":s2_lookups"] = {"value": f"{st['s2_lookup'][0]}/{st['s2_lookup'][1]}",
                                       "interrupted": st["interrupted"], "n": st["n"], "source": src}
        N[base + ":cost_usd"] = {"value": st["cost_usd"], "source": src}
        if e["within_sim_ci"] is not None:
            within.append(e["within_sim_ci"])
    sc = [r for r in rows if not r.get("refusal")]
    N["real.n_episodes"] = {"value": len(rows), "planned": 0, "source": src}
    try:
        sys.path.insert(0, os.path.join(ROOT, "src_real"))
        import run_real
        N["real.n_episodes"]["planned"] = len(run_real.plan(cfg))
    except Exception:
        pass
    N["real.agreement.ledger_vs_stripe"] = {
        "value": sum(1 for r in sc if r["agree"]) / len(sc) if sc else None,
        "k": sum(1 for r in sc if r["agree"]), "n": len(sc),
        "k_exact": sum(1 for r in sc if r["agree_exact"]), "source": src,
        "criterion": "R1-H2: simulator verdict on the same calls = Stripe's record, every episode"}
    N["real.H1.within"] = {"value": sum(within), "n": len(within), "source": src}
    chk = os.path.join(cfg["out_dir"], "stripe_check.json")
    if os.path.exists(chk):
        with open(chk) as fh:
            c = json.load(fh)
        N["real.stripe_check.ok"] = {"value": bool(c.get("ok")), "utc": c.get("utc"),
                                     "api_version": c.get("api_version"),
                                     "key_reuse_other_params": c.get("key_reuse_other_params"),
                                     "source": [os.path.relpath(chk, ROOT)]}
    srch = last_by_id(read_jsonl(cfg["search_path"]))
    if srch:
        att = {r["episode_id"]: r.get("attempt_id") for r in rows}
        srch = [x for x in srch if att.get(x["episode_id"]) == x["attempt_id"]]
        N["real.search_check.agree"] = {"value": sum(1 for x in srch if x["agree"]) / len(srch)
                                        if srch else None,
                                        "k": sum(1 for x in srch if x["agree"]), "n": len(srch),
                                        "source": [os.path.relpath(cfg["search_path"], ROOT)]}
    out_path = out_path or os.path.join(OUT, "real_stripe.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict(sorted(N.items())), f, indent=1)
    return N


def main():
    import run_real
    cfg = run_real.load_config(os.path.join(ROOT, "configs", "real_stripe.json"))
    if not os.path.exists(cfg["raw_path"]):
        print("real_stripe: no data yet (results/real_stripe/raw.jsonl) - skipped")
        return 0
    N = build(cfg)
    rows = []
    for k, v in N.items():
        if k.endswith(":ivr") or k.endswith(":givr"):
            f = lambda x: "-" if x is None else f"{x:.3f}"
            rows.append([k, f(v["value"]), f"[{f(v['ci'][0])}, {f(v['ci'][1])}]", v["n"],
                         v["sim_key"] or "-", f(v["sim_value"]),
                         "-" if not v["sim_ci"] else f"[{f(v['sim_ci'][0])}, {f(v['sim_ci'][1])}]",
                         {True: "yes", False: "NO", None: "-"}[v["within_sim_ci"]], v["status"]])
    w = csv.writer(sys.stdout, delimiter="\t")
    w.writerow(["key", "real", "real_ci_task", "n", "sim_key", "sim", "sim_ci", "within", "status"])
    w.writerows(rows)
    a = N["real.agreement.ledger_vs_stripe"]
    print(f"ledger vs Stripe agreement: {a['k']}/{a['n']} (exact counts {a['k_exact']}/{a['n']})")
    print(f"real_stripe.json: {len(N)} values -> {os.path.relpath(os.path.join(OUT, 'real_stripe.json'), ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
