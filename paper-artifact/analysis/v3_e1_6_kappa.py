"""Agreement between the rule-based E1-6 belief coder and the hand codes. Run AFTER filling the `human_code` column of
results/analysis_v3/e1_6_handcheck.csv (and, optionally, the extra 60 in
e1_6_handcheck_extra60.csv) with one of:
  possibly_committed | committed | not_committed | handoff | no_text | uncoded
Writes results/analysis_v3/e1_6_kappa.json: n coded, raw agreement, Cohen's
kappa on the 6-way code and on the binary entertains_commit
(possibly_committed/committed vs the rest, no_text excluded), and the
disagreements. Prints PENDING and exits 0 if nothing is coded yet.
"""
import csv, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v3_common import OUT

FILES = ["e1_6_handcheck.csv", "e1_6_handcheck_extra60.csv"]
CODES = {"possibly_committed", "committed", "not_committed", "handoff", "no_text", "uncoded"}


def kappa(pairs):
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    rows = []
    for fn in FILES:
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            rows += [r for r in csv.DictReader(open(p)) if r.get("human_code", "").strip()]
    bad = [r["hc_id"] for r in rows if r["human_code"].strip() not in CODES]
    if bad:
        sys.exit(f"unknown human_code in {bad}; allowed: {sorted(CODES)}")
    out = {"status": "PENDING (no hand codes yet)" if not rows else "done", "n": len(rows)}
    if rows:
        pairs = [(r["coder_code"], r["human_code"].strip()) for r in rows]
        bin_ = [(a in ("possibly_committed", "committed"), b in ("possibly_committed", "committed"))
                for a, b in pairs if a != "no_text" and b != "no_text"]
        out.update({"agreement": sum(a == b for a, b in pairs) / len(pairs), "kappa_6way": kappa(pairs),
                    "n_binary": len(bin_), "agreement_binary": sum(a == b for a, b in bin_) / len(bin_) if bin_ else None,
                    "kappa_binary": kappa(bin_),
                    "disagreements": [{"hc_id": r["hc_id"], "coder": r["coder_code"], "human": r["human_code"],
                                       "note": r.get("human_note", "")} for r in rows
                                      if r["coder_code"] != r["human_code"].strip()]})
    json.dump(out, open(os.path.join(OUT, "e1_6_kappa.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}))


if __name__ == "__main__":
    main()
