"""R5 real error texts: catalogue loading, the registered selection rule, and
registration of the injected texts as World failure modes.

No tracked file is edited. `register()` adds one entry per injected text to
`world.WORDING_VARIANTS` at run time, so the existing World code path is used
unchanged: the effect COMMITS on the first valid effectful call, then a
timeout- or server-class exception carries the text, and the model loop
(model_agent / sweep2) shows it as a tool result "ERROR: <text>" with
is_error true. Mode name: f"real_{id}_after_commit".

World.call formats the text with str.format(request_id=...), so braces in a
catalogue text are escaped here ({ -> {{) and the model sees the text
verbatim.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
V32 = os.path.dirname(HERE)
REPO = os.path.dirname(V32)
CATALOGUE = os.path.join(V32, "error_texts", "catalogue.json")

CLASSES = ("U", "N", "R", "G")
CLASS_NAMES = {"U": "signals-outcome-unknown", "N": "says-not-applied",
               "R": "says-retry-safe", "G": "generic-error"}
QUOTA = {"U": 3, "G": 3, "N": 1, "R": 1}          # registered subset shape
MAX_PER_PROVIDER = 2
CONDITIONS = ("timeout", "5xx", "ambiguous")
# Anchors: the paper's own strings, same batch (R2/R3 names, unchanged modes)
ANCHORS = {"error_500_after_commit": "anchor_e500",
           "status_unknown_after_commit": "anchor_w3"}
PREFIX, SUFFIX = "real_", "_after_commit"
REQUIRED_FIELDS = ("id", "provider", "status", "condition", "source_kind", "text", "url",
                   "accessed", "retrieval", "class", "vendor_name_in_text", "borderline")


def load(path=CATALOGUE) -> dict:
    with open(path) as f:
        cat = json.load(f)
    ids = [t["id"] for t in cat["texts"]]
    assert len(ids) == len(set(ids)), "duplicate ids in the catalogue"
    for t in cat["texts"]:
        missing = [k for k in REQUIRED_FIELDS if k not in t]
        assert not missing, (t.get("id"), missing)
        assert t["class"] in CLASSES, t["id"]
        assert t["text"].strip() == t["text"] and t["text"], t["id"]
        assert t["url"].startswith("https://"), t["id"]
    return cat


def by_id(cat) -> dict:
    return {t["id"]: t for t in cat["texts"]}


def fm_of(text_id: str) -> str:
    return f"{PREFIX}{text_id}{SUFFIX}"


def id_of(fm: str):
    if fm.startswith(PREFIX) and fm.endswith(SUFFIX):
        return fm[len(PREFIX):-len(SUFFIX)]
    return None


def _vendor_free(t, cat) -> bool:
    if t["vendor_name_in_text"]:
        return False
    low = t["text"].lower()
    names = [p.lower() for p in cat["provider_order"]] + ["dynamodb", "amazon", "google", "microsoft",
                                                           "stripe", "twilio", "shopify", "paypal"]
    return not any(n in low for n in names)


def select_subset(cat) -> list:
    """The registered selection rule (catalogue 'selection_rule')."""
    prov_rank = {p: i for i, p in enumerate(cat["provider_order"])}
    kind_rank = {k: i for i, k in enumerate(cat["source_kind_order"])}
    chosen, per_prov = [], {}
    for cls in ("U", "G", "N", "R"):
        elig = [t for t in cat["texts"] if t["class"] == cls and not t["borderline"]
                and t["condition"] in CONDITIONS and _vendor_free(t, cat)]
        elig.sort(key=lambda t: (kind_rank[t["source_kind"]], prov_rank.get(t["provider"], 99), t["id"]))
        got = 0
        for t in elig:
            if got == QUOTA[cls]:
                break
            if per_prov.get(t["provider"], 0) >= MAX_PER_PROVIDER:
                continue
            chosen.append(t["id"])
            per_prov[t["provider"]] = per_prov.get(t["provider"], 0) + 1
            got += 1
        assert got == QUOTA[cls], f"class {cls}: only {got} eligible texts"
    return chosen


def kind_of(t) -> str:
    """Exception class for the World: timeout-class for timeout texts,
    server-class otherwise. The model sees the same 'ERROR: <text>' either
    way (is_error true); the class is kept for bookkeeping only."""
    return "timeout" if t["condition"] == "timeout" else "server"


def register(cfg_r5: dict, cat=None) -> dict:
    """Add the injected texts to world.WORDING_VARIANTS (idempotent).
    Returns {failure_mode: text_id}."""
    import world
    cat = cat or load(_abs(cfg_r5.get("catalogue", CATALOGUE)))
    ids = by_id(cat)
    out = {}
    for tid in cfg_r5["injected"]:
        t = ids[tid]
        fm = fm_of(tid)
        entry = (kind_of(t), t["text"].replace("{", "{{").replace("}", "}}"))
        prev = world.WORDING_VARIANTS.get(fm)
        assert prev in (None, entry), f"{fm} already registered with a different text"
        world.WORDING_VARIANTS[fm] = entry
        out[fm] = tid
    return out


def class_of_fm(fm: str, cat) -> str:
    """'U'/'N'/'R'/'G' for an injected text, 'anchor_e500'/'anchor_w3' for an anchor."""
    if fm in ANCHORS:
        return ANCHORS[fm]
    return by_id(cat)[id_of(fm)]["class"]


def _abs(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)
