#!/usr/bin/env python3
"""Launcher for the response-window quick experiments (QUICK-EXPERIMENTS.md).

Adds the few new texts below to the harness's wording tables IN MEMORY, then
hands over to the unchanged src/tier1.py. No repo file is edited; every run
writes only under its config's out_dir (results/response_v31/...).

Run from the repo root (v27_fix/), e.g.
  $PY -B ../response-prep-v31/scripts/run_qx.py --config ../response-prep-v31/configs/qx1_note_control.json --dry-run
  $PY -B ../response-prep-v31/scripts/run_qx.py --config ../response-prep-v31/configs/qx1_note_control.json --estimate
  $PY -B ../response-prep-v31/scripts/run_qx.py --config ../response-prep-v31/configs/qx1_note_control.json --launch
--dry-run and --estimate make no API call. --launch bills the keys in .env
(sourced by your shell, never by this script).

Texts follow the harness convention: the model sees "ERROR: " + text for tool
errors (so error texts omit "ERROR: "), and a note is appended to the task
instruction after a blank line, exactly like bare_hint.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("IBENCH_REPO") or os.path.normpath(os.path.join(HERE, "..", "..", "v27_fix"))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "src2"))

import world   # noqa: E402
import tier1   # noqa: E402

# QX1: notes with the same frame as bare_hint but no interruption content.
NEW_NOTES = {
    # same "NOTE: ... Finish the task." frame, no event at all
    "note_null": "NOTE: Finish the task.",
    # length-matched to bare_hint, carries an irrelevant fact instead of the event
    "note_irrelevant": "NOTE: this task was assigned to you from the operations "
                       "queue. Finish the task.",
}

# QX2: lexical ablation of the status (after commit, timeout class like W3).
# W3 = "The request may have been processed; its outcome is unknown."
NEW_AFTER_COMMIT = {
    "status_processed_only_after_commit": ("timeout", "The request may have been processed."),
    "status_unknown_only_after_commit": ("timeout", "The outcome of the request is unknown."),
}

# QX3: a "never sent" status BEFORE a commit (nothing commits, as in R4).
NEW_BEFORE_COMMIT = {
    "status_never_sent_before_commit": ("timeout", "The request was not sent; nothing was processed."),
}


def install():
    sweep2 = tier1._sweep2()          # the module object tier1 itself uses
    for k, v in NEW_NOTES.items():
        assert k not in sweep2.HINT_PARAPHRASES, k
        sweep2.HINT_PARAPHRASES[k] = v
    for k, v in NEW_AFTER_COMMIT.items():
        assert k not in world.WORDING_VARIANTS, k
        world.WORDING_VARIANTS[k] = v
    for k, v in NEW_BEFORE_COMMIT.items():
        assert k not in world.PRECOMMIT_VARIANTS, k
        world.PRECOMMIT_VARIANTS[k] = v


if __name__ == "__main__":
    os.chdir(REPO)
    install()
    sys.exit(tier1.main(sys.argv[1:]))
