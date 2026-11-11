"""One live process per (output dir, runner, model set).

Lets several Terminal tabs run different models into the same output dir
without two processes ever running the same episodes. The OS releases the
lock when the process exits, so a crash never leaves a stale lock.
"""
from __future__ import annotations

import fcntl
import os
import re

_HELD = []  # keep file objects alive for the life of the process


def acquire(out_dir: str, runner: str, models) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", ",".join(sorted(models)))
    path = os.path.join(out_dir, f".lock_{runner}_{tag}")
    f = open(path, "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        print(f"{runner}: {', '.join(models)} is already running in another "
              f"process (lock {path}); skipping here.")
        return False
    f.write(str(os.getpid())); f.flush()
    _HELD.append(f)
    return True
