"""Machine-wide GPU locks, so GPU jobs queue instead of starving each other.

Two locks exist. "gpu" (the default) is for anything that loads more than one model or generates long texts,
and for the edit lab's forward passes. "batch" is for single-model scoring sweeps, which are small enough to run
next to a lab pass; they still queue among themselves. A job that needs both takes them in `LOCK_ORDER`.

Hold a lock for one unit of work, not for a whole sweep: a sweep that takes the lock once per configuration lets
an Astra lab pass in between configurations. (In the night of 2026-11-23 a scan that held "batch" for its whole
run kept a lab session waiting for 6.5 hours.) Waiting is in arrival order: a job that releases the lock and asks
for it again goes behind a job that is already waiting.

In code:      with gpu_lock("exp07 chen-s10", name="batch"): ...
              with gpu_locks("patching dev-00012", ("batch", "gpu")): ...
As a wrapper: python -m wsp.gpu_lock [--name batch] -- python some_script.py ...
              The wrapper holds the lock for the whole command. The command sees which locks its parent holds
              (environment variable WSP_GPU_LOCKS_HELD) and does not wait for them again.

Taking a lock also keeps the Mac awake while on AC power (see wsp/keep_awake.py).
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

from wsp.keep_awake import ensure_awake

LOCK_DIR = Path(__file__).resolve().parents[1] / "logs"
LOCK_ORDER = ("batch", "gpu")  # every holder of both takes them in this order, so two holders cannot deadlock
HELD_ENV = "WSP_GPU_LOCKS_HELD"


def held_by_parent() -> list[str]:
    return [name for name in os.environ.get(HELD_ENV, "").split(",") if name]


@contextmanager
def gpu_lock(label: str = "", name: str = "gpu"):
    """Block until no other holder of the named lock is running."""
    if name in held_by_parent():  # the wrapper process that started us holds it for the whole command
        yield
        return
    ensure_awake()
    LOCK_DIR.mkdir(exist_ok=True)
    with open(LOCK_DIR / f"{name}.turn", "a+") as turn, open(LOCK_DIR / f"{name}.lock", "a+") as handle:
        waited = time.time()
        # Stand in line first. Whoever waits for the lock keeps the line closed behind it, so a job that has just
        # released the lock cannot take it again ahead of the one that was waiting.
        fcntl.flock(turn, fcntl.LOCK_EX)
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        finally:
            fcntl.flock(turn, fcntl.LOCK_UN)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {label} (waited {time.time() - waited:.0f}s)\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def gpu_locks(label: str, names: tuple[str, ...] | list[str]):
    """Hold several locks, always taken in `LOCK_ORDER`."""
    with ExitStack() as stack:
        for name in sorted(names, key=lambda n: LOCK_ORDER.index(n) if n in LOCK_ORDER else len(LOCK_ORDER)):
            stack.enter_context(gpu_lock(label, name=name))
        yield


def main() -> int:
    if "--" not in sys.argv:
        print(__doc__)
        return 2
    split = sys.argv.index("--")
    options, command = sys.argv[1:split], sys.argv[split + 1 :]
    name = options[options.index("--name") + 1] if "--name" in options else "gpu"
    with gpu_lock(" ".join(command)[:200], name=name):
        held = ",".join(dict.fromkeys([*held_by_parent(), name]))
        return subprocess.run(command, check=False, env={**os.environ, HELD_ENV: held}).returncode


if __name__ == "__main__":
    sys.exit(main())
