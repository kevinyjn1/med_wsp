"""Run a command while holding a machine-wide GPU lock, so GPU jobs queue instead of starving each other.

Two locks exist. "gpu" (the default) is for anything that loads more than one model or generates long texts,
and for the edit lab's forward passes. "batch" is for single-model scoring sweeps, which are small enough to run
next to a lab pass; they still queue among themselves.

Usage:  python -m wsp.gpu_lock [--name batch] -- python experiments/.../run.py ...
"""

from __future__ import annotations

import fcntl
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parents[1] / "logs"


@contextmanager
def gpu_lock(label: str = "", name: str = "gpu"):
    """Block until no other holder of the named lock is running."""
    LOCK_DIR.mkdir(exist_ok=True)
    with open(LOCK_DIR / f"{name}.lock", "a+") as handle:
        waited = time.time()
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {label} (waited {time.time() - waited:.0f}s)\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> int:
    if "--" not in sys.argv:
        print(__doc__)
        return 2
    split = sys.argv.index("--")
    options, command = sys.argv[1:split], sys.argv[split + 1 :]
    name = options[options.index("--name") + 1] if "--name" in options else "gpu"
    with gpu_lock(" ".join(command)[:200], name=name):
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
