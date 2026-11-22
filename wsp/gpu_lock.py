"""Run a command while holding the machine-wide GPU lock, so GPU jobs queue instead of slowing each other.

Usage:  python -m wsp.gpu_lock -- python experiments/.../run.py ...
"""

from __future__ import annotations

import fcntl
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_FILE = Path(__file__).resolve().parents[1] / "logs" / "gpu.lock"


@contextmanager
def gpu_lock(label: str = ""):
    """Block until no other holder is using the GPU."""
    LOCK_FILE.parent.mkdir(exist_ok=True)
    with open(LOCK_FILE, "a+") as handle:
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
    command = sys.argv[sys.argv.index("--") + 1 :]
    with gpu_lock(" ".join(command)[:200]):
        return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
