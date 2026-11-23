"""Keep the Mac from going to sleep while a long job runs.

macOS puts an idle machine to sleep even while a GPU job is running; the job is then suspended and continues only
in short maintenance wakes (a 200 s step took up to 4,300 s in the night of 2026-11-23). `ensure_awake()` starts
`caffeinate -s -w <this process>`, which holds a "prevent system sleep" assertion until this process exits.

The assertion is valid on AC power only: on battery the machine may still sleep, which protects the battery
(a GPU job empties it in under an hour). Closing the lid always puts the machine to sleep.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

_started: subprocess.Popen | None = None


def ensure_awake() -> bool:
    """Hold a no-sleep assertion for the rest of this process's life. Safe to call many times.

    Returns False where there is nothing to do (not macOS, or `caffeinate` missing)."""
    global _started
    if _started is not None and _started.poll() is None:
        return True
    tool = shutil.which("caffeinate") if sys.platform == "darwin" else None
    if tool is None:
        return False
    _started = subprocess.Popen(  # exits by itself when this process does
        [tool, "-s", "-w", str(os.getpid())],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return True
