"""Run model-written Python in an isolated process (macOS `sandbox-exec`).

The MIT harness used bubblewrap on Linux: no network, a clean environment, one working folder, a time
limit and an output cap per run. This is the macOS equivalent:

- no network at all;
- writes only inside the run's working folder (and the process's private temp folder);
- reads inside the home directory only from an allow-list (the sandbox Python environment, the working
  folder, and read-only inputs such as a weights file), so answer keys and credentials are unreachable;
- a clean environment, `python -I`, a wall-clock limit, and the printed output cut at `max_out` characters.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SANDBOX_VENV = REPO / ".venv-sandbox"
SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def _real(path: Path | str) -> str:
    return os.path.realpath(str(path))


def _python_roots(python: Path) -> list[str]:
    """Folders the interpreter itself needs: the venv and the base installation it points to."""
    venv = python.parent.parent
    roots = {_real(venv), str(venv)}
    cfg = venv / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.startswith("home"):
                base_bin = Path(line.split("=", 1)[1].strip())
                roots |= {_real(base_bin.parent), str(base_bin.parent)}
    roots.add(_real(Path(_real(python)).parent.parent))
    return sorted(roots)


def build_profile(work_dir: Path, read_only: list[Path], python: Path) -> str:
    home = _real(Path.home())
    allow_read = _python_roots(python) + [_real(work_dir)] + [_real(p) for p in read_only]
    allow_write = [_real(work_dir), _real(tempfile.gettempdir()), "/private/var/folders", "/dev"]

    def subpaths(paths: list[str]) -> str:
        return " ".join(f'(subpath "{p}")' for p in paths)

    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
        f"(allow file-write* {subpaths(allow_write)})",
        f'(deny file-read* (subpath "{home}"))',
        # Path resolution needs to stat the folders above an allowed path.
        f'(allow file-read-metadata (subpath "{home}"))',
        f"(allow file-read* {subpaths(allow_read)})",
    ])  # fmt: skip


def run(
    code: str,
    work_dir: Path,
    *,
    prelude: str = "",
    read_only: list[Path] | None = None,
    timeout_s: int = 600,
    max_out: int = 20_000,
    python: Path | None = None,
    log: Path | None = None,
    threads: int = 8,
) -> str:
    """Run `prelude + code` in a fresh sandboxed process inside `work_dir`; return what it printed."""
    python = python or (SANDBOX_VENV / "bin" / "python")
    work_dir.mkdir(parents=True, exist_ok=True)
    profile = build_profile(work_dir, read_only or [], python)
    env = {
        "HOME": _real(work_dir), "PATH": "/usr/bin:/bin", "TMPDIR": _real(tempfile.gettempdir()),
        "OMP_NUM_THREADS": str(threads), "VECLIB_MAXIMUM_THREADS": str(threads),
        "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
    }  # fmt: skip
    cmd = [SANDBOX_EXEC, "-p", profile, str(python), "-I", "-c", prelude + "\n" + code]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=work_dir, env=env, check=False
        )
        out = proc.stdout + (("\n[stderr]\n" + proc.stderr[-6000:]) if proc.stderr.strip() else "")
    except subprocess.TimeoutExpired:
        out = f"[timed out after {timeout_s} s]"
    seconds = round(time.time() - start, 2)
    if len(out) > max_out:
        out = out[:max_out] + f"\n[output cut at {max_out:,} characters]"
    out = out or "(no output)"
    if log is not None:
        with open(log, "a") as f:
            f.write(
                json.dumps({"t": time.time(), "kind": "python", "s": seconds, "code": code, "out": out})
                + "\n"
            )
    return out


if __name__ == "__main__":
    # Self-check: python runs, numpy imports, network and out-of-sandbox reads/writes are blocked.
    work = Path(sys.argv[1])
    probe = r"""
import os, socket, numpy as np
print("numpy", np.__version__, "sum", int(np.arange(10).sum()))
open("inside.txt", "w").write("ok"); print("write inside:", open("inside.txt").read())
for label, fn in [
    ("write outside", lambda: open(OUTSIDE, "w")),
    ("read ssh dir", lambda: os.listdir(SECRET_DIR)),
    ("read hf token", lambda: open(HF_TOKEN).read()),
    ("network", lambda: socket.create_connection(("1.1.1.1", 80), timeout=3)),
]:
    try:
        fn(); print(label, "-> ALLOWED (bad)")
    except Exception as e:
        print(label, "-> blocked:", type(e).__name__)
"""
    home = str(Path.home())
    pre = f"OUTSIDE={home + '/sandbox_escape_test.txt'!r}\nSECRET_DIR={home + '/.ssh'!r}\nHF_TOKEN={home + '/.cache/huggingface/token'!r}\n"
    print(run(probe, work, prelude=pre, timeout_s=60))
