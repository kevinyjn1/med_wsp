"""A persistent sandboxed Python kernel: variables survive between code blocks, like a notebook.

The MIT experiment 8 harness gave the model such a kernel over the weights. Ours runs inside the same macOS
sandbox as one-shot runs (no network, writes only in the work folder, reads only from an allow-list). If a
block exceeds the time limit the kernel is killed and restarted, and the caller is told its variables are gone.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from wsp.harness import sandbox

SENTINEL = "\x00KERNEL\x00"

# Runs inside the sandbox. Requests arrive as JSON lines on stdin; each reply is one sentinel-prefixed JSON
# line on the real stdout. Whatever the block prints is captured and returned in the reply.
SERVER = r"""
import ast, contextlib, io, json, os, sys, traceback
_reply = os.fdopen(os.dup(1), "w")
_ns = {"__name__": "__main__"}
for _line in sys.stdin:
    _req = json.loads(_line)
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):
        try:
            _tree = ast.parse(_req["code"], "<block>")
            _last = _tree.body.pop() if _tree.body and isinstance(_tree.body[-1], ast.Expr) else None
            exec(compile(_tree, "<block>", "exec"), _ns)
            if _last is not None:
                _value = eval(compile(ast.Expression(_last.value), "<block>", "eval"), _ns)
                if _value is not None:
                    print(repr(_value))
        except BaseException:
            traceback.print_exc(limit=6)
    _reply.write("\x00KERNEL\x00" + json.dumps({"out": _buf.getvalue()}) + "\n")
    _reply.flush()
"""


class Kernel:
    def __init__(
        self,
        work_dir: Path,
        *,
        prelude: str = "",
        read_only: list[Path] | None = None,
        timeout_s: int = 900,
        max_out: int = 10_000,
        python: Path | None = None,
        threads: int = 8,
        memory_limit_gb: float = 14.0,
    ):
        self.work_dir, self.prelude, self.read_only = work_dir, prelude, read_only or []
        self.timeout_s, self.max_out, self.threads = timeout_s, max_out, threads
        self.python = python or (sandbox.SANDBOX_VENV / "bin" / "python")
        self.proc: subprocess.Popen | None = None
        self.restarts = 0
        self.memory_limit_gb = memory_limit_gb
        self._over_memory = False
        self._lines: queue.Queue = queue.Queue()

    def start(self) -> str:
        """Start the process and run the prelude; returns what the prelude printed."""
        self.work_dir.mkdir(parents=True, exist_ok=True)
        profile = sandbox.build_profile(self.work_dir, self.read_only, self.python)
        env = {
            "HOME": os.path.realpath(self.work_dir), "PATH": "/usr/bin:/bin", "TMPDIR": os.path.realpath(tempfile.gettempdir()),
            "OMP_NUM_THREADS": str(self.threads), "VECLIB_MAXIMUM_THREADS": str(self.threads),
            "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
        }  # fmt: skip
        self.proc = subprocess.Popen(
            [sandbox.SANDBOX_EXEC, "-p", profile, str(self.python), "-I", "-c", SERVER],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=self.work_dir, env=env,
        )  # fmt: skip
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()
        threading.Thread(target=self._watch_memory, args=(self.proc,), daemon=True).start()
        return self._exchange(self.prelude, timeout_s=600) if self.prelude else ""

    @staticmethod
    def _pump(proc: subprocess.Popen, lines: queue.Queue) -> None:
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _watch_memory(self, proc: subprocess.Popen) -> None:
        """Kill the kernel when its resident memory passes the limit (macOS does not enforce address-space limits)."""
        while proc.poll() is None:
            ps = subprocess.run(
                ["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True, check=False
            )
            rss_kb = int(ps.stdout.strip() or 0)
            if rss_kb / 1e6 > self.memory_limit_gb:
                self._over_memory = True
                proc.kill()
                return
            time.sleep(3)

    def _exchange(self, code: str, timeout_s: int) -> str:
        self.proc.stdin.write(json.dumps({"code": code}) + "\n")
        self.proc.stdin.flush()
        stray = []
        while True:
            line = self._lines.get(timeout=timeout_s)
            if line is None:
                raise RuntimeError("kernel exited: " + "".join(stray)[-2000:])
            if line.startswith(SENTINEL):
                return json.loads(line[len(SENTINEL) :])["out"]
            stray.append(
                line
            )  # output written below Python's stdout (a C library), kept only for crash reports

    def run(self, code: str) -> str:
        """Execute one block in the persistent namespace and return its printed output."""
        note = ""
        if self.proc is None or self.proc.poll() is not None:
            note = (
                "[note: the kernel was restarted, so its variables are gone]\n"
                if self.proc is not None
                else ""
            )
            self.restarts += self.proc is not None
            self.start()
        try:
            out = self._exchange(code, self.timeout_s)
        except queue.Empty:
            self.stop()
            self.restarts += 1
            self.start()
            return (
                f"[timed out after {self.timeout_s} s; the kernel was restarted and its variables are lost]"
            )
        except (RuntimeError, BrokenPipeError) as error:
            over_memory, self._over_memory = self._over_memory, False
            self.stop()
            self.restarts += 1
            self.start()
            if over_memory:
                return (
                    f"[memory limit exceeded: the kernel used more than {self.memory_limit_gb:g} GB and was restarted; "
                    "its variables are lost. Keep fewer large tensors in memory at once]"
                )
            return f"[the kernel crashed and was restarted; its variables are lost]\n{str(error)[-1500:]}"
        out = note + out
        if len(out) > self.max_out:
            out = out[: self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        return out or "(no output)"

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=10)


if __name__ == "__main__":
    import sys

    kernel = Kernel(
        Path(sys.argv[1]), prelude="import numpy as np\nX = np.arange(6)\nprint('prelude ok')", timeout_s=5
    )
    print("start:", kernel.start().strip())
    print("1:", kernel.run("y = int(X.sum())\ny"))
    print("2:", kernel.run("print(y * 2)\n1/0").strip().splitlines()[-1])
    print("3:", kernel.run("import time\ntime.sleep(30)"))
    print("4:", kernel.run("print('y' in dir(), int(X[2]))"))
    kernel.memory_limit_gb, kernel.timeout_s = 0.4, 30
    print(
        "5m:",
        kernel.run(
            "import numpy as np\nbig = np.ones((120_000_000,))\nimport time\ntime.sleep(8)\nprint(big.nbytes)"
        ),
    )
    kernel.memory_limit_gb, kernel.timeout_s = 14.0, 5
    print(
        "5:",
        kernel.run(
            "import socket\ntry:\n    socket.create_connection(('1.1.1.1', 80), timeout=2)\nexcept Exception as e:\n    print('net blocked', type(e).__name__)"
        ),
    )
    kernel.stop()
