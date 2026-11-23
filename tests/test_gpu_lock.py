"""The GPU job locks: mutual exclusion, arrival order, lock order and the wrapper's hand-over to its command."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from wsp import gpu_lock as locks


@pytest.fixture(autouse=True)
def private_lock_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(locks, "LOCK_DIR", tmp_path)
    monkeypatch.setattr(locks, "ensure_awake", lambda: False)
    monkeypatch.delenv(locks.HELD_ENV, raising=False)


def test_a_waiting_job_goes_before_the_job_that_releases_and_asks_again():
    """A sweep takes the lock once per configuration. A lab pass that starts waiting during configuration 1 must
    run before configuration 2, even though the sweep asks for the lock again immediately."""
    order: list[str] = []
    sweep_has_lock, pass_is_waiting = threading.Event(), threading.Event()

    def sweep():
        for step in range(3):
            with locks.gpu_lock(f"config {step}", name="batch"):
                order.append(f"config {step}")
                if step == 0:
                    sweep_has_lock.set()
                    assert pass_is_waiting.wait(5)
                    time.sleep(0.2)  # the lab pass is now blocked inside gpu_lock

    def lab_pass():
        assert sweep_has_lock.wait(5)
        pass_is_waiting.set()
        with locks.gpu_lock("lab pass", name="batch"):
            order.append("lab pass")

    threads = [threading.Thread(target=sweep), threading.Thread(target=lab_pass)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert order == ["config 0", "lab pass", "config 1", "config 2"]


def test_only_one_holder_at_a_time():
    inside, most = 0, 0
    guard = threading.Lock()

    def job():
        nonlocal inside, most
        for _ in range(5):
            with locks.gpu_lock("job", name="gpu"):
                with guard:
                    inside += 1
                    most = max(most, inside)
                time.sleep(0.01)
                with guard:
                    inside -= 1

    threads = [threading.Thread(target=job) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)
    assert most == 1


def test_several_locks_are_taken_in_the_fixed_order(monkeypatch):
    taken: list[str] = []
    real = locks.gpu_lock

    def recording(label="", name="gpu"):
        taken.append(name)
        return real(label, name=name)

    monkeypatch.setattr(locks, "gpu_lock", recording)
    with locks.gpu_locks("two models", ("gpu", "batch")):
        pass
    assert taken == list(locks.LOCK_ORDER)


def test_a_lock_held_by_the_wrapper_is_not_waited_for_again(tmp_path):
    """`python -m wsp.gpu_lock --name batch -- script` holds "batch" for the script. The script takes the same lock
    per unit of work; it must pass straight through instead of waiting for its own parent forever."""
    script = tmp_path / "child.py"
    script.write_text(
        "import os\n"
        "from pathlib import Path\n"
        "from wsp import gpu_lock as locks\n"
        f"locks.LOCK_DIR = Path({str(tmp_path)!r})\n"
        "locks.ensure_awake = lambda: False\n"
        "with locks.gpu_locks('unit of work', ('batch', 'gpu')):\n"
        "    print('held by parent:', os.environ[locks.HELD_ENV])\n"
    )
    wrapper = (
        "import sys\n"
        "from pathlib import Path\n"
        "from wsp import gpu_lock as locks\n"
        f"locks.LOCK_DIR = Path({str(tmp_path)!r})\n"
        "locks.ensure_awake = lambda: False\n"
        f"sys.argv = ['gpu_lock', '--name', 'batch', '--', sys.executable, {str(script)!r}]\n"
        "sys.exit(locks.main())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", wrapper],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(Path(locks.__file__).resolve().parents[1])},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "held by parent: batch"
