"""EXP-01: replicate MIT weight-space-programming experiment 3 (which text was a char-level LM trained on?).

Astra gets only the four transformer blocks of a 4x6x384 character-level language model (no embeddings,
no output layer), may analyze them with Python (no forward pass, up to 100 runs), and must say which of
three corpora the network was trained on and what each of the 24 heads does.

We reuse MIT's released networks and their exact first message, and rerun the reading with our own
Astra access. Usage:  python experiments/exp_01_mit3_language/run.py [--runs r0 r1] [--parallel 3]
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp.harness import sandbox
from wsp.harness.loop import Loop, State

EXP = Path(__file__).resolve().parent
MIT = REPO / "external" / "lm_understand-logs" / "exp3_language"
LANGUAGES = ("python", "romaji", "english")
PRELUDE = 'import numpy as np\nW = dict(np.load("weights.npz"))\n'


def neutral_work_dir(run_id: str) -> Path:
    """A folder whose name says nothing about the network, so the answer cannot leak through a path."""
    digest = hashlib.sha256(f"exp01:{run_id}".encode()).hexdigest()[:10]
    return EXP / "work" / f"net_{digest}"


def run_one(language: str, rep: str, effort: str) -> tuple[str, State]:
    run_id = f"{language}_{rep}"
    work = neutral_work_dir(run_id)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(MIT / "networks" / language / "blocks.npz", work / "weights.npz")
    # The first message is identical for all six conversations; take MIT's verbatim.
    message = (MIT / "runs" / "python_r0" / "message.txt").read_text()

    def run_python(code: str, state: State) -> str:
        # Each run is a fresh process and nothing is kept between runs (as the prompt states).
        for leftover in work.iterdir():
            if leftover.name != "weights.npz":
                shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
        return sandbox.run(code, work, prelude=PRELUDE, timeout_s=600, max_out=20_000)

    loop = Loop(run_dir=EXP / "raw" / "runs" / run_id, first_message=message,
                handlers={"python": run_python}, max_runs=100, effort=effort)  # fmt: skip
    return run_id, loop.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES), choices=LANGUAGES)
    parser.add_argument("--runs", nargs="+", default=["r0", "r1"])
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--effort", default="xhigh")
    args = parser.parse_args()

    jobs = [(language, rep) for language in args.languages for rep in args.runs]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for run_id, state in pool.map(lambda job: run_one(*job, args.effort), jobs):
            print(f"{run_id}: done={state.done} turns={state.turn} runs={state.runs} "
                  f"failures={state.failures} wall={state.wall_s:.0f}s", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
