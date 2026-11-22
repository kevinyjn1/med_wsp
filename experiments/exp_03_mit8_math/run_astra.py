"""EXP-03c: our own Astra session on Qwen3.5-0.8B math (MIT experiment 8, scaled to this Mac).

Astra gets the weights, MIT's practice problems, a persistent kernel and a budget of forward passes (see
wsp/qwen_lab.py for what differs from MIT's harness). The session's submitted edit is the content of
raw/<name>/work/edited/ when it ends; evaluate it with eval_mlx.py --edits.

Usage:  python experiments/exp_03_mit8_math/run_astra.py --name p10_local --budget 10 [--effort max]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp.qwen_lab import run_session


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--effort", default="max")
    args = parser.parse_args()
    run_dir = Path(__file__).resolve().parent / "raw" / args.name
    state = run_session(run_dir, budget=args.budget, effort=args.effort)
    print(f"session: done={state.done} turns={state.turn} runs={state.runs} handoffs={state.handoffs} "
          f"failures={state.failures} wall={state.wall_s:.0f}s", flush=True)  # fmt: skip
    edited = run_dir / "work" / "edited"
    print("submitted tensors:", sorted(p.name for p in edited.glob("*.npy")) if edited.is_dir() else [])


if __name__ == "__main__":
    main()
