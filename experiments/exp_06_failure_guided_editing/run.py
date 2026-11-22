"""EXP-06: failure-guided weight editing of Phi-3.5-mini-instruct on MedQA, with no donor model.

Astra gets the weights, 4,272 practice questions (MedQA dev + the first 3,000 training questions) with the
model's own result on each (so it knows the failures), a persistent Python kernel and a budget of forward
passes (see wsp/lab.py and wsp/lab_session.py). When the session ends, the submitted edit is scored against the
unedited model on the MedQA test split, which Astra never sees.

EXP-05c uses the same driver with a donor checkpoint:  --donor mediphi  (or mediphi_instruct).

Usage:  python experiments/exp_06_failure_guided_editing/run.py --name run1 --budget 30 [--donor mediphi]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa
from wsp.lab_session import final_evaluation, run_session

BASELINE = REPO / "experiments" / "exp_04_medqa_baseline" / "processed"


def practice_pool(model: str) -> tuple[list[dict], list[dict]]:
    items, results = [], []
    for split in ("dev", "train"):
        records = medqa.load_jsonl(BASELINE / f"{model}_{split}.jsonl")
        items += medqa.load_medqa(split)[: len(records)]
        results += records
    return items, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--donor", default=None)
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.out) / "raw" / args.name
    practice, baseline = practice_pool("phi")
    donor_baseline = practice_pool(args.donor)[1] if args.donor else None
    if not args.eval_only:
        state = run_session(run_dir, model="phi", donor=args.donor, budget=args.budget, effort=args.effort,
                            practice=practice, baseline=baseline, donor_baseline=donor_baseline)  # fmt: skip
        print(f"session: done={state.done} turns={state.turn} runs={state.runs} handoffs={state.handoffs} "
              f"failures={state.failures} wall={state.wall_s:.0f}s", flush=True)  # fmt: skip
        if not state.done:
            return
    result = final_evaluation(run_dir, medqa.load_medqa("test"))
    print(json.dumps({k: v for k, v in result.items() if k != "edited_tensors"}, indent=1), flush=True)
    print(
        "edited tensors:",
        [(t["name"], round(t["relative_change"], 5)) for t in result["edited_tensors"]],
        flush=True,
    )


if __name__ == "__main__":
    main()
