"""EXP-05c: Astra moves a fine-tune's MedQA advantage into the base model by editing weights (donor session).

Setting chosen by the researcher on 2026-11-22: donor = MediPhi-Instruct, primary measure = the bare
lm-evaluation-harness prompt (where the donor beats Phi-3.5), guardrail = the chat prompt (where they tie).
Astra gets both sets of weights, 4,272 practice questions with both models' results under both prompts, a
persistent kernel, and a budget of forward passes: score, teacher-forced and activation patching
(wsp/lab.py, wsp/lab_session.py). The submitted edit is then scored on the MedQA test split under both prompts.

Usage:  python experiments/exp_05_medical_circuit/run_astra.py --name transfer1 --donor mediphi_instruct --budget 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa
from wsp.lab_session import final_evaluation, run_transfer_session

BASELINE = REPO / "experiments" / "exp_04_medqa_baseline" / "processed"


def practice_pool(donor: str) -> tuple[list[dict], dict[tuple[str, str], list[dict]]]:
    """MedQA dev + the first 3,000 training questions, with both models' results under both prompts."""
    items: list[dict] = []
    results: dict[tuple[str, str], list[dict]] = {
        (who, prompt): [] for who in ("original", "donor") for prompt in ("bare", "chat")
    }
    for split in ("dev", "train"):
        n = len(medqa.load_jsonl(BASELINE / f"phi_{split}.jsonl"))
        items += medqa.load_medqa(split)[:n]
        for who, name in (("original", "phi"), ("donor", donor)):
            for prompt, prefix in (("bare", "bare_"), ("chat", "")):
                rows = medqa.load_jsonl(BASELINE / f"{prefix}{name}_{split}.jsonl")
                assert len(rows) == n, (prefix, name, split, len(rows), n)
                results[who, prompt] += rows
    assert all([r["id"] for r in rows] == [i["id"] for i in items] for rows in results.values())
    return items, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--donor", default="mediphi_instruct")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--effort", default="xhigh")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent / "raw" / args.name
    practice, results = practice_pool(args.donor)
    if not args.eval_only:
        state = run_transfer_session(run_dir, donor=args.donor, budget=args.budget, effort=args.effort,
                                     practice=practice, results=results)  # fmt: skip
        print(f"session: done={state.done} turns={state.turn} runs={state.runs} handoffs={state.handoffs} "
              f"failures={state.failures} wall={state.wall_s:.0f}s", flush=True)  # fmt: skip
        if not state.done:
            return
    result = final_evaluation(run_dir, medqa.load_medqa("test"), styles=("bare", "chat"))
    print(json.dumps(result["by_prompt"], indent=1), flush=True)
    print("share of weights changed:", round(result["share_of_weights_changed"], 6), flush=True)
    print(
        "edited tensors:",
        [(t["name"], round(t["relative_change"], 5)) for t in result["edited_tensors"]],
        flush=True,
    )


if __name__ == "__main__":
    main()
