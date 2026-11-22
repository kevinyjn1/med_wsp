"""EXP-03b: behaviour of Qwen3.5-0.8B with and without MIT's weight edit, on this Mac (MLX).

A scaled-down version of MIT's evaluation: same prompt (thinking enabled), same sampling (temperature 0.6,
top_p 0.95, top_k 20, presence penalty 1.5), same grading (last \\boxed{} answer checked with math-verify), but
a sample of problems, one sample per problem and a configurable token limit.

Usage:
    python experiments/exp_03_mit8_math/eval_mlx.py --model base   --set gsm8k --n 48 --max-tokens 16384
    python experiments/exp_03_mit8_math/eval_mlx.py --model p50c   --set gsm8k --n 48 --max-tokens 16384
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp.models import snapshot_dir

EXP = Path(__file__).resolve().parent
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def load_problems(name: str, n: int, seed: int = 0) -> list[dict]:
    from datasets import load_dataset

    if name == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        problems = [{"id": f"gsm8k-{i}", "problem": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
                    for i, r in enumerate(rows)]  # fmt: skip
    elif name == "math500":
        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"]} for r in rows]
    else:
        raise ValueError(name)
    return random.Random(seed).sample(problems, min(n, len(problems)))


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (balanced braces); None when there is no closed box."""
    starts = [m.end() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    for start in reversed(starts):
        depth = 1
        for position in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[position], 0)
            if depth == 0:
                return text[start:position]
    return None


def grade(prediction: str | None, gold: str) -> bool:
    if prediction is None:
        return False
    from math_verify import parse, verify

    try:
        return bool(verify(parse(f"\\boxed{{{gold}}}"), parse(f"\\boxed{{{prediction}}}")))
    except Exception:  # noqa: BLE001 - an unparseable answer is a wrong answer
        return False


def repetition_share(token_ids: list[int], window: int = 512, ngram: int = 8) -> float:
    """Share of the last `window` tokens covered by an 8-gram that already occurred earlier in the answer."""
    if len(token_ids) < ngram * 2:
        return 0.0
    seen, repeated = set(), np.zeros(len(token_ids), bool)
    for i in range(len(token_ids) - ngram + 1):
        gram = tuple(token_ids[i : i + ngram])
        if gram in seen:
            repeated[i : i + ngram] = True
        seen.add(gram)
    return float(repeated[-window:].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="base", help="base, or the name of an MIT edit run (p10c, p25c, p50c)"
    )
    parser.add_argument("--set", default="gsm8k", choices=["gsm8k", "math500"])
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import mlx.core as mx
    import mlx_lm
    from mlx.utils import tree_flatten
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    model, tokenizer = mlx_lm.load(str(snapshot_dir("Qwen/Qwen3.5-0.8B")))
    if args.model != "base":
        params = dict(tree_flatten(model.parameters()))
        updates = []
        for path in sorted((EXP / "work" / f"edit_{args.model}").glob("*.weight.npy")):
            name = path.name[: -len(".npy")].replace("model.language_model.", "language_model.model.")
            assert name in params, name
            new = mx.array(np.load(path)).astype(params[name].dtype)
            assert new.shape == params[name].shape
            print(
                f"edit {name}: relative change {float(mx.linalg.norm((new - params[name]).astype(mx.float32)) / mx.linalg.norm(params[name].astype(mx.float32))):.4f}"
            )
            updates.append((name, new))
        assert updates, "no edited tensors found; run apply_edit.py first"
        from mlx.utils import tree_unflatten

        model.update(tree_unflatten(updates))
        mx.eval(model.parameters())

    problems = load_problems(args.set, args.n)
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p["problem"] + "\n" + INSTRUCTION}],
                                             tokenize=True, add_generation_prompt=True, enable_thinking=True) for p in problems]  # fmt: skip
    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
    processors = make_logits_processors(presence_penalty=1.5, presence_context_size=args.max_tokens)
    mx.random.seed(0)

    tag = f"{args.model}_{args.set}_n{len(problems)}_t{args.max_tokens}"
    out_path = EXP / "processed" / f"eval_{tag}.jsonl"
    done = (
        {json.loads(line)["id"] for line in out_path.read_text().splitlines()} if out_path.exists() else set()
    )
    start = time.time()
    for begin in range(0, len(problems), args.batch_size):
        chunk = [(p, t) for p, t in zip(problems[begin : begin + args.batch_size], prompts[begin : begin + args.batch_size], strict=True)
                 if p["id"] not in done]  # fmt: skip
        if not chunk:
            continue
        response = mlx_lm.batch_generate(model, tokenizer, [t for _, t in chunk], max_tokens=args.max_tokens,
                                         sampler=sampler, logits_processors=processors)  # fmt: skip
        with open(out_path, "a") as f:
            for (problem, _), text in zip(chunk, response.texts, strict=True):
                ids = tokenizer.encode(text, add_special_tokens=False)
                answer_part = text.split("</think>")[-1] if "</think>" in text else text
                prediction = last_boxed(answer_part) if "</think>" in text else last_boxed(text)
                f.write(json.dumps({
                    "id": problem["id"], "gold": problem["answer"], "pred": prediction, "correct": grade(prediction, problem["answer"]),
                    "tokens": len(ids), "cut": len(ids) >= args.max_tokens - 2, "closed_thinking": "</think>" in text,
                    "repetition": round(repetition_share(ids), 4), "text_head": text[:600], "text_tail": text[-600:],
                }) + "\n")  # fmt: skip
        print(
            f"{min(begin + args.batch_size, len(problems))}/{len(problems)} [{time.time() - start:.0f}s]",
            flush=True,
        )

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    summary = {"model": args.model, "set": args.set, "n": len(rows), "max_tokens": args.max_tokens,
               "acc": float(np.mean([r["correct"] for r in rows])), "cut_share": float(np.mean([r["cut"] for r in rows])),
               "mean_tokens": float(np.mean([r["tokens"] for r in rows])), "closed_thinking": float(np.mean([r["closed_thinking"] for r in rows])),
               "mean_repetition": float(np.mean([r["repetition"] for r in rows])), "seconds": round(time.time() - start)}  # fmt: skip
    (EXP / "processed" / f"eval_{tag}.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
