"""EXP-04c: MedQA scored the way lm-evaluation-harness's `medqa_4options` task does it.

The MediPhi paper reports MedQA only in an appendix table and does not describe the protocol. A common default
is lm-evaluation-harness, whose `medqa_4options` task (dataset GBaker/MedQA-USMLE-4-options-hf, test split) uses

    Question: {question}
    A. {option}
    B. {option}
    C. {option}
    D. {option}
    Answer:

as a plain completion prompt (no chat template, no system prompt) and picks the option whose continuation " A",
" B", " C" or " D" has the highest log-likelihood. For single-token continuations that equals comparing the
next-token log-probabilities at the end of the prompt, which is what this script does.

Variants: `raw` (the harness default for causal models: no BOS token), `raw_bos` (with BOS), and `chat` (the same
text wrapped in the model's chat template, as with the harness's --apply_chat_template).

Usage:  python experiments/exp_04_medqa_baseline/run_lmeval_style.py --model phi --variants raw raw_bos chat
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa, models

OUT = Path(__file__).resolve().parent / "processed"
LETTERS = ("A", "B", "C", "D")


def load_docs() -> list[dict]:
    from datasets import load_dataset

    rows = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="test")
    return [{"id": f"gbaker-test-{i:05d}", "question": r["sent1"], "options": [r[f"ending{k}"] for k in range(4)], "label": int(r["label"])}
            for i, r in enumerate(rows)]  # fmt: skip


def doc_to_text(doc: dict) -> str:
    answers = "".join(f"{letter}. {option}\n" for letter, option in zip(LETTERS, doc["options"], strict=True))
    return f"Question: {doc['question']}\n{answers}Answer:"


@torch.no_grad()
def score(model, tokenizer, docs: list[dict], variant: str, batch_size: int = 16) -> list[dict]:
    # The harness encodes context + continuation together and splits at the context length, so the continuation
    # " A" is whatever follows "Answer:" in the joint encoding (one token here; " A" encoded alone would be two).
    stem = tokenizer.encode("Answer:", add_special_tokens=False)
    choice_ids = []
    for letter in LETTERS:
        continuation = tokenizer.encode("Answer: " + letter, add_special_tokens=False)[len(stem) :]
        assert len(continuation) == 1, f"' {letter}' after 'Answer:' is not a single token: {continuation}"
        choice_ids.append(continuation[0])
    texts = [doc_to_text(doc) for doc in docs]
    if variant == "chat":
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True
            )
            for t in texts
        ]
    bos = [tokenizer.bos_token_id] if variant == "raw_bos" else []
    encoded = [bos + tokenizer.encode(t, add_special_tokens=False) for t in texts]
    order = sorted(range(len(docs)), key=lambda i: len(encoded[i]))
    records: dict[int, dict] = {}
    for begin in range(0, len(order), batch_size):
        rows = order[begin : begin + batch_size]
        width = max(len(encoded[i]) for i in rows)
        input_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for k, i in enumerate(rows):  # left padding: the last column is the end of every prompt
            input_ids[k, width - len(encoded[i]) :] = torch.tensor(encoded[i])
            mask[k, width - len(encoded[i]) :] = 1
        logits = model(
            input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False
        ).logits[:, -1]
        logprobs = torch.log_softmax(logits.float(), dim=-1)[:, choice_ids].cpu()
        del logits, input_ids, mask
        models.release_gpu_cache()
        for k, i in enumerate(rows):
            pred = int(logprobs[k].argmax())
            records[i] = {"id": docs[i]["id"], "label": docs[i]["label"], "pred": pred, "correct": pred == docs[i]["label"],
                          "choice_logprobs": [round(float(x), 4) for x in logprobs[k]], "choice_mass": float(logprobs[k].exp().sum())}  # fmt: skip
    return [records[i] for i in range(len(docs))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--variants", nargs="+", default=["raw"], choices=["raw", "raw_bos", "chat"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    docs = load_docs()[: args.limit]
    tokenizer = models.load_tokenizer("phi")
    model = models.load_model(args.model)
    for variant in args.variants:
        out = (
            OUT / f"lmeval_{variant}_{args.model}_test{'' if args.limit is None else f'_{args.limit}'}.jsonl"
        )
        if out.exists():
            records = medqa.load_jsonl(out)
        else:
            start = time.time()
            records = score(model, tokenizer, docs, variant)
            medqa.save_jsonl(records, out)
            print(f"  ({time.time() - start:.0f}s)", flush=True)
        n = len(records)
        print(json.dumps({"model": args.model, "variant": variant, "n": n, "acc": round(sum(r["correct"] for r in records) / n, 4),
                          "choice_mass": round(sum(r["choice_mass"] for r in records) / n, 4),
                          "predicted": {letter: sum(r["pred"] == k for r in records) for k, letter in enumerate(LETTERS)}}), flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
