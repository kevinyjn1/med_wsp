"""MedQA (USMLE, 4 options): data loading, the prompt, and single-forward-pass letter scoring.

Scoring reads the next-token logits right after the assistant header and compares the four answer
letters. One forward pass per question, deterministic, and it gives a logit margin that activation
patching and weight edits can be measured against.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

LETTERS = ("A", "B", "C", "D")
SPLIT_FILES = {"train": "train", "dev": "validation", "test": "test"}
INSTRUCTION = (
    "The following is a multiple choice question about medical knowledge. "
    "Answer with the letter of the correct option only."
)


def load_medqa(split: str) -> list[dict]:
    """Official MedQA-USMLE 4-option split: train 10,178 / dev 1,272 / test 1,273 questions."""
    path = hf_hub_download(
        "bigbio/med_qa",
        f"med_qa_en_4options_source/{SPLIT_FILES[split]}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    items = []
    for index, row in enumerate(pd.read_parquet(path).to_dict("records")):
        options = {opt["key"]: opt["value"] for opt in row["options"]}
        items.append({
            "id": f"{split}-{index:05d}", "question": row["question"], "options": options,
            "answer": row["answer_idx"], "meta": row.get("meta_info", ""),
        })  # fmt: skip
    return items


def user_message(item: dict) -> str:
    options = "\n".join(f"{letter}. {item['options'][letter]}" for letter in LETTERS)
    return f"{INSTRUCTION}\n\n{item['question']}\n\n{options}\n\nAnswer:"


STYLES = ("chat", "bare")


def bare_prompt(item: dict) -> str:
    """The plain-text prompt of lm-evaluation-harness's `medqa_4options` task: no chat template, no instruction."""
    options = "".join(f"{letter}. {item['options'][letter]}\n" for letter in LETTERS)
    return f"Question: {item['question']}\n{options}Answer:"


def build_prompt(tokenizer, item: dict, style: str = "chat") -> str:
    """ "chat": our instruction wrapped in the model's chat template. "bare": the harness's plain-text prompt."""
    if style == "bare":
        return bare_prompt(item)
    if style != "chat":
        raise ValueError(f"unknown prompt style {style!r}: use one of {STYLES}")
    messages = [{"role": "user", "content": user_message(item)}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def letter_token_ids(tokenizer, style: str = "chat") -> dict[str, list[int]]:
    """Token ids that count as each answer letter.

    "chat": every single-token spelling ("A" and the word-initial " A"). "bare": only the continuation the harness
    scores, " A" as it is tokenized right after "Answer:" (one token; encoded on its own it would be two).
    """
    if style == "bare":
        stem = tokenizer.encode("Answer:", add_special_tokens=False)
        ids = {}
        for letter in LETTERS:
            continuation = tokenizer.encode("Answer: " + letter, add_special_tokens=False)[len(stem) :]
            if len(continuation) != 1:
                raise ValueError(f"' {letter}' after 'Answer:' is not a single token: {continuation}")
            ids[letter] = continuation
        return ids
    ids = {}
    for letter in LETTERS:
        found = set()
        for text in (letter, " " + letter):
            pieces = tokenizer.encode(text, add_special_tokens=False)
            if len(pieces) == 1:
                found.add(pieces[0])
        for piece in (letter, "▁" + letter):
            token_id = tokenizer.convert_tokens_to_ids(piece)
            if token_id is not None and token_id != tokenizer.unk_token_id:
                found.add(token_id)
        ids[letter] = sorted(found)
    return ids


def letter_scores(last_logits: torch.Tensor, letter_ids: dict[str, list[int]]) -> torch.Tensor:
    """(batch, vocab) next-token logits -> (batch, 4) letter log-probabilities (summed over spellings)."""
    logprobs = torch.log_softmax(last_logits.float(), dim=-1)
    return torch.stack(
        [torch.logsumexp(logprobs[:, letter_ids[letter]], dim=-1) for letter in LETTERS], dim=-1
    )


def gold_margins(last_logits: torch.Tensor, letter_ids: dict[str, list[int]], gold: str) -> torch.Tensor:
    """(batch, vocab) next-token logits -> (batch,) log-prob of the gold letter minus the best other letter."""
    scores = letter_scores(last_logits, letter_ids).cpu()
    index = LETTERS.index(gold)
    others = torch.cat([scores[:, :index], scores[:, index + 1 :]], dim=1).max(dim=1).values
    return scores[:, index] - others


@torch.no_grad()
def evaluate(model, tokenizer, items: list[dict], batch_size: int = 8, progress: bool = True,
             style: str = "chat") -> list[dict]:  # fmt: skip
    """Score every item; returns one record per item with letter log-probs, prediction and margin."""
    from tqdm import tqdm

    letter_ids = letter_token_ids(tokenizer, style)
    prompts = [build_prompt(tokenizer, item, style) for item in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))  # similar lengths per batch
    records: dict[int, dict] = {}
    for start in tqdm(range(0, len(order), batch_size), disable=not progress):
        batch = order[start : start + batch_size]
        enc = tokenizer(
            [prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        scores = letter_scores(model(**enc, use_cache=False).logits[:, -1, :], letter_ids).cpu()
        for row, i in enumerate(batch):
            item, lp = items[i], scores[row]
            gold = LETTERS.index(item["answer"])
            others = torch.cat([lp[:gold], lp[gold + 1 :]])
            records[i] = {
                "id": item["id"], "answer": item["answer"], "pred": LETTERS[int(lp.argmax())],
                "correct": bool(lp.argmax() == gold), "margin": float(lp[gold] - others.max()),
                "letter_logprobs": [round(float(x), 4) for x in lp], "letter_mass": float(lp.exp().sum()),
            }  # fmt: skip
    return [records[i] for i in range(len(items))]


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
