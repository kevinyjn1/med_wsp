"""Math answer extraction and grading, as in MIT experiment 8: the last \\boxed{} answer, checked with math-verify."""

from __future__ import annotations

import re

import numpy as np

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} or \\fbox{...} with balanced braces; None when there is no closed box."""
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


def judge(text: str, gold: str, n_tokens: int, max_tokens: int) -> dict:
    """Grade one generated answer the way the evaluation does, and say how it failed if it did."""
    cut = n_tokens >= max_tokens - 2
    prediction = last_boxed(text)
    correct = grade(prediction, gold)
    failure = (
        None
        if correct
        else ("cut_at_max_tokens" if cut else "no_answer_finished" if prediction is None else "wrong_answer")
    )
    return {"pred": prediction, "correct": correct, "tokens": n_tokens, "cut": cut, "failure": failure}
