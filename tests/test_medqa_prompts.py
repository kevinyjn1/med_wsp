"""Prompt formats and answer-letter tokens for MedQA scoring."""

from __future__ import annotations

import pytest

from wsp import medqa, models

ITEM = {"id": "x", "question": "Which vitamin prevents scurvy?", "answer": "C",
        "options": {"A": "Vitamin A", "B": "Vitamin B12", "C": "Vitamin C", "D": "Vitamin D"}}  # fmt: skip


@pytest.fixture(scope="module")
def tokenizer():
    try:
        return models.load_tokenizer("phi")
    except FileNotFoundError:
        pytest.skip("the Phi-3.5 tokenizer is not in the local Hugging Face cache")


def test_bare_prompt_is_the_harness_format(tokenizer):
    expected = "Question: Which vitamin prevents scurvy?\nA. Vitamin A\nB. Vitamin B12\nC. Vitamin C\nD. Vitamin D\nAnswer:"
    assert medqa.build_prompt(tokenizer, ITEM, "bare") == expected


def test_chat_prompt_has_the_template_and_the_instruction(tokenizer):
    prompt = medqa.build_prompt(tokenizer, ITEM)
    assert prompt.startswith("<|user|>") and prompt.endswith("<|assistant|>\n")
    assert "Answer with the letter of the correct option only." in prompt


def test_unknown_style_is_rejected(tokenizer):
    with pytest.raises(ValueError, match="unknown prompt style"):
        medqa.build_prompt(tokenizer, ITEM, "raw")


def test_bare_letter_tokens_are_the_single_continuation_after_the_prompt(tokenizer):
    ids = medqa.letter_token_ids(tokenizer, "bare")
    for letter in medqa.LETTERS:
        assert len(ids[letter]) == 1
        joint = tokenizer.encode(medqa.bare_prompt(ITEM) + " " + letter, add_special_tokens=False)
        assert joint[-1] == ids[letter][0]
        assert joint[:-1] == tokenizer.encode(medqa.bare_prompt(ITEM), add_special_tokens=False)
    assert set(ids["A"]) <= set(medqa.letter_token_ids(tokenizer, "chat")["A"])
