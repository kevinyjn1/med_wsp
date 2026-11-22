"""Answer extraction, grading and the repetition measure used in the Qwen math evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "eval_mlx", Path(__file__).resolve().parents[1] / "experiments" / "exp_03_mit8_math" / "eval_mlx.py"
)
eval_mlx = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_mlx)


def test_last_boxed_takes_the_last_closed_box_with_nested_braces():
    assert eval_mlx.last_boxed(r"first \boxed{7}, finally \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert eval_mlx.last_boxed(r"closed \boxed{3} then unclosed \boxed{\frac{3}{4}") == "3"
    assert eval_mlx.last_boxed("no box here") is None


def test_grading_accepts_equivalent_forms_only():
    assert eval_mlx.grade("42", "42")
    assert eval_mlx.grade(r"\frac{1}{2}", "0.5")
    assert not eval_mlx.grade("41", "42")
    assert not eval_mlx.grade(None, "1")


def test_repetition_share():
    assert eval_mlx.repetition_share(list(range(600))) == 0.0
    assert eval_mlx.repetition_share(list(range(10)) * 80) == 1.0
    assert eval_mlx.repetition_share([1, 2, 3]) == 0.0
