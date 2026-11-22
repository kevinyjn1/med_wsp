"""Answer extraction, grading and the repetition measure used in the Qwen math evaluation."""

from __future__ import annotations

from wsp import mathqa as eval_mlx


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


def test_judge_names_the_failure():
    assert eval_mlx.judge(r"so \boxed{4}", "4", 50, 1000)["failure"] is None
    assert eval_mlx.judge(r"so \boxed{5}", "4", 50, 1000)["failure"] == "wrong_answer"
    assert eval_mlx.judge("thinking forever", "4", 1000, 1000)["failure"] == "cut_at_max_tokens"
    assert eval_mlx.judge("I stop here", "4", 50, 1000)["failure"] == "no_answer_finished"
