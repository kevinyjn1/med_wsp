"""Run one Astra weight-editing session on a Phi-3.5-family model with the MedQA edit lab.

The task message follows the structure of MIT experiment 8's prompt (TASK, HOW IT WILL BE MEASURED, WHAT
COUNTS, WHAT YOU HAVE, HOW TO WORK, RULES, HOW THE SESSION RUNS), rewritten for multiple-choice medical
questions scored in one forward pass. With a donor checkpoint the session becomes "find what the fine-tune
changed and move it into the base"; without one it is failure-guided editing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from wsp import medqa, models
from wsp.gpu_lock import gpu_lock
from wsp.harness.loop import BLOCK_RE, Loop, State
from wsp.lab import MedQALab
from wsp.patching import swapped_weights

EXPLAIN = (
    "EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to "
    "8 sentences): what the last result showed, what you conclude from it, what you will do now and why, and what "
    "outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that "
    "led to your edits. These explanations are read by the experimenters."
)
UNDELIVERABLE = (
    "[The output of your last step could not be delivered: the provider refused the message or the call failed "
    "repeatedly. Medical question texts can trigger content filters. Repeat what you need, but print only short "
    "excerpts (at most about 300 characters each), question ids and statistics, not whole question texts.]"
)

DONOR_TASK = (
    " You also have the weights of a second model of the same architecture, {donor_id}, a medical fine-tune of the "
    "first (the DONOR). A second goal is to find out which weight changes in the donor matter for medical questions "
    "and to move only those into the first model, rather than copying the donor wholesale."
)
DONOR_COUNTS = (
    " Copying whole donor tensors is allowed as an experiment, but the final edit should be targeted: say which "
    "components you took from the donor (or rebuilt from it) and why those and not the rest. Replacing most of the "
    "model by the donor is not what we are looking for."
)
DONOR_HAVE = (
    " load_donor(name) gives the donor's version of any tensor (float32), and PRACTICE_DIR/donor_results.jsonl has "
    "the donor's result on every practice question in the same format."
)
DONOR_FORWARD = ' "model" can also be "donor" (the donor\'s unedited weights) in both kinds of pass.'


def task_message(*, model_id: str, donor_id: str | None, budget: int, n_practice: int, accuracy: float,
                 donor_accuracy: float | None, max_runs: int) -> str:  # fmt: skip
    donor = donor_id is not None
    return f"""TASK. You are given the weights of a language model, {model_id} (a decoder-only transformer: 32 layers of width 3,072; no biases anywhere; RMSNorm before attention (input_layernorm) and before the MLP (post_attention_layernorm); self-attention with 32 heads of size 96 and rotary position embeddings, stored as one fused matrix self_attn.qkv_proj.weight of shape 9,216 x 3,072 whose rows 0..3,071 are the queries, 3,072..6,143 the keys and 6,144..9,215 the values, head h using rows 96h..96h+95 of each part, and self_attn.o_proj.weight of shape 3,072 x 3,072 whose columns 96h..96h+95 belong to head h; a gated MLP of width 8,192: mlp.gate_up_proj.weight of shape 16,384 x 3,072 with rows 0..8,191 the gate and 8,192..16,383 the up projection, hidden = up * SiLU(gate), and mlp.down_proj.weight of shape 3,072 x 8,192; vocabulary 32,064 with separate input embedding and output layer; tensors are named model.layers.<l>.<...>, matrices are (out, in)). Your job is to make it better at answering medical licensing exam questions by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network on the questions it answers incorrectly and repair that with targeted changes.{DONOR_TASK.format(donor_id=donor_id) if donor else ""}

HOW IT WILL BE MEASURED. After you finish, your edited model and the unedited model are evaluated on a held-out set of four-option medical licensing questions (USMLE style) with this fixed protocol: the chat prompt shown in PRACTICE_DIR/eval_prompt_format.txt (an instruction, the question, the options A to D and "Answer:"), one forward pass and no generation; the answer is the letter among A, B, C and D with the highest next-token probability at the end of the prompt (the probabilities of the spellings "A" and " A" are added, likewise for the other letters). The config, tokenizer and prompt are fixed: only weight values can change (same tensor names and shapes). You never see the evaluation questions.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when it answers practice questions incorrectly, identify the components responsible (specific neurons, heads, directions or circuits) and change them in a way you can justify with evidence. Out of scope, and refused by save_edit and by the forward-pass tool: the final norm (model.norm.weight), the input embedding and the output layer (model.embed_tokens.weight, lm_head.weight), and any edit that only multiplies a whole tensor by a constant. Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for. A systematic preference for or against particular answer letters is a real failure mechanism and repairing it is in scope, but then say clearly how much of any gain comes from that and how much from changes in how the model uses the content of the question.{DONOR_COUNTS if donor else ""}

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU): variables survive between your code blocks, like a notebook. The model's files are read-only in MODEL_DIR, the practice files read-only in PRACTICE_DIR, and your current directory (WORK) is your read-write folder. Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, networkx, safetensors, tokenizers. Defined in advance: NAMES (all tensor names), load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited() and tok (the tokenizer, a `tokenizers` object).{DONOR_HAVE if donor else ""} Gradients are disabled. To run code, reply with your explanation followed by exactly one block that starts with ```python; if its last line is an expression, its value is shown. One execution may take up to 900 seconds and should stay under about 12 GB of memory (one large matrix as float32 is 0.1 to 0.2 GB); output is cut at 10,000 characters. You may run code up to {max_runs} times. If the kernel is restarted (time limit or crash), its variables are lost and you are told so.
2. Practice questions with answers (none of them is in the evaluation): PRACTICE_DIR/questions.jsonl, one JSON object per line with id, question, options (A to D) and answer, {n_practice:,} questions; and PRACTICE_DIR/original_results.jsonl with the original model's result on each of them under the evaluation protocol: pred, correct, margin (log-probability of the correct letter minus the best other letter) and letter_logprobs (A, B, C, D). The original model answers {100 * accuracy:.1f} % of them correctly; the others are its failures.{f" The donor answers {100 * donor_accuracy:.1f} % of them correctly." if donor else ""}
3. A forward-pass tool with a budget of {budget} forward passes in total. A forward pass is one of two things, and each counts as one pass:
 (a) SCORE: the model answers up to 1,024 practice questions with the evaluation protocol. Reply with your explanation followed by exactly one block that starts with ```forward and contains JSON:
 {{"type": "score", "model": "original", "edited" or "both", "questions": ["<practice id>", ...]}}
 "original" runs the unedited weights, "edited" your current edits (the tensors in WORK/edited, rounded to bfloat16 like a saved checkpoint), and "both" runs both on the same questions and returns a paired comparison (the accuracy difference edited - original with its standard error, how many questions went from wrong to right and back with a sign test, and the mean change of the margin with its standard error). You get the accuracy, the mean margin and the counts of predicted letters; per-question results are in WORK/score_<k>.jsonl. A pass with 1,024 questions takes about 5 minutes per model.
 (b) TEACHER-FORCED: the model reads practice questions (as their evaluation prompts) and/or raw texts, with no generation. Reply with a ```forward block containing JSON:
 {{"type": "teacher_forced", "model": "original" or "edited", "questions": ["<practice id>", ...], "texts": ["raw text", ...] (optional), "return": {{"positions": "last" or "all", "token_logprobs": true, "topk": k (at most 50), "hidden_states": [layers 0..32], "mlp_act": [layers 0..31], "attn_layers": [layers 0..31]}}}}
 hidden_states layer 0 is the embeddings and layer l the residual stream after layer l; mlp_act is the input of mlp.down_proj (the 8,192 hidden units, after the gate); attn_layers gives attention probabilities per head. With "positions": "last" (the default) every array is taken at the last token of each text, where the answer letter is predicted, and up to 1,024 texts are allowed; with "all" arrays cover every token, right-padded, for at most 64 texts and 40,000 tokens in total. Attention is returned for at most 4 texts of at most 700 tokens each. The results are float16 .npy files with meta.json in WORK/fwd_<k>/.{DONOR_FORWARD if donor else ""}
 A request that fails does not count against the budget.

HOW TO WORK.
1. Diagnose first. Look at how the model fails on practice questions: which letters it over- or under-predicts, how confident it is when wrong, whether failures cluster by kind of question. Go after the largest category you can explain.
2. Localize causally. That a component is more active on correctly answered questions is not enough (those questions are easier). Before relying on a component, show that changing it changes the behaviour you predicted, and try several strengths to see a dose-response.
3. Measure with enough power. Accuracy over n questions has a standard error of about 50/sqrt(n) percentage points: about 3.1 points at 256 questions, 1.6 at 1,024. Compare original and edited weights on the SAME questions with "model": "both": the paired difference has a much smaller standard error than two separate measurements. The margin is a denser signal than accuracy. Keep an edit only if its gain is clearly larger than its standard error on questions that were NOT used to choose it: set aside a part of the practice questions for a final confirmation and do not look at it before. Edits that change a tensor by a fraction of a percent usually have effects too small to see; test strengths large enough to matter and back off if they hurt.
4. Plan your forward passes (see the budget above): for example teacher-forced passes to read activations on failures and successes, score passes to test candidate edits, and a final paired score pass on questions you have not used.
5. Work efficiently: state the hypothesis of each code block in its first comment line; load things once into kernel variables; print compact summaries, never whole arrays or whole question texts (short excerpts of at most about 300 characters; outputs are shown in ASCII only); keep WORK/notes.md up to date (findings with numbers, edits tried and their measured effect, the current best edit, which questions are set aside; it is shown to you again if the conversation is restarted); keep copies of good edits (for example copy WORK/edited to WORK/best_1).

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not implement or simulate the network's computation on any input in the kernel; (c) internet access or search (there is none). Allowed: any analysis of the weights and of forward-pass results, any use of the practice questions, and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. It continues until all {budget} forward passes are used (replying without a code block does not end it earlier); after that, make your final choice (the tensors in WORK/edited when you finish are submitted; leave it empty if no edit earned its place) and reply without a code block, saying what you changed, the evidence for it and why you expect it to help. {EXPLAIN}"""


def ascii_only(text: str) -> str:
    return text.encode("ascii", "replace").decode()


def drive(lab, message: str, *, run_dir: Path, budget: int, max_runs: int = 300, effort: str = "xhigh",
          max_turns: int = 400) -> State:  # fmt: skip
    """Run one session of any budgeted lab: explanations required, restarts every 40 replies, until the passes are used."""

    def format_output(kind: str, output: str, state: State) -> str:
        if kind == "forward":
            return ascii_only(output)
        return f"OUTPUT of Python run {state.runs.get('python', 0)} of at most {max_runs}:\n```\n{ascii_only(output)}\n```"

    def precheck(answer: str, kind: str, state: State) -> str | None:
        """A block must come with an explanation; a bare block is sent back unrun, at most twice in a row."""
        before = answer[: BLOCK_RE.search(answer).start()].strip()
        if len(re.sub(r"\s+", " ", before)) >= 80 or state.unexplained >= 2:
            state.unexplained = 0
            return None
        state.unexplained += 1
        return ("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). "
                + EXPLAIN + " Send the block again, preceded by your explanation.")  # fmt: skip

    def on_final(answer: str, state: State) -> str | None:
        if lab.passes_left() > 0:
            return (f"You have {lab.passes_left()} of {budget} forward passes left, and the session continues until all "
                    "are used. Use them to test what you have not tested yet (for example a confirmation on questions you "
                    "set aside, or a stronger or more targeted variant of your best edit), then give your final answer.")  # fmt: skip
        return None

    def handoff(state: State, last_message: str) -> str:
        notes = lab.work / "notes.md"
        notebook = notes.read_text()[-8000:] if notes.exists() else "(empty)"
        edits = (
            sorted(p.name[:-4] for p in (lab.work / "edited").glob("*.npy"))
            if (lab.work / "edited").is_dir()
            else []
        )
        files = sorted(p.name for p in lab.work.iterdir() if p.name != "practice")
        return (f"THIS IS A RESTARTED CONVERSATION (to keep it short): {state.turn} replies happened before. Forward passes "
                f"used: {lab.state['passes']} of {budget}; Python runs used: {state.runs.get('python', 0)} of {max_runs}. Your "
                f"kernel is still running with its variables. Tensors currently edited: {edits}. Files in WORK: {files}.\n"
                f"YOUR NOTEBOOK (WORK/notes.md, last 8,000 characters):\n```\n{ascii_only(notebook)}\n```\nContinue from here.\n\n"
                f"Last result:\n{last_message}")  # fmt: skip

    loop = Loop(
        run_dir=run_dir, first_message=message, handlers={"python": lab.run_python, "forward": lab.forward},
        max_runs=max_runs, count_as_run=("python",), effort=effort, max_turns=max_turns, on_final=on_final,
        format_output=format_output, precheck=precheck, handoff_every=40, handoff=handoff, undeliverable_notice=UNDELIVERABLE,
    )  # fmt: skip
    try:
        lab.kernel.start()
        return loop.run()
    finally:
        lab.kernel.stop()


def run_session(run_dir: Path, *, model: str = "phi", donor: str | None = None, budget: int = 30, effort: str = "xhigh",
                practice: list[dict], baseline: list[dict], donor_baseline: list[dict] | None = None,
                max_runs: int = 300, max_turns: int = 400) -> State:  # fmt: skip
    run_dir.mkdir(parents=True, exist_ok=True)
    lab = MedQALab(
        run_dir,
        model=model,
        donor=donor,
        budget=budget,
        practice=practice,
        baseline=baseline,
        max_python_runs=max_runs,
    )
    if donor_baseline is not None:
        medqa.save_jsonl(donor_baseline, lab.practice_dir / "donor_results.jsonl")
    message = task_message(
        model_id=models.MODEL_IDS[model].split("/")[-1], donor_id=models.MODEL_IDS[donor].split("/")[-1] if donor else None,
        budget=budget, n_practice=len(practice), accuracy=float(np.mean([r["correct"] for r in baseline])),
        donor_accuracy=float(np.mean([r["correct"] for r in donor_baseline])) if donor_baseline else None, max_runs=max_runs,
    )  # fmt: skip

    return drive(
        lab, message, run_dir=run_dir, budget=budget, max_runs=max_runs, effort=effort, max_turns=max_turns
    )


@torch.no_grad()
def final_evaluation(run_dir: Path, items: list[dict], *, model: str = "phi") -> dict:
    """Score the original and the submitted edit on held-out questions (paired) and describe the edit."""
    edited_dir = run_dir / "work" / "edited"
    lab = MedQALab.__new__(
        MedQALab
    )  # only current_edits is needed, with the same validation as during the session
    lab.work = run_dir / "work"
    tokenizer = models.load_tokenizer("phi")
    with gpu_lock(f"final evaluation {run_dir.name}"):
        net = models.load_model(model)
        edits = lab.current_edits(net)
        params = dict(net.named_parameters())
        described = [{"name": name, "relative_change": float((tensor.float() - params[name].detach().cpu().float()).norm()
                                                               / params[name].detach().cpu().float().norm()),
                      "entries_changed": int((tensor.float() != params[name].detach().cpu().to(torch.bfloat16).float()).sum())}
                     for name, tensor in edits.items()]  # fmt: skip
        original = medqa.evaluate(net, tokenizer, items, 16, progress=False)
        if edits:
            with swapped_weights(net, edits):
                edited = medqa.evaluate(net, tokenizer, items, 16, progress=False)
        else:
            edited = original
        del net
    o = np.array([r["correct"] for r in original])
    e = np.array([r["correct"] for r in edited])
    from scipy.stats import binomtest

    fixed, broken = int((e & ~o).sum()), int((~e & o).sum())
    delta = np.array([b["margin"] - a["margin"] for a, b in zip(original, edited, strict=True)])
    result = {
        "n": len(items), "original_accuracy": float(o.mean()), "edited_accuracy": float(e.mean()),
        "accuracy_difference": float(e.mean() - o.mean()), "wrong_to_right": fixed, "right_to_wrong": broken,
        "sign_test_p": float(binomtest(fixed, fixed + broken, 0.5).pvalue) if fixed + broken else 1.0,
        "margin_difference": float(delta.mean()), "margin_difference_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "predicted_original": {letter: sum(r["pred"] == letter for r in original) for letter in medqa.LETTERS},
        "predicted_edited": {letter: sum(r["pred"] == letter for r in edited) for letter in medqa.LETTERS},
        "edited_tensors": described, "edited_dir_exists": edited_dir.is_dir(),
    }  # fmt: skip
    (run_dir / "final_eval.json").write_text(json.dumps(result, indent=1))
    medqa.save_jsonl([{"id": a["id"], "original": a, "edited": b} for a, b in zip(original, edited, strict=True)],
                     run_dir / "final_eval_per_question.jsonl")  # fmt: skip
    return result
