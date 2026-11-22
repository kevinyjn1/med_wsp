"""The edit lab for Qwen3.5-0.8B and math: a scaled-down rebuild of the MIT experiment 8 harness.

Same pieces as MIT's (persistent kernel over the weights, practice problems, a forward-pass tool with a budget of
GENERATION and TEACHER-FORCED passes, the same rules), sized for one Apple M4 Max:

  generation   MLX, the evaluation's sampling; at most 64 answers per pass (32 per model with "both") and at most
               16,384 new tokens (MIT: 512 answers, 32,768 tokens on a rented GPU)
  teacher      PyTorch; at most 256 texts of at most 4,096 tokens, 100,000 tokens in total (MIT: 1,024 texts,
               524,288 tokens)

Not rebuilt: the "activations" option of generation passes, and per-problem sampling seeds shared between the
original and the edited model (MLX samples a batch from one random stream).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from wsp import mathqa
from wsp.checkpoints import Checkpoint
from wsp.lab import BudgetedLab, make_prelude, validate_edit
from wsp.lab_session import EXPLAIN, drive
from wsp.models import snapshot_dir

QWEN = "Qwen/Qwen3.5-0.8B"
PREFIX = "model.language_model."
FROZEN = (PREFIX + "norm.weight", PREFIX + "embed_tokens.weight", "lm_head.weight")
MAX_GENERATIONS, MAX_PER_MODEL_BOTH, MAX_TOKENS = 64, 32, 16_384
MAX_TEXTS, MAX_TEXT_TOKENS, MAX_TOTAL_TOKENS, MAX_ATTN_TEXTS, MAX_ATTN_TOKENS = 256, 4096, 100_000, 2, 2048
EVAL_PREFIX = "<|im_start|>user\n"
EVAL_SUFFIX = "\n" + mathqa.INSTRUCTION + "<|im_end|>\n<|im_start|>assistant\n<think>\n"
REPO = Path(__file__).resolve().parents[1]
MIT_PRACTICE = REPO / "external" / "lm_understand-logs" / "exp8_math" / "practice"


class QwenMathLab(BudgetedLab):
    lock_name = (
        "batch"  # passes are long; they queue with other sweeps and do not block a Phi lab's short passes
    )

    def __init__(self, run_dir: Path, *, budget: int, max_python_runs: int = 300):
        self.work = run_dir / "work"
        self.practice_dir = self.work / "practice"
        self.practice_dir.mkdir(parents=True, exist_ok=True)
        self.practice: dict[str, dict] = {}
        for name in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
            text = (MIT_PRACTICE / name).read_text()
            (self.practice_dir / name).write_text(text)
            for line in text.split("\n"):  # not splitlines(): problem texts contain other Unicode line breaks
                if not line.strip():
                    continue
                row = json.loads(line)
                self.practice[row["id"]] = row
        (self.practice_dir / "eval_prompt_format.txt").write_text(
            f"PREFIX = {EVAL_PREFIX!r}\nSUFFIX = {EVAL_SUFFIX!r}\nThe evaluation prompt is PREFIX + problem + SUFFIX.\n"
        )
        model_dir = snapshot_dir(QWEN)
        self.checkpoint = Checkpoint(model_dir)
        prelude = make_prelude(model_dir, self.practice_dir, FROZEN, prefix=PREFIX)
        self._init_runtime(
            run_dir,
            budget=budget,
            prelude=prelude,
            read_only=[model_dir.parents[1]],
            max_python_runs=max_python_runs,
        )

    # ---------------------------------------------------------------------------------------------- edits
    def edited_dir(self) -> Path | None:
        """Validate every file in edited/ (same rules as save_edit) and return the folder, or None when it is empty."""
        folder = self.work / "edited"
        files = sorted(folder.glob("*.npy")) if folder.is_dir() else []
        for path in files:
            name = path.name[: -len(".npy")]
            if name not in self.checkpoint.names() or not name.startswith(PREFIX):
                raise ValueError(f"edited/{path.name}: {name} is not a tensor of the language model")
            validate_edit(
                name,
                torch.from_numpy(np.load(path)).to(torch.float32),
                self.checkpoint.get(name),
                frozen=FROZEN,
            )
        return folder if files else None

    def _worker(self, kind: str, job: dict, index: int) -> None:
        job_path = self.run_dir / "jobs" / f"pass_{index}_{job.get('tag', kind)}.json"
        job_path.parent.mkdir(exist_ok=True)
        job_path.write_text(json.dumps(job))
        result = subprocess.run([sys.executable, "-m", "wsp.qwen_worker", kind, str(job_path)], capture_output=True, text=True,
                                cwd=REPO, check=False)  # fmt: skip
        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "the model run failed"
            )

    # ------------------------------------------------------------------------------------------- dispatch
    def _dispatch(self, request: dict, index: int) -> tuple[dict, str]:
        kind = request.get("type")
        if kind == "generate":
            return self._generate(request, index)
        if kind == "teacher_forced":
            return self._teacher_forced(request, index)
        raise ValueError(f'unknown type {kind!r}: use "generate" or "teacher_forced"')

    def _generate(self, request: dict, index: int) -> tuple[dict, str]:
        which = request.get("model", "original")
        if which not in ("original", "edited", "both"):
            raise ValueError('"model" must be "original", "edited" or "both"')
        if "activations" in request:
            raise ValueError(
                'the "activations" option is not available in this lab; read activations with a teacher-forced pass'
            )
        ids = list(request.get("problems", []))
        unknown = [i for i in ids if i not in self.practice]
        if unknown:
            raise ValueError(f"unknown practice ids: {unknown[:5]}")
        items = [{"id": i, "source": self.practice[i]["source"], "answer": str(self.practice[i]["answer"]),
                  "message": self.practice[i]["problem"] + "\n" + mathqa.INSTRUCTION} for i in ids]  # fmt: skip
        items += [
            {"id": f"prompt-{k}", "source": "prompt", "answer": None, "message": text}
            for k, text in enumerate(request.get("prompts", []))
        ]
        n = int(request.get("n", 1))
        max_tokens = int(request.get("max_tokens", MAX_TOKENS))
        limit = MAX_PER_MODEL_BOTH if which == "both" else MAX_GENERATIONS
        if not items or n < 1 or len(items) * n > limit:
            raise ValueError(
                f"items x n must be between 1 and {limit}"
                + (' per model with "both"' if which == "both" else "")
            )
        if not 1 <= max_tokens <= MAX_TOKENS:
            raise ValueError(f"max_tokens is at most {MAX_TOKENS}")
        edited_dir = self.edited_dir() if which in ("edited", "both") else None
        if which in ("edited", "both") and edited_dir is None:
            raise ValueError("there are no edits in edited/ to run")

        results, files = {}, {}
        for model in ("original", "edited") if which == "both" else (which,):
            out = self.run_dir / "jobs" / f"pass_{index}_{model}.out.json"
            self._worker("generate", {
                "tag": model, "edited_dir": str(edited_dir) if model == "edited" else None, "max_tokens": max_tokens,
                "seed": index, "batch_size": 32, "out": str(out), "user_messages": [item["message"] for item in items for _ in range(n)],
            }, index)  # fmt: skip
            generations = json.loads(out.read_text())
            rows = []
            for k, item in enumerate(items):
                samples = []
                for generation in generations[k * n : (k + 1) * n]:
                    judged = mathqa.judge(generation["text"], item["answer"], generation["tokens"], max_tokens) if item["answer"] is not None else {
                        "pred": mathqa.last_boxed(generation["text"]), "correct": None, "tokens": generation["tokens"],
                        "cut": generation["tokens"] >= max_tokens - 2, "failure": None}  # fmt: skip
                    samples.append({"text": generation["text"], **judged})
                rows.append(
                    {"id": item["id"], "source": item["source"], "answer": item["answer"], "samples": samples}
                )
            name = f"gen_{index}_{model}.jsonl" if which == "both" else f"gen_{index}.jsonl"
            (self.work / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            results[model], files[model] = rows, name

        def block(rows: list[dict]) -> dict:
            samples = [(r["source"], s) for r in rows for s in r["samples"]]
            graded = [(src, s) for src, s in samples if s["correct"] is not None]
            by_source = {
                src: round(float(np.mean([s["correct"] for g, s in graded if g == src])), 4)
                for src in sorted({g for g, _ in graded})
            }
            return {
                "generations": len(samples), "graded": len(graded),
                "accuracy": round(float(np.mean([s["correct"] for _, s in graded])), 4) if graded else None,
                "accuracy_by_source": by_source, "mean_tokens": round(float(np.mean([s["tokens"] for _, s in samples])), 1),
                "cut_share": round(float(np.mean([s["cut"] for _, s in samples])), 4),
                "failures": {kind: sum(s["failure"] == kind for _, s in graded) for kind in ("wrong_answer", "no_answer_finished", "cut_at_max_tokens")},
            }  # fmt: skip

        summary = {"ok": True, "type": "generate", "model": which, "max_tokens": max_tokens,
                   "edited_tensors": sorted(p.name[:-4] for p in edited_dir.glob("*.npy")) if edited_dir else []}  # fmt: skip
        if which == "both":
            summary["files"] = files
            summary.update({model: block(rows) for model, rows in results.items()})
            original = np.array(
                [
                    np.mean([s["correct"] for s in r["samples"]])
                    for r in results["original"]
                    if r["answer"] is not None
                ]
            )
            edited = np.array(
                [
                    np.mean([s["correct"] for s in r["samples"]])
                    for r in results["edited"]
                    if r["answer"] is not None
                ]
            )
            if len(original):
                difference = edited - original
                summary["paired"] = {
                    "items": len(difference), "accuracy_difference": round(float(difference.mean()), 4),
                    "standard_error": round(float(difference.std(ddof=1) / np.sqrt(len(difference))), 4) if len(difference) > 1 else None,
                    "items_wrong_to_right": int((difference > 0).sum()), "items_right_to_wrong": int((difference < 0).sum()),
                    "note": "the two models are sampled independently (no shared random numbers)",
                }  # fmt: skip
            return summary, " and ".join(files.values())
        summary.update({"file": files[which], **block(results[which])})
        return summary, files[which]

    def _teacher_forced(self, request: dict, index: int) -> tuple[dict, str]:
        which = request.get("model", "original")
        if which not in ("original", "edited"):
            raise ValueError('"model" must be "original" or "edited" for a teacher-forced pass')
        want = request.get("return", {})
        texts = list(request.get("texts", []))
        if request.get("chats"):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(snapshot_dir(QWEN))
            thinking = bool(request.get("enable_thinking", False))
            texts += [
                tokenizer.apply_chat_template(chat, tokenize=False, enable_thinking=thinking)
                for chat in request["chats"]
            ]
        if not texts:
            raise ValueError('give "texts" (raw texts) or "chats"')
        from tokenizers import Tokenizer

        fast = Tokenizer.from_file(str(snapshot_dir(QWEN) / "tokenizer.json"))
        lengths = [len(fast.encode(text, add_special_tokens=False).ids) for text in texts]
        if len(texts) > MAX_TEXTS or max(lengths) > MAX_TEXT_TOKENS or sum(lengths) > MAX_TOTAL_TOKENS:
            raise ValueError(f"at most {MAX_TEXTS} texts of at most {MAX_TEXT_TOKENS} tokens, {MAX_TOTAL_TOKENS} tokens in total "
                             f"(got {len(texts)} texts, longest {max(lengths)}, total {sum(lengths)})")  # fmt: skip
        if want.get("attn_layers") and (len(texts) > MAX_ATTN_TEXTS or max(lengths) > MAX_ATTN_TOKENS):
            raise ValueError(
                f"attention is returned for at most {MAX_ATTN_TEXTS} texts of at most {MAX_ATTN_TOKENS} tokens"
            )
        if int(want.get("topk", 0)) > 50:
            raise ValueError("topk is at most 50")
        if any(not 0 <= layer <= 24 for layer in want.get("hidden_states", [])) or any(
            not 0 <= layer <= 23 for layer in want.get("mlp_act", [])
        ):
            raise ValueError("hidden_states layers are 0..24 and mlp_act layers 0..23")
        width = 1024 * len(want.get("hidden_states", [])) + 3584 * len(want.get("mlp_act", []))
        if 2 * width * len(texts) * max(lengths) > 4e9:
            raise ValueError(
                "the requested arrays would exceed 4 GB; ask for fewer layers or fewer or shorter texts"
            )
        edited_dir = self.edited_dir() if which == "edited" else None
        if which == "edited" and edited_dir is None:
            raise ValueError("there are no edits in edited/")
        out = self.run_dir / "jobs" / f"pass_{index}_tf.out.json"
        self._worker("teacher_forced", {"tag": "tf", "edited_dir": str(edited_dir) if edited_dir else None, "texts": texts,
                                        "return": want, "out": str(out), "out_dir": str(self.work / f"fwd_{index}")}, index)  # fmt: skip
        summary = {"ok": True, "type": "teacher_forced", "model": which, **json.loads(out.read_text()),
                   "edited_tensors": sorted(p.name[:-4] for p in edited_dir.glob("*.npy")) if edited_dir else []}  # fmt: skip
        return summary, f"fwd_{index}/"


def task_message(budget: int, max_runs: int = 300) -> str:
    """MIT experiment 8's task message with this lab's limits (see the module docstring for what differs)."""
    return f"""TASK. You are given the weights of a language model, Qwen3.5-0.8B (text part: 24 layers of width 1,024; layers 3, 7, 11, 15, 19 and 23 are full-attention layers with 8 query heads and 2 key/value heads of size 256 and a gated output, the other 18 are Gated DeltaNet linear-attention layers with 16 heads of size 128; each layer has a SwiGLU MLP of width 3,584; RMSNorm; vocabulary 248,320; input embedding and output layer tied). Your job is to make it better at mathematics by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it fails at math problems and repair that with targeted changes.

HOW IT WILL BE MEASURED. After you finish, your edited model and the unedited model are evaluated on problems from MATH-500 and GSM8K with this fixed protocol: the chat template with thinking enabled (enable_thinking=True: the model reasons between <think> and </think>, then answers); the user message is the problem, a newline and "{mathqa.INSTRUCTION}"; sampling with temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5 and at most {MAX_TOKENS:,} new tokens for reasoning and answer together; the last \\boxed{{}} answer is graded against the reference answer. The config, tokenizer, chat template and decoding settings are fixed: only weight values can change (same tensor names and shapes). You never see these benchmarks.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when it fails at math problems, identify the components responsible (specific neurons, heads, directions or circuits) and change them in a way you can justify with evidence. Repairing a specific failure mechanism is in scope: for example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives that repetition counts. Out of scope, and refused by save_edit: the final norm (model.language_model.norm.weight), the input embedding / output layer (model.language_model.embed_tokens.weight, lm_head.weight), and any edit that only multiplies a whole tensor by a constant. Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for, and neither is making the model skip its thinking or close it early regardless of the problem; repairing a specific failure inside the thinking (such as repetition) is in scope.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU): variables survive between your code blocks, like a notebook. The model's files are read-only in MODEL_DIR, practice problems read-only in PRACTICE_DIR, and your current directory (WORK) is your read-write folder. Available: numpy, scipy, scikit-learn, PyTorch (CPU), pandas, sympy, networkx, safetensors, tokenizers. Defined in advance: NAMES (all tensor names of the language model, model.language_model.*), load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited() and tok (the tokenizer). Gradients are disabled. To run code, reply with your explanation followed by exactly one block that starts with ```python; if its last line is an expression, its value is shown. One execution may take up to 900 seconds and should stay under about 12 GB of memory; output is cut at 10,000 characters. You may run code up to {max_runs} times. If the kernel is restarted (time or memory limit), its variables are lost and you are told so.
2. Practice problems with answers (none of them is in the evaluation), one JSON object per line with id, source, problem, answer and, for GSM8K and MATH, a reference solution: PRACTICE_DIR/gsm8k_train.jsonl (7,469 grade-school problems), PRACTICE_DIR/math_train.jsonl (7,275 competition problems, with subject and level 1-5) and PRACTICE_DIR/aime.jsonl (714 AIME problems, 1983-2023, answers only). Choose whichever serve your experiments. PRACTICE_DIR/eval_prompt_format.txt gives the exact evaluation prompt.
3. A forward-pass tool with a budget of {budget} forward passes in total. A forward pass is one of two things, and each counts as one pass:
 (a) GENERATION: the model answers up to {MAX_GENERATIONS} generations (problems x samples) with the evaluation's protocol (thinking enabled, the same sampling). Reply with your explanation followed by exactly one block that starts with ```forward and contains JSON:
 {{"type": "generate", "model": "original", "edited" or "both", "problems": ["<practice id>", ...], "prompts": ["<your own user message>", ...] (optional), "n": samples per item (items x n <= {MAX_GENERATIONS}; with "both" <= {MAX_PER_MODEL_BOTH} per model), "max_tokens": at most {MAX_TOKENS}}}
 "original" runs the unedited weights, "edited" your current edits, and "both" runs both on the same items and returns a paired comparison (the accuracy difference edited - original over items with its standard error, and how many items went from wrong to right and back). The two models are sampled independently (they do not share random numbers). Every answer to a practice problem is graded. You get the accuracy (overall and by source), the mean length, the share cut at max_tokens and the failure counts; texts, extracted answers and grades are in WORK/gen_<k>.jsonl (with "both": WORK/gen_<k>_original.jsonl and WORK/gen_<k>_edited.jsonl). A generation pass takes up to about half an hour at full size (answers can be {MAX_TOKENS:,} tokens long).
 (b) TEACHER-FORCED: the model reads up to {MAX_TEXTS} texts of up to {MAX_TEXT_TOKENS:,} tokens each, at most {MAX_TOTAL_TOKENS:,} tokens in total (no generation; about a minute per 5,000 tokens). Reply with a ```forward block containing JSON:
 {{"type": "teacher_forced", "model": "original" or "edited", "texts": ["raw text", ...] or "chats": [[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], ...] (add "enable_thinking": true for the thinking template), "return": {{"token_logprobs": true, "topk": k (at most 50), "hidden_states": [layers 0..24], "mlp_act": [layers 0..23], "attn_layers": [full-attention layers only; at most {MAX_ATTN_TEXTS} texts of at most {MAX_ATTN_TOKENS:,} tokens]}}}}
 hidden_states layer 0 is the embeddings and layer l the residual stream after layer l; mlp_act is the input of the MLP's down_proj (3,584 hidden units). The results of (b) are float16 .npy files (with meta.json) in WORK/fwd_<k>/, right-padded to the longest text (at most 4 GB per pass). To score a generated answer with a teacher-forced pass, send prompt + answer as one raw text in "texts" ("chats" re-renders assistant turns and drops their reasoning).
 A request that fails does not count against the budget. Your budget counts passes, not their size: small passes give noisy measurements (about 50/sqrt(n) percentage points), so measure with as many answers as a pass allows unless you have a specific reason not to. Separate generation passes do not reproduce each other exactly: compare edited and original only within one "model": "both" pass.

HOW TO WORK.
1. Diagnose first. Look at how the model fails on practice problems: a wrong final answer, finishing without an answer, or being cut off at the token limit (often while repeating itself in its thinking). The generation summary counts these; read some failing texts. Go after the largest category you can explain.
2. Localize causally. That a component is more active on correct answers is not enough (correct answers come from easier and shorter problems). Before relying on a component, show that changing it changes the behaviour you predicted, and try several strengths to see a dose-response.
3. Measure with enough power. Accuracy over n generations has a standard error of about 50/sqrt(n) percentage points: about 8.8 points at 32 generations, 6.3 at 64. Compare original and edited weights on the SAME problems with "model": "both"; do not compare with an accuracy measured on other problems. Keep an edit only if its gain is clearly larger than its standard error on problems that were not used to choose it. A lower share of cut-off answers does not by itself mean more correct answers: judge edits by accuracy. Dense measurements help too and are cheap: one teacher-forced pass can read reference solutions of practice problems (or the model's own good versus failing continuations) and gives a signal on every token. Edits that change a tensor by a fraction of a percent usually have effects too small to see; test strengths large enough to matter and back off if they hurt.
4. Plan your forward passes (see the budget above): for example one diagnostic generation, teacher-forced passes for dense paired measurements, and a final paired generation ("model": "both") on problems you have not used to confirm.
5. Work efficiently: state the hypothesis of each code block in its first comment line; work through the model in parts; load things once into kernel variables; print compact summaries, never whole arrays (and never raw lists of vocabulary tokens: the vocabulary contains offensive and adult strings and printing them can get the session refused by the provider's content filter; print digits, math symbols, common math words or token ids instead; outputs are shown in ASCII only, other characters appear as '?'; likewise print short excerpts of generated texts, at most about 300 characters each, not whole transcripts); before every pass, write to WORK/notes.md the exact current edit (tensors, what was changed, strength) and the problem set, so that you can rebuild them (findings with numbers, edits tried and their measured effect, the current best edit; the notebook is shown to you again if the conversation is restarted); keep copies of good edits (for example copy WORK/edited to WORK/best_1).

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not implement or simulate the network's computation on any input in the kernel; (c) internet access or search (there is none). Allowed: any analysis of the weights and of forward-pass results, any use of the practice problems, and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. It continues until all {budget} forward passes are used (replying without a code block does not end it earlier); after that, make your final choice (the tensors in WORK/edited when you finish are submitted) and reply without a code block, saying what you changed, the evidence for it and why you expect it to help. {EXPLAIN}"""


def run_session(run_dir: Path, *, budget: int = 10, effort: str = "max", max_runs: int = 300):
    run_dir.mkdir(parents=True, exist_ok=True)
    lab = QwenMathLab(run_dir, budget=budget, max_python_runs=max_runs)
    return drive(
        lab, task_message(budget, max_runs), run_dir=run_dir, budget=budget, max_runs=max_runs, effort=effort
    )
