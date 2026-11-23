"""The edit lab: what Astra gets when it is asked to change a model's weights (Phi-3.5 family, MedQA).

Modeled on MIT weight-space-programming experiment 8:

  - a persistent Python kernel over the weights (CPU, sandboxed): NAMES, load, load_original, save_edit,
    drop_edit, edited, tok (and load_donor when a donor checkpoint is provided);
  - practice questions with answers and the original model's result on each of them;
  - a forward-pass tool with a budget. One pass is either a SCORE pass (letter scoring of up to 1,024 practice
    questions with the original weights, the edited weights, or both, paired) or a TEACHER-FORCED read
    (hidden states, MLP activations, attention, log-probs on chosen questions or raw texts).

The model itself only ever runs here, on the host, never in the kernel. Edits are whole tensors saved as
float32 `.npy` files under <work>/edited/ and are re-validated on the host before every pass.
"""

from __future__ import annotations

import contextlib
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from wsp import medqa, models
from wsp.gpu_lock import gpu_lock
from wsp.harness.kernel import Kernel
from wsp.looping import decoder_layers
from wsp.patching import COMPONENTS, Patch, capture, run_patched, swapped_weights

FROZEN = ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight")
MAX_SCORE_QUESTIONS = 1024
MAX_PATCH_QUESTIONS, MAX_PATCHES, PATCH_ROWS = 64, 256, 24
REPO = Path(__file__).resolve().parents[1]
MAX_TEXTS_LAST, MAX_TEXTS_ALL, MAX_TOKENS_ALL, MAX_ATTN_TEXTS, MAX_ATTN_TOKENS = 1024, 64, 40_000, 4, 700

PRELUDE = r'''
import json, os
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer

torch.set_grad_enabled(False)
np.set_printoptions(precision=4, suppress=True, linewidth=160)
MODEL_DIR, DONOR_DIR, PRACTICE_DIR, WORK = __MODEL_DIR__, __DONOR_DIR__, __PRACTICE_DIR__, os.getcwd()
FROZEN = __FROZEN__

def _index(folder):
    where = {}
    for f in sorted(os.listdir(folder)):
        if f.endswith(".safetensors"):
            with safe_open(os.path.join(folder, f), "pt") as h:
                for k in h.keys():
                    where[k] = os.path.join(folder, f)
    return where

_where = _index(MODEL_DIR)
_donor = _index(DONOR_DIR) if DONOR_DIR else {}
NAMES = sorted(n for n in _where if n.startswith(__PREFIX__))

def _read(where, name):
    with safe_open(where[name], "pt") as h:
        return h.get_tensor(name).to(torch.float32)

def load_original(name):
    """The original weights of one tensor, as float32."""
    return _read(_where, name)

def load_donor(name):
    """The donor checkpoint's weights of one tensor, as float32 (only when a donor is provided)."""
    if not _donor:
        raise RuntimeError("no donor checkpoint in this session")
    return _read(_donor, name)

def _edit_path(name):
    return os.path.join(WORK, "edited", name + ".npy")

def edited():
    """Names of the tensors that currently have an edit."""
    folder = os.path.join(WORK, "edited")
    return sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".npy")) if os.path.isdir(folder) else []

def load(name):
    """Your current edit of a tensor if there is one, else the original (float32)."""
    path = _edit_path(name)
    return torch.from_numpy(np.load(path)) if os.path.exists(path) else load_original(name)

def drop_edit(name):
    if os.path.exists(_edit_path(name)):
        os.remove(_edit_path(name))

def save_edit(name, tensor):
    """Store an edited version of a tensor (same name and shape). Out-of-scope edits are refused."""
    if name not in _where:
        raise ValueError(f"unknown tensor {name!r}")
    if name in FROZEN:
        raise ValueError(f"{name} is out of scope (final norm, input embedding, output layer)")
    new = torch.as_tensor(tensor, dtype=torch.float32)
    old = load_original(name)
    if new.shape != old.shape:
        raise ValueError(f"{name}: shape {tuple(new.shape)} differs from the original {tuple(old.shape)}")
    if not torch.isfinite(new).all():
        raise ValueError(f"{name}: the edit contains non-finite values")
    scale = float((new * old).sum() / (old * old).sum())
    if abs(scale - 1.0) > 1e-9 and float((new - scale * old).norm()) <= 1e-6 * float(new.norm()):
        raise ValueError(f"{name}: an edit that only multiplies the whole tensor by a constant ({scale:.4f}) is out of scope")
    os.makedirs(os.path.join(WORK, "edited"), exist_ok=True)
    np.save(_edit_path(name), new.numpy())
    return f"saved {name}: relative change {float((new - old).norm() / old.norm()):.5f}"

tok = Tokenizer.from_file(os.path.join(MODEL_DIR, "tokenizer.json"))
print("kernel ready:", len(NAMES), "tensors; edited:", edited())
'''


def validate_edit(name: str, new: torch.Tensor, old: torch.Tensor, frozen: tuple[str, ...] = FROZEN) -> None:
    """Host-side copy of the save_edit rules (the kernel could write files without calling save_edit)."""
    if name in frozen:
        raise ValueError(f"{name} is out of scope (final norm, input embedding, output layer)")
    if new.shape != old.shape:
        raise ValueError(f"{name}: shape {tuple(new.shape)} differs from the original {tuple(old.shape)}")
    if not torch.isfinite(new).all():
        raise ValueError(f"{name}: the edit contains non-finite values")
    old32 = old.to(torch.float32)
    scale = float((new * old32).sum() / (old32 * old32).sum())
    if abs(scale - 1.0) > 1e-9 and float((new - scale * old32).norm()) <= 1e-6 * float(new.norm()):
        raise ValueError(
            f"{name}: an edit that only multiplies the whole tensor by a constant is out of scope"
        )


def make_prelude(
    model_dir: Path,
    practice_dir: Path,
    frozen: tuple[str, ...],
    donor_dir: Path | None = None,
    prefix: str = "",
) -> str:
    """The kernel prelude for one checkpoint folder."""
    return (
        PRELUDE.replace("__MODEL_DIR__", repr(str(model_dir)))
        .replace("__DONOR_DIR__", repr(str(donor_dir)) if donor_dir else "None")
        .replace("__PRACTICE_DIR__", repr(str(practice_dir)))
        .replace("__FROZEN__", repr(tuple(frozen)))
        .replace("__PREFIX__", repr(prefix))
    )


class PassError(Exception):
    """A forward pass that failed in a worker process; `label` is the original exception's type name."""

    def __init__(self, label: str, message: str):
        super().__init__(message)
        self.label = label


LOCK_ORDER = ("batch", "gpu")  # every holder of both takes them in this order, so two holders cannot deadlock


class BudgetedLab:
    """Kernel, forward-pass budget and logging shared by the labs. A subclass implements `_dispatch`."""

    lock_name = "gpu"

    def _locks(self, request: dict) -> tuple[str, ...]:
        """Which job locks one request needs (see wsp/gpu_lock.py). Two-model requests need both."""
        return (self.lock_name,)

    def _init_runtime(
        self, run_dir: Path, *, budget: int, prelude: str, read_only: list[Path], max_python_runs: int = 300
    ) -> None:
        self.run_dir, self.work, self.budget, self.max_python_runs = (
            run_dir,
            run_dir / "work",
            budget,
            max_python_runs,
        )
        self.kernel = Kernel(self.work, prelude=prelude, read_only=read_only, timeout_s=900, max_out=10_000)
        self.state_path = run_dir / "lab_state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"passes": 0, "requests": 0}
        )

    def _dispatch(self, request: dict, index: int) -> tuple[dict, str]:
        raise NotImplementedError

    # ------------------------------------------------------------------------------------------- kernel
    def run_python(self, code: str, _state=None) -> str:
        return self.kernel.run(code)

    def passes_left(self) -> int:
        return self.budget - self.state["passes"]

    # -------------------------------------------------------------------------------------- forward tool
    def forward(self, code: str, _state=None) -> str:
        """Handle one ```forward block. A failed request is reported and not counted."""
        self.state["requests"] += 1
        index = self.state["passes"] + 1
        try:
            request = json.loads(code)
            if self.passes_left() <= 0:
                raise RuntimeError("the forward-pass budget is used up")
            start = time.time()
            with contextlib.ExitStack() as stack, torch.no_grad():
                for name in sorted(
                    self._locks(request),
                    key=lambda n: LOCK_ORDER.index(n) if n in LOCK_ORDER else len(LOCK_ORDER),
                ):
                    stack.enter_context(gpu_lock(f"lab {self.run_dir.name} pass {index}", name=name))
                summary, where = self._dispatch(request, index)
            summary["s"] = round(time.time() - start)
        except Exception as error:  # noqa: BLE001 - every failure is reported to the model, uncounted
            self._save_state()
            self._log(
                {
                    "t": time.time(),
                    "ok": False,
                    "request": code[:4000],
                    "error": f"{getattr(error, 'label', type(error).__name__)}: {error}",
                }
            )
            label = getattr(error, "label", type(error).__name__)
            return f"The forward pass failed and was not counted: {label}: {str(error)[:600]}"
        finally:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        self.state["passes"] = index
        self._save_state()
        self._log({"t": time.time(), "ok": True, "pass": index, "request": request, "summary": summary})
        return (f"Forward pass {index} of {self.budget} done ({request['type']}); results in {where}. Summary:\n"
                f"```\n{json.dumps(summary, indent=1)}\n```\nForward passes left: {self.passes_left()}.")  # fmt: skip

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state))

    def _log(self, record: dict) -> None:
        with open(self.run_dir / "forward_calls.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")


class MedQALab(BudgetedLab):
    """Kernel + forward-pass tool + budget for one Astra session on a Phi-3.5-family model."""

    def __init__(self, run_dir: Path, *, model: str = "phi", donor: str | None = None, budget: int = 30,
                 practice: list[dict], baseline: list[dict], max_python_runs: int = 300, style: str = "chat"):  # fmt: skip
        self.work = run_dir / "work"
        self.model_name, self.donor_name, self.style = model, donor, style
        self.practice = {item["id"]: item for item in practice}
        self.practice_dir = self.work / "practice"
        self.practice_dir.mkdir(parents=True, exist_ok=True)
        model_dir = models.snapshot_dir(models.MODEL_IDS[model])
        donor_dir = models.snapshot_dir(models.MODEL_IDS[donor]) if donor else None
        medqa.save_jsonl(
            [
                {"id": i["id"], "question": i["question"], "options": i["options"], "answer": i["answer"]}
                for i in practice
            ],
            self.practice_dir / "questions.jsonl",
        )
        medqa.save_jsonl(baseline, self.practice_dir / "original_results.jsonl")
        self.tokenizer = models.load_tokenizer("phi")
        if style == "chat":
            (self.practice_dir / "eval_prompt_format.txt").write_text(
                "The exact evaluation prompt for a question (the answer letter is read from the next-token logits at its end):\n\n"
                + medqa.build_prompt(self.tokenizer, practice[0]) + "\n"
            )  # fmt: skip
        else:
            (self.practice_dir / "eval_prompt_format.txt").write_text(
                "The two evaluation prompts for one question. In both, the answer letter is read from the next-token\n"
                "logits at the end of the prompt.\n\n===== BARE prompt (primary measure; plain text, no chat template) =====\n"
                + medqa.build_prompt(self.tokenizer, practice[0], "bare")
                + "\n\n===== CHAT prompt (must not get worse) =====\n"
                + medqa.build_prompt(self.tokenizer, practice[0], "chat") + "\n"
            )  # fmt: skip
        prelude = make_prelude(model_dir, self.practice_dir, FROZEN, donor_dir)
        # The Hugging Face snapshot files are links into the repo's blob folder, so the whole repo folder is readable.
        read_only = [model_dir.parents[1]] + ([donor_dir.parents[1]] if donor_dir else [])
        self._init_runtime(
            run_dir, budget=budget, prelude=prelude, read_only=read_only, max_python_runs=max_python_runs
        )

    @classmethod
    def for_worker(cls, run_dir: Path, model: str, donor: str | None, style: str) -> MedQALab:
        """The model-running half of a lab, rebuilt inside a worker process from the files of a session."""
        lab = cls.__new__(cls)
        lab.run_dir, lab.work = run_dir, run_dir / "work"
        lab.model_name, lab.donor_name, lab.style = model, donor, style
        lab.tokenizer = models.load_tokenizer("phi")
        lab.practice = {
            item["id"]: item for item in medqa.load_jsonl(lab.work / "practice" / "questions.jsonl")
        }
        return lab

    def _locks(self, request: dict) -> tuple[str, ...]:
        # A patching pass holds the base and the donor at once (15 GB): nothing else may run next to it.
        return ("batch", "gpu") if request.get("type") == "patch" else ("gpu",)

    def _dispatch(self, request: dict, index: int) -> tuple[dict, str]:
        """Run the pass in a fresh process, so every byte of model memory is returned when it ends."""
        jobs = self.run_dir / "jobs"
        jobs.mkdir(exist_ok=True)
        job, out = jobs / f"pass_{index}.json", jobs / f"pass_{index}.out.json"
        out.unlink(missing_ok=True)
        job.write_text(json.dumps({"run_dir": str(self.run_dir), "model": self.model_name, "donor": self.donor_name,
                                   "style": self.style, "request": request, "index": index, "out": str(out)}))  # fmt: skip
        result = subprocess.run([sys.executable, "-m", "wsp.lab_worker", str(job)], capture_output=True, text=True,
                                cwd=REPO, check=False)  # fmt: skip
        if not out.exists():
            tail = (
                result.stderr.strip().splitlines()[-1]
                if result.stderr.strip()
                else f"exit code {result.returncode}"
            )
            raise PassError("WorkerError", f"the model process ended without a result ({tail})")
        answer = json.loads(out.read_text())
        if "error" in answer:
            raise PassError(answer["error_type"], answer["error"])
        return answer["summary"], answer["where"]

    # --------------------------------------------------------------------------------------------- edits
    def current_edits(self, model) -> dict[str, torch.Tensor]:
        """Validated edits as tensors ready to be swapped in (rounded through bfloat16, like a saved checkpoint)."""
        folder = self.work / "edited"
        params = dict(model.named_parameters())
        edits = {}
        for path in sorted(folder.glob("*.npy")) if folder.is_dir() else []:
            name = path.name[: -len(".npy")]
            if name not in params:
                raise ValueError(f"edited/{path.name}: {name} is not a tensor of the model")
            new = torch.from_numpy(np.load(path)).to(torch.float32)
            validate_edit(name, new, params[name].detach().cpu())
            edits[name] = new.to(torch.bfloat16)
        return edits

    def _dispatch_local(self, request: dict, index: int) -> tuple[dict, str]:
        """The pass itself; runs inside the worker process."""
        kind, which = request.get("type"), request.get("model", "original")
        if kind == "score":
            return self._score(request, which, index)
        if kind == "teacher_forced":
            return self._teacher_forced(request, which, index)
        if kind == "patch":
            return self._patch(request, which, index)
        kinds = '"score", "teacher_forced" or "patch"' if self.donor_name else '"score" or "teacher_forced"'
        raise ValueError(f"unknown type {kind!r}: use {kinds}")

    def _style(self, request: dict) -> str:
        style = request.get("prompt", self.style)
        if style not in medqa.STYLES:
            raise ValueError('"prompt" must be "bare" or "chat"')
        return style

    def _items(self, ids: list[str]) -> list[dict]:
        unknown = [i for i in ids if i not in self.practice]
        if unknown:
            raise ValueError(f"unknown practice ids: {unknown[:5]}")
        return [self.practice[i] for i in ids]

    def _load(self, which: str):
        if which == "donor" and not self.donor_name:
            raise ValueError("no donor checkpoint in this session")
        return models.load_model(self.donor_name if which == "donor" else self.model_name)

    def _score(self, request: dict, which: str, index: int) -> tuple[dict, str]:
        if which not in ("original", "edited", "both", "donor"):
            raise ValueError('"model" must be "original", "edited", "both" or "donor"')
        items = self._items(request["questions"])
        if not 1 <= len(items) <= MAX_SCORE_QUESTIONS:
            raise ValueError(f"a score pass takes 1 to {MAX_SCORE_QUESTIONS} questions")
        style = self._style(request)
        model = self._load(which)
        results = {}
        if which in ("original", "both", "donor"):
            results["donor" if which == "donor" else "original"] = medqa.evaluate(
                model, self.tokenizer, items, 16, progress=False, style=style
            )
        if which in ("edited", "both"):
            edits = self.current_edits(model)
            if not edits:
                raise ValueError("there are no edits in edited/ to score")
            with swapped_weights(model, edits):
                results["edited"] = medqa.evaluate(
                    model, self.tokenizer, items, 16, progress=False, style=style
                )
        del model

        def block(records: list[dict]) -> dict:
            return {"accuracy": round(float(np.mean([r["correct"] for r in records])), 4),
                    "mean_margin": round(float(np.mean([r["margin"] for r in records])), 4),
                    "predicted": {letter: sum(r["pred"] == letter for r in records) for letter in medqa.LETTERS}}  # fmt: skip

        summary = {"ok": True, "type": "score", "model": which, "prompt": style, "questions": len(items),
                   "edited_tensors": sorted(p.name[:-4] for p in (self.work / "edited").glob("*.npy")) if which in ("edited", "both") else []}  # fmt: skip
        summary.update({name: block(records) for name, records in results.items()})
        rows = [{"id": item["id"], "answer": item["answer"]} for item in items]
        for name, records in results.items():
            for row, record in zip(rows, records, strict=True):
                row[name] = {k: record[k] for k in ("pred", "correct", "margin", "letter_logprobs")}
        if which == "both":
            original = np.array([r["correct"] for r in results["original"]])
            edited = np.array([r["correct"] for r in results["edited"]])
            delta = np.array(
                [
                    e["margin"] - o["margin"]
                    for o, e in zip(results["original"], results["edited"], strict=True)
                ]
            )
            fixed, broken = int((edited & ~original).sum()), int((~edited & original).sum())
            diff = edited.astype(float) - original.astype(float)
            summary["paired"] = {
                "accuracy_difference": round(float(diff.mean()), 4),
                "standard_error": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 4) if len(diff) > 1 else None,
                "wrong_to_right": fixed, "right_to_wrong": broken,
                "sign_test_p": round(float(binomtest(fixed, fixed + broken, 0.5).pvalue), 4) if fixed + broken else 1.0,
                "margin_difference": round(float(delta.mean()), 4),
                "margin_difference_se": round(float(delta.std(ddof=1) / np.sqrt(len(delta))), 4) if len(delta) > 1 else None,
            }  # fmt: skip
        medqa.save_jsonl(rows, self.work / f"score_{index}.jsonl")
        return summary, f"score_{index}.jsonl"

    def _teacher_forced(self, request: dict, which: str, index: int) -> tuple[dict, str]:
        if which not in ("original", "edited", "donor"):
            raise ValueError('"model" must be "original", "edited" or "donor" for a teacher-forced pass')
        want = request.get("return", {})
        positions = want.get("positions", "last")
        if positions not in ("last", "all"):
            raise ValueError('"positions" must be "last" or "all"')
        style = self._style(request)
        texts = [
            medqa.build_prompt(self.tokenizer, item, style)
            for item in self._items(request.get("questions", []))
        ]
        texts += list(request.get("texts", []))
        ids = list(request.get("questions", [])) + [f"text-{i}" for i in range(len(request.get("texts", [])))]
        encoded = [self.tokenizer(text, add_special_tokens=False)["input_ids"] for text in texts]
        lengths = [len(e) for e in encoded]
        if not texts:
            raise ValueError('give "questions" (practice ids) and/or "texts" (raw texts)')
        limit = MAX_TEXTS_LAST if positions == "last" else MAX_TEXTS_ALL
        if len(texts) > limit:
            raise ValueError(f'at most {limit} texts with "positions": "{positions}"')
        if positions == "all" and sum(lengths) > MAX_TOKENS_ALL:
            raise ValueError(
                f'at most {MAX_TOKENS_ALL} tokens in total with "positions": "all" (got {sum(lengths)})'
            )
        hidden_layers, mlp_layers = list(want.get("hidden_states", [])), list(want.get("mlp_act", []))
        attn_layers, topk = list(want.get("attn_layers", [])), int(want.get("topk", 0))
        model = self._load(which)
        n_layers = model.config.num_hidden_layers
        if any(not 0 <= layer <= n_layers for layer in hidden_layers) or any(
            not 0 <= layer < n_layers for layer in mlp_layers + attn_layers
        ):
            raise ValueError(
                f"hidden_states layers are 0..{n_layers}; mlp_act and attn_layers are 0..{n_layers - 1}"
            )
        if attn_layers and (len(texts) > MAX_ATTN_TEXTS or max(lengths) > MAX_ATTN_TOKENS):
            raise ValueError(
                f"attention is returned for at most {MAX_ATTN_TEXTS} texts of at most {MAX_ATTN_TOKENS} tokens"
            )
        if topk > 50:
            raise ValueError("topk is at most 50")
        edits = self.current_edits(model) if which == "edited" else {}
        if which == "edited" and not edits:
            raise ValueError("there are no edits in edited/")

        longest = max(lengths)
        out: dict[str, np.ndarray] = {}

        def store(name: str, row: int, value: torch.Tensor, per_token: bool) -> None:
            """value is [T, ...] for this text (unpadded). Kept at the last token or right-padded over all tokens."""
            array = value.float().cpu().numpy()
            if positions == "last" or not per_token:
                array = array[-1] if per_token else array
                out.setdefault(name, np.zeros((len(texts), *array.shape), np.float16))[row] = array
            else:
                target = out.setdefault(name, np.zeros((len(texts), longest, *array.shape[1:]), np.float16))
                target[row, : array.shape[0]] = array

        blocks = decoder_layers(model)
        captured: dict[int, torch.Tensor] = {}
        last_block: dict[str, torch.Tensor] = {}
        handles = [blocks[layer].mlp.down_proj.register_forward_pre_hook(lambda module, args, layer=layer: captured.__setitem__(layer, args[0]))
                   for layer in mlp_layers]  # fmt: skip
        # transformers reports the last "hidden state" after the final norm; the residual stream after the last
        # layer is taken from that layer's own output instead (see tests/test_hidden_state_convention.py).
        handles.append(blocks[-1].register_forward_hook(
            lambda module, args, output: last_block.__setitem__("x", output[0] if isinstance(output, tuple) else output)))  # fmt: skip
        order = sorted(range(len(texts)), key=lambda i: lengths[i])
        batch_size = 1 if attn_layers else 8
        logprob_sum, logprob_count = 0.0, 0
        try:
            with swapped_weights(model, edits):
                for begin in range(0, len(order), batch_size):
                    rows = order[begin : begin + batch_size]
                    width = max(lengths[i] for i in rows)
                    input_ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long)
                    mask = torch.zeros((len(rows), width), dtype=torch.long)
                    for k, i in enumerate(rows):  # left padding: the last column is every text's last token
                        input_ids[k, width - lengths[i] :] = torch.tensor(encoded[i])
                        mask[k, width - lengths[i] :] = 1
                    result = model(input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False,
                                   output_hidden_states=bool(hidden_layers), output_attentions=bool(attn_layers))  # fmt: skip
                    logprobs = torch.log_softmax(result.logits.float(), dim=-1)
                    for k, i in enumerate(rows):
                        span = slice(width - lengths[i], width)
                        token_ids = input_ids[k, span].to(model.device)
                        own = logprobs[k, span]
                        if want.get("token_logprobs") and lengths[i] > 1:
                            token_lp = own[:-1].gather(1, token_ids[1:, None])[:, 0]
                            logprob_sum, logprob_count = (
                                logprob_sum + float(token_lp.sum()),
                                logprob_count + len(token_lp),
                            )
                            target = out.setdefault(
                                "token_logprobs", np.zeros((len(texts), longest - 1), np.float16)
                            )
                            target[i, : len(token_lp)] = token_lp.cpu().numpy()
                        if topk:
                            top = own.topk(topk, dim=-1)
                            store("topk_logprobs", i, top.values, per_token=True)
                            ids_target = out.setdefault(
                                "topk_ids",
                                np.zeros(
                                    (len(texts), topk)
                                    if positions == "last"
                                    else (len(texts), longest, topk),
                                    np.int32,
                                ),
                            )
                            if positions == "last":
                                ids_target[i] = top.indices[-1].cpu().numpy()
                            else:
                                ids_target[i, : lengths[i]] = top.indices.cpu().numpy()
                        for layer in hidden_layers:
                            stream = last_block["x"] if layer == n_layers else result.hidden_states[layer]
                            store(f"hidden_{layer}", i, stream[k, span], per_token=True)
                        for layer in mlp_layers:
                            store(f"mlp_act_{layer}", i, captured[layer][k, span], per_token=True)
                        for layer in attn_layers:  # [heads, T, T], right-padded on both token axes
                            attention = result.attentions[layer][k][:, span, span].float().cpu().numpy()
                            target = out.setdefault(
                                f"attn_{layer}",
                                np.zeros((len(texts), attention.shape[0], longest, longest), np.float16),
                            )
                            target[i, :, : lengths[i], : lengths[i]] = attention
                    del result, logprobs
        finally:
            for handle in handles:
                handle.remove()
        del model

        folder = self.work / f"fwd_{index}"
        folder.mkdir(parents=True, exist_ok=True)
        padded = np.full((len(texts), longest), -1, np.int32)
        for i, e in enumerate(encoded):
            padded[i, : len(e)] = e
        np.save(folder / "input_ids.npy", padded)
        for name, array in out.items():
            np.save(folder / f"{name}.npy", array)
        meta = {"ids": ids, "lengths": lengths, "positions": positions, "model": which,
                "note": "arrays are float16; with positions 'last' they are taken at each text's last token; "
                        "with 'all' they are right-padded to the longest text; input_ids is padded with -1"}  # fmt: skip
        (folder / "meta.json").write_text(json.dumps(meta))
        summary = {"ok": True, "type": "teacher_forced", "model": which, "texts": len(texts), "positions": positions,
                   "longest": longest, "files": {"input_ids.npy": list(padded.shape), **{f"{k}.npy": list(v.shape) for k, v in out.items()}},
                   "edited_tensors": sorted(edits)}  # fmt: skip
        if logprob_count:
            summary["mean_token_logprob"] = round(logprob_sum / logprob_count, 4)
        return summary, f"fwd_{index}/"

    def _patch(self, request: dict, which: str, index: int) -> tuple[dict, str]:
        """Activation patching: donor activations placed into the base model's forward pass, one patch at a time."""
        if not self.donor_name:
            raise ValueError("no donor checkpoint in this session")
        if which not in ("original", "edited"):
            raise ValueError('"model" (the model that receives the patches) must be "original" or "edited"')
        style = self._style(request)
        items = self._items(request.get("questions", []))
        if not 1 <= len(items) <= MAX_PATCH_QUESTIONS:
            raise ValueError(f"a patch pass takes 1 to {MAX_PATCH_QUESTIONS} questions")
        specs = request.get("patches", [])
        if not isinstance(specs, list) or not 1 <= len(specs) <= MAX_PATCHES:
            raise ValueError(f'"patches" is a list of 1 to {MAX_PATCHES} patches')
        base = self._load("original")
        n_layers, n_heads = base.config.num_hidden_layers, base.config.num_attention_heads
        patches = []
        for spec in specs:
            layer, component, head = spec.get("layer"), spec.get("component"), spec.get("head")
            if not isinstance(layer, int) or not 0 <= layer < n_layers:
                raise ValueError(f"patch layer must be 0..{n_layers - 1}: {spec}")
            if component not in (*COMPONENTS, "head"):
                raise ValueError(f'patch component must be "attn", "mlp", "resid" or "head": {spec}')
            if component == "head" and (not isinstance(head, int) or not 0 <= head < n_heads):
                raise ValueError(f'a "head" patch needs "head" in 0..{n_heads - 1}: {spec}')
            patches.append(Patch(layer, component, head if component == "head" else None))
        edits = self.current_edits(base) if which == "edited" else {}
        if which == "edited" and not edits:
            raise ValueError("there are no edits in edited/")
        donor = self._load("donor")
        letter_ids = medqa.letter_token_ids(self.tokenizer, style)
        layers = sorted({p.layer for p in patches})
        components = tuple(c for c in COMPONENTS if any(p.component == c for p in patches))
        heads = any(p.component == "head" for p in patches)
        head_dim = base.config.hidden_size // n_heads
        rows = []
        with swapped_weights(base, edits):
            for item in items:
                enc = self.tokenizer(
                    [medqa.build_prompt(self.tokenizer, item, style)],
                    return_tensors="pt",
                    add_special_tokens=False,
                )
                enc = {k: v.to(base.device) for k, v in enc.items()}
                cache = capture(donor, enc, layers=layers, components=components, heads=heads)
                base_margin = float(
                    medqa.gold_margins(
                        base(**enc, use_cache=False).logits[:, -1, :], letter_ids, item["answer"]
                    )[0]
                )
                donor_margin = float(medqa.gold_margins(cache["logits"], letter_ids, item["answer"])[0])
                margins: list[float] = []
                for begin in range(0, len(patches), PATCH_ROWS):
                    logits = run_patched(
                        base, enc, cache, patches[begin : begin + PATCH_ROWS], head_dim=head_dim
                    )
                    margins += [float(x) for x in medqa.gold_margins(logits, letter_ids, item["answer"])]
                rows.append({"id": item["id"], "answer": item["answer"], "base": round(base_margin, 4), "donor": round(donor_margin, 4),
                             "patched": {p.label(): round(m, 4) for p, m in zip(patches, margins, strict=True)}})  # fmt: skip
                del cache
        del base, donor
        medqa.save_jsonl(rows, self.work / f"patch_{index}.jsonl")

        base_margins = np.array([r["base"] for r in rows])
        donor_margins = np.array([r["donor"] for r in rows])
        gap = float((donor_margins - base_margins).mean())
        table = []
        for patch in patches:
            patched = np.array([r["patched"][patch.label()] for r in rows])
            change = patched - base_margins
            table.append({
                "patch": patch.label(), "margin_change": round(float(change.mean()), 4),
                "se": round(float(change.std(ddof=1) / np.sqrt(len(change))), 4) if len(change) > 1 else None,
                "share_of_donor_gap": round(float(change.mean()) / gap, 3) if abs(gap) > 1e-6 else None,
                "accuracy": round(float((patched > 0).mean()), 4),
            })  # fmt: skip
        table.sort(key=lambda row: -row["margin_change"])
        shown = table if len(table) <= 40 else table[:25] + table[-15:]
        summary = {
            "ok": True, "type": "patch", "model": which, "prompt": style, "questions": len(items), "patches": len(patches),
            "base": {"accuracy": round(float((base_margins > 0).mean()), 4), "mean_margin": round(float(base_margins.mean()), 4)},
            "donor": {"accuracy": round(float((donor_margins > 0).mean()), 4), "mean_margin": round(float(donor_margins.mean()), 4)},
            "note": "margin = log-prob of the correct letter minus the best other letter; a patch puts the donor's activation "
                    "(all token positions) into the receiving model; rows are sorted by margin_change"
                    + ("" if len(table) <= 40 else "; the 25 largest and 15 smallest are shown, all are in the file"),
            "results": shown, "edited_tensors": sorted(edits),
        }  # fmt: skip
        return summary, f"patch_{index}.jsonl"
