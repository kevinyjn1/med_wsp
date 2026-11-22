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

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from wsp import medqa, models
from wsp.gpu_lock import gpu_lock
from wsp.harness.kernel import Kernel
from wsp.looping import decoder_layers
from wsp.patching import swapped_weights

FROZEN = ("model.norm.weight", "model.embed_tokens.weight", "lm_head.weight")
MAX_SCORE_QUESTIONS = 1024
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
NAMES = sorted(_where)

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


def validate_edit(name: str, new: torch.Tensor, old: torch.Tensor) -> None:
    """Host-side copy of the save_edit rules (the kernel could write files without calling save_edit)."""
    if name in FROZEN:
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


class MedQALab:
    """Kernel + forward-pass tool + budget for one Astra session on a Phi-3.5-family model."""

    def __init__(self, run_dir: Path, *, model: str = "phi", donor: str | None = None, budget: int = 30,
                 practice: list[dict], baseline: list[dict], max_python_runs: int = 300):  # fmt: skip
        self.run_dir, self.work = run_dir, run_dir / "work"
        self.model_name, self.donor_name, self.budget, self.max_python_runs = (
            model,
            donor,
            budget,
            max_python_runs,
        )
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
        example = medqa.build_prompt(self.tokenizer, practice[0])
        (self.practice_dir / "eval_prompt_format.txt").write_text(
            "The exact evaluation prompt for a question (the answer letter is read from the next-token logits at its end):\n\n"
            + example + "\n"
        )  # fmt: skip
        prelude = (PRELUDE.replace("__MODEL_DIR__", repr(str(model_dir)))
                   .replace("__DONOR_DIR__", repr(str(donor_dir)) if donor_dir else "None")
                   .replace("__PRACTICE_DIR__", repr(str(self.practice_dir))).replace("__FROZEN__", repr(FROZEN)))  # fmt: skip
        # The Hugging Face snapshot files are links into the repo's blob folder, so the whole repo folder is readable.
        read_only = [model_dir.parents[1]] + ([donor_dir.parents[1]] if donor_dir else [])
        self.kernel = Kernel(self.work, prelude=prelude, read_only=read_only, timeout_s=900, max_out=10_000)
        self.state_path = run_dir / "lab_state.json"
        self.state = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"passes": 0, "requests": 0}
        )

    # ------------------------------------------------------------------------------------------- kernel
    def run_python(self, code: str, _state=None) -> str:
        return self.kernel.run(code)

    def passes_left(self) -> int:
        return self.budget - self.state["passes"]

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
            with gpu_lock(f"lab {self.run_dir.name} pass {index}"), torch.no_grad():
                summary, where = self._dispatch(request, index)
            summary["s"] = round(time.time() - start)
        except Exception as error:  # noqa: BLE001 - every failure is reported to the model, uncounted
            self._save_state()
            self._log(
                {
                    "t": time.time(),
                    "ok": False,
                    "request": code[:4000],
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            return f"The forward pass failed and was not counted: {type(error).__name__}: {str(error)[:600]}"
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

    def _dispatch(self, request: dict, index: int) -> tuple[dict, str]:
        kind, which = request.get("type"), request.get("model", "original")
        if kind == "score":
            return self._score(request, which, index)
        if kind == "teacher_forced":
            return self._teacher_forced(request, which, index)
        raise ValueError(f'unknown type {kind!r}: use "score" or "teacher_forced"')

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
        model = self._load(which)
        results = {}
        if which in ("original", "both", "donor"):
            results["donor" if which == "donor" else "original"] = medqa.evaluate(
                model, self.tokenizer, items, 16, progress=False
            )
        if which in ("edited", "both"):
            edits = self.current_edits(model)
            if not edits:
                raise ValueError("there are no edits in edited/ to score")
            with swapped_weights(model, edits):
                results["edited"] = medqa.evaluate(model, self.tokenizer, items, 16, progress=False)
        del model

        def block(records: list[dict]) -> dict:
            return {"accuracy": round(float(np.mean([r["correct"] for r in records])), 4),
                    "mean_margin": round(float(np.mean([r["margin"] for r in records])), 4),
                    "predicted": {letter: sum(r["pred"] == letter for r in records) for letter in medqa.LETTERS}}  # fmt: skip

        summary = {"ok": True, "type": "score", "model": which, "questions": len(items),
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
        texts = [
            medqa.build_prompt(self.tokenizer, item) for item in self._items(request.get("questions", []))
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
        handles = [blocks[layer].mlp.down_proj.register_forward_pre_hook(lambda module, args, layer=layer: captured.__setitem__(layer, args[0]))
                   for layer in mlp_layers]  # fmt: skip
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
                            store(f"hidden_{layer}", i, result.hidden_states[layer][k, span], per_token=True)
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
