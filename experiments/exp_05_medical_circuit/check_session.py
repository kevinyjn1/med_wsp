"""EXP-05c: check the donor session `transfer1` without using any of Astra's own code or summaries.

Astra ended the session by withdrawing its candidate edit, so the harness's final evaluation compares the
unedited model with itself. What the session found therefore rests on Astra's own confirmation passes. This
script checks them three ways and writes `processed/check_<name>.json`, plus `check_<name>_questions.jsonl`
with the per-question result of every score pass (the session's work folder is not in git):

  records  recount every score pass from the per-question files the lab wrote, with the answer keys taken from
           the MedQA files of this repo, and compare with the lab's pass summaries and with Astra's final report;
           check that the confirmation questions appear in no earlier forward request.
  weights  compare the saved candidate with the base and the donor checkpoint: which entries differ from the
           base, whether they equal the donor's, and whether the MLP entries are the listed units.
  rescore  (GPU, optional) score the saved candidate again on the confirmation questions and compare, question
           by question, with what the lab recorded when Astra's passes ran. The lab records the names of the
           edited tensors of a pass, not their content, so this is what ties the saved files to those passes.

Usage:  python experiments/exp_05_medical_circuit/check_session.py --name transfer1 [--rescore]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa, models
from wsp.checkpoints import Checkpoint
from wsp.gpu_lock import gpu_lock
from wsp.patching import swapped_weights

HERE = Path(__file__).resolve().parent
BASELINE = REPO / "experiments" / "exp_04_medqa_baseline" / "processed"
# File names Astra chose in this session's work folder.
SPLITS = ("confirmation", "development", "development_check", "validation", "tuning", "explore", "discovery")
CANDIDATE, MANIFEST, REPORT = "candidate_sparse_full", "candidate_manifest.json", "final_report.json"
UNITS = 8192  # MLP units per layer; unit u is rows u and u + UNITS of gate_up_proj and column u of down_proj
# The six weight tensors of a decoder layer, in the order they are used.
PARTS = ("input_layernorm", "self_attn.qkv_proj", "self_attn.o_proj", "post_attention_layernorm",
         "mlp.gate_up_proj", "mlp.down_proj")  # fmt: skip


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def paired(original: np.ndarray, edited: np.ndarray, margin_change: np.ndarray) -> dict:
    diff = edited.astype(float) - original.astype(float)
    fixed, broken = int((edited & ~original).sum()), int((original & ~edited).sum())
    n = len(diff)
    return {
        "n": n, "original_correct": int(original.sum()), "edited_correct": int(edited.sum()),
        "difference_points": 100 * float(diff.mean()), "standard_error_points": 100 * float(diff.std(ddof=1) / np.sqrt(n)),
        "fixed": fixed, "broken": broken,
        "sign_test_p": float(binomtest(fixed, fixed + broken, 0.5).pvalue) if fixed + broken else 1.0,
        "margin_change": float(margin_change.mean()), "margin_change_se": float(margin_change.std(ddof=1) / np.sqrt(n)),
    }  # fmt: skip


def scored(work: Path, index: int, answers: dict[str, str]) -> dict:
    """One score pass, recounted from letter log-probabilities and this repo's answer keys."""
    rows = load_lines(work / f"score_{index}.jsonl")
    gold = np.array([medqa.LETTERS.index(answers[r["id"]]) for r in rows])
    out = {"ids": [r["id"] for r in rows]}
    for who in ("original", "edited"):
        logprobs = np.array([r[who]["letter_logprobs"] for r in rows])
        out[who] = {
            "logprobs": logprobs, "pred": logprobs.argmax(axis=1), "correct": logprobs.argmax(axis=1) == gold,
            "margin": np.array([r[who]["margin"] for r in rows]), "letter_mass": np.exp(logprobs).sum(axis=1),
            "differs_from_stored": int(sum((logprobs[k].argmax() == gold[k]) != r[who]["correct"] for k, r in enumerate(rows))),
        }  # fmt: skip
    return out


def donor_reference(donor: str, ids: list[str]) -> dict:
    """The donor's own EXP-04 results on the given questions, per prompt: what a complete transfer would give."""
    out = {}
    for prompt, prefix in (("bare", "bare_"), ("chat", "")):
        rows = {
            r["id"]: r
            for split in ("dev", "train")
            for r in medqa.load_jsonl(BASELINE / f"{prefix}{donor}_{split}.jsonl")
        }
        picked = [rows[i] for i in ids]
        out[prompt] = {
            "correct": int(sum(r["correct"] for r in picked)), "n": len(picked),
            "letter_mass": float(np.mean([r["letter_mass"] for r in picked])),
            "by_id": {r["id"]: {"correct": r["correct"], "margin": r["margin"]} for r in picked},
        }  # fmt: skip
    return out


def check_records(run: Path, donor: str) -> dict:
    work = run / "work"
    answers = {item["id"]: item["answer"] for split in ("dev", "train") for item in medqa.load_medqa(split)}
    splits = {name: json.loads((work / f"{name}_ids.json").read_text()) for name in SPLITS}
    confirmation = set(splits["confirmation"])
    calls = load_lines(run / "forward_calls.jsonl")
    passes, first_confirmation, rejected, per_question = [], None, 0, []
    for call in calls:
        if not call.get("ok"):
            rejected += 1
            continue
        request, summary, index = call["request"], call["summary"], call["pass"]
        questions = request.get("questions", [])
        if confirmation & set(questions) and first_confirmation is None:
            first_confirmation = index
        entry = {
            "pass": index, "type": request["type"], "model": request.get("model"), "prompt": request.get("prompt"),
            "questions": len(questions), "confirmation_questions": len(confirmation & set(questions)),
            "question_set": next((name for name in SPLITS if questions and set(questions) == set(splits[name])), None),
        }  # fmt: skip
        if request["type"] == "score" and "paired" in summary:
            data = scored(work, index, answers)
            assert data["ids"] == questions, index
            mine = paired(data["original"]["correct"], data["edited"]["correct"],
                          data["edited"]["margin"] - data["original"]["margin"])  # fmt: skip
            lab = summary["paired"]
            entry.update(mine)
            per_question += [
                {"pass": index, "id": i,
                 **{who: {"pred": medqa.LETTERS[data[who]["pred"][k]], "correct": bool(data[who]["correct"][k]),
                          "margin": float(data[who]["margin"][k])} for who in ("original", "edited")}}
                for k, i in enumerate(questions)
            ]  # fmt: skip
            entry["edited_tensors"] = summary["edited_tensors"]
            entry["recount_differs_from_stored"] = (
                data["original"]["differs_from_stored"] + data["edited"]["differs_from_stored"]
            )
            entry["letter_mass"] = [float(data[who]["letter_mass"].mean()) for who in ("original", "edited")]
            entry["predicted"] = {who: np.bincount(data[who]["pred"], minlength=4).tolist() for who in ("original", "edited")}  # fmt: skip
            # the lab rounds its summary to 4 decimals
            entry["matches_lab_summary"] = bool(
                mine["fixed"] == lab["wrong_to_right"] and mine["broken"] == lab["right_to_wrong"]
                and round(mine["difference_points"] / 100, 4) == lab["accuracy_difference"]
                and round(mine["standard_error_points"] / 100, 4) == lab["standard_error"]
                and round(mine["margin_change"], 4) == lab["margin_difference"]
                and round(mine["margin_change_se"], 4) == lab["margin_difference_se"]
            )  # fmt: skip
        passes.append(entry)

    report = json.loads((work / REPORT).read_text())
    confirmation_passes = {
        p["prompt"]: p for p in passes if p["question_set"] == "confirmation" and "fixed" in p
    }
    against_report = {}
    for prompt, mine in confirmation_passes.items():
        theirs = report[f"confirmation_{prompt}"]
        against_report[prompt] = bool(
            mine["fixed"] == theirs["rescues"] and mine["broken"] == theirs["regressions"]
            and abs(mine["difference_points"] - theirs["difference_pp"]) < 1e-9
            and abs(mine["standard_error_points"] - theirs["paired_se_pp"]) < 1e-9
            and abs(mine["margin_change"] - theirs["margin_difference"]) < 1e-9
            and abs(mine["letter_mass"][1] - theirs["candidate_letter_mass"]) < 1e-6
        )  # fmt: skip

    # Astra's "extraction" diagnostic: the bare-prompt outcome, split by whether the unedited model is right
    # under the chat prompt.
    bare = scored(work, confirmation_passes["bare"]["pass"], answers)
    chat = scored(work, confirmation_passes["chat"]["pass"], answers)
    chat_right = dict(zip(chat["ids"], chat["original"]["correct"], strict=True))
    right_under_chat = np.array([chat_right[i] for i in bare["ids"]])
    change = bare["edited"]["margin"] - bare["original"]["margin"]
    conditional = {
        str(flag): paired(bare["original"]["correct"][keep], bare["edited"]["correct"][keep], change[keep])
        for flag, keep in ((True, right_under_chat), (False, ~right_under_chat))
    }
    names = [n for n in SPLITS if n != "confirmation"]
    reference = donor_reference(donor, bare["ids"])
    per_question += [{"pass": "donor", "id": i, **{prompt: reference[prompt]["by_id"][i] for prompt in reference}}
                     for i in bare["ids"]]  # fmt: skip
    medqa.save_jsonl(per_question, HERE / "processed" / f"check_{run.name}_questions.jsonl")
    return {
        "practice_questions": len(confirmation | set(splits["development"])),
        "splits": {name: len(ids) for name, ids in splits.items()},
        "confirmation_disjoint_from_all_other_splits": all(not confirmation & set(splits[n]) for n in names),
        "split_overlaps": {f"{a} & {b}": len(set(splits[a]) & set(splits[b]))
                           for i, a in enumerate(names) for b in names[i + 1:] if set(splits[a]) & set(splits[b])},
        "first_pass_with_confirmation_questions": first_confirmation,
        "forward_requests": len(calls), "rejected_requests": rejected,
        "all_score_passes_match_lab_summaries": all(p.get("matches_lab_summary", True) for p in passes),
        "confirmation_matches_astra_report": against_report,
        "bare_outcome_by_unedited_chat_result": conditional,
        "donor_on_confirmation": {prompt: {k: v for k, v in block.items() if k != "by_id"} for prompt, block in reference.items()},
        "passes": passes,
    }  # fmt: skip


def check_weights(run: Path, donor: str) -> dict:
    work = run / "work"
    manifest = json.loads((work / MANIFEST).read_text())
    units = {int(layer): sorted(ids) for layer, ids in manifest["unit_map"].items()}
    base, other = Checkpoint("phi"), Checkpoint(donor)
    total = sum(base.get(name).numel() for name in base.names())
    tensors = []
    for path in sorted((work / CANDIDATE).glob("*.npy")):
        name = path.name[: -len(".npy")]
        layer = int(name.split("layers.")[1].split(".")[0])
        old, new = base.get(name), other.get(name)
        saved = torch.from_numpy(np.load(path)).to(torch.float32).to(torch.bfloat16)  # as the lab loads it
        differs = saved != old
        if "mlp." in name:
            # the listed units' rows (gate_up_proj) or columns (down_proj); everything else must be the base's
            support = torch.zeros_like(differs)
            ids = torch.tensor(units[layer])
            if "gate_up" in name:
                support[torch.cat([ids, ids + UNITS])] = True
            else:
                support[:, ids] = True
        else:
            support = torch.ones_like(differs)
        delta, full = (saved.float() - old.float()), (new.float() - old.float())
        tensors.append({
            "name": name, "layer": layer, "entries": old.numel(), "support": int(support.sum()),
            "changed": int(differs.sum()), "changed_outside_support": int((differs & ~support).sum()),
            "support_not_equal_to_donor": int(((saved != new) & support).sum()),
            "largest_difference_from_donor": float((saved.float() - new.float())[support].abs().max()),
            "relative_change": float(delta.norm() / old.float().norm()),
            "donor_relative_change": float(full.norm() / old.float().norm()),
        })  # fmt: skip
    present = {t["name"] for t in tensors}
    absent = [name for layer in sorted(units) for part in PARTS
              if (name := f"model.layers.{layer}.{part}.weight") not in present]  # fmt: skip
    n_layers = 1 + max(int(name.split(".")[2]) for name in base.names() if name.startswith("model.layers."))
    donor_change = [
        [float((other.get(name).float() - base.get(name).float()).norm() / base.get(name).float().norm())
         for layer in range(n_layers) for name in [f"model.layers.{layer}.{part}.weight"]]
        for part in PARTS
    ]  # fmt: skip
    return {
        "model_parameters": total, "tensors_saved": len(tensors), "layers": sorted(units),
        "units_per_layer": sorted({len(ids) for ids in units.values()}),
        "support": sum(t["support"] for t in tensors), "changed": sum(t["changed"] for t in tensors),
        "share_of_model_in_support": sum(t["support"] for t in tensors) / total,
        "share_of_model_changed": sum(t["changed"] for t in tensors) / total,
        "changed_outside_support": sum(t["changed_outside_support"] for t in tensors),
        "support_not_equal_to_donor": sum(t["support_not_equal_to_donor"] for t in tensors),
        "largest_difference_from_donor": max(t["largest_difference_from_donor"] for t in tensors),
        "matches_manifest": bool(sum(t["support"] for t in tensors) == manifest["support_count"]
                                 and sum(t["changed"] for t in tensors) == manifest["changed_bfloat16_count"]
                                 and total == manifest["total_parameter_count"]),
        # tensors of the edited layers that are not in the snapshot, and whether the donor's differ from the base's there
        "not_saved": [{"name": name, "donor_differs_in": int((base.get(name) != other.get(name)).sum())} for name in absent],
        # relative change of every layer tensor if it were the donor's: one row per part, one column per layer
        "parts": list(PARTS), "donor_relative_change": donor_change,
        "units": {str(layer): ids for layer, ids in units.items()},
        "tensors": tensors,
    }  # fmt: skip


def rescore(run: Path, records: dict) -> dict:
    """Score the saved candidate on the confirmation questions again; compare per question with the lab's files."""
    work = run / "work"
    by_id = {item["id"]: item for split in ("dev", "train") for item in medqa.load_medqa(split)}
    tokenizer = models.load_tokenizer("phi")
    out = {}
    with gpu_lock(f"exp05 rescore {run.name}"):
        net = models.load_model("phi")
        edits = {path.name[: -len(".npy")]: torch.from_numpy(np.load(path)).to(torch.float32).to(torch.bfloat16)
                 for path in sorted((work / CANDIDATE).glob("*.npy"))}  # fmt: skip
        for entry in records["passes"]:
            if entry["question_set"] != "confirmation" or "fixed" not in entry:
                continue
            rows = load_lines(work / f"score_{entry['pass']}.jsonl")
            items = [by_id[r["id"]] for r in rows]
            with swapped_weights(net, edits):
                again = medqa.evaluate(net, tokenizer, items, 16, progress=False, style=entry["prompt"])
            then = np.array([r["edited"]["letter_logprobs"] for r in rows])
            now = np.array([r["letter_logprobs"] for r in again])
            unedited = np.array([r["original"]["letter_logprobs"] for r in rows])
            out[entry["prompt"]] = {
                "pass": entry["pass"], "questions": len(rows), "correct": int(sum(r["correct"] for r in again)),
                "largest_logprob_difference": float(np.abs(now - then).max()),
                "questions_with_any_difference": int((np.abs(now - then).max(axis=1) > 0).sum()),
                "predictions_that_differ": int(sum(a["pred"] != r["edited"]["pred"] for a, r in zip(again, rows, strict=True))),
                # for scale: how far the unedited model's numbers are from the candidate's on the same questions
                "mean_distance_to_unedited": float(np.abs(unedited - then).max(axis=1).mean()),
            }  # fmt: skip
        del net
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="transfer1")
    parser.add_argument("--donor", default="mediphi_instruct")
    parser.add_argument("--rescore", action="store_true", help="also score the saved candidate again (GPU)")
    args = parser.parse_args()
    run = HERE / "raw" / args.name
    target = HERE / "processed" / f"check_{args.name}.json"
    result = json.loads(target.read_text()) if target.exists() else {}
    result["records"] = check_records(run, args.donor)
    result["weights"] = check_weights(run, args.donor)
    if args.rescore:
        result["rescore"] = rescore(run, result["records"])
    target.write_text(json.dumps(result, indent=1))
    records, weights = result["records"], result["weights"]
    print({k: v for k, v in records.items() if k not in ("passes", "bare_outcome_by_unedited_chat_result")})
    for entry in records["passes"]:
        if "fixed" in entry:
            print(f"  pass {entry['pass']:>2} {entry['prompt']:<4} {entry['question_set'] or '?':<17} n={entry['n']:<4} "
                  f"tensors={len(entry['edited_tensors']):<2} {entry['difference_points']:+.2f} points "
                  f"(SE {entry['standard_error_points']:.2f}) {entry['fixed']}/{entry['broken']} "
                  f"p={entry['sign_test_p']:.4f} margin {entry['margin_change']:+.4f} ({entry['margin_change_se']:.4f})")  # fmt: skip
    print(records["bare_outcome_by_unedited_chat_result"])
    print({k: v for k, v in weights.items() if k not in ("tensors", "units", "donor_relative_change")})
    print(result.get("rescore", "not rescored"))
    print("wrote", target)


if __name__ == "__main__":
    main()
