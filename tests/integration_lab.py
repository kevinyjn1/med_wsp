"""End-to-end check of the edit lab on the real Phi-3.5 weights (needs the GPU; run by hand):

    python tests/integration_lab.py <empty scratch folder>

Part 1 (no donor, chat prompt): kernel prelude, save_edit refusals, a paired score pass, a teacher-forced pass,
budget accounting, a failed request that is not counted, and that hidden_32 is the pre-norm residual stream.
Part 2 (donor, bare prompt): the bare and chat prompt options of a score pass and an activation-patching pass.
Every pass runs in a worker process, as in a real session.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wsp import medqa
from wsp.checkpoints import Checkpoint
from wsp.lab import MedQALab

BASELINE = REPO / "experiments" / "exp_04_medqa_baseline" / "processed"
MedQALab._locks = lambda self, request: ("selftest",)


def summary_of(reply: str) -> dict:
    return json.loads(reply.split("```")[1])


def part_one(scratch: Path, items: list[dict], baseline: list[dict]) -> None:
    lab = MedQALab(scratch / "run", model="phi", budget=3, practice=items, baseline=baseline)
    print(lab.kernel.start().strip())
    try:
        out = lab.run_python(
            "name = 'model.layers.20.mlp.down_proj.weight'\n"
            "w = load_original(name)\n"
            "import json\n"
            "rows = [json.loads(l) for l in open(PRACTICE_DIR + '/original_results.jsonl')]\n"
            "print(tuple(w.shape), w.dtype, len(NAMES), len(rows), sum(r['correct'] for r in rows))\n"
            "for bad_name, bad in [('lm_head.weight', load_original('lm_head.weight')), (name, w * 1.5), (name, w[:10])]:\n"
            "    try:\n        save_edit(bad_name, bad)\n        print('NOT refused', bad_name)\n"
            "    except ValueError as e:\n        print('refused:', str(e)[:70])\n"
            "e = w.clone(); e[:, :400] = 0\n"
            "save_edit(name, e)"
        )
        print(out.strip())
        assert out.count("refused:") == 3 and "NOT refused" not in out and "saved model.layers.20" in out, out

        ids = [item["id"] for item in items]
        bad = lab.forward(json.dumps({"type": "score", "model": "both", "questions": ["nope"]}))
        print(bad)
        assert "not counted: ValueError" in bad and lab.state["passes"] == 0

        reply = lab.forward(json.dumps({"type": "score", "model": "both", "questions": ids}))
        summary = summary_of(reply)
        print("score:", {k: summary[k] for k in ("prompt", "original", "edited", "paired")})
        expected = float(np.mean([r["correct"] for r in baseline]))
        assert abs(summary["original"]["accuracy"] - expected) < 1e-3, (summary["original"], expected)
        assert (
            summary["paired"]["margin_difference"] != 0
            and lab.state["passes"] == 1
            and summary["prompt"] == "chat"
        )

        reply = lab.forward(json.dumps({
            "type": "teacher_forced", "model": "edited", "questions": ids[:3], "texts": ["The capital of France is"],
            "return": {"positions": "all", "token_logprobs": True, "topk": 5, "hidden_states": [0, 16, 32],
                       "mlp_act": [20], "attn_layers": [5]},
        }))  # fmt: skip
        files = summary_of(reply)["files"]
        longest = files["input_ids.npy"][1]
        assert files["hidden_16.npy"] == [4, longest, 3072] and files["mlp_act_20.npy"] == [4, longest, 8192]
        assert files["attn_5.npy"] == [4, 32, longest, longest] and files["topk_ids.npy"] == [4, longest, 5]
        attn = np.load(lab.work / "fwd_2" / "attn_5.npy")
        length = json.loads((lab.work / "fwd_2" / "meta.json").read_text())["lengths"][3]
        assert abs(float(attn[3, 0, length - 1, :length].astype(np.float32).sum()) - 1.0) < 0.02

        # hidden_32 must be the residual stream BEFORE the final norm: norm + output layer reproduce the top token
        # (the edit leaves the final norm and the output layer untouched).
        checkpoint = Checkpoint("phi")
        hidden = np.load(lab.work / "fwd_2" / "hidden_32.npy")[3, length - 1].astype(np.float32)
        normed = hidden / np.sqrt((hidden**2).mean() + 1e-5) * checkpoint.f32("model.norm.weight")
        top = int(np.argmax(checkpoint.f32("lm_head.weight") @ normed))
        assert top == int(np.load(lab.work / "fwd_2" / "topk_ids.npy")[3, length - 1, 0])

        assert lab.passes_left() == 1 and sorted(p.name for p in (lab.run_dir / "jobs").glob("*.out.json")) == [
            "pass_1.out.json", "pass_2.out.json"]  # fmt: skip
        print("PART 1 OK")
    finally:
        lab.kernel.stop()


def part_two(scratch: Path, items: list[dict]) -> None:
    donor = "mediphi_instruct"
    bare = (
        {r["id"]: r for r in medqa.load_jsonl(BASELINE / "bare_phi_dev.jsonl")}
        if (BASELINE / "bare_phi_dev.jsonl").exists()
        else {}
    )
    lab = MedQALab(scratch / "donor_run", model="phi", donor=donor, budget=4, practice=items,
                   baseline=[bare.get(i["id"], {"id": i["id"]}) for i in items], style="bare")  # fmt: skip
    print(lab.kernel.start().strip())
    try:
        out = lab.run_python("d = load_donor('model.layers.3.mlp.down_proj.weight') - load_original('model.layers.3.mlp.down_proj.weight')\n"
                             "print(round(float(d.norm()), 3), open(PRACTICE_DIR + '/eval_prompt_format.txt').read().count('====='))")  # fmt: skip
        print("donor delta norm and prompt-format markers:", out.strip())
        assert out.strip().endswith(" 4") and not out.startswith("0.0 ")

        ids = [item["id"] for item in items[:12]]
        by_prompt = {}
        for prompt in ("bare", "chat"):
            summary = summary_of(
                lab.forward(
                    json.dumps({"type": "score", "model": "original", "questions": ids, "prompt": prompt})
                )
            )
            by_prompt[prompt] = summary["original"]
            assert summary["prompt"] == prompt
        print("score by prompt:", by_prompt)
        assert by_prompt["bare"]["mean_margin"] != by_prompt["chat"]["mean_margin"]
        if bare:
            expected = float(np.mean([bare[i]["correct"] for i in ids]))
            assert abs(by_prompt["bare"]["accuracy"] - expected) < 1e-3, (by_prompt["bare"], expected)

        patches = [{"layer": 5, "component": "attn"}, {"layer": 5, "component": "mlp"}, {"layer": 31, "component": "resid"},
                   {"layer": 20, "component": "head", "head": 3}, {"layer": 0, "component": "resid"}]  # fmt: skip
        reply = lab.forward(json.dumps({"type": "patch", "questions": ids[:4], "patches": patches}))
        summary = summary_of(reply)
        print(
            "patch:",
            summary["base"],
            summary["donor"],
            [(r["patch"], r["margin_change"]) for r in summary["results"]],
        )
        rows = medqa.load_jsonl(lab.work / "patch_3.jsonl")
        assert len(rows) == 4 and set(rows[0]["patched"]) == {
            "L5.attn",
            "L5.mlp",
            "L31.resid",
            "L20.head3",
            "L0.resid",
        }
        # Replacing the last residual stream by the donor's gives the donor's hidden state under the base's own head;
        # the two models share the final norm and nearly share the head, so the margin lands near the donor's.
        last = np.array([r["patched"]["L31.resid"] for r in rows])
        assert np.abs(last - np.array([r["donor"] for r in rows])).max() < 1.0, (
            last,
            [r["donor"] for r in rows],
        )

        bad = lab.forward(
            json.dumps(
                {"type": "patch", "questions": ids[:2], "patches": [{"layer": 99, "component": "mlp"}]}
            )
        )
        assert "not counted: ValueError" in bad and lab.passes_left() == 1, bad
        print("PART 2 OK")
    finally:
        lab.kernel.stop()


def main(scratch: Path) -> None:
    items = medqa.load_medqa("dev")[:48]
    baseline = medqa.load_jsonl(BASELINE / "phi_dev.jsonl")[:48]
    part_one(scratch, items, baseline)
    part_two(scratch, items)
    print("LAB OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
