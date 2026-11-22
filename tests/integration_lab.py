"""End-to-end check of the edit lab on the real Phi-3.5 weights (needs the GPU; run by hand):

    python tests/integration_lab.py <empty scratch folder>

Covers: kernel prelude, save_edit refusals, a paired score pass, a teacher-forced pass, budget accounting,
and that a failed request is not counted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from wsp import medqa
from wsp.lab import MedQALab


def main(scratch: Path) -> None:
    baseline_dir = REPO / "experiments" / "exp_04_medqa_baseline" / "processed"
    items = medqa.load_medqa("dev")[:48]
    baseline = medqa.load_jsonl(baseline_dir / "phi_dev.jsonl")[:48]
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
        assert "not counted" in bad and lab.state["passes"] == 0

        reply = lab.forward(json.dumps({"type": "score", "model": "both", "questions": ids}))
        print(reply[:900])
        summary = json.loads(reply.split("```")[1])
        expected = float(np.mean([r["correct"] for r in baseline]))
        assert abs(summary["original"]["accuracy"] - expected) < 1e-3, (summary["original"], expected)
        assert summary["paired"]["margin_difference"] != 0 and lab.state["passes"] == 1

        reply = lab.forward(json.dumps({
            "type": "teacher_forced", "model": "edited", "questions": ids[:3], "texts": ["The capital of France is"],
            "return": {"positions": "all", "token_logprobs": True, "topk": 5, "hidden_states": [0, 16, 32],
                       "mlp_act": [20], "attn_layers": [5]},
        }))  # fmt: skip
        print(reply[:1200])
        files = json.loads(reply.split("```")[1])["files"]
        longest = files["input_ids.npy"][1]
        assert files["hidden_16.npy"] == [4, longest, 3072] and files["mlp_act_20.npy"] == [4, longest, 8192]
        assert files["attn_5.npy"] == [4, 32, longest, longest] and files["topk_ids.npy"] == [4, longest, 5]
        acts = np.load(lab.work / "fwd_2" / "mlp_act_20.npy")
        attn = np.load(lab.work / "fwd_2" / "attn_5.npy")
        length = json.loads((lab.work / "fwd_2" / "meta.json").read_text())["lengths"][3]
        print("attention rows sum to", float(attn[3, 0, length - 1, :length].astype(np.float32).sum()),
              "| mlp act nonzero share", float((acts[3, :length] != 0).mean()))  # fmt: skip
        assert abs(float(attn[3, 0, length - 1, :length].astype(np.float32).sum()) - 1.0) < 0.02

        # hidden_32 must be the residual stream BEFORE the final norm: applying the norm and the output layer in the
        # kernel-free way below has to reproduce the model's own top token (texts were read with the edited weights,
        # which leave the final norm and the output layer untouched).
        from wsp.checkpoints import Checkpoint

        checkpoint = Checkpoint("phi")
        hidden = np.load(lab.work / "fwd_2" / "hidden_32.npy")[3, length - 1].astype(np.float32)
        normed = hidden / np.sqrt((hidden**2).mean() + 1e-5) * checkpoint.f32("model.norm.weight")
        top = int(np.argmax(checkpoint.f32("lm_head.weight") @ normed))
        expected = int(np.load(lab.work / "fwd_2" / "topk_ids.npy")[3, length - 1, 0])
        print("top token from hidden_32 through norm and head:", top, "| model's top token:", expected)
        assert top == expected

        out = lab.run_python("import os\nsorted(os.listdir('.')), edited()")
        print(out.strip())
        assert lab.passes_left() == 1
        print("LAB OK")
    finally:
        lab.kernel.stop()


if __name__ == "__main__":
    main(Path(sys.argv[1]))
