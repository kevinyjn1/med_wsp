"""Export weights and analysis results to the 3D viewer in `viz/` (manifest.json + raw binary panels).

A dataset is a tower of layers; a layer holds groups (attention, MLP, ...); a group holds panels. A panel
is one matrix drawn cell by cell. Values are stored either as float32 or as uint8 quantized over a
symmetric (signed) or zero-based (magnitude) range that the manifest records, so the viewer can show the
real number under the cursor.

Usage:
    python -m wsp.viz_export mit3          # the three character-level networks of MIT experiment 3
    python -m wsp.viz_export mit8          # the Qwen3.5-0.8B edit of MIT experiment 8
    python -m wsp.viz_export weightdiff    # MediPhi minus Phi-3.5, per head and per neuron
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "viz" / "public" / "data"


def write_panel(
    folder: Path,
    panel_id: str,
    matrix: np.ndarray,
    *,
    signed: bool,
    quantize: bool,
    color_max: float | None = None,
) -> dict:
    """Write one matrix and return its storage description for the manifest.

    Quantized panels store sign(x) * sqrt(|x| / vmax) in one byte: fine steps near zero, where most
    weights live, and no clipping of the largest ones. `colorMax` is where the color scale saturates.
    """
    matrix = np.asarray(matrix, np.float32)
    rows, cols = matrix.shape
    vmax = float(np.abs(matrix).max()) or 1.0
    file = f"{panel_id}.bin"
    if quantize:
        root = np.sign(matrix) * np.sqrt(np.abs(matrix) / vmax)
        q = (
            np.clip(np.rint(root * 127.0) + 128.0, 1, 255)
            if signed
            else np.clip(np.rint(root * 255.0), 0, 255)
        )
        q.astype(np.uint8).tofile(folder / file)
    else:
        matrix.tofile(folder / file)
    return {
        "file": file, "dtype": "uint8" if quantize else "float32", "encoding": "sqrt" if quantize else "linear",
        "rows": rows, "cols": cols, "signed": signed, "vmax": vmax, "colorMax": float(color_max or vmax),
    }  # fmt: skip


def bands(count: int, size: int, prefix: str) -> list[dict]:
    return [{"start": i * size, "size": size, "label": f"{prefix}{i}"} for i in range(count)]


# --------------------------------------------------------------------------------------------- MIT exp 3
def export_mit3() -> None:
    mit = REPO / "external" / "lm_understand-logs" / "exp3_language"
    truth = json.loads((mit / "eval" / "heads.json").read_text())
    scores_path = REPO / "experiments" / "exp_01_mit3_language" / "processed" / "scores.json"
    ours = {row["run"]: row for row in json.loads(scores_path.read_text())}
    layers_n, heads_n, d_model, head_dim = 4, 6, 384, 64
    for language in ("python", "romaji", "english"):
        folder = DATA / f"mit3_{language}"
        folder.mkdir(parents=True, exist_ok=True)
        weights = np.load(mit / "networks" / language / "blocks.npz")
        head_truth = truth[f"{language}_L4_H6_d384_s0"]
        runs = [ours[f"{language}_r0"], ours[f"{language}_r1"]]
        layers = []
        for layer in range(layers_n):
            head_info = []
            for head in range(heads_n):
                key = f"L{layer}H{head}"
                measured = head_truth[key]
                labels = [run["head_labels"].get(key, "unclear") for run in runs]
                is_copy = measured["induct"] > 0.2
                named = "copy" in labels
                status = (
                    "good" if is_copy and named else "serious" if is_copy else "warning" if named else None
                )
                short = {
                    "good": "copy",
                    "serious": "copy (missed)",
                    "warning": "copy? (not measured)",
                    None: "",
                }[status]
                head_info.append(
                    {
                        "head": head,
                        "astra": labels,
                        "measured": measured,
                        "status": status,
                        "statusShort": short,
                    }
                )

            def highlights(axis: str, info=head_info) -> list[dict]:
                verdicts = {
                    "good": "copying head: named by Astra and measured",
                    "serious": "measured copying head that Astra did not name",
                    "warning": "named a copying head by Astra, not measured",
                }
                return [
                    {
                        "axis": axis, "start": h["head"] * head_dim, "size": head_dim, "status": h["status"],
                        "short": f"H{h['head']} {h['statusShort']}",
                        "label": f"H{h['head']}: {verdicts[h['status']]} (induction score {h['measured']['induct']:.3f})",
                    }
                    for h in info if h["status"]
                ]  # fmt: skip

            def panel(name: str, label: str, *, transpose: bool = False, row_heads: bool = False,
                      col_heads: bool = False, note: str = "", own_row: bool = False, layer=layer, highlights=highlights,
                      weights=weights, folder=folder) -> dict:  # fmt: skip
                matrix = weights[f"{name}.{layer}"]
                matrix = matrix[None, :] if matrix.ndim == 1 else matrix
                matrix = matrix.T if transpose else matrix
                saturate = float(np.percentile(np.abs(matrix), 99.5))
                spec = write_panel(
                    folder, f"L{layer}_{name}", matrix, signed=True, quantize=True, color_max=saturate
                )
                spec.update(
                    {
                        "id": f"L{layer}.{name}",
                        "label": label,
                        "note": note,
                        "unit": "weight",
                        "ownRow": own_row,
                    }
                )
                if row_heads:
                    spec["rowBands"], spec["highlights"] = bands(heads_n, head_dim, "H"), highlights("row")
                if col_heads:
                    spec["colBands"], spec["highlights"] = bands(heads_n, head_dim, "H"), highlights("col")
                return spec

            layers.append({
                "name": f"Block {layer}", "heads": head_info,
                "groups": [
                    {"name": "Attention", "kind": "attention", "panels": [
                        panel("ln1_w", "LayerNorm 1 gain", note="1 x 384, scales the residual stream before attention", own_row=True),
                        panel("W_q", "W_q (query)", row_heads=True, note="rows 64h..64h+63 belong to head h"),
                        panel("W_k", "W_k (key)", row_heads=True, note="rows 64h..64h+63 belong to head h"),
                        panel("W_v", "W_v (value)", row_heads=True, note="rows 64h..64h+63 belong to head h"),
                        panel("W_o", "W_o (output)", col_heads=True, note="columns 64h..64h+63 belong to head h"),
                    ]},
                    {"name": "MLP", "kind": "mlp", "panels": [
                        panel("ln2_w", "LayerNorm 2 gain", note="1 x 384, scales the residual stream before the MLP", own_row=True),
                        panel("W_fc", "W_fc (in), shown transposed", transpose=True,
                              note="384 x 1536: column j is the input weights of hidden unit j", own_row=True),
                        panel("W_proj", "W_proj (out)", note="384 x 1536: column j is what hidden unit j writes", own_row=True),
                    ]},
                ],
            })  # fmt: skip
        copy_true = sorted(k for k, v in head_truth.items() if v["induct"] > 0.2)
        manifest = {
            "id": f"mit3_{language}",
            "title": f"Character-level transformer trained on {language}",
            "subtitle": "MIT weight-space-programming experiment 3: the four blocks Astra was given (7,084,032 weights)",
            "family": "MIT experiment 3 replication",
            "facts": [
                (
                    f"Astra's corpus guess (two conversations): {runs[0]['corpus_guess']} (P={runs[0]['p_correct']}), "
                    f"{runs[1]['corpus_guess']} (P={runs[1]['p_correct']})"
                ),
                f"Measured copying heads (induction score > 0.2): {', '.join(copy_true)}",
                f"Copying heads named by Astra: run 0 {', '.join(runs[0]['copy_named'])}; run 1 {', '.join(runs[1]['copy_named'])}",
                "Embeddings and the output layer were withheld, so the 384 residual dimensions carry no labels.",
            ],
            "colorNote": "Signed weights. Each matrix has its own scale and saturates at the 99.5th percentile of its magnitudes.",
            "d_model": d_model, "layers": layers,
        }  # fmt: skip
        (folder / "manifest.json").write_text(json.dumps(manifest))
        print("wrote", folder, f"{sum(f.stat().st_size for f in folder.iterdir()) / 1e6:.1f} MB")


# ------------------------------------------------------------------------------------------- MIT exp 8
def export_mit8(run: str = "p50c") -> None:
    """The Qwen3.5-0.8B weight edit of MIT experiment 8: base matrix, the change, and the edited columns."""
    from wsp.checkpoints import Checkpoint

    exp = REPO / "experiments" / "exp_03_mit8_math"
    report = json.loads((exp / "processed" / f"edit_{run}_report.json").read_text())
    checkpoint = Checkpoint("Qwen/Qwen3.5-0.8B")
    folder = DATA / f"mit8_qwen_{run}"
    folder.mkdir(parents=True, exist_ok=True)
    layers = []
    for tensor in report["tensors"]:
        name = tensor["name"]
        layer = int(name.split("layers.")[1].split(".")[0])
        base = checkpoint.f32(name)
        edited = np.load(exp / "work" / f"edit_{run}" / f"{name}.npy")
        direction = np.load(exp / "work" / f"edit_{run}" / f"{name}.direction.npy")
        delta = edited - base
        columns = np.array(tensor["changed_columns"])
        mask = np.zeros((1, base.shape[1]), np.float32)
        mask[0, columns] = 1.0
        saturate = float(np.percentile(np.abs(base), 99.5))
        key = f"L{layer}"

        def spec(
            panel_id, label, matrix, *, signed=True, color_max=None, note="", folder=folder, key=key, **extra
        ):
            out = write_panel(
                folder, f"{key}_{panel_id}", matrix, signed=signed, quantize=True, color_max=color_max
            )
            out.update({"id": f"{key}.{panel_id}", "label": label, "note": note, "unit": "weight", **extra})
            return out

        column_note = (
            f"the {len(columns)} edited hidden units, in index order: "
            + ", ".join(map(str, columns[:12]))
            + ", ..."
        )
        layers.append({
            "name": f"Layer {layer} MLP output (down_proj)",
            "stats": {"relative change": tensor["relative_change"]},
            "groups": [
                {"name": "Whole matrix", "kind": "attention", "panels": [
                    spec("mask", "Edited columns (64 of 3,584)", mask, signed=False, ownRow=True,
                         note="bright = a hidden unit whose output column was edited"),
                    spec("base", "down_proj before the edit", base, color_max=saturate, ownRow=True,
                         note="1,024 x 3,584: column j is what hidden unit j writes to the residual stream"),
                    spec("delta", "Change (edited minus original)", delta, color_max=float(np.abs(delta).max()) * 0.6,
                         ownRow=True, note="zero everywhere except the 64 edited columns"),
                ]},
                {"name": "Edited columns", "kind": "mlp", "panels": [
                    spec("u", "Loop direction u", direction[:, None], cellW=40, ownRow=False,
                         note="unit vector in the residual stream (1,024 dims) that the edit removes from the 64 columns"),
                    spec("cols_base", "64 columns before", base[:, columns], color_max=saturate, cellW=6, note=column_note),
                    spec("cols_edit", "64 columns after", edited[:, columns], color_max=saturate, cellW=6, note=column_note),
                    spec("cols_delta", "Change", delta[:, columns], color_max=float(np.abs(delta).max()) * 0.6, cellW=6,
                         note="rank one: every column is a multiple of u"),
                ]},
            ],
        })  # fmt: skip
    cosine = report.get("direction_cosine_between_tensors")
    manifest = {
        "id": f"mit8_qwen_{run}",
        "title": "Qwen3.5-0.8B: the weight edit that stops the thinking loop",
        "subtitle": "MIT weight-space-programming experiment 8 (50 forward passes): 131,072 weights in two MLP output matrices",
        "family": "MIT experiment 8 replication",
        "maxRowWidth": 3700,
        "facts": [
            "Rebuilt from the public base weights with MIT's released diff: both tensors match MIT's sha256 bit for bit.",
            "The edit is W' = W - 6 u (u^T W) on 64 columns per layer: rank one "
            f"({100 * min(t['rank1_energy'] for t in report['tensors']):.3f} % of the change's energy), recovered strength "
            + " and ".join(f"{t['strength_c']:.3f}" for t in report["tensors"]) + ".",
            f"The two layers use different directions (cosine {cosine:.3f}).",
            "MIT's reported effect: MATH-500 7.9 % to 48.1 %, GSM8K 13.8 % to 67.1 %, answers cut at the token limit 93.0 % to 33.1 %.",
        ],
        "colorNote": "Signed weights. Matrices saturate at the 99.5th percentile of the original matrix; the change has its own scale.",
        "layers": layers,
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest))
    print("wrote", folder, f"{sum(f.stat().st_size for f in folder.iterdir()) / 1e6:.1f} MB")


# ------------------------------------------------------------------------------------- MediPhi - Phi-3.5
def export_weightdiff() -> None:
    src = REPO / "experiments" / "exp_05_medical_circuit" / "processed"
    summary = json.loads((src / "weight_diff_summary.json").read_text())
    maps = np.load(src / "weight_diff_maps.npz")
    meta = summary["meta"]
    by = {(t["layer"], t["component"]): t for t in summary["tensors"] if "layer" in t}
    folder = DATA / "weightdiff_mediphi_phi"
    folder.mkdir(parents=True, exist_ok=True)
    head_max = float(max(maps[f"head_{p}"].max() for p in "qkvo"))
    neuron_max = float(
        np.percentile(np.concatenate([maps[f"neuron_{p}"].ravel() for p in ("gate", "up", "down")]), 99.9)
    )
    layers = []
    for layer in range(meta["n_layers"]):
        heads = np.stack([maps[f"head_{p}"][layer] for p in "qkvo"])  # 4 x 32
        attn = write_panel(folder, f"L{layer}_heads", heads, signed=False, quantize=False, color_max=head_max)
        attn.update({
            "id": f"L{layer}.heads", "label": "Heads (Q, K, V, O)", "unit": "relative change",
            "cell": 24, "rowLabels": ["Q", "K", "V", "O"], "colBands": bands(meta["n_heads"], 1, "H"),
            "note": "rows Q, K, V, O; columns are the 32 heads. Norm of the change in a head's weights divided by the norm of those weights in Phi-3.5",
        })  # fmt: skip
        mlp_panels = []
        for part, label in (("gate", "MLP gate"), ("up", "MLP up"), ("down", "MLP down")):
            tile = maps[f"neuron_{part}"][layer].reshape(64, 128)
            spec = write_panel(
                folder, f"L{layer}_{part}", tile, signed=False, quantize=False, color_max=neuron_max
            )
            spec.update({"id": f"L{layer}.{part}", "label": label, "unit": "relative change",
                         "cell": 6, "note": "8,192 neurons drawn as 64 x 128: cell (r, c) is neuron 128 r + c"})  # fmt: skip
            mlp_panels.append(spec)
        stats = {c.split(".")[-1]: round(by[(layer, c)]["rel_delta"], 4)
                 for c in ("self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj")}  # fmt: skip
        layers.append({"name": f"Layer {layer}", "stats": stats, "groups": [
            {"name": "Attention", "kind": "attention", "panels": [attn]},
            {"name": "MLP", "kind": "mlp", "panels": mlp_panels},
        ]})  # fmt: skip
    total = np.sqrt(sum(t["delta_norm"] ** 2 for t in summary["tensors"])) / np.sqrt(
        sum(t["base_norm"] ** 2 for t in summary["tensors"])
    )
    manifest = {
        "id": "weightdiff_mediphi_phi",
        "title": "Where MediPhi's weights differ from Phi-3.5-mini-instruct",
        "subtitle": "Relative size of the weight change, per attention head and per MLP neuron, all 32 layers",
        "family": "Phi-3.5 vs MediPhi",
        "facts": [
            f"Whole model: the change has {100 * total:.2f} % of the norm of the base weights.",
            "MediPhi is a breadcrumbs_ties merge of five domain experts onto Phi-3.5 (mergekit_config.yml in the model repo).",
            "The change is dense and high-rank: 50-65 % of entries differ and 90 % of the energy needs 1,500-2,400 of 3,072 singular values.",
            "Attention output (O) changes most in the first layers; query/key/value change more with depth; the MLP is nearly uniform.",
        ],
        "maxRowWidth": 2600,
        "colorNote": f"Sequential scale shared by all layers: heads 0 to {head_max:.3f}, neurons 0 to {neuron_max:.3f} (99.9th percentile).",
        "layers": layers,
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest))
    print("wrote", folder, f"{sum(f.stat().st_size for f in folder.iterdir()) / 1e6:.1f} MB")


def write_index() -> None:
    entries = []
    for manifest_path in sorted(DATA.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        entries.append({k: manifest[k] for k in ("id", "title", "subtitle", "family")})
    (DATA / "index.json").write_text(json.dumps(entries, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("what", nargs="+", choices=["mit3", "mit8", "weightdiff"])
    args = parser.parse_args()
    for what in args.what:
        {"mit3": export_mit3, "mit8": export_mit8, "weightdiff": export_weightdiff}[what]()
    write_index()


if __name__ == "__main__":
    main()
