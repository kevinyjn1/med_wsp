"""Export weights and analysis results to the 3D viewer in `viz/` (manifest.json + raw binary panels).

A dataset is a tower of layers; a layer holds groups (attention, MLP, ...); a group holds panels. A panel
is one matrix drawn cell by cell. Values are stored either as float32 or as uint8 quantized over a
symmetric (signed) or zero-based (magnitude) range that the manifest records, so the viewer can show the
real number under the cursor.

Usage:
    python -m wsp.viz_export mit3          # the three character-level networks of MIT experiment 3
    python -m wsp.viz_export mit8          # the Qwen3.5-0.8B edit of MIT experiment 8
    python -m wsp.viz_export weightdiff    # MediPhi minus Phi-3.5, per head and per neuron
    python -m wsp.viz_export exp06         # the single-unit edit Astra tested and rejected (EXP-06 run1)
    python -m wsp.viz_export exp05c        # the donor transfer Astra built and withdrew (EXP-05c transfer1)
    python -m wsp.viz_export interventions looping   # EXP-05b patching and swaps, EXP-07 looping scan
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "viz" / "public" / "data"


MISSING = -1.0e30  # stored in float32 panels for cells that were not measured (NaN in the exporter's matrix)


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
    NaN marks a cell that was not measured; the viewer draws it hatched and says so.
    """
    matrix = np.asarray(matrix, np.float32)
    rows, cols = matrix.shape
    missing = np.isnan(matrix)
    if missing.any():
        assert not quantize, "cells that were not measured need a float32 panel"
        matrix = np.where(missing, 0.0, matrix).astype(np.float32)
    vmax = float(np.abs(matrix).max()) or 1.0
    matrix[missing] = MISSING
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
        **({"missing": MISSING} if missing.any() else {}),
    }  # fmt: skip


def bands(count: int, size: int, prefix: str) -> list[dict]:
    return [{"start": i * size, "size": size, "label": f"{prefix}{i}"} for i in range(count)]


def tile(folder: Path, panel_id: str, label: str, values, shape: tuple[int, int], *, signed: bool,
         color_max: float | None = None, cell: int = 6, **extra) -> dict:  # fmt: skip
    """A float32 panel from a flat list of values laid out row by row. A list shorter than the grid leaves the
    last cells empty (drawn hatched; the viewer says there is no value there)."""
    values = np.asarray(values, np.float32).ravel()
    grid = np.full(shape[0] * shape[1], np.nan, np.float32)
    grid[: len(values)] = values
    spec = write_panel(
        folder, panel_id, grid.reshape(shape), signed=signed, quantize=False, color_max=color_max
    )
    spec.update({"id": panel_id, "label": label, "cell": cell, **extra})
    if len(values) < len(grid):
        spec["missingText"] = "no value here (the grid is larger than the list)"
    return spec


def question_grid(folder: Path, key: str, title: str, rows: list[dict], shape: tuple[int, int]) -> tuple[list[dict], dict]:  # fmt: skip
    """Three panels with one cell per question, sorted by the unedited model's margin: that margin, whether the
    edit fixed or broke the question, and the change of the margin. `rows` hold `original` and `edited`, each with
    `correct` and `margin`. Also returns the paired counts and the sorted arrays."""
    rows = sorted(rows, key=lambda r: r["original"]["margin"])
    before = np.array([r["original"]["correct"] for r in rows], bool)
    after = np.array([r["edited"]["correct"] for r in rows], bool)
    outcome = (after & ~before).astype(np.float32) - (before & ~after).astype(np.float32)
    margin_before = np.array([r["original"]["margin"] for r in rows], np.float32)
    change = np.array([r["edited"]["margin"] for r in rows], np.float32) - margin_before
    fixed, broken, n = int((outcome > 0).sum()), int((outcome < 0).sum()), len(rows)
    order = f"{title}, sorted by the unedited model's margin: most wrong first (top left) to most right (bottom right)"
    panels = [
        tile(folder, f"{key}_before", "Margin before the edit", margin_before, shape, signed=True,
             color_max=float(np.percentile(np.abs(margin_before), 98)), cell=14, unit="margin",
             note=order + ". Margin = log-probability of the correct letter minus the best wrong letter; positive = answered correctly"),
        tile(folder, f"{key}_outcome", f"Outcome: {fixed} fixed, {broken} broken", outcome, shape, signed=True,
             color_max=2.0, cell=14, unit="outcome", note=order),
        tile(folder, f"{key}_change", "Change of the margin", change, shape, signed=True,
             color_max=float(np.percentile(np.abs(change), 98)), cell=14, unit="margin change", note=order),
    ]  # fmt: skip
    return panels, {
        "n": n, "fixed": fixed, "broken": broken, "before": float(before.mean()), "after": float(after.mean()),
        "se": float(np.std(after.astype(float) - before.astype(float), ddof=1) / np.sqrt(n)),
        "margin_before": margin_before, "change": change, "ids": [r["id"] for r in rows],
    }  # fmt: skip


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


# ------------------------------------------------------------------------------- interventions and looping
MODEL_NAMES = {"phi": "Phi-3.5", "mediphi": "MediPhi", "mediphi_instruct": "MediPhi-Instruct"}


def export_interventions(donor: str = "mediphi_instruct", style: str = "bare") -> None:
    """Activation patching and weight swapping (EXP-05b), per layer: the mean change of the correct answer's
    margin and the share of questions answered correctly afterwards, for each of the three question sets."""
    suffix = "_bare" if style == "bare" else ""
    src = REPO / "experiments" / "exp_05_medical_circuit" / "processed" / f"patch_{donor}{suffix}"
    name = MODEL_NAMES.get(donor, donor)
    sets, n_layers = ("gain", "loss", "both"), 32
    folder = DATA / f"interventions_{donor}{suffix}_phi"
    folder.mkdir(parents=True, exist_ok=True)
    layer_labels = [str(i) for i in range(n_layers)]
    groups, facts = [], []

    def pair(prefix: str, change: np.ndarray, right: np.ndarray, labels: list[str], what: str) -> list[dict]:
        """Two stacked panels over the same rows: margin change (signed) and share answered correctly."""
        common = {"cell": 44, "rowLabels": labels, "colLabels": layer_labels, "ownRow": True}
        margin = write_panel(folder, f"{prefix}_margin", change, signed=True, quantize=False)
        margin.update({"id": f"{prefix}_margin", "label": "Change of the correct answer's margin", "unit": "margin change",
                       "note": f"mean over the set's questions, log-prob units, when {what}; columns are layers", **common})  # fmt: skip
        share = write_panel(folder, f"{prefix}_right", right, signed=False, quantize=False, color_max=1.0)
        share.update({"id": f"{prefix}_right", "label": "Share of questions answered correctly", "unit": "share",
                      "note": f"when {what}. Unchanged Phi-3.5 answers none of the gain questions and all of the loss "
                              "and both questions correctly.", **common})  # fmt: skip
        return [margin, share]

    rows = (
        [json.loads(line) for line in (src / "activation.jsonl").read_text().splitlines()]
        if (src / "activation.jsonl").exists()
        else []
    )
    if rows:
        components = {"resid": "residual stream", "attn": "attention output", "mlp": "MLP output"}
        change = np.zeros((len(sets) * len(components), n_layers), np.float32)
        right = np.zeros_like(change)
        labels, gaps = [], {}
        for si, set_name in enumerate(sets):
            members = [r for r in rows if r["set"] == set_name]
            base = np.array([r["base"] for r in members])
            gaps[set_name] = float(np.mean([r["donor"] for r in members]) - base.mean())
            for ci, (component, text) in enumerate(components.items()):
                labels.append(f"{set_name}: {text}")
                patched = np.array([[r["patched"][f"L{layer}.{component}"] for layer in range(n_layers)] for r in members])  # fmt: skip
                change[si * len(components) + ci] = (patched - base[:, None]).mean(axis=0)
                right[si * len(components) + ci] = (patched > 0).mean(axis=0)
            facts.append(f"{set_name.capitalize()} set ({len(members)} questions): Phi-3.5's mean margin is {base.mean():+.2f}, "
                         f"{name}'s is {base.mean() + gaps[set_name]:+.2f} (difference {gaps[set_name]:+.2f}).")  # fmt: skip
        closed = change[0] / gaps["gain"]  # gain set, residual stream
        half, most = int(np.argmax(closed >= 0.5)), int(np.argmax(closed >= 0.95))
        single = {text: (int(np.argmax(np.abs(change[ci]))), float(change[ci][np.argmax(np.abs(change[ci]))]))
                  for ci, text in enumerate(components.values()) if text != "residual stream"}  # fmt: skip
        facts.append(f"Residual stream, gain set: taking {name}'s stream after layer {half} closes half of the difference, "
                     f"after layer {most} at least 95 % of it; every later layer stays between "
                     f"{100 * closed[most:].min():.0f} % and {100 * closed[most:].max():.0f} %.")  # fmt: skip
        facts.append("Single components, gain set: the largest effects are "
                     + " and ".join(f"{text} of layer {layer} ({value:+.2f}, {100 * value / gaps['gain']:.0f} % of the difference)"
                                    for text, (layer, value) in single.items()) + ".")  # fmt: skip
        groups.append({"name": f"Activations taken from {name}", "kind": "attention",
                       "panels": pair("activation", change, right, labels,
                                      f"one layer's activation (all token positions) comes from {name}")})  # fmt: skip

    swaps = (
        {
            row["config"]: row["margins"]
            for row in map(json.loads, (src / "weights.jsonl").read_text().splitlines())
        }
        if (src / "weights.jsonl").exists()
        else {}
    )
    if swaps:
        question_sets = json.loads((src / "question_sets.json").read_text())
        parts = {"attn": "attention weights", "mlp": "MLP weights"}
        change = np.zeros((len(sets) * len(parts), n_layers), np.float32)
        right = np.zeros_like(change)
        labels = []
        for si, set_name in enumerate(sets):
            ids = question_sets[set_name]
            base = np.array([swaps["base"][i] for i in ids])
            for pi, (part, text) in enumerate(parts.items()):
                labels.append(f"{set_name}: {text}")
                for layer in range(n_layers):
                    if f"L{layer}.{part}" in swaps:
                        swapped = np.array([swaps[f"L{layer}.{part}"][i] for i in ids])
                        change[si * len(parts) + pi, layer] = (swapped - base).mean()
                        right[si * len(parts) + pi, layer] = (swapped > 0).mean()
        gain_ids = question_sets["gain"]
        everything = (
            float(np.mean([swaps["all"][i] - swaps["base"][i] for i in gain_ids])) if "all" in swaps else None
        )
        row, layer = np.unravel_index(np.argmax(np.abs(change[: len(parts)])), change[: len(parts)].shape)
        fact = (f"Weight swaps, gain set: the largest single-layer effect is {change[row, layer]:+.2f} "
                f"({list(parts.values())[row]} of layer {layer})")  # fmt: skip
        if everything is not None:
            fact += (f", {100 * change[row, layer] / everything:.0f} % of the {everything:+.2f} that copying every weight gives. "
                     f"The 32 single-layer attention swaps add up to {change[0].sum():+.2f}, the MLP swaps to {change[1].sum():+.2f}")  # fmt: skip
        facts.append(fact + ".")
        groups.append({"name": f"Weights copied from {name}", "kind": "mlp",
                       "panels": pair("weights", change, right, labels,
                                      f"one layer's attention weights or MLP weights are {name}'s")})  # fmt: skip
    if not groups:
        print("no intervention results yet")
        return
    manifest = {
        "id": f"interventions_{donor}{suffix}_phi", "title": f"Which layers carry the difference between Phi-3.5 and {name}?",
        "subtitle": f"MedQA, {style} prompt: what happens to Phi-3.5's answers when one layer's activations or weights come from {name}",
        "family": "Phi-3.5 vs MediPhi", "maxRowWidth": 2600,
        "facts": [(f"Question sets (60 each, from the practice pool): gain = only {name} answers correctly; "
                   "loss = only Phi-3.5 answers correctly; both = both answer correctly."),
                  "Margin = log-probability of the correct letter minus the best wrong letter; above zero means answered correctly.",
                  *facts],
        "colorNote": ("Margin panels: red raises the correct answer, blue lowers it; each panel has its own scale. "
                      "Share panels: from 0 % (dark) to 100 % of the set's questions answered correctly."),
        "layers": [{"name": "All 32 layers (columns)", "groups": groups}],
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest))
    print("wrote", folder)


def export_looping(tag: str = "phi_dev_600") -> None:
    """Layer-looping scan (EXP-07): change of accuracy and margin against the unmodified model, by window start."""
    src = REPO / "experiments" / "exp_07_layer_looping" / "processed" / f"scan_{tag}.jsonl"
    # the "defaults" and "position" sweeps share two configurations; keep one row per configuration
    rows = list({r["config"]: r for r in map(json.loads, src.read_text().splitlines())}.values())
    baseline = next(r for r in rows if r["config"] == "baseline")
    scan = [
        r
        for r in rows
        if r.get("length") == 4
        and r.get("steps") == 2
        and r.get("rule") in ("chen", "naive")
        and r.get("beta", 0) == 0
    ]
    starts = sorted({r["start"] for r in scan})
    rules = ("chen", "naive")
    folder = DATA / f"looping_{tag}"
    folder.mkdir(parents=True, exist_ok=True)
    margin = np.zeros((len(rules), len(starts)), np.float32)
    accuracy = np.zeros((len(rules), len(starts)), np.float32)
    changed = np.zeros((len(rules), len(starts)), np.float32)
    for r in scan:
        cell = rules.index(r["rule"]), starts.index(r["start"])
        margin[cell], accuracy[cell], changed[cell] = (
            r["margin_delta"],
            100 * r["acc_delta"],
            r["fixed"] + r["broken"],
        )
    common = {"cell": 70, "rowLabels": ["damped (Chen), 2 sub-steps", "plain repetition, twice"], "ownRow": True,
              "colLabels": [f"{s}-{s + 3}" for s in starts]}  # fmt: skip
    panels = []
    for panel_id, label, grid, unit, signed, note in (
        ("accuracy", "Change of accuracy (percentage points)", accuracy, "points", True,
         "accuracy with the window repeated minus the unmodified model, same questions; columns are the repeated layers"),
        ("margin", "Change of the correct answer's margin", margin, "margin change", True,
         "mean over questions, log-prob units; columns are the repeated layers"),
        ("changed", "Questions whose result changed (fixed + broken)", changed, "count", False,
         f"of {baseline['n']} questions: how many went from wrong to right or from right to wrong"),
    ):  # fmt: skip
        spec = write_panel(folder, panel_id, grid, signed=signed, quantize=False)
        spec.update({"id": panel_id, "label": label, "unit": unit, "note": note, **common})
        panels.append(spec)

    def plain(r: dict) -> str:
        rule = {"chen": "damped rule (Chen)", "naive": "plain repetition", "lys": "averaged loop (Lys)"}[
            r["rule"]
        ]
        return f"{rule} on layers {r['start']}-{r['start'] + r['length'] - 1}, {r['steps']} steps"

    def outcome(r: dict) -> str:
        p = "p < 0.001" if r["mcnemar_p"] < 0.001 else f"p = {r['mcnemar_p']:.3f}"
        return (f"accuracy {100 * r['acc_delta']:+.2f} points ({r['fixed']} fixed, {r['broken']} broken, "
                f"sign test {p}), margin {r['margin_delta']:+.2f}")  # fmt: skip

    tested = [r for r in rows if r["config"] != "baseline"]
    best, worst = max(tested, key=lambda r: r["acc_delta"]), min(tested, key=lambda r: r["acc_delta"])
    higher = [r for r in tested if r["acc_delta"] > 0]
    manifest = {
        "id": f"looping_{tag}", "title": "Layer looping on Phi-3.5: which 4-layer window to repeat?",
        "subtitle": f"MedQA {tag.split('_')[1]} split, {baseline['n']} questions, chat prompt, single-pass letter scoring",
        "family": "Training-free layer looping", "maxRowWidth": 2600,
        "facts": [
            f"Unmodified model: accuracy {100 * baseline['acc']:.1f} %, mean margin {baseline['mean_margin']:+.2f}.",
            f"Largest gain: {plain(best)}: {outcome(best)}.",
            f"Largest loss: {plain(worst)}: {outcome(worst)}.",
            (f"{len(higher)} of {len(tested)} configurations have higher accuracy than the unmodified model; "
             f"{sum(r['mcnemar_p'] < 0.05 for r in higher)} of those reach p < 0.05 (not corrected for {len(tested)} comparisons)."),
            "The papers' default recipes on layers 14-17: " + "; ".join(
                f"{plain(r).split(' on ')[0]}, {r['steps']} steps: {100 * r['acc_delta']:+.2f} points"
                for r in tested if r["start"] == 14 and r["length"] == 4) + ".",
        ],
        "colorNote": ("Accuracy and margin: red = better than the unmodified model, blue = worse; each panel has its own scale. "
                      "Changed questions: brighter = more questions changed."),
        "layers": [{"name": "Repeated layers (columns)", "groups": [{"name": "4-layer window, 2 steps", "kind": "attention", "panels": panels}]}],
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest))
    print("wrote", folder)


# ------------------------------------------------------------------- EXP-06: failure-guided editing
def export_exp06(run: str = "run1") -> None:
    """The one-unit edit Astra locked and then rejected: where it sits, what it does to the unit's output
    direction, and what it did to each practice question (the questions used to choose it vs untouched ones)."""
    from wsp.checkpoints import Checkpoint

    src = REPO / "experiments" / "exp_06_failure_guided_editing" / "raw" / run
    work = src / "work"
    lock = json.loads((work / "candidate_lock.json").read_text())
    name, unit = lock["tensor"], int(lock["column"])
    layer = int(name.split("layers.")[1].split(".")[0])
    base = Checkpoint("phi").f32(name)
    edited = np.load(work / f"locked_l{layer}_orth_minus{-int(lock['orthogonal_scale'])}" / f"{name}.npy")
    changed = np.flatnonzero(np.abs(edited - base).max(axis=0) > 0)
    assert changed.tolist() == [unit], changed
    scale = float(lock["orthogonal_scale"])
    # edited = kept + scale * reversed, base = kept + reversed
    reversed_part = (base[:, unit] - edited[:, unit]) / (1.0 - scale)
    kept_part = base[:, unit] - reversed_part
    relative = float(np.linalg.norm(edited - base) / np.linalg.norm(base))
    folder = DATA / f"exp06_edit_{run}"
    folder.mkdir(parents=True, exist_ok=True)

    panel = partial(tile, folder)

    mask = np.zeros(base.shape[1], np.float32)
    mask[unit] = 1.0
    norms = np.linalg.norm(base, axis=0)
    # Astra's reason for picking the unit: each unit's mean direct contribution to (logit of A minus the mean
    # logit of A-D) at the answer position, on the 128 discovery questions of forward pass 1. The readout is the
    # output layer's rows for the tokens "_A".."_D" (319, 350, 315, 360) times the final norm's weight.
    checkpoint = Checkpoint("phi")
    readout = checkpoint.f32("lm_head.weight")[[319, 350, 315, 360]] * checkpoint.f32("model.norm.weight")
    toward_a = (readout[0] - readout.mean(axis=0)) @ base
    activation = np.load(work / "fwd_1" / f"mlp_act_{layer}.npy").astype(np.float32)
    push = activation.mean(axis=0) * toward_a
    runner_up = float(np.sort(np.abs(push))[-2])
    where = f"8,192 units drawn as 64 x 128: cell (r, c) is unit 128 r + c. Unit {unit} is row {unit // 128}, column {unit % 128}"
    saturate = float(np.percentile(np.abs(edited[:, unit]), 99.5))
    direction = "3,072 residual-stream dimensions drawn as 48 x 64: cell (r, c) is dimension 64 r + c"
    layers = [
        {"name": f"Layer {layer} MLP output (down_proj): 8,192 units", "stats": {"relative change": relative}, "groups": [
            {"name": "Location", "kind": "mlp", "panels": [
                panel("mask", f"Edited unit: {unit}", mask, (64, 128), signed=False, unit="weight",
                     note=where + ". Bright = edited"),
                panel("push", "Each unit's direct push on answer A", push, (64, 128), signed=True, unit="push",
                     color_max=2 * float(np.abs(push).max()),  # the one strong unit lands on the saturated hue, not on near-white
                     note=where + f". Mean over {len(activation)} discovery questions of the unit's activation times its "
                     f"direct effect on the logit of A minus the mean logit of A to D. Unit {unit}: {push[unit]:+.2f}; "
                     f"the next largest unit: {runner_up:.2f} in size. Descriptive, not a causal effect"),
            ]},
            {"name": f"Unit {unit}", "kind": "attention", "panels": [
                panel("before", "Output direction before", base[:, unit], (48, 64), signed=True, color_max=saturate, cell=8,
                     unit="weight", note=direction + f". Norm {norms[unit]:.2f} (median over units {np.median(norms):.2f})"),
                panel("kept", "Letter-readout part (kept)", kept_part, (48, 64), signed=True,
                     color_max=saturate, cell=8, unit="weight",
                     note=direction + f". Norm {np.linalg.norm(kept_part):.2f}; lies in the three directions that separate A, B, C, D at the output"),
                panel("reversed", f"The rest (multiplied by {scale:g})", reversed_part, (48, 64), signed=True,
                     color_max=saturate, cell=8, unit="weight",
                     note=direction + f". Norm {np.linalg.norm(reversed_part):.2f} before the edit; shown before the multiplication"),
                panel("after", "Output direction after", edited[:, unit], (48, 64), signed=True, color_max=saturate, cell=8,
                     unit="weight", note=direction + f". Norm {np.linalg.norm(edited[:, unit]):.2f} (was {norms[unit]:.2f})"),
            ]},
        ]},
    ]  # fmt: skip

    def scored(passes: list[int]) -> list[dict]:
        return [json.loads(line) for k in passes
                for line in (work / f"score_{k}.jsonl").read_text().split("\n") if line.strip()]  # fmt: skip

    facts, groups = [], []
    for key, short, title, passes, shape in (
        ("selection", "Selection", "512 questions used to choose the edit", [21, 22], (16, 32)),
        ("reserve", "Reserve", "256 untouched questions (reserve)", [26], (8, 32)),
        ("confirmation", "Confirmation", "1,024 untouched questions (final confirmation)", [27, 28, 29, 30], (32, 32)),
    ):  # fmt: skip
        panels, stats = question_grid(folder, key, title, scored(passes), shape)
        facts.append(f"{title}: {100 * stats['before']:.2f} % to {100 * stats['after']:.2f} % "
                     f"({100 * (stats['after'] - stats['before']):+.2f} points, standard error {100 * stats['se']:.2f}; "
                     f"{stats['fixed']} fixed, {stats['broken']} broken).")  # fmt: skip
        if key == "confirmation":
            margin_before, change = stats["margin_before"], stats["change"]
            right = margin_before > 0
            facts.append(
                f"On those {stats['n']:,} questions the edit mostly sharpened the answer the model already gave: the margin of the "
                f"correct answer moved by {change[right].mean():+.2f} on average where the model was right and by "
                f"{change[~right].mean():+.2f} where it was wrong (correlation with the margin before: "
                f"{np.corrcoef(margin_before, change)[0, 1]:.2f})."
            )
        groups.append({"name": short, "kind": "attention", "panels": panels})
    layers.append({"name": "What the edit did to each practice question", "groups": groups})

    final = json.loads((src / "final_eval.json").read_text())
    manifest = {
        "id": f"exp06_edit_{run}",
        "title": "Failure-guided editing of Phi-3.5: the one unit Astra tried to change",
        "subtitle": f"EXP-06 {run}: layer {layer}, MLP unit {unit}. It helped on the questions used to choose it and did not hold on untouched ones",
        "family": "Failure-guided weight editing",
        "maxRowWidth": 2400,
        "facts": [
            f"Astra submitted no edit. MedQA test accuracy is {100 * final['original_accuracy']:.2f} % before and after ({final['n']:,} questions).",
            (f"The candidate changes one of 8,192 output directions in one matrix ({100 * relative:.2f} % of that matrix's norm): "
             f"the part outside the answer-letter readout is multiplied by {scale:g}."),
            (f"Why this unit: the unedited model answers A too rarely, and on {len(activation)} questions this unit's direct push on "
             f"A was {push[unit]:+.1f}, against at most {runner_up:.1f} in size for any other unit of the layer."),
            *facts,
            ("Astra's rule, written before it looked at the confirmation questions: keep the edit only if the gain there "
             "is more than 2 standard errors and the sign test gives p < 0.05. Not met, so it restored the original weights."),
        ],
        "colorNote": "Red = positive, blue = negative. Weights of the unit share one scale. Outcome: red = fixed by the edit, "
        "blue = broken, gray = same result. Margins are in log-probability units and saturate at the 98th percentile.",
        "layers": layers,
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest))
    print("wrote", folder, f"{sum(f.stat().st_size for f in folder.iterdir()) / 1e6:.2f} MB")
    for fact in manifest["facts"]:
        print("  -", fact)


# ------------------------------------------------------------------- EXP-05c: the donor transfer session
# Every transfer of donor weights that Astra scored, in the order it first tried them, with the forward passes
# that scored it. Which tensors a pass touched is in the lab's record; which heads or units inside them is from
# Astra's notes.
EXP05C_TRIALS = (
    ("Layer 11: the whole MLP", (4,)),
    ("Layer 0: the norm before attention", (5,)),
    ("Six attention heads in layers 7, 18 and 25", (7,)),
    ("Layers 0-7: attention", (8,)),
    ("Four norms in layers 0-2", (9,)),
    ("12 early layers: MLP output weights", (10,)),
    ("Layers 0-7: everything", (11, 21)),
    ("Layers 0-7: MLP input weights", (12,)),
    ("<b>Candidate</b>: layers 0-7, attention + 256 MLP units", (13, 14, 18, 19, 29, 30)),
    ("Candidate at half strength", (15,)),
    ("Layer 0: everything", (16,)),
    ("Control: same size, in 8 later layers", (17, 24)),
    ("Control: candidate with random MLP units", (20,)),
    ("Candidate at 1.5 times strength", (22, 23)),
)
EXP05C_SETS = {  # Astra's question sets: short name, what the questions are, grid shape (rows x columns = size)
    "explore": ("Exploration", "questions used to choose between transfers", (8, 32)),
    "validation": (
        "Validation",
        "other development questions, first used after the candidate was built",
        (16, 32),
    ),
    "confirmation": ("Confirmation", "questions kept untouched until the last two passes", (32, 32)),
}
PART_LABELS = {
    "input_layernorm": "norm before attention", "self_attn.qkv_proj": "attention Q, K, V",
    "self_attn.o_proj": "attention output", "post_attention_layernorm": "norm before the MLP",
    "mlp.gate_up_proj": "MLP gate and up", "mlp.down_proj": "MLP down",
}  # fmt: skip


def export_exp05c(run: str = "transfer1", donor: str = "mediphi_instruct") -> None:
    """The donor session (EXP-05c): every transfer Astra scored, the candidate's weights, and what the candidate
    did to each question. Reads `processed/check_<run>.json`, `check_<run>_questions.jsonl` and, when the
    candidate was scored on the test split, `check_<run>_test_questions.jsonl` (all written by
    `experiments/exp_05_medical_circuit/check_session.py`)."""
    src = REPO / "experiments" / "exp_05_medical_circuit"
    check = json.loads((src / "processed" / f"check_{run}.json").read_text())
    records, weights = check["records"], check["weights"]
    per_pass: dict = {}
    for row in map(json.loads, (src / "processed" / f"check_{run}_questions.jsonl").read_text().splitlines()):
        per_pass.setdefault(row["pass"], []).append(row)
    final = json.loads((src / "raw" / run / "final_eval.json").read_text())["by_prompt"]
    # The candidate on the test split: not one of Astra's passes, scored afterwards at the researcher's request.
    test, test_file = check.get("test"), src / "processed" / f"check_{run}_test_questions.jsonl"
    test_rows = [json.loads(line) for line in test_file.read_text().splitlines()] if test else []
    sets = dict(EXP05C_SETS)
    if test:
        sets["test"] = (
            "Test",
            "MedQA test questions, scored once after the session at the researcher's request",
            (36, 36),  # 1,273 questions: the last 23 cells of the grid stay empty
        )
    name = MODEL_NAMES[donor]
    folder = DATA / f"exp05c_transfer_{run}"
    folder.mkdir(parents=True, exist_ok=True)
    panel = partial(tile, folder)
    scored = {p["pass"]: p for p in records["passes"] if "fixed" in p}
    parts, n_layers = weights["parts"], len(weights["donor_relative_change"][0])
    layer_labels = [str(i) for i in range(n_layers)]

    # ---- 1. every transfer Astra scored: where it took donor weights, and what it did
    columns = [(key, prompt) for key in sets for prompt in ("bare", "chat")]
    taken = np.zeros((len(EXP05C_TRIALS), n_layers), np.float32)
    accuracy = np.full((len(EXP05C_TRIALS), len(columns)), np.nan, np.float32)
    margin = np.full_like(accuracy, np.nan)
    for t, (_, passes) in enumerate(EXP05C_TRIALS):
        for tensor in scored[passes[0]]["edited_tensors"]:
            taken[t, int(tensor.split(".")[2])] += 1
        for k in passes:
            cell = t, columns.index((scored[k]["question_set"], scored[k]["prompt"]))
            accuracy[cell], margin[cell] = scored[k]["difference_points"], scored[k]["margin_change"]
    trial_labels = [label for label, _ in EXP05C_TRIALS]
    is_candidate = ["Candidate</b>" in label for label in trial_labels].index(True)
    for prompt in ("bare", "chat") if test else ():
        cell = is_candidate, columns.index(("test", prompt))
        accuracy[cell], margin[cell] = test[prompt]["difference_points"], test[prompt]["margin_change"]
    size = {key: shape[0] * shape[1] for key, (_, _, shape) in sets.items()} | (
        {"test": len(test_rows)} if test else {}
    )
    sets_text = "; ".join(f"{size[key]:,} = {text}" for key, (_, text, _) in sets.items())
    # cell widths chosen so that the whole row, framed with its row labels, still shows every column label
    measured = {"cell": 30, "cellW": 96, "rowNames": trial_labels,
                "colLabels": [f"{size[key]:,} {prompt}" for key, prompt in columns]}  # fmt: skip
    board = [
        panel("trials_taken", f"Where the weights are {name}'s (columns: layers)", taken, taken.shape, signed=False,
              color_max=float(len(parts)), cell=30, cellW=22, unit="tensors", rowLabels=trial_labels, colLabels=layer_labels,
              note="From the lab's record of each pass. A tensor counts when any part of it was changed: the candidate "
                   "takes all attention weights but only 256 of 8,192 MLP units per layer"),
        panel("trials_accuracy", "Change of accuracy (points)", accuracy, accuracy.shape, signed=True, unit="points",
              note=f"Edited model minus unedited model on the same questions. Columns: {sets_text}. Hatched = not measured",
              **measured),
        panel("trials_margin", "Change of the correct answer's margin", margin, margin.shape, signed=True,
              unit="margin change", note=f"Mean over the questions, log-probability units. Columns: {sets_text}", **measured),
    ]  # fmt: skip

    # ---- 2. the candidate's weights
    by_name = {t["name"]: t for t in weights["tensors"]}
    donor_change = np.array(weights["donor_relative_change"], np.float32)
    applied = np.array([[by_name.get(f"model.layers.{layer}.{part}.weight", {}).get("relative_change", 0.0)
                         for layer in range(n_layers)] for part in parts], np.float32)  # fmt: skip
    matrices = [i for i, part in enumerate(parts) if "norm" not in part]
    scale = float(donor_change[matrices].max())  # the norm vectors of layer 0 change far more than any matrix
    grid = {"cell": 44, "unit": "relative change", "rowLabels": [PART_LABELS[part] for part in parts],
            "colLabels": layer_labels, "ownRow": True}  # fmt: skip
    maps = [
        panel("donor_change", f"How much {name} differs from Phi-3.5", donor_change, donor_change.shape, signed=False,
              color_max=scale, note="Norm of the difference divided by the norm of Phi-3.5's tensor; columns are layers. "
              f"The color saturates at {100 * scale:.1f} %, the largest value of a weight matrix", **grid),
        panel("candidate_change", "How much of that the candidate takes", applied, applied.shape, signed=False,
              color_max=scale, note="Same measure and color scale for the candidate: equal to the panel above where a whole "
              "tensor is taken, smaller where only 256 MLP units are, zero elsewhere", **grid),
    ]  # fmt: skip
    units = {int(layer): ids for layer, ids in weights["units"].items()}
    masks = []
    for layer, ids in sorted(units.items()):
        mask = np.zeros(8192, np.float32)
        mask[ids] = 1.0
        masks.append(panel(f"units_L{layer}", f"Layer {layer}: {len(ids)} units", mask, (64, 128), signed=False, cell=4,
                           unit="weight", note="8,192 MLP units drawn as 64 x 128: cell (r, c) is unit 128 r + c. Bright = "
                           f"the unit's input and output weights are {name}'s"))  # fmt: skip

    # ---- 3. what the candidate did to each question
    candidate = EXP05C_TRIALS[is_candidate][1]

    def rows_of(key: str, prompt: str) -> list[dict]:
        """The candidate's per-question records on one question set under one prompt."""
        if key == "test":
            return [{"id": r["id"], **r[prompt]} for r in test_rows]
        k = next(k for k in candidate if (scored[k]["question_set"], scored[k]["prompt"]) == (key, prompt))
        return per_pass[k]

    # the donor itself on the questions where it was looked up (EXP-04 results), by question id and prompt
    whole_donor = {
        "confirmation": {r["id"]: r for r in per_pass["donor"]},
        "test": {r["id"]: r["donor"] for r in test_rows},
    }
    facts, groups, stats = [], [], {}
    for key, (short, text, shape) in sets.items():
        for prompt in ("bare", "chat"):
            rows = rows_of(key, prompt)
            assert len(rows) == size[key], (key, prompt, len(rows))
            panels, stat = question_grid(
                folder, f"{key}_{prompt}", f"{size[key]:,} {text}, {prompt} prompt", rows, shape
            )
            stats[key, prompt] = stat
            if key in whole_donor:  # the donor itself on the same questions, in the same order
                correct = {r["id"]: r["original"]["correct"] for r in rows}
                before = np.array([correct[i] for i in stat["ids"]])
                whole = np.array([whole_donor[key][i][prompt]["correct"] for i in stat["ids"]])
                outcome = (whole & ~before).astype(np.float32) - (before & ~whole).astype(np.float32)
                panels.append(panel(
                    f"{key}_{prompt}_donor", f"With all of {name}: {int((outcome > 0).sum())} fixed, {int((outcome < 0).sum())} broken",
                    outcome, shape, signed=True, color_max=2.0, cell=14, unit="outcome",
                    note=f"{name} itself against the unedited Phi-3.5 on the same questions in the same order (EXP-04 results)"))  # fmt: skip
            groups.append({"name": f"{short}, {prompt} prompt", "kind": "attention", "panels": panels})
        bare, chat = stats[key, "bare"], stats[key, "chat"]
        facts.append(
            f"{short} ({bare['n']:,} {text}): bare prompt {100 * bare['before']:.2f} % to {100 * bare['after']:.2f} % "
            f"({100 * (bare['after'] - bare['before']):+.2f} points, standard error {100 * bare['se']:.2f}; {bare['fixed']} fixed, "
            f"{bare['broken']} broken); chat prompt {100 * (chat['after'] - chat['before']):+.2f} points "
            f"(standard error {100 * chat['se']:.2f}; {chat['fixed']} fixed, {chat['broken']} broken)."
        )
    # the bare-prompt outcome, with the questions ordered by how the unedited model does under the chat prompt
    for key in whole_donor if test else ("confirmation",):
        short, _, shape = sets[key]
        chat_margin = {r["id"]: r["original"]["margin"] for r in rows_of(key, "chat")}
        ordered = sorted(rows_of(key, "bare"), key=lambda r: chat_margin[r["id"]])
        outcome = np.array(
            [float(r["edited"]["correct"]) - float(r["original"]["correct"]) for r in ordered], np.float32
        )
        keys = np.array([chat_margin[r["id"]] for r in ordered], np.float32)
        order = (f"the {size[key]:,} {short.lower()} questions, sorted by the unedited model's margin under the chat prompt: "
                 "most wrong first (top left) to most right (bottom right)")  # fmt: skip
        groups.append({"name": f"{short}: bare outcome by chat result", "kind": "attention", "panels": [
            panel(f"cross_{key}_chat_margin", "Unedited model's margin, chat prompt", keys, shape, signed=True,
                  color_max=float(np.percentile(np.abs(keys), 98)), cell=14, unit="margin", note=order),
            panel(f"cross_{key}_bare_outcome", "Candidate's outcome, bare prompt", outcome, shape, signed=True,
                  color_max=2.0, cell=14, unit="outcome", note=order),
        ]})  # fmt: skip

    def by_chat(split: dict, what: str) -> str:
        return (f"on the {split['True']['n']} {what} questions the unedited model gets right under the chat prompt, the "
                f"candidate's bare-prompt accuracy changes by {split['True']['difference_points']:+.2f} points "
                f"({split['True']['fixed']} fixed, {split['True']['broken']} broken); on the other {split['False']['n']} by "
                f"{split['False']['difference_points']:+.2f} points ({split['False']['fixed']} fixed, {split['False']['broken']} broken)")  # fmt: skip

    reference = records["donor_on_confirmation"]["bare"]
    scale = (f"For scale, {name} itself under the bare prompt: {100 * reference['correct'] / reference['n']:.2f} % of the "
             f"confirmation questions (unedited Phi-3.5: {100 * stats['confirmation', 'bare']['before']:.2f} %)")  # fmt: skip
    where = "Where the gain sits: " + by_chat(records["bare_outcome_by_unedited_chat_result"], "confirmation")
    rule = ("Astra's rule, written before the confirmation passes: keep the edit only if the bare-prompt gain exceeds 2 "
            "standard errors and chat-prompt accuracy does not fall. Neither held, so it removed the edit.")  # fmt: skip
    if test:
        scale += (f", and {100 * test['bare']['donor']['correct'] / test['bare']['donor']['n']:.2f} % of the test questions "
                  f"({100 * stats['test', 'bare']['before']:.2f} %)")  # fmt: skip
        where += ". Test split: " + by_chat(test["bare_outcome_by_unedited_chat_result"], "test")
        rule += (f" On the test split the bare-prompt difference is {test['rule']['bare_gain_in_standard_errors']:+.1f} standard "
                 f"errors and the chat-prompt difference is {test['chat']['difference_points']:+.2f} points.")  # fmt: skip
    manifest = {
        "id": f"exp05c_transfer_{run}",
        "title": f"Moving {name}'s MedQA advantage into Phi-3.5: the weight transfer Astra built and withdrew",
        "subtitle": (f"EXP-05c {run}: {name}'s attention weights and 256 MLP units per layer in layers 0 to 7. Astra withdrew "
                     "it after its confirmation passes and submitted no edit"),
        "family": "Phi-3.5 vs MediPhi",
        "maxRowWidth": 2400,
        "facts": [
            (f"Astra submitted no edit, so MedQA test accuracy is {100 * final['bare']['original_accuracy']:.2f} % (bare prompt) "
             f"and {100 * final['chat']['original_accuracy']:.2f} % (chat prompt) before and after ({final['bare']['n']:,} questions)."),
            (f"The candidate takes {name}'s weights in layers {min(units)} to {max(units)}: all attention weights and norms, and "
             f"256 of 8,192 MLP units per layer. That is {100 * weights['share_of_model_in_support']:.2f} % of the model's weights; "
             f"{100 * weights['share_of_model_changed']:.2f} % actually differ from Phi-3.5's."),
            *facts, scale + ".", where + ".", rule,
        ],
        "colorNote": ("Red = positive, blue = negative, hatched = not measured. Outcome panels: red = fixed by the edit, "
                      "blue = broken, gray = same result. Margins are in log-probability units and saturate at the 98th "
                      "percentile. Weight panels: brighter = larger share or larger change."),
        "layers": [
            {"name": "Every transfer Astra scored (rows, in the order it tried them)",
             "groups": [{"name": "Transfers", "kind": "attention", "panels": board}]},
            {"name": f"The candidate's weights: layers {min(units)} to {max(units)} of {n_layers}",
             "stats": {"of the model in the transfer": weights["share_of_model_in_support"],
                       "actually different": weights["share_of_model_changed"]},
             "groups": [{"name": "All 32 layers (columns)", "kind": "attention", "panels": maps},
                        {"name": "MLP units taken", "kind": "attention", "panels": masks}]},
            {"name": "What the candidate did to each question", "groups": groups},
        ],
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest))
    print("wrote", folder, f"{sum(f.stat().st_size for f in folder.iterdir()) / 1e6:.2f} MB")
    for fact in manifest["facts"]:
        print("  -", fact)


def write_index() -> None:
    entries = []
    for manifest_path in sorted(DATA.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        entries.append({k: manifest[k] for k in ("id", "title", "subtitle", "family")})
    (DATA / "index.json").write_text(json.dumps(entries, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    exporters = {
        "mit3": export_mit3,
        "mit8": export_mit8,
        "weightdiff": export_weightdiff,
        "interventions": export_interventions,
        "looping": export_looping,
        "exp06": export_exp06,
        "exp05c": export_exp05c,
    }
    parser.add_argument("what", nargs="+", choices=list(exporters))
    args = parser.parse_args()
    for what in args.what:
        exporters[what]()
    write_index()


if __name__ == "__main__":
    main()
