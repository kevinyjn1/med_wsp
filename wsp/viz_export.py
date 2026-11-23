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
    python -m wsp.viz_export interventions looping   # EXP-05b patching and swaps, EXP-07 looping scan
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


# ------------------------------------------------------------------------------- interventions and looping
def export_interventions(donor: str = "mediphi_instruct", style: str = "bare") -> None:
    """Activation patching and weight swapping (EXP-05b): mean change of the gold-answer margin, per layer."""
    suffix = "_bare" if style == "bare" else ""
    src = REPO / "experiments" / "exp_05_medical_circuit" / "processed" / f"patch_{donor}{suffix}"
    sets, components = ("gain", "loss", "both"), ("attn", "mlp", "resid")
    folder = DATA / f"interventions_{donor}{suffix}_phi"
    folder.mkdir(parents=True, exist_ok=True)
    panels, facts = [], []
    rows = (
        [json.loads(line) for line in (src / "activation.jsonl").read_text().splitlines()]
        if (src / "activation.jsonl").exists()
        else []
    )
    n_layers = 32
    if rows:
        grid = np.zeros((len(sets) * len(components), n_layers), np.float32)
        labels = []
        for si, name in enumerate(sets):
            members = [r for r in rows if r["set"] == name]
            gap = np.mean([r["donor"] - r["base"] for r in members])
            for ci, component in enumerate(components):
                labels.append(f"{name}: {component}")
                for layer in range(n_layers):
                    grid[si * len(components) + ci, layer] = np.mean(
                        [r["patched"][f"L{layer}.{component}"] - r["base"] for r in members]
                    )
            facts.append(
                f"Activation patching, {name} set ({len(members)} questions): donor margin minus base margin is {gap:+.2f} on average."
            )
        spec = write_panel(folder, "activation", grid, signed=True, quantize=False)
        spec.update({"id": "activation", "label": "Activation patching: donor activation placed into the base", "unit": "margin change",
                     "cell": 44, "rowLabels": labels, "colLabels": [str(i) for i in range(n_layers)], "ownRow": True,
                     "note": "mean change of the gold-answer margin (log-prob units) when one layer's attention output, MLP output or residual stream comes from the donor; columns are layers"})  # fmt: skip
        panels.append(spec)
    swaps = (
        [json.loads(line) for line in (src / "weights.jsonl").read_text().splitlines()]
        if (src / "weights.jsonl").exists()
        else []
    )
    if swaps:
        question_sets = json.loads((src / "question_sets.json").read_text())
        by = {row["config"]: row["margins"] for row in swaps}
        grid = np.zeros((len(sets) * 2, n_layers), np.float32)
        labels = []
        for si, name in enumerate(sets):
            ids = question_sets[name]
            for pi, part in enumerate(("attn", "mlp")):
                labels.append(f"{name}: {part} weights")
                for layer in range(n_layers):
                    if f"L{layer}.{part}" in by:
                        grid[si * 2 + pi, layer] = np.mean(
                            [by[f"L{layer}.{part}"][i] - by["base"][i] for i in ids]
                        )
            if "all" in by:
                facts.append(f"Weight swap, {name} set: copying every donor weight changes the margin by "
                             f"{np.mean([by['all'][i] - by['base'][i] for i in ids]):+.2f} on average.")  # fmt: skip
        spec = write_panel(folder, "weights", grid, signed=True, quantize=False)
        spec.update({"id": "weights", "label": "Weight swap: one layer's weights copied from the donor", "unit": "margin change",
                     "cell": 44, "rowLabels": labels, "colLabels": [str(i) for i in range(n_layers)], "ownRow": True,
                     "note": "mean change of the gold-answer margin when one layer's attention weights or MLP weights are the donor's; columns are layers"})  # fmt: skip
        panels.append(spec)
    if not panels:
        print("no intervention results yet")
        return
    manifest = {
        "id": f"interventions_{donor}{suffix}_phi", "title": f"Which layers carry the difference between Phi-3.5 and {donor}?",
        "subtitle": f"MedQA, {style} prompt: change of the gold-answer margin under activation patching and weight swapping, by layer",
        "family": "Phi-3.5 vs MediPhi", "maxRowWidth": 2600,
        "facts": [*facts, "gain = donor right, base wrong; loss = base right, donor wrong; both = both right (60 questions each from the discovery pool)."],
        "colorNote": "Signed change of the margin in log-probability units: red raises the gold answer, blue lowers it. Each panel has its own scale.",
        "layers": [{"name": "All 32 layers (columns)", "groups": [{"name": "Interventions", "kind": "attention", "panels": panels}]}],
    }  # fmt: skip
    (folder / "manifest.json").write_text(json.dumps(manifest))
    print("wrote", folder)


def export_looping(tag: str = "phi_dev_600") -> None:
    """Layer-looping scan (EXP-07): change of accuracy and margin against the unmodified model, by window start."""
    src = REPO / "experiments" / "exp_07_layer_looping" / "processed" / f"scan_{tag}.jsonl"
    rows = [json.loads(line) for line in src.read_text().splitlines()]
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
    for r in scan:
        margin[rules.index(r["rule"]), starts.index(r["start"])] = r["margin_delta"]
        accuracy[rules.index(r["rule"]), starts.index(r["start"])] = 100 * r["acc_delta"]
    common = {"cell": 70, "rowLabels": ["damped (Chen), 2 sub-steps", "naive, applied twice"], "ownRow": True,
              "colLabels": [f"{s}-{s + 3}" for s in starts]}  # fmt: skip
    panels = []
    for panel_id, label, grid, unit, note in (
        ("margin", "Change of the gold-answer margin", margin, "margin change", "mean over questions, log-prob units; columns are the looped 4-layer window"),
        ("accuracy", "Change of accuracy (percentage points)", accuracy, "points", "accuracy of the looped model minus the unmodified model on the same questions"),
    ):  # fmt: skip
        spec = write_panel(folder, panel_id, grid, signed=True, quantize=False)
        spec.update({"id": panel_id, "label": label, "unit": unit, "note": note, **common})
        panels.append(spec)
    best = max((r for r in rows if r["config"] != "baseline"), key=lambda r: r["margin_delta"])
    manifest = {
        "id": f"looping_{tag}", "title": "Layer looping on Phi-3.5: which 4-layer window to repeat?",
        "subtitle": f"MedQA {tag.split('_')[1]} split, {baseline['n']} questions, single-pass letter scoring",
        "family": "Training-free layer looping", "maxRowWidth": 2600,
        "facts": [
            f"Unmodified model: accuracy {100 * baseline['acc']:.1f} %, mean margin {baseline['mean_margin']:+.2f}.",
            (f"Best configuration by margin: {best['config']} ({best['margin_delta']:+.3f} +/- {best['margin_delta_se']:.3f}; accuracy {100 * best['acc_delta']:+.1f} points, "
            f"{best['fixed']} questions fixed, {best['broken']} broken, McNemar p = {best['mcnemar_p']:.2f})."),
            (f"{sum(r.get('acc_delta', 0) > 0 for r in rows)} of {len(rows) - 1} configurations have higher accuracy than the unmodified model; "
            f"{sum(r.get('mcnemar_p', 1) < 0.05 and r.get('acc_delta', 0) > 0 for r in rows)} of those are significant at 0.05 (uncorrected)."),
        ],
        "colorNote": "Red = better than the unmodified model, blue = worse. Each panel has its own scale.",
        "layers": [{"name": "Window start (columns)", "groups": [{"name": "4-layer window, 2 steps", "kind": "attention", "panels": panels}]}],
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

    def tile(panel_id, label, vector, shape, *, signed, color_max=None, cell=6, **extra):
        spec = write_panel(folder, panel_id, np.asarray(vector, np.float32).reshape(shape), signed=signed,
                           quantize=False, color_max=color_max)  # fmt: skip
        spec.update({"id": panel_id, "label": label, "cell": cell, **extra})
        return spec

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
                tile("mask", f"Edited unit: {unit}", mask, (64, 128), signed=False, unit="weight",
                     note=where + ". Bright = edited"),
                tile("push", "Each unit's direct push on answer A", push, (64, 128), signed=True, unit="push",
                     color_max=2 * float(np.abs(push).max()),  # the one strong unit lands on the saturated hue, not on near-white
                     note=where + f". Mean over {len(activation)} discovery questions of the unit's activation times its "
                     f"direct effect on the logit of A minus the mean logit of A to D. Unit {unit}: {push[unit]:+.2f}; "
                     f"the next largest unit: {runner_up:.2f} in size. Descriptive, not a causal effect"),
            ]},
            {"name": f"Unit {unit}", "kind": "attention", "panels": [
                tile("before", "Output direction before", base[:, unit], (48, 64), signed=True, color_max=saturate, cell=8,
                     unit="weight", note=direction + f". Norm {norms[unit]:.2f} (median over units {np.median(norms):.2f})"),
                tile("kept", "Letter-readout part (kept)", kept_part, (48, 64), signed=True,
                     color_max=saturate, cell=8, unit="weight",
                     note=direction + f". Norm {np.linalg.norm(kept_part):.2f}; lies in the three directions that separate A, B, C, D at the output"),
                tile("reversed", f"The rest (multiplied by {scale:g})", reversed_part, (48, 64), signed=True,
                     color_max=saturate, cell=8, unit="weight",
                     note=direction + f". Norm {np.linalg.norm(reversed_part):.2f} before the edit; shown before the multiplication"),
                tile("after", "Output direction after", edited[:, unit], (48, 64), signed=True, color_max=saturate, cell=8,
                     unit="weight", note=direction + f". Norm {np.linalg.norm(edited[:, unit]):.2f} (was {norms[unit]:.2f})"),
            ]},
        ]},
    ]  # fmt: skip

    def scored(passes: list[int]) -> list[dict]:
        rows = []
        for k in passes:
            rows += [
                json.loads(line)
                for line in (work / f"score_{k}.jsonl").read_text().split("\n")
                if line.strip()
            ]
        return sorted(rows, key=lambda r: r["original"]["margin"])

    facts, groups = [], []
    for key, short, title, passes, shape in (
        ("selection", "Selection", "512 questions used to choose the edit", [21, 22], (16, 32)),
        ("reserve", "Reserve", "256 untouched questions (reserve)", [26], (8, 32)),
        (
            "confirmation",
            "Confirmation",
            "1,024 untouched questions (final confirmation)",
            [27, 28, 29, 30],
            (32, 32),
        ),
    ):
        rows = scored(passes)
        before = np.array([r["original"]["correct"] for r in rows], bool)
        after = np.array([r["edited"]["correct"] for r in rows], bool)
        outcome = (after & ~before).astype(np.float32) - (before & ~after).astype(np.float32)
        margin_before = np.array([r["original"]["margin"] for r in rows], np.float32)
        change = np.array([r["edited"]["margin"] for r in rows], np.float32) - margin_before
        fixed, broken, n = int((outcome > 0).sum()), int((outcome < 0).sum()), len(rows)
        se = 100 * float(np.std(after.astype(float) - before.astype(float), ddof=1) / np.sqrt(n))
        facts.append(f"{title}: {100 * before.mean():.2f} % to {100 * after.mean():.2f} % "
                     f"({100 * (after.mean() - before.mean()):+.2f} points, standard error {se:.2f}; {fixed} fixed, {broken} broken).")  # fmt: skip
        if key == "confirmation":
            right = margin_before > 0
            facts.append(
                f"On those {n:,} questions the edit mostly sharpened the answer the model already gave: the margin of the "
                f"correct answer moved by {change[right].mean():+.2f} on average where the model was right and by "
                f"{change[~right].mean():+.2f} where it was wrong (correlation with the margin before: "
                f"{np.corrcoef(margin_before, change)[0, 1]:.2f})."
            )
        order = f"{title}, sorted by the unedited model's margin: most wrong first (top left) to most right (bottom right)"
        margin_max = float(np.percentile(np.abs(margin_before), 98))
        groups.append({"name": short, "kind": "attention", "panels": [
            tile(f"{key}_before", "Margin before the edit", margin_before, shape, signed=True,
                 color_max=margin_max, cell=14, unit="margin",
                 note=order + ". Margin = log-probability of the correct letter minus the best wrong letter; positive = answered correctly"),
            tile(f"{key}_outcome", f"Outcome: {fixed} fixed, {broken} broken", outcome, shape, signed=True,
                 color_max=2.0, cell=14, unit="outcome", note=order),
            tile(f"{key}_change", "Change of the margin", change, shape, signed=True,
                 color_max=float(np.percentile(np.abs(change), 98)), cell=14, unit="margin change", note=order),
        ]})  # fmt: skip
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


def write_index() -> None:
    entries = []
    for manifest_path in sorted(DATA.glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        entries.append({k: manifest[k] for k in ("id", "title", "subtitle", "family")})
    (DATA / "index.json").write_text(json.dumps(entries, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "what", nargs="+", choices=["mit3", "mit8", "weightdiff", "interventions", "looping", "exp06"]
    )
    args = parser.parse_args()
    for what in args.what:
        {
            "mit3": export_mit3,
            "mit8": export_mit8,
            "weightdiff": export_weightdiff,
            "interventions": export_interventions,
            "looping": export_looping,
            "exp06": export_exp06,
        }[what]()
    write_index()


if __name__ == "__main__":
    main()
