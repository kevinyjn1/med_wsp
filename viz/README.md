# Weight viewer

An interactive 3D view of network weights and of our analysis results, in the spirit of
[bbycroft.net/llm](https://bbycroft.net/llm): every matrix is a slab drawn cell by cell from the real
numbers, layers stack into a tower, and attention and MLP groups sit at different depths.

## Run

```bash
python -m wsp.viz_export mit3 mit8 weightdiff exp06   # from the repo root: writes viz/public/data/
npm install --prefix viz
npm run dev --prefix viz                   # http://localhost:5173
```

`viz/public/data/` is not in git; the exporter rebuilds it from `external/lm_understand-logs` (MIT's released
networks) and from `experiments/*/processed/`.

## Datasets

| Id | What it shows |
|---|---|
| `mit3_python`, `mit3_romaji`, `mit3_english` | The four transformer blocks Astra was given in MIT experiment 3 (7,084,032 weights each), with the copying heads marked: named by Astra and confirmed by measurement, named but not confirmed, or measured but missed. |
| `mit8_qwen_p50c` | MIT experiment 8: the two edited MLP output matrices of Qwen3.5-0.8B (before, change, the 64 edited columns, the loop direction). Needs `python experiments/exp_03_mit8_math/apply_edit.py` first. |
| `exp06_edit_run1` | EXP-06: the one MLP unit of Phi-3.5 that Astra edited and then rejected (location, each unit's push on answer A, the unit's output direction before and after), and one cell per practice question: margin before, fixed / broken, change of the margin. Needs the run's `work` folder (local only). |
| `weightdiff_mediphi_phi` | Relative size of MediPhi minus Phi-3.5, per attention head (Q, K, V, O) and per MLP neuron (gate, up, down), for all 32 layers. |

## Controls

Drag to pan, right-drag to rotate, scroll to zoom at the cursor, click a matrix to frame it, `R` to reset.
Hovering a cell shows its row, column, head and value. The contrast slider changes the exponent applied to
the color scale.

## Data format

`data/<id>/manifest.json` lists layers -> groups -> panels. A panel is one matrix stored row-major in
`<panel>.bin`, either `float32` or one byte per cell holding `sign(x) * sqrt(|x| / vmax)` (fine steps near
zero, no clipping). `colorMax` is where the color scale saturates. Signed values use a blue / gray / red
diverging scale; magnitudes use a one-hue blue scale. Marks carry a status (`good`, `warning`, `serious`)
that is always shown with an icon and a text label, never by color alone.
