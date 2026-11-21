# ZLabPlot Style Guide

Summary of the visual style enforced by `matplotlibrc` and the
`ZLabPlot` class overrides in `ZLabPlot.py`, with enough detail
to reproduce the look without the package (e.g. via a standalone
`matplotlib.rcParams` block).

Two layers apply, in order:
1. `matplotlibrc` — baseline rc file (loaded if `MATPLOTLIBRC` env var points
   to it, or copied into matplotlib's user config dir).
2. `ZLabPlot.__init__()` — overrides a handful of `matplotlibrc` values at
   runtime (ticks are noticeably bolder than the raw rc file). These
   overrides are the ones that actually apply when using `ZLabPlot`.

## Typography

- LaTeX rendering: `text.usetex: True` (requires a LaTeX install on `PATH`).
  Disable with `rcmap={'text.usetex': False}` if LaTeX is unavailable.
- Font family: `font.family: serif`.
- Base font size: `font.size: 22` (rc default) — large, intended for
  print/poster figures that get scaled down, not screen-native sizing.
- Math fallback font (used when not rendering via LaTeX):
  `mathtext.fontset: dejavusans`.
- `ZLabPlot` sets a `default_fontsize = 'large'` used for tick labels
  (`xtick.labelsize` / `ytick.labelsize`), and passes it through as the
  default for legend/label/tick fontsize kwargs in `add_data()`
  unless overridden per-call.
- Axes title size: `'large'` in `ZLabPlot` (rc default is `'medium'`).

## Axes and ticks

- Ticks point **inward**: `xtick.direction: in`, `ytick.direction: in`.
- Ticks drawn on **all four sides** (top, bottom, left, right), both major
  and minor (`xtick.top/bottom`, `ytick.left/right`, `*.minor.visible: True`).
- `ZLabPlot` overrides tick geometry to be bolder than the raw rc defaults:

  | param                          | rc default | ZLabPlot override |
  |---------------------------------|-----------|--------------------|
  | `x/ytick.major.size`            | 3.5       | 9                  |
  | `x/ytick.minor.size`            | 2         | 6                  |
  | `x/ytick.major.width`           | 0.8       | 2                  |
  | `x/ytick.minor.width`           | 0.6       | 2                  |
  | `x/ytick.major.pad`             | 3.5       | 10                 |
  | `x/ytick.minor.pad`             | 3.4       | 10                 |
  | `x/ytick.labelsize`             | `medium`  | `large`            |

- Axes spines shown on all four sides (`axes.spines.left/right/top/bottom: True`),
  black edge color, `axes.linewidth: 1.0`.
- No gridlines by default (`axes.grid: False`).
- Scientific notation only outside a 1e-5–1e6 data range
  (`axes.formatter.limits: -5, 6`), with offset notation enabled.
- Axis labels centered (`xaxis.labellocation`/`yaxis.labellocation: center`).

## Colors

- Default color cycle is matplotlib's classic tab10:
  `1f77b4, ff7f0e, 2ca02c, d62728, 9467bd, 8c564b, e377c2, 7f7f7f, bcbd22, 17becf`
  (`axes.prop_cycle`).
- For sequential/continuous coloring across many curves, use
  `ZLabPlot.get_color_series(color_index, cmap)`, which normalizes a list of
  integer indices against a named colormap (e.g. `'OrRd'`, `'plasma'`) via
  `matplotlib.cm.ScalarMappable`.
- `ZLabPlot.add_data(..., cmap=..., color=[...])` accepts a list of integer
  indices to look up colors from a colormap directly, instead of pre-computing
  RGBA tuples.
- `get_Color_from_RGB((r, g, b))` converts an 8-bit RGB tuple to a hex string.

## Legend

- No legend frame: `legend.frameon: False`.
- Small legend font: `legend.fontsize: small`.
- Tight spacing: `legend.borderpad: 0.1`, `legend.labelspacing: 0.2`,
  `legend.handlelength: 1.0`, `legend.handleheight: 0.7`,
  `legend.handletextpad: 0.4`, `legend.borderaxespad: 0.5`,
  `legend.columnspacing: 1.0`.
- `legend.loc: best` by default.
- `ZLabPlot.add_data()` calls `ax.legend(..., labelspacing=0.5, frameon=False)`
  when a `legend` list is supplied, and accepts `ncol`, `legendtitle`,
  `legend_location` kwargs.

## Figure size and aspect ratio

- Default figure is **square**: `figure.figsize: 6.94, 6.94` (inches),
  i.e. a 1:1 aspect ratio at the figure level. There is no built-in
  widescreen/landscape default — override via `rcmap={'figure.figsize': (w, h)}`
  if a different ratio is needed (the gallery example uses `(12, 12)`, still
  square).
- White background, transparent on save by default
  (`figure.facecolor: white`, `savefig.transparent: True` — `ZLabPlot.save()`
  also exposes `transparent` as an explicit kwarg, default `True`).
- Axes box placement is **not** matplotlib's normal subplot-fraction layout;
  `ZLabPlot.add_subplot()` builds it with
  `mpl_toolkits.axes_grid1.Divider`/`Size.Fixed` for 2D plots:

  ```python
  h = [Size.Fixed(1.0), Size.Fixed(figsize[0] * margin_ratio)]
  v = [Size.Fixed(0.7), Size.Fixed(figsize[1] * margin_ratio)]
  divider = Divider(fig, (0.1, 0.1, 0.9, 0.9), h, v, aspect=False)
  ax = fig.add_axes(divider.get_position(), axes_locator=divider.new_locator(nx=1, ny=1))
  ```

  This reserves a fixed margin (1.0 in horizontal, 0.7 in vertical) for
  labels/ticks, and sizes the axes box to `margin_ratio` (default `0.7`) of
  the figure's width/height in absolute inches. `aspect=False` means the data
  aspect ratio is not locked to the box shape. With a square figure and a
  uniform `margin_ratio`, the resulting axes box is also approximately square.
  3D subplots (`projection='3d'`) instead use a normal `plt.subplot(...,
  projection='3d')` and are not size-fixed this way.
- `image.aspect: equal` in the rc file affects `imshow`-style pixel aspect
  only, not general line/scatter plots.

## Lines, markers, patches

- `lines.linewidth: 1.0`, `lines.markersize: 6`, `lines.markeredgewidth: 1.0`.
- Solid line style by default (`lines.linestyle: -`), round joins, projecting
  solid cap style.
- `patch.linewidth: 1`, patches default to color-cycle facecolor with black
  edge only if forced.
- Scatter default marker is a circle (`scatter.marker: o`) with
  face-colored edges (`scatter.edgecolors: face`).
- `ZLabPlot.add_data()` per-curve style kwargs (`lw`, `ls`, `ms`, `msize`,
  `mf`, `color`, `zorder`) are "padded" across a list of curves via
  `feature_padding()` — pass a scalar to apply it to every curve, or a list
  shorter than the curve count to have the remainder fall back to
  matplotlib defaults.

## Axis-label conventions (`plottype`)

`ZLabPlot` ships built-in LaTeX axis labels for common MD/liquid-structure
quantities, keyed by a `plottype` string (e.g. returned by
`LiquidLibIO.read()`), so labels don't need to be typed out manually:

| `plottype` key(s)        | x-label                          | y-label |
|---------------------------|-----------------------------------|---------|
| `PDF`, `RDF`, `gr`         | `r (Å)`                          | `g(r)` |
| `gtheta`, `ADF`            | `θ (rad)`                        | `g(θ)` |
| `sk`, `sq`                 | `Q (Å⁻¹)`                        | `S(Q)` |
| `msd`, `r2t`                | `t (ps)`                        | `⟨r²⟩(t) (Å²)` / `(Å²/s)` for y-label variants |
| `vacf`/`eacf`/`hacf`/`sacf` | `t (ps)`                        | normalized autocorrelation, e.g. `⟨v(t)v(0)⟩/⟨v(0)²⟩` |
| `fskt`, `fkt`, `fthetakt`, `ipdkt` | `t (ps)`                 | scattering function, e.g. `F(Q,t)` |
| `gsrt`, `grt`               | `r (Å)`                          | van Hove function, e.g. `G(r,t)` |
| `alpha2t`, `chi4t`          | `t (ps)`                        | `α₂(t)`, `χ₄(t)` |

This mapping is domain-specific to Z Lab's liquid/MD simulation plots; for a
different domain (e.g. this repo's tendon-mechanics figures), supply
`xlabel`/`ylabel` explicitly instead of relying on `plottype`.

## Minimal usage example (from `gallery/gallery.py`)

```python
from ZLabPlot import ZLabPlot

zp = ZLabPlot(rcmap={'axes.titlesize': 'small', 'legend.title_fontsize': 'small'})
zp.add_subplot(plottitle="Collective Intermediate Scattering Function")
zp.add_data(
    t_list, fkt_list, plottype="fkt",
    lw=2, ylim=(0.0, 1.1), xlog=True,
    legendtitle=r'$Q=\ \sim %.1f\ \mathrm{Å^{-1}}$' % k,
    cmap='OrRd', legend=legend_labels, color=color_indices, ncol=2,
)
zp.save(filename='fkt', transparent=False)
zp.clear()
```

## Reproducing without the `ZLabPlot` class

To replicate the look in a plain `matplotlib` script (e.g. for this
repository's paper figures) without importing `ZLabPlot.py`, set:

```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    'text.usetex': True,              # or False if no LaTeX available
    'font.family': 'serif',
    'font.size': 22,
    'axes.titlesize': 'large',
    'axes.grid': False,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True, 'xtick.bottom': True,
    'ytick.left': True, 'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size': 9, 'ytick.major.size': 9,
    'xtick.minor.size': 6, 'ytick.minor.size': 6,
    'xtick.major.width': 2, 'ytick.major.width': 2,
    'xtick.minor.width': 2, 'ytick.minor.width': 2,
    'xtick.major.pad': 10, 'ytick.major.pad': 10,
    'xtick.minor.pad': 10, 'ytick.minor.pad': 10,
    'xtick.labelsize': 'large', 'ytick.labelsize': 'large',
    'legend.frameon': False,
    'legend.fontsize': 'small',
    'figure.figsize': (6.94, 6.94),
    'figure.facecolor': 'white',
    'savefig.transparent': True,
})
```

This reproduces the tick/legend/font style; the fixed-margin `Divider`-based
axes box sizing is `ZLabPlot`-specific and only matters if an exact
reproduction of axes-box proportions (not just rc styling) is required.

## Source files

Paths below are relative to `.claude/skills/figure-production/`.

- `matplotlibrc` — baseline rc file.
- `ZLabPlot.py` — `ZLabPlot` class, runtime rc overrides,
  `plottype` label maps, per-curve styling helpers.
- `gallery/gallery.py` — usage examples.
- `README` — dependencies and setup notes; contact: Shao-Chun
  Lee (scl6@illinois.edu), Y Zhang (zhyang@illinois.edu), Z Lab, UIUC.
