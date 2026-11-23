// 3D weight viewer: every matrix is a slab whose face is drawn cell by cell from the raw numbers.
// Layers stack downward into a tower; attention and MLP groups sit at different depths.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

THREE.ColorManagement.enabled = false; // colors below are sRGB values used as-is, as in the 2D legend

// ---------------------------------------------------------------------------------------- palette
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);
const PALETTE = {
  plane: "#0d0d0d",
  slabSide: "#232322",
  neutral: "#383835", // diverging midpoint (gray, never a hue)
  negative: ["#3987e5", "#cde2fb"], // blue arm: mid, strongest
  positive: ["#e66767", "#fbd5d5"], // red arm: mid, strongest
  sequential: ["#1f2226", "#184f95", "#3987e5", "#cde2fb"], // one hue, recedes toward the surface at zero
  status: { good: "#0ca30c", warning: "#fab219", serious: "#ec835a" },
};
const STATUS_ORDER = ["good", "warning", "serious"];
const STATUS_ICON = { good: "✓", warning: "!", serious: "×" };
const STATUS_TEXT = {
  good: "Copying head: named by Astra and confirmed by measurement",
  warning: "Named a copying head by Astra, not confirmed by measurement",
  serious: "Measured copying head that Astra did not name",
};

// ----------------------------------------------------------------------------------------- layout
const GAP = { panel: 44, row: 64, group: 110, layer: 360 };
const MIN_STRIP = 16; // height given to 1-row matrices so they stay visible
const SLAB_DEPTH = 10;
const GROUP_Z = { attention: 0, mlp: -170 };
const TRUNK_X = -130;

const dom = {
  canvas: document.getElementById("canvas"),
  stage: document.getElementById("stage"),
  labels: document.getElementById("labels"),
  tooltip: document.getElementById("tooltip"),
  loading: document.getElementById("loading"),
  dataset: document.getElementById("dataset"),
  title: document.getElementById("title"),
  subtitle: document.getElementById("subtitle"),
  facts: document.getElementById("facts"),
  legendBars: document.getElementById("legend-bars"),
  colorNote: document.getElementById("color-note"),
  gamma: document.getElementById("gamma"),
  gammaOut: document.getElementById("gamma-out"),
  focus: document.getElementById("focus"),
  focusWrap: document.getElementById("focus-wrap"),
  statusLegend: document.getElementById("status-legend"),
  statusList: document.getElementById("status-list"),
  layers: document.getElementById("layers"),
};

const renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, antialias: true });
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setClearColor(PALETTE.plane);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(28, 1, 5, 400000);
const controls = new OrbitControls(camera, dom.canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.screenSpacePanning = true;
controls.zoomToCursor = true;
controls.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };
controls.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_ROTATE };
controls.minPolarAngle = Math.PI * 0.15;
controls.maxPolarAngle = Math.PI * 0.85;
controls.minAzimuthAngle = -Math.PI * 0.45;
controls.maxAzimuthAngle = Math.PI * 0.45;

// ---------------------------------------------------------------------------------------- shaders
const VERTEX = /* glsl */ `
out vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const FRAGMENT = /* glsl */ `
precision highp float;
precision highp sampler2D;
uniform sampler2D tData;
uniform vec2 uSize;          // cols, rows
uniform float uSigned, uQuantized, uSqrt, uVmax, uColorMax, uGamma, uDim;
uniform float uHasMissing, uMissing;   // cells holding uMissing were not measured
uniform vec2 uHover;         // col, row of the hovered cell, or -1
uniform vec2 uBand;          // band size in cols, rows (0 = none)
uniform vec3 uNeutral, uNegMid, uNegMax, uPosMid, uPosMax, uSeq0, uSeq1, uSeq2, uSeq3, uGap;
uniform vec3 uStatus[3];
uniform vec4 uMarks[8];      // axis (0 row, 1 col), start, size, status index
uniform int uMarkCount;
in vec2 vUv;
out vec4 fragColor;

vec3 ramp3(vec3 a, vec3 b, vec3 c, float t) {
  return t < 0.5 ? mix(a, b, t * 2.0) : mix(b, c, t * 2.0 - 1.0);
}
vec3 sequential(float t) {
  if (t < 0.34) return mix(uSeq0, uSeq1, t / 0.34);
  if (t < 0.67) return mix(uSeq1, uSeq2, (t - 0.34) / 0.33);
  return mix(uSeq2, uSeq3, (t - 0.67) / 0.33);
}

void main() {
  vec2 cellf = vec2(vUv.x * uSize.x, (1.0 - vUv.y) * uSize.y);
  vec2 cell = clamp(floor(cellf), vec2(0.0), uSize - 1.0);
  float raw = texelFetch(tData, ivec2(cell), 0).r;
  float v = raw / uVmax;                               // value as a fraction of the matrix's largest magnitude
  if (uQuantized > 0.5) {
    float q = uSigned > 0.5 ? (raw * 255.0 - 128.0) / 127.0 : raw;
    v = uSqrt > 0.5 ? sign(q) * q * q : q;
  }
  float mag = pow(clamp(abs(v) * uVmax / uColorMax, 0.0, 1.0), uGamma);
  vec3 color = uSigned > 0.5
    ? (v < 0.0 ? ramp3(uNeutral, uNegMid, uNegMax, mag) : ramp3(uNeutral, uPosMid, uPosMax, mag))
    : sequential(mag);

  vec2 perPixel = max(fwidth(cellf), vec2(1e-6));   // cells per screen pixel
  vec2 pixels = 1.0 / perPixel;                      // screen pixels per cell
  vec2 f = fract(cellf);

  // Not measured: the background with a diagonal hatch, so it cannot be read as a value of zero.
  if (uHasMissing > 0.5 && raw == uMissing) {
    float stripes = max(2.0, floor(min(pixels.x, pixels.y) / 9.0));
    float hatch = smoothstep(0.42, 0.5, abs(fract((f.x + f.y) * 0.5 * stripes) - 0.5));
    color = mix(uGap, uNeutral, mix(0.3, hatch * 0.75, smoothstep(10.0, 22.0, min(pixels.x, pixels.y))));
  }
  vec2 edge = min(f, 1.0 - f) / perPixel;            // distance to the cell border, in pixels

  // Marks: dim what is outside, draw a 2 px frame around each marked band.
  float inside = 0.0;
  vec3 frameColor = vec3(0.0);
  float frame = 0.0;
  for (int i = 0; i < 8; i++) {
    if (i >= uMarkCount) break;
    vec4 m = uMarks[i];
    float along = m.x < 0.5 ? cellf.y : cellf.x;
    float across = m.x < 0.5 ? cellf.x : cellf.y;
    float pp = m.x < 0.5 ? perPixel.y : perPixel.x;
    float ppAcross = m.x < 0.5 ? perPixel.x : perPixel.y;
    float extent = m.x < 0.5 ? uSize.x : uSize.y;
    float d0 = (along - m.y) / pp;
    float d1 = (m.y + m.z - along) / pp;
    if (d0 >= 0.0 && d1 >= 0.0) {
      inside = 1.0;
      float border = min(min(d0, d1), min(across, extent - across) / ppAcross);
      if (border < 2.0) { frame = 1.0; frameColor = uStatus[int(m.w)]; }
    }
  }
  if (uMarkCount > 0) color = mix(color, uGap, uDim * (1.0 - inside) * 0.82);

  // Cell gaps appear once cells are large enough to read individually.
  float gapX = smoothstep(5.0, 11.0, pixels.x) * (1.0 - smoothstep(0.0, 0.9, edge.x));
  float gapY = smoothstep(5.0, 11.0, pixels.y) * (1.0 - smoothstep(0.0, 0.9, edge.y));
  color = mix(color, uGap, max(gapX, gapY) * 0.9);

  // Band separators (head boundaries): a 1 px line that stays visible when zoomed out.
  if (uBand.x > 0.0) {
    float d = abs(fract(cellf.x / uBand.x + 0.5) - 0.5) * uBand.x / perPixel.x;
    if (cellf.x > 0.5 * uBand.x && cellf.x < uSize.x - 0.5 * uBand.x) color = mix(color, uGap, (1.0 - smoothstep(0.4, 1.1, d)) * 0.9);
  }
  if (uBand.y > 0.0) {
    float d = abs(fract(cellf.y / uBand.y + 0.5) - 0.5) * uBand.y / perPixel.y;
    if (cellf.y > 0.5 * uBand.y && cellf.y < uSize.y - 0.5 * uBand.y) color = mix(color, uGap, (1.0 - smoothstep(0.4, 1.1, d)) * 0.9);
  }
  if (frame > 0.5) color = frameColor;

  // Hovered cell: a white outline at least 1.5 px wide.
  if (uHover.x >= 0.0 && cell == uHover) {
    float ring = min(edge.x, edge.y);
    if (ring < 1.5 || min(pixels.x, pixels.y) < 4.0) color = vec3(1.0);
  }
  fragColor = vec4(color, 1.0);
}`;

function panelMaterial(panel, texture, gamma) {
  const marks = (panel.highlights || []).slice(0, 8);
  const markVectors = Array.from({ length: 8 }, (_, i) => {
    const m = marks[i];
    return m
      ? new THREE.Vector4(m.axis === "row" ? 0 : 1, m.start, m.size, STATUS_ORDER.indexOf(m.status))
      : new THREE.Vector4(0, 0, 0, 0);
  });
  const v3 = (h) => new THREE.Vector3(...hex(h));
  return new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3,
    vertexShader: VERTEX,
    fragmentShader: FRAGMENT,
    uniforms: {
      tData: { value: texture },
      uSize: { value: new THREE.Vector2(panel.cols, panel.rows) },
      uSigned: { value: panel.signed ? 1 : 0 },
      uQuantized: { value: panel.dtype === "uint8" ? 1 : 0 },
      uSqrt: { value: panel.encoding === "sqrt" ? 1 : 0 },
      uVmax: { value: panel.vmax },
      uColorMax: { value: panel.colorMax ?? panel.vmax },
      uGamma: { value: gamma },
      uDim: { value: 0 },
      uHasMissing: { value: panel.missing === undefined ? 0 : 1 },
      uMissing: { value: panel.missing ?? 0 },
      uHover: { value: new THREE.Vector2(-1, -1) },
      uBand: { value: new THREE.Vector2(panel.colBands ? panel.colBands[0].size : 0, panel.rowBands ? panel.rowBands[0].size : 0) },
      uNeutral: { value: v3(PALETTE.neutral) },
      uNegMid: { value: v3(PALETTE.negative[0]) },
      uNegMax: { value: v3(PALETTE.negative[1]) },
      uPosMid: { value: v3(PALETTE.positive[0]) },
      uPosMax: { value: v3(PALETTE.positive[1]) },
      uSeq0: { value: v3(PALETTE.sequential[0]) },
      uSeq1: { value: v3(PALETTE.sequential[1]) },
      uSeq2: { value: v3(PALETTE.sequential[2]) },
      uSeq3: { value: v3(PALETTE.sequential[3]) },
      uGap: { value: v3(PALETTE.plane) },
      uStatus: { value: STATUS_ORDER.map((s) => v3(PALETTE.status[s])) },
      uMarks: { value: markVectors },
      uMarkCount: { value: marks.length },
    },
  });
}

// ------------------------------------------------------------------------------------------ state
let current = null; // { manifest, panels: [...], labels: [...], layerBoxes: [...], bounds }
let hovered = null;
let flight = null;
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const sideMaterial = new THREE.MeshBasicMaterial({ color: PALETTE.slabSide });

function disposeCurrent() {
  if (!current) return;
  for (const p of current.panels) {
    p.mesh.geometry.dispose();
    p.material.dispose();
    p.texture.dispose();
  }
  scene.clear();
  redraw = true;
  dom.labels.replaceChildren();
  current = null;
  hovered = null;
}

function addLabel(labels, className, html, position, opts = {}) {
  const el = document.createElement("div");
  el.className = `label ${className}`;
  el.innerHTML = html;
  dom.labels.appendChild(el);
  labels.push({
    el, position, anchor: opts.anchor || "left-bottom", minPixels: opts.minPixels || 0, span: opts.span || 0, clip: !!opts.clip,
    room: opts.room || 0, roomPixels: opts.roomPixels || 17,
    // `tick`: this label is one of an axis of column labels, { axis, index, units }; see tickStride
    tick: opts.tick || null,
  });
  return el;
}

// Column labels thin out like the ticks of a chart axis: with the camera far away only every 2nd, 4th, ... is
// shown, so that the widest label of the axis always fits into the columns it stands for.
const STRIDES = [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 50, 64, 100];
function tickStride(axis, units, pixelsPerUnit) {
  const need = (axis.maxWidth + 4) / (units * pixelsPerUnit);
  return STRIDES.find((stride) => stride >= need) ?? Infinity;
}

function layoutGroup(group, maxRowWidth) {
  // Flex-wrap: panels fill a row left to right and wrap when the row would exceed maxRowWidth.
  const rows = [[]];
  let x = 0;
  for (const panel of group.panels) {
    const w = Math.max(panel.cols * (panel.cellW || panel.cell || 1), panel.cols === 1 ? MIN_STRIP : 0);
    const h = Math.max(panel.rows * (panel.cellH || panel.cell || 1), MIN_STRIP);
    const previous = rows.at(-1).at(-1);
    if (previous && (panel.ownRow || previous.panel.ownRow || x + w > maxRowWidth)) {
      rows.push([]);
      x = 0;
    }
    rows.at(-1).push({ panel, x, w, h });
    x += w + GAP.panel;
  }
  return rows;
}

async function loadDataset(id) {
  dom.loading.hidden = false;
  disposeCurrent();
  const base = `data/${id}`;
  const manifest = await (await fetch(`${base}/manifest.json`)).json();
  const gamma = Number(dom.gamma.value);
  const panels = [];
  const labels = [];
  const layerBoxes = [];
  const maxRowWidth = manifest.maxRowWidth || 1800;
  let y = 0;
  let maxX = 0;

  const buffers = new Map();
  await Promise.all(
    manifest.layers.flatMap((layer) =>
      layer.groups.flatMap((group) =>
        group.panels.map(async (panel) => {
          const raw = await (await fetch(`${base}/${panel.file}`)).arrayBuffer();
          buffers.set(panel.id, panel.dtype === "uint8" ? new Uint8Array(raw) : new Float32Array(raw));
        }),
      ),
    ),
  );

  manifest.layers.forEach((layer, layerIndex) => {
    const top = y;
    let layerMaxX = 0;
    let labelPixels = 0; // widest row label left of the layer's first column of panels
    layer.groups.forEach((group, groupIndex) => {
      const z = GROUP_Z[group.kind] ?? 0;
      addLabel(labels, "group", group.name, new THREE.Vector3(-24, y + 8, z), { anchor: "right-bottom" });
      const groupTop = y;
      const groupRows = layoutGroup(group, maxRowWidth);
      for (const [rowIndex, row] of groupRows.entries()) {
        const rowHeight = Math.max(...row.map((r) => r.h));
        // Free space under this row. Column labels share it with the title of the panel below. Under the last
        // row of a layer it reaches down to the next layer's name.
        const lastRow = rowIndex === groupRows.length - 1;
        const lastGroup = groupIndex === layer.groups.length - 1;
        const below = { room: !lastRow ? GAP.row : lastGroup ? GAP.layer - 110 : GAP.group, roomPixels: 46 };
        for (const { panel, x, w, h } of row) {
          const data = buffers.get(panel.id);
          const texture = new THREE.DataTexture(
            data, panel.cols, panel.rows, THREE.RedFormat,
            panel.dtype === "uint8" ? THREE.UnsignedByteType : THREE.FloatType,
          );
          texture.internalFormat = panel.dtype === "uint8" ? "R8" : "R32F";
          texture.minFilter = texture.magFilter = THREE.NearestFilter;
          texture.unpackAlignment = 1;
          texture.needsUpdate = true;
          const material = panelMaterial(panel, texture, gamma);
          const geometry = new THREE.BoxGeometry(w, h, SLAB_DEPTH);
          const mesh = new THREE.Mesh(geometry, [sideMaterial, sideMaterial, sideMaterial, sideMaterial, material, sideMaterial]);
          mesh.position.set(x + w / 2, y - h / 2, z - SLAB_DEPTH / 2);
          scene.add(mesh);
          const record = { panel, data, texture, material, mesh, layerIndex, layerName: layer.name, x, y, w, h, z };
          mesh.userData.record = record;
          panels.push(record);

          addLabel(labels, "clip", `${panel.label} <span style="color:var(--muted)">${panel.rows} x ${panel.cols}</span>`,
            new THREE.Vector3(x, y + 8, z), { minPixels: 96, span: w, clip: true, room: GAP.row });
          const cell = panel.cell || 1;
          const cellW = panel.cellW || cell;
          if (panel.rowBands && panel.rowBands.length <= 8) {
            for (const band of panel.rowBands) {
              addLabel(labels, "band", band.label, new THREE.Vector3(x - 8, y - (band.start + band.size / 2) * cell, z),
                { anchor: "right-middle", minPixels: 260, span: w });
            }
          }
          if (panel.colBands && panel.colBands.length <= 8) {
            for (const band of panel.colBands) {
              addLabel(labels, "band", band.label, new THREE.Vector3(x + (band.start + band.size / 2) * cellW, y - h - 6, z),
                { anchor: "center-top", minPixels: 260, span: w, ...below });
            }
          }
          if (panel.colLabels) {
            const axis = { maxWidth: 0, frame: -1, stride: 1 };
            panel.colLabels.forEach((text, c) => {
              // one word per line, so a label of several words still fits under a narrow column
              const el = addLabel(labels, "band column", text.replaceAll(" ", "<br>"), new THREE.Vector3(x + (c + 0.5) * cellW, y - h - 6, z),
                { anchor: "center-top", minPixels: 200, span: w, tick: { axis, index: c, units: cellW }, ...below });
              axis.maxWidth = Math.max(axis.maxWidth, el.offsetWidth);
            });
          }
          if (panel.rowLabels) {
            panel.rowLabels.forEach((text, r) => {
              const el = addLabel(labels, "band", text, new THREE.Vector3(x - 8, y - (r + 0.5) * cell, z),
                { anchor: "right-middle", minPixels: 200, span: w });
              if (x === 0) labelPixels = Math.max(labelPixels, el.offsetWidth);
            });
          }
          for (const mark of panel.highlights || []) {
            const isRow = mark.axis === "row";
            const at = isRow
              ? new THREE.Vector3(x + w + 10, y - (mark.start + mark.size / 2) * cell, z)
              : new THREE.Vector3(x + (mark.start + mark.size / 2) * cellW, y + 26, z);
            addLabel(labels, `mark ${mark.status}`, `<i>${STATUS_ICON[mark.status]}</i>${mark.short}`, at,
              { anchor: isRow ? "left-middle" : "center-bottom", minPixels: 330, span: w });
          }
          layerMaxX = Math.max(layerMaxX, x + w);
        }
        y -= rowHeight + GAP.row;
      }
      // Connector from the residual stream to this group.
      const mid = (groupTop + y + GAP.row) / 2;
      const connector = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(TRUNK_X, mid, 0), new THREE.Vector3(-14, mid, z)]);
      scene.add(new THREE.Line(connector, new THREE.LineBasicMaterial({ color: PALETTE.neutral })));
      y -= GAP.group - GAP.row;
    });
    const bottom = y + GAP.group;
    layerBoxes.push({ top, bottom, width: layerMaxX, name: layer.name, labelPixels });
    addLabel(labels, "layer", layer.name, new THREE.Vector3(0, top + 96, 0), { anchor: "left-bottom" });
    maxX = Math.max(maxX, layerMaxX);
    y -= GAP.layer - GAP.group;
  });

  // Residual stream: the trunk every block reads from and writes back to.
  const trunkBottom = layerBoxes.at(-1).bottom;
  const trunk = new THREE.Mesh(new THREE.BoxGeometry(8, -trunkBottom + 180, 8), new THREE.MeshBasicMaterial({ color: PALETTE.neutral }));
  trunk.position.set(TRUNK_X, trunkBottom / 2 + 90, 0);
  scene.add(trunk);
  addLabel(labels, "group", "Residual stream", new THREE.Vector3(TRUNK_X, 190, 0), { anchor: "center-bottom" });

  current = { manifest, panels, labels, layerBoxes, width: maxX };
  renderSidebar(manifest);
  applyFocus();
  frameLayer(0, false);
  dom.loading.hidden = true;
}

// ---------------------------------------------------------------------------------------- sidebar
function gradient(stops) {
  return `linear-gradient(90deg, ${stops.join(", ")})`;
}

function renderSidebar(manifest) {
  dom.title.textContent = manifest.title;
  dom.subtitle.textContent = manifest.subtitle;
  dom.facts.replaceChildren(...manifest.facts.map((fact) => Object.assign(document.createElement("li"), { textContent: fact })));
  dom.colorNote.textContent = manifest.colorNote || "";

  const all = current.panels.map((p) => p.panel);
  const bars = [];
  if (all.some((p) => p.signed)) {
    bars.push({
      name: "Signed value",
      ends: ["most negative", "0", "most positive"],
      css: gradient([PALETTE.negative[1], PALETTE.negative[0], PALETTE.neutral, PALETTE.positive[0], PALETTE.positive[1]]),
    });
  }
  if (all.some((p) => !p.signed)) {
    bars.push({ name: "Magnitude", ends: ["0", "largest"], css: gradient(PALETTE.sequential) });
  }
  dom.legendBars.replaceChildren(
    ...bars.map((bar) => {
      const el = document.createElement("div");
      el.className = "bar";
      el.innerHTML = `<div class="name">${bar.name}</div><div class="ramp" style="background:${bar.css}"></div>
        <div class="ends">${bar.ends.map((e) => `<span>${e}</span>`).join("")}</div>`;
      return el;
    }),
  );

  const used = STATUS_ORDER.filter((s) => all.some((p) => (p.highlights || []).some((m) => m.status === s)));
  dom.statusLegend.hidden = dom.focusWrap.hidden = used.length === 0;
  dom.statusList.replaceChildren(
    ...used.map((s) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="swatch ${s}">${STATUS_ICON[s]}</span><span>${STATUS_TEXT[s]}</span>`;
      return li;
    }),
  );

  dom.layers.replaceChildren(
    ...manifest.layers.map((layer, index) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      let meta = "";
      if (layer.stats) {
        meta = Object.entries(layer.stats).map(([k, v]) => `<span>${k.replace("_proj", "")} ${(100 * v).toFixed(2)} %</span>`).join("");
      } else if (layer.heads) {
        const marked = layer.heads.filter((h) => h.status);
        meta = marked.length
          ? marked.map((h) => `<span class="chip"><span class="dot" style="background:${PALETTE.status[h.status]}"></span>H${h.head} ${h.statusShort}</span>`).join("")
          : "<span>no copying heads</span>";
      }
      button.innerHTML = `<span>${layer.name}</span><span class="meta">${meta}</span>`;
      button.addEventListener("click", () => frameLayer(index, true));
      li.appendChild(button);
      return li;
    }),
  );
}

function markCurrentLayer(index) {
  [...dom.layers.querySelectorAll("button")].forEach((b, i) => b.setAttribute("aria-current", String(i === index)));
}

// ----------------------------------------------------------------------------------------- camera
function flyTo(target, distance, animate, tilt = true) {
  const direction = tilt ? new THREE.Vector3(0.26, 0.14, 1).normalize() : new THREE.Vector3(0, 0, 1);
  const position = target.clone().addScaledVector(direction, distance);
  if (!animate) {
    camera.position.copy(position);
    controls.target.copy(target);
    controls.update();
    return;
  }
  flight = { fromP: camera.position.clone(), fromT: controls.target.clone(), toP: position, toT: target, start: performance.now(), ms: 650 };
}

function fitDistance(width, height) {
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const fitH = height / (2 * Math.tan(vFov / 2));
  const fitW = width / (2 * Math.tan(vFov / 2) * camera.aspect);
  return Math.max(fitH, fitW) * 1.22;
}

// Row labels keep their size in pixels, so the further away the camera is, the more of the scene they cover.
// Returns the distance at which a box of `width` units and its `labelPixels` wide labels fit side by side, and
// how many scene units the labels take at that distance.
function roomForLabels(width, labelPixels, minDistance, slack = 1) {
  const { clientWidth, clientHeight } = dom.stage;
  const unitsToPixels = clientHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  const forLabels = (slack * unitsToPixels * width) / Math.max(clientWidth - labelPixels - 80, 240);
  const distance = Math.max(minDistance, labelPixels ? forLabels : 0);
  return { distance, labelUnits: (labelPixels * distance) / unitsToPixels };
}

function frameLayer(index, animate) {
  const box = current.layerBoxes[index];
  const height = box.top - box.bottom;
  // (1.12: the view is tilted, so the right-hand side is nearer to the camera and drawn larger)
  const { distance, labelUnits } = roomForLabels(box.width + 120, box.labelPixels, fitDistance(box.width + 320, height + 160), 1.12);
  const target = new THREE.Vector3(box.width / 2 - Math.max(40, labelUnits / 2), (box.top + box.bottom) / 2, -60);
  flyTo(target, distance, animate);
  markCurrentLayer(index);
}

function framePanel(record) {
  markCurrentLayer(record.layerIndex);
  if (record.panel.rowNames) {
    // Its rows are labelled once, left of the first panel of the row: show the whole row with those labels.
    const row = current.panels.filter((p) => p.layerIndex === record.layerIndex && p.y === record.y && p.z === record.z);
    const left = Math.min(...row.map((p) => p.x));
    const width = Math.max(...row.map((p) => p.x + p.w)) - left;
    const height = Math.max(...row.map((p) => p.h));
    const labelPixels = current.layerBoxes[record.layerIndex].labelPixels;
    const { distance, labelUnits } = roomForLabels(width + 60, labelPixels, fitDistance(width * 1.08, height * 1.25));
    flyTo(new THREE.Vector3(left + (width - labelUnits) / 2, record.y - height / 2, record.z), distance, true, false);
    return;
  }
  const labelRoom = record.panel.rowLabels || record.panel.rowBands ? record.w * 0.3 : 0;
  const target = new THREE.Vector3(record.x + (record.w - labelRoom) / 2, record.y - record.h / 2, record.z);
  flyTo(target, fitDistance(record.w * 1.15 + labelRoom, Math.max(record.h, record.w * 0.2) * 1.25), true, false);
}

// ------------------------------------------------------------------------------------ interaction
function decode(record, row, col) {
  const { panel, data } = record;
  const raw = data[row * panel.cols + col];
  if (panel.missing !== undefined && raw === Math.fround(panel.missing)) return null;
  if (panel.dtype !== "uint8") return raw;
  const q = panel.signed ? (raw - 128) / 127 : raw / 255;
  return (panel.encoding === "sqrt" ? Math.sign(q) * q * q : q) * panel.vmax;
}

function bandLabel(bands, index) {
  const band = bands?.find((b) => index >= b.start && index < b.start + b.size);
  return band ? band.label : null;
}

const UNIT_LABEL = {
  "relative change": "Relative change",
  "margin change": "Margin change",
  margin: "Margin",
  push: "Push on A (logits)",
  points: "Accuracy change",
  outcome: "Outcome",
  share: "Answered correctly",
  count: "Changed",
  tensors: "Taken from the donor",
};

function formatValue(value, unit) {
  if (value === null) return "not measured";
  if (unit === "relative change") return `${(100 * value).toFixed(2)} %`;
  if (unit === "margin change" || unit === "margin" || unit === "push") return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
  if (unit === "outcome") return value > 0 ? "fixed (wrong to right)" : value < 0 ? "broken (right to wrong)" : "same result as before";
  if (unit === "share") return `${(100 * value).toFixed(1)} % of the set`;
  if (unit === "count") return `${value.toFixed(0)} questions`;
  if (unit === "tensors") return `${value.toFixed(0)} of the layer's 6 weight tensors`;
  if (unit === "points") return `${value >= 0 ? "+" : ""}${value.toFixed(2)} points`;
  const abs = Math.abs(value);
  return abs !== 0 && abs < 0.001 ? value.toExponential(2) : value.toFixed(4);
}

function pick(event) {
  if (!current) return null;
  const rect = dom.canvas.getBoundingClientRect();
  pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(current.panels.map((p) => p.mesh), false).find((h) => h.face.materialIndex === 4);
  if (!hit) return null;
  const record = hit.object.userData.record;
  const col = Math.min(record.panel.cols - 1, Math.floor(hit.uv.x * record.panel.cols));
  const row = Math.min(record.panel.rows - 1, Math.floor((1 - hit.uv.y) * record.panel.rows));
  return { record, row, col };
}

function clearHover() {
  if (hovered) {
    hovered.record.material.uniforms.uHover.value.set(-1, -1);
    redraw = true;
  }
  hovered = null;
  dom.tooltip.hidden = true;
}

function onMove(event) {
  if (dragging) return clearHover();
  const hit = pick(event);
  if (!hit) return clearHover();
  if (hovered && hovered.record !== hit.record) hovered.record.material.uniforms.uHover.value.set(-1, -1);
  hovered = hit;
  const { record, row, col } = hit;
  const { panel } = record;
  record.material.uniforms.uHover.value.set(col, row);
  redraw = true;
  const value = decode(record, row, col);
  const rowBand = bandLabel(panel.rowBands, row) ?? panel.rowLabels?.[row] ?? panel.rowNames?.[row];
  const colBand = bandLabel(panel.colBands, col) ?? panel.colLabels?.[col];
  const mark = (panel.highlights || []).find((m) => {
    const i = m.axis === "row" ? row : col;
    return i >= m.start && i < m.start + m.size;
  });
  const lines = [
    ["Row", `${row}${rowBand ? ` (${rowBand})` : ""}`],
    ["Column", `${col}${colBand ? ` (${colBand})` : ""}`],
    [UNIT_LABEL[panel.unit] ?? "Value", value === null && panel.missingText ? panel.missingText : formatValue(value, panel.unit)],
  ];
  if (!panel.unit || panel.unit === "weight") {
    lines.push(["Largest in matrix", `±${panel.vmax.toFixed(4)}`]);
    lines.push(["Color saturates at", `±${(panel.colorMax ?? panel.vmax).toFixed(4)}`]);
  }
  dom.tooltip.innerHTML = `<div class="t-title">${record.layerName} - ${panel.label}</div>
    <dl>${lines.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>
    ${mark ? `<div class="t-note">${STATUS_ICON[mark.status]} ${mark.label}</div>` : ""}
    ${panel.note ? `<div class="t-note">${panel.note}</div>` : ""}`;
  dom.tooltip.hidden = false;
  const stage = dom.stage.getBoundingClientRect();
  const tip = dom.tooltip.getBoundingClientRect();
  let left = event.clientX - stage.left + 16;
  let top = event.clientY - stage.top + 16;
  if (left + tip.width > stage.width - 8) left = event.clientX - stage.left - tip.width - 16;
  if (top + tip.height > stage.height - 8) top = event.clientY - stage.top - tip.height - 16;
  dom.tooltip.style.transform = `translate(${Math.max(8, left)}px, ${Math.max(8, top)}px)`;
  dom.tooltip.style.left = dom.tooltip.style.top = "0";
}

let dragging = false;
let downAt = null;
dom.canvas.addEventListener("pointerdown", (e) => {
  downAt = [e.clientX, e.clientY];
});
dom.canvas.addEventListener("pointermove", (e) => {
  if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) {
    dragging = true;
    dom.canvas.classList.add("dragging");
  }
  onMove(e);
});
window.addEventListener("pointerup", (e) => {
  const wasClick = downAt && !dragging && e.target === dom.canvas;
  downAt = null;
  dragging = false;
  dom.canvas.classList.remove("dragging");
  if (wasClick) {
    const hit = pick(e);
    if (hit) framePanel(hit.record);
  }
});
dom.canvas.addEventListener("pointerleave", clearHover);
dom.canvas.addEventListener("contextmenu", (e) => e.preventDefault());
window.addEventListener("keydown", (e) => {
  if (e.key.toLowerCase() === "r" && current && !e.metaKey && !e.ctrlKey && e.target.tagName !== "SELECT") frameLayer(0, true);
});

function applyFocus() {
  if (!current) return;
  for (const p of current.panels) p.material.uniforms.uDim.value = dom.focus.checked ? 1 : 0;
  redraw = true;
}
dom.focus.addEventListener("change", applyFocus);
dom.gamma.addEventListener("input", () => {
  dom.gammaOut.textContent = Number(dom.gamma.value).toFixed(2);
  if (current) for (const p of current.panels) p.material.uniforms.uGamma.value = Number(dom.gamma.value);
  redraw = true;
});
dom.gammaOut.textContent = Number(dom.gamma.value).toFixed(2);

// --------------------------------------------------------------------------------------- frame loop
const projected = new THREE.Vector3();
const forward = new THREE.Vector3();
const ANCHOR = {
  "left-bottom": "translate(0, -100%)",
  "left-middle": "translate(0, -50%)",
  "right-middle": "translate(-100%, -50%)",
  "right-top": "translate(-100%, 0)",
  "right-bottom": "translate(-100%, -100%)",
  "center-top": "translate(-50%, 0)",
  "center-bottom": "translate(-50%, -100%)",
};

let labelFrame = 0;

function updateLabels(width, height) {
  camera.getWorldDirection(forward);
  const unitsToPixels = height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  labelFrame += 1;
  for (const label of current.labels) {
    const depth = projected.copy(label.position).sub(camera.position).dot(forward);
    projected.copy(label.position).project(camera);
    const pixelsPerUnit = unitsToPixels / Math.max(depth, 1);
    if (label.tick && label.tick.axis.frame !== labelFrame) {
      // one stride per axis and frame (taken at its first label), so a tilted view does not thin it unevenly
      label.tick.axis.frame = labelFrame;
      label.tick.axis.stride = tickStride(label.tick.axis, label.tick.units, pixelsPerUnit);
    }
    const visible = depth > 0 && Math.abs(projected.x) < 1.15 && Math.abs(projected.y) < 1.15
      && (label.span ? label.span * pixelsPerUnit >= label.minPixels : true)
      && (!label.tick || label.tick.index % label.tick.axis.stride === 0)
      && label.room * pixelsPerUnit >= (label.room ? label.roomPixels : 0); // labels in a gap need free pixels there
    label.el.style.display = visible ? "" : "none";
    if (!visible) continue;
    if (label.clip) label.el.style.maxWidth = `${Math.floor(label.span * pixelsPerUnit)}px`;
    const sx = (projected.x * 0.5 + 0.5) * width;
    const sy = (-projected.y * 0.5 + 0.5) * height;
    label.el.style.transform = `translate(${sx.toFixed(1)}px, ${sy.toFixed(1)}px) ${ANCHOR[label.anchor]}`;
  }
}

function resize() {
  const { clientWidth: width, clientHeight: height } = dom.stage;
  if (dom.canvas.width !== Math.floor(width * renderer.getPixelRatio()) || dom.canvas.height !== Math.floor(height * renderer.getPixelRatio())) {
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    redraw = true;
  }
  return { width, height };
}

// A frame is drawn only when the camera moved, a flight is running, the canvas was resized or `redraw` was set,
// so a still view leaves the GPU idle.
let redraw = true;
const drawnCamera = new THREE.Matrix4();

function tick(now) {
  const { width, height } = resize();
  if (flight) {
    const t = Math.min(1, (now - flight.start) / flight.ms);
    const k = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    camera.position.lerpVectors(flight.fromP, flight.toP, k);
    controls.target.lerpVectors(flight.fromT, flight.toT, k);
    if (t >= 1) flight = null;
  }
  controls.update();
  camera.updateMatrixWorld();
  if (redraw || flight || !drawnCamera.equals(camera.matrixWorld)) {
    redraw = false;
    drawnCamera.copy(camera.matrixWorld);
    renderer.render(scene, camera);
    if (current) updateLabels(width, height);
  }
  requestAnimationFrame(tick);
}

// ------------------------------------------------------------------------------------------- boot
async function boot() {
  const index = await (await fetch("data/index.json")).json();
  const families = [...new Set(index.map((d) => d.family))];
  for (const family of families) {
    const group = document.createElement("optgroup");
    group.label = family;
    for (const entry of index.filter((d) => d.family === family)) {
      group.appendChild(Object.assign(document.createElement("option"), { value: entry.id, textContent: entry.title }));
    }
    dom.dataset.appendChild(group);
  }
  const wanted = new URLSearchParams(location.search).get("d");
  dom.dataset.value = index.some((d) => d.id === wanted) ? wanted : index[0].id;
  dom.dataset.addEventListener("change", () => {
    history.replaceState(null, "", `?d=${dom.dataset.value}`);
    loadDataset(dom.dataset.value);
  });
  requestAnimationFrame(tick);
  await loadDataset(dom.dataset.value);
}

boot();
