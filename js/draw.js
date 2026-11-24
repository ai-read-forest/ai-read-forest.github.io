// Shared canvas helpers. Figure colours follow the team deck: forest green = bird sound / signal,
// copper = predator sound / negative weight, muted gray-green = context, magma = spectrograms.

export const C = {
  ink: "#1F2622",
  muted: "#66736A",
  hair: "#DCDFD6",
  forest: "#183F2D", // emphasis text, bird sound labels
  green: "#2D6A4F", // marks: bird sound, signal, positive weights
  lime: "#C1D485", // glow / deco (never small text on white)
  copper: "#C47E4C", // predator sound, negative weights (marks only)
  copperInk: "#9A5A2C", // copper for small text
  gray: "#A7B0A9",
  bg: "#FFFFFF",
};

export const FONT = '"Pretendard Variable", Pretendard, -apple-system, BlinkMacSystemFont, system-ui, sans-serif';
export const MONO = 'ui-monospace, "SF Mono", Menlo, Consolas, monospace';

export const clamp = (v, a = 0, b = 1) => (v < a ? a : v > b ? b : v);
export const lerp = (a, b, u) => a + (b - a) * u;
export const smooth = (u) => {
  const v = clamp(u);
  return v * v * (3 - 2 * v);
};
// progress of t inside [a, b], eased
export const seg = (t, a, b) => smooth((t - a) / (b - a));

export function font(ctx, size, weight = 400, mono = false, italic = false) {
  ctx.font = `${italic ? "italic " : ""}${weight} ${size}px ${mono ? MONO : FONT}`;
}

export function text(ctx, s, x, y, { size = 12, weight = 400, color = C.ink, align = "left", base = "alphabetic", mono = false, italic = false, alpha = 1 } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  font(ctx, size, weight, mono, italic);
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = base;
  ctx.fillText(s, x, y);
  ctx.restore();
}

export function line(ctx, x1, y1, x2, y2, { color = C.gray, width = 1, dash = null, alpha = 1 } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  ctx.restore();
}

export function rect(ctx, x, y, w, h, { stroke = C.gray, width = 1, fill = null, alpha = 1, dash = null } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fillRect(x, y, w, h);
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  }
  ctx.restore();
}

export function dot(ctx, x, y, r, color, alpha = 1) {
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function rampFrom(stops) {
  return (u) => {
    const v = clamp(u);
    for (let i = 1; i < stops.length; i++) {
      if (v <= stops[i][0]) {
        const [a, ca] = stops[i - 1];
        const [b, cb] = stops[i];
        const k = (v - a) / (b - a);
        return [0, 1, 2].map((j) => Math.round(lerp(ca[j], cb[j], k)));
      }
    }
    return stops[stops.length - 1][1];
  };
}

// matplotlib "magma" (as in the team deck's spectrograms), for sound pictures
export const magma = rampFrom([
  [0, [0, 0, 4]],
  [0.125, [28, 16, 68]],
  [0.25, [79, 18, 123]],
  [0.375, [129, 37, 129]],
  [0.5, [181, 54, 122]],
  [0.625, [229, 80, 100]],
  [0.75, [251, 135, 97]],
  [0.875, [254, 194, 135]],
  [1, [252, 253, 191]],
]);
// white -> lime -> forest, for the network's insides (feature maps, vector)
export const greens = rampFrom([
  [0, [255, 255, 255]],
  [0.35, [193, 212, 133]],
  [0.7, [45, 106, 79]],
  [1, [24, 63, 45]],
]);
export const magmaCss = (u) => `rgb(${magma(u).join(",")})`;
export const greensCss = (u) => `rgb(${greens(u).join(",")})`;

// Diverging: copper (negative) - white - forest green (positive), u in [-1, 1].
export function diverge(u) {
  const v = clamp(u, -1, 1);
  const end = v >= 0 ? [45, 106, 79] : [196, 126, 76];
  const k = Math.abs(v);
  return `rgb(${[0, 1, 2].map((j) => Math.round(lerp(255, end[j], k))).join(",")})`;
}

export function percentile(data, q) {
  const step = Math.max(1, Math.floor(data.length / 20000));
  const s = [];
  for (let i = 0; i < data.length; i += step) s.push(data[i]);
  s.sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

// rows x cols Float32Array -> offscreen canvas coloured with a ramp (magma or greens).
export function imageCanvas(data, rows, cols, lo, hi, map = magma) {
  const cv = document.createElement("canvas");
  cv.width = cols;
  cv.height = rows;
  const g = cv.getContext("2d");
  const img = g.createImageData(cols, rows);
  const span = hi - lo || 1;
  for (let i = 0; i < rows * cols; i++) {
    const [r, gg, b] = map((data[i] - lo) / span);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = gg;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  return cv;
}

// Min/max envelope of x over `px` columns, for drawing long waveforms.
export function envelope(x, px, n = x.length) {
  const mins = new Float32Array(px);
  const maxs = new Float32Array(px);
  for (let i = 0; i < px; i++) {
    const a = Math.floor((i * n) / px);
    const b = Math.max(a + 1, Math.floor(((i + 1) * n) / px));
    let lo = Infinity;
    let hi = -Infinity;
    for (let k = a; k < b && k < n; k++) {
      const v = x[k];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    mins[i] = lo === Infinity ? 0 : lo;
    maxs[i] = hi === -Infinity ? 0 : hi;
  }
  return { mins, maxs };
}

export function drawEnvelope(ctx, env, x0, yMid, w, halfH, { upto = 1, color = C.green, scale = 1 } = {}) {
  const px = env.mins.length;
  const n = Math.floor(px * clamp(upto));
  ctx.save();
  ctx.fillStyle = color;
  for (let i = 0; i < n; i++) {
    const x = x0 + (i / px) * w;
    const top = yMid - env.maxs[i] * scale * halfH;
    const bot = yMid - env.mins[i] * scale * halfH;
    ctx.fillRect(x, top, Math.max(1, w / px), Math.max(1, bot - top));
  }
  ctx.restore();
}

// deterministic pseudo-random in [0, 1)
export function hash01(i) {
  let x = (i + 1) * 2654435761;
  x ^= x >>> 13;
  x = Math.imul(x, 1274126177);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

export function fmt(v, d = 2) {
  const s = Math.abs(v).toFixed(d);
  return (v < 0 ? "−" : "") + s;
}
export function fmtSigned(v, d = 2) {
  return (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);
}

// ---- soft drawing helpers (team palette: white -> butter -> lime -> green -> forest) ----
export const leaf = rampFrom([
  [0, [255, 255, 255]],
  [0.22, [254, 239, 197]],
  [0.5, [193, 212, 133]],
  [0.78, [45, 106, 79]],
  [1, [24, 63, 45]],
]);
export const leafCss = (u) => `rgb(${leaf(u).join(",")})`;

// Separable Gaussian blur of a rows x cols array (edge-clamped).
export function blur2d(data, rows, cols, sr = 1, sc = 1) {
  const kern = (s) => {
    const r = Math.max(1, Math.ceil(s * 2.5));
    const w = [];
    let sum = 0;
    for (let i = -r; i <= r; i++) {
      const v = Math.exp(-(i * i) / (2 * s * s));
      w.push(v);
      sum += v;
    }
    return { r, w: w.map((v) => v / sum) };
  };
  const tmp = new Float32Array(data.length);
  const out = new Float32Array(data.length);
  const kc = kern(sc);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let s = 0;
      for (let k = -kc.r; k <= kc.r; k++) s += kc.w[k + kc.r] * data[i * cols + Math.min(cols - 1, Math.max(0, j + k))];
      tmp[i * cols + j] = s;
    }
  }
  const kr = kern(sr);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let s = 0;
      for (let k = -kr.r; k <= kr.r; k++) s += kr.w[k + kr.r] * tmp[Math.min(rows - 1, Math.max(0, i + k)) * cols + j];
      out[i * cols + j] = s;
    }
  }
  return out;
}

// Blurred, leaf-coloured image of a rows x cols array (values scaled lo..hi).
export function softImage(data, rows, cols, lo, hi, { sr = 1, sc = 1.4 } = {}) {
  return imageCanvas(blur2d(data, rows, cols, sr, sc), rows, cols, lo, hi, leaf);
}

// Smooth closed/open path through points with quadratic midpoints.
export function smoothPath(ctx, pts, close = false, cont = false) {
  if (pts.length < 2) return;
  if (cont) ctx.lineTo(pts[0][0], pts[0][1]);
  else ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2;
    const my = (pts[i][1] + pts[i + 1][1]) / 2;
    ctx.quadraticCurveTo(pts[i][0], pts[i][1], mx, my);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last[0], last[1]);
  if (close) ctx.closePath();
}

// Draw an offscreen image into a rounded rectangle.
export function roundImage(ctx, img, x, y, w, h, r = 10, src = null) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  if (src) ctx.drawImage(img, src[0], src[1], src[2], src[3], x, y, w * (src[4] ?? 1), h);
  else ctx.drawImage(img, x, y, w, h);
  ctx.restore();
}

export function roundBox(ctx, x, y, w, h, r, { fill = null, stroke = null, width = 1.5, alpha = 1, dash = null } = {}) {
  if (alpha <= 0) return;
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
  if (fill) {
    ctx.fillStyle = fill;
    ctx.fill();
  }
  if (stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = width;
    if (dash) ctx.setLineDash(dash);
    ctx.stroke();
  }
  ctx.restore();
}

// Smooth filled waveform envelope: top = max, bottom = min, both smoothed.
export function softEnvelope(ctx, env, x0, yMid, w, half, { upto = 1, color = C.green, scale = 1, alpha = 0.9 } = {}) {
  const n = Math.max(2, Math.floor(env.maxs.length * clamp(upto)));
  if (n < 2) return;
  const px = env.maxs.length;
  const top = [];
  const bot = [];
  for (let i = 0; i < n; i++) {
    const x = x0 + (i / (px - 1)) * w;
    top.push([x, yMid - Math.max(0.02, env.maxs[i] * scale) * half]);
    bot.push([x, yMid - Math.min(-0.02, env.mins[i] * scale) * half]);
  }
  ctx.save();
  ctx.globalAlpha *= alpha;
  ctx.fillStyle = color;
  ctx.beginPath();
  smoothPath(ctx, top);
  const rb = bot.reverse();
  ctx.lineTo(rb[0][0], rb[0][1]);
  for (let i = 1; i < rb.length - 1; i++) {
    const mx = (rb[i][0] + rb[i + 1][0]) / 2;
    const my = (rb[i][1] + rb[i + 1][1]) / 2;
    ctx.quadraticCurveTo(rb[i][0], rb[i][1], mx, my);
  }
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}
