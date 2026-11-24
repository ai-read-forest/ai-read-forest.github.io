// The stage: one canvas, one soft-style scene per step. Two modes:
//   bird  - the demo clip; later steps show BirdNET V2.4's real (precomputed) output
//   voice - a microphone recording; later steps show the tiny voice model's real layers (js/voice.js)
// Scene time t (seconds) restarts when a step is shown or "다시 재생" is pressed.

import { BANDS, EXAMPLE_KERNELS, N, SR, sigmoid } from "./dsp.js";
import * as audio from "./audio.js";
import {
  C, clamp, diverge, dot, envelope, hash01, leafCss, lerp, line, roundBox, roundImage, seg, smoothPath,
  softEnvelope, text,
} from "./draw.js";

const LABELS = {
  input: "녹음: 3초 소리와 시간 띠",
  wave: "파형과 확대한 숫자들",
  fft: "짧은 조각의 주파수와 세로 한 줄",
  spec: "스펙트로그램",
  conv: "필터가 지나가며 만드는 feature map",
  vector: "층을 거쳐 작아지는 그림과 벡터",
  score: "sigmoid 곡선과 점수",
  result: "결과",
};

export class Stage {
  constructor(canvas, state, { reduce = false, displayName = (c) => c.ko || c.en } = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.S = state;
    this.name = displayName;
    this.reduce = reduce;
    this.scene = "input";
    this.t0 = performance.now();
    this.cache = {};
    this.visible = true;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    new IntersectionObserver(([e]) => {
      this.visible = e.isIntersecting;
      if (this.visible) this.loop();
    }).observe(canvas);
    this.loop();
  }

  resize() {
    const r = this.cv.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.W = Math.max(1, r.width);
    this.H = Math.max(1, r.height);
    this.cv.width = Math.round(this.W * dpr);
    this.cv.height = Math.round(this.H * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.cache = {};
    this.narrow = this.W < 520;
    this.pad = this.narrow ? 14 : 24;
    this.fs = this.narrow ? 12 : 14;
    this.draw();
  }

  setScene(name) {
    this.scene = name;
    this.t0 = performance.now();
    this.cv.setAttribute("aria-label", LABELS[name] || "단계별 시각화");
    this.loop();
  }

  replay() {
    this.t0 = performance.now();
    this.loop();
  }

  dataChanged() {
    this.cache = {};
    this.replay();
  }

  loop() {
    if (this.raf) return;
    const tick = () => {
      this.raf = null;
      this.draw();
      if (this.visible) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  message(msg, sub) {
    const { ctx, W, H } = this;
    text(ctx, msg, W / 2, H / 2, { size: 16, weight: 700, color: C.forest, align: "center" });
    if (sub) text(ctx, sub, W / 2, H / 2 + 24, { size: 13, color: C.muted, align: "center" });
  }

  draw() {
    const { ctx, W, H, S } = this;
    ctx.clearRect(0, 0, W, H);
    const a = S.a;
    if (!a) return this.message("불러오는 중…");
    const t = this.reduce ? 60 : (performance.now() - this.t0) / 1000;
    const late = ["conv", "vector", "score", "result"].includes(this.scene);
    if (late && S.mode === "voice" && !S.voice.out) {
      const st = S.voice.status;
      if (st === "quiet") return this.message("소리가 너무 작아요", "조금 더 가까이에서 다시 녹음해 보세요");
      if (st === "error") return this.message("목소리 모델을 불러오지 못했어요", "인터넷 연결을 확인한 뒤 다시 녹음해 보세요");
      return this.message("목소리 모델을 불러오는 중…");
    }
    if (late && S.mode === "bird" && !S.demo) return this.message("BirdNET 출력을 불러오는 중…");
    ctx.save();
    this[this.scene](t, a);
    ctx.restore();
  }

  env(key, x, px, n) {
    const k = `${key}:${Math.round(px)}:${n ?? ""}`;
    this.cache[k] ??= envelope(x, Math.max(2, Math.round(px)), n);
    return this.cache[k];
  }

  playFrac() {
    const p = this.S.play;
    if (!p) return null;
    const u = (audio.now() - p.start) / p.dur;
    return u >= 0 && u <= 1 ? u : null;
  }

  playhead(x0, w, y0, y1) {
    const u = this.playFrac();
    if (u === null) return;
    line(this.ctx, x0 + u * w, y0, x0 + u * w, y1, { color: C.copper, width: 2.5, alpha: 0.9 });
  }

  band() {
    return this.S.mode === "voice" ? BANDS.low : BANDS.high;
  }

  bandImg(a) {
    return this.S.mode === "voice" ? a.img.low : a.img.high;
  }

  small(s, x, y, opts = {}) {
    text(this.ctx, s, x, y, { size: this.fs - 1, color: C.muted, ...opts });
  }

  // ---------------------------------------------------------------- 1 input
  input(t) {
    const { ctx, W, H, pad, fs, S } = this;
    const a = S.a;
    const rec = S.rec;
    const cx = W / 2;
    const cy = H * 0.3;
    const R = Math.min(W, H) * 0.14;
    const pulse = rec.active ? 1 + 0.05 * Math.sin(t * 6) : 1;
    dot(ctx, cx, cy, R * 1.25 * pulse, C.lime, 0.35);
    dot(ctx, cx, cy, R, C.bg, 1);
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineWidth = 6;
    ctx.strokeStyle = "rgba(193,212,133,0.6)";
    ctx.beginPath();
    ctx.arc(cx, cy, R, 0, Math.PI * 2);
    ctx.stroke();
    const prog = rec.active && rec.buf ? rec.n / rec.buf.length : 0;
    if (rec.active) {
      ctx.strokeStyle = C.forest;
      ctx.beginPath();
      ctx.arc(cx, cy, R, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * prog);
      ctx.stroke();
    }
    // microphone
    const mw = R * 0.42;
    const mh = R * 0.78;
    ctx.strokeStyle = C.forest;
    ctx.lineWidth = Math.max(3, R * 0.07);
    ctx.beginPath();
    ctx.roundRect(cx - mw / 2, cy - mh / 2 - R * 0.1, mw, mh, mw / 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy - R * 0.05, R * 0.4, 0.12 * Math.PI, 0.88 * Math.PI);
    ctx.moveTo(cx, cy + R * 0.35);
    ctx.lineTo(cx, cy + R * 0.55);
    ctx.stroke();
    ctx.restore();

    let src;
    if (rec.active) src = `녹음 중 ${((prog * N) / SR).toFixed(1)}초`;
    else if (S.mode === "voice") src = "내 목소리 3초";
    else src = "새소리 데모 · 국민대 직박구리";
    text(ctx, src, cx, cy + R * 1.25 + fs * 1.8, { size: fs + 3, weight: 700, color: C.forest, align: "center" });

    // soft 3-s strip
    const x0 = pad;
    const w = W - 2 * pad;
    const yMid = H * 0.76;
    const half = H * 0.1;
    roundBox(ctx, x0 - 8, yMid - half - 10, w + 16, 2 * half + 20, 16, { fill: "rgba(254,239,197,0.55)" });
    if (rec.active && rec.buf) {
      const frac = rec.n / rec.buf.length;
      const env = envelope(rec.buf, Math.max(2, Math.round(80 * frac)), rec.n);
      let peak = 1e-4;
      for (let i = 0; i < env.maxs.length; i++) peak = Math.max(peak, env.maxs[i], -env.mins[i]);
      softEnvelope(ctx, env, x0, yMid, w * frac, half, { scale: 1 / peak });
    } else {
      softEnvelope(ctx, this.env("raw", a.raw, 160), x0, yMid, w, half, { upto: seg(t, 0.1, 1.3), scale: 1 / a.peak });
      this.playhead(x0, w, yMid - half - 6, yMid + half + 6);
    }
    this.small("0초", x0, yMid + half + fs + 12);
    this.small("3초", x0 + w, yMid + half + fs + 12, { align: "right" });
    text(ctx, "3초 = 숫자 144,000개", W / 2, yMid - half - 20, { size: fs, weight: 600, color: C.forest, align: "center", alpha: seg(t, 1.2, 1.7) });
  }

  // ---------------------------------------------------------------- 2 waveform
  wave(t, a) {
    const { ctx, W, H, pad, fs } = this;
    const x0 = pad;
    const w = W - 2 * pad;
    const yMid = H * 0.3;
    const half = H * 0.2;
    const p = seg(t, 0, 2.2);
    softEnvelope(ctx, this.env("xn", a.xn, 220), x0, yMid, w, half, { upto: p });
    if (p < 1) dot(ctx, x0 + p * w, yMid, 6, C.forest);
    this.small("0초", x0, yMid + half + fs + 8);
    this.small("3초", x0 + w, yMid + half + fs + 8, { align: "right" });
    this.playhead(x0, w, yMid - half, yMid + half);

    // magnifier on 240 samples (5 ms) around the loudest sample
    const z = a.zoom;
    const zx = x0 + (z.c / N) * w;
    const za = seg(t, 2.3, 2.8);
    ctx.save();
    ctx.globalAlpha = za;
    ctx.strokeStyle = C.copper;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(zx, yMid, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    const bx = x0;
    const by = H * 0.62;
    const bw = w;
    const bh = H * 0.3;
    line(ctx, zx, yMid + 14, bx + bw / 2, by - 4, { color: C.copper, width: 2, alpha: za * 0.8 });
    roundBox(ctx, bx, by, bw, bh, 18, { fill: "rgba(254,239,197,0.6)", alpha: za });
    const g = seg(t, 2.8, 3.8);
    if (g > 0) {
      let m = 1e-6;
      for (let i = 0; i < 240; i++) m = Math.max(m, Math.abs(a.xn[z.c - 120 + i]));
      const mid = by + bh / 2;
      const pts = [];
      for (let i = 0; i < 240; i++) pts.push([bx + 14 + ((i + 0.5) / 240) * (bw - 28), mid - (a.xn[z.c - 120 + i] / m) * (bh / 2 - 14) * g]);
      ctx.save();
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 2.5;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      smoothPath(ctx, pts);
      ctx.stroke();
      ctx.restore();
      for (let i = 0; i < 240; i += this.narrow ? 4 : 2) dot(ctx, pts[i][0], pts[i][1], 2.6, C.forest, g);
    }
    text(ctx, "점 하나 = 숫자 하나", bx + 16, by + fs + 10, { size: fs, weight: 600, color: C.forest, alpha: za });
  }

  // ---------------------------------------------------------------- 3 FFT
  fft(t, a) {
    const { ctx, W, H, pad, fs } = this;
    const f = a.frame;
    const band = this.band();
    const colW = clamp(W * 0.045, 12, 24);
    const colX = W - pad - colW;
    const x0 = pad;
    const x1 = colX - (this.narrow ? 30 : 48);
    const w = x1 - x0;

    // A: where the slice sits in the 3 s
    const yA = H * 0.06;
    const hA = H * 0.1;
    softEnvelope(ctx, this.env("xnA", a.xn, 120), x0, yA + hA / 2, w, hA / 2, { color: C.gray, alpha: 0.8 });
    const fx = x0 + (f.start / N) * w;
    const fw = Math.max(10, (band.nfft / N) * w);
    roundBox(ctx, fx - 4, yA - 4, fw + 8, hA + 8, 8, { stroke: C.copper, width: 3, alpha: seg(t, 0.1, 0.5) });

    // B: the slice, softened at both ends by the Hann window
    const yB = H * 0.24;
    const hB = H * 0.2;
    const midB = yB + hB / 2;
    const ha = seg(t, 0.9, 1.4);
    ctx.save();
    ctx.globalAlpha = ha * 0.55;
    ctx.fillStyle = "rgba(254,239,197,1)";
    ctx.beginPath();
    for (let i = 0; i <= 48; i++) {
      const u = i / 48;
      const v = 0.5 - 0.5 * Math.cos(2 * Math.PI * u);
      const px = x0 + u * w;
      if (i === 0) ctx.moveTo(px, midB);
      ctx.lineTo(px, midB - v * (hB / 2));
    }
    for (let i = 48; i >= 0; i--) {
      const u = i / 48;
      const v = 0.5 - 0.5 * Math.cos(2 * Math.PI * u);
      ctx.lineTo(x0 + u * w, midB + v * (hB / 2));
    }
    ctx.fill();
    ctx.restore();
    const mu = seg(t, 1.5, 2.3);
    const step = Math.max(1, Math.floor(band.nfft / 320));
    const pts = [];
    for (let i = 0; i < band.nfft; i += step) pts.push([x0 + (i / (band.nfft - 1)) * w, midB - (lerp(f.raw[i], f.windowed[i], mu) / f.peak) * (hB / 2)]);
    ctx.save();
    ctx.globalAlpha = seg(t, 0.4, 0.8);
    ctx.strokeStyle = C.green;
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.beginPath();
    smoothPath(ctx, pts);
    ctx.stroke();
    ctx.restore();

    // C: spectrum as a soft hill, from 0 Hz to the band's top
    const yC = H * 0.52;
    const hC = H * 0.3;
    const base = yC + hC;
    const fmaxShow = band.fmax;
    const nb = f.bars.length;
    const kmax = Math.min(nb - 1, Math.ceil((fmaxShow / (SR / 2)) * (nb - 1)));
    const xf = (hz) => x0 + (hz / fmaxShow) * w;
    let vmax = 1e-9;
    for (let k = 0; k <= kmax; k++) vmax = Math.max(vmax, f.bars[k]);
    const g = seg(t, 2.5, 3.5);
    if (g > 0) {
      const hill = [];
      const stride = Math.max(1, Math.floor(kmax / 160));
      for (let k = 0; k <= kmax; k += stride) {
        let v = 0;
        for (let j = k; j < Math.min(kmax + 1, k + stride); j++) v = Math.max(v, f.bars[j]);
        hill.push([xf((k * SR) / 2 / (nb - 1)), base - (v / vmax) * hC * 0.92 * g]);
      }
      ctx.save();
      ctx.fillStyle = C.green;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.moveTo(x0, base);
      smoothPath(ctx, hill, false, true);
      ctx.lineTo(x0 + w, base);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    roundBox(ctx, x0 - 2, base, w + 4, 3, 2, { fill: C.hair });
    this.small("낮은 소리", x0, base + fs + 6);
    this.small("높은 소리", x0 + w, base + fs + 6, { align: "right" });

    // mel humps (soft), then 96 numbers gathered into one column
    const tri = seg(t, 3.7, 4.7);
    const edges = a.melEdgesHz;
    if (tri > 0) {
      ctx.save();
      ctx.strokeStyle = C.copper;
      ctx.globalAlpha = 0.35;
      ctx.lineWidth = 1.2;
      const n = Math.floor(tri * 96);
      for (let b = 0; b < n; b += 2) {
        ctx.beginPath();
        ctx.moveTo(xf(edges[b]), base);
        ctx.quadraticCurveTo(xf(edges[b + 1]), base - hC * 0.6, xf(edges[b + 2]), base);
        ctx.stroke();
      }
      ctx.restore();
    }
    const sa = seg(t, 4.8, 5.3);
    if (sa > 0) {
      const top = H * 0.06;
      const ch = (H * 0.88 - top) / 96;
      ctx.save();
      ctx.globalAlpha = sa;
      for (let b = 0; b < 96; b++) {
        const sx = xf(edges[b + 1]);
        const mb = seg(t, 5.0 + (1.2 * b) / 96, 5.6 + (1.2 * b) / 96);
        const x = lerp(sx, colX, mb);
        const y = lerp(base - 10, H * 0.88 - (b + 1) * ch, mb);
        const sz = lerp(6, colW, mb);
        const hh = lerp(6, ch + 0.6, mb);
        ctx.fillStyle = leafCss(f.melU[b]);
        ctx.beginPath();
        ctx.roundRect(x, y, sz, hh, mb > 0.9 ? 0 : 3);
        ctx.fill();
      }
      ctx.restore();
      const la = seg(t, 6.6, 7.1);
      roundBox(ctx, colX - 3, top - 3, colW + 6, H * 0.88 - top + 6, 8, { stroke: C.forest, width: 2.5, alpha: la });
      text(ctx, "한 줄", colX + colW / 2, top - 10, { size: fs, weight: 700, color: C.forest, align: "center", alpha: la });
    }
  }

  // ---------------------------------------------------------------- 4 spectrogram
  spec(t, a) {
    const { ctx, W, H, pad, fs } = this;
    const x0 = pad;
    const w = W - 2 * pad;
    const y0 = H * 0.1;
    const h = H * 0.72;
    const img = this.bandImg(a);
    const cols = img.width;
    roundBox(ctx, x0, y0, w, h, 16, { fill: "rgba(254,239,197,0.35)" });
    const p = seg(t, 0.3, 3.6);
    const k = Math.max(1, Math.round(p * cols));
    roundImage(ctx, img, x0, y0, w, h, 16, [0, 0, k, img.height, k / cols]);
    if (p < 1) roundBox(ctx, x0 + (k / cols) * w - 3, y0 - 4, 6, h + 8, 3, { fill: C.copper, alpha: 0.8 });
    this.small("높은 소리", x0 + 4, y0 - 8);
    this.small("낮은 소리", x0 + 4, y0 + h + fs + 6);
    this.small("시간 →", x0 + w, y0 + h + fs + 6, { align: "right" });
    if (this.S.mode === "voice") this.small("0–3 kHz", x0 + w, y0 - 8, { align: "right" });
    this.playhead(x0, w, y0, y0 + h);
  }

  // ---------------------------------------------------------------- 5 convolution
  conv(t, a) {
    const { ctx, W, H, pad, fs, S } = this;
    const voice = S.mode === "voice";
    let map;
    let kernel;
    let kname;
    if (voice) {
      const v = S.voice;
      map = { img: v.imgs[0], rows: v.out.acts[0].H, cols: v.out.acts[0].W, best: v.best0, padTop: 1, padLeft: 3 };
      kernel = v.kernel0;
      kname = `학습된 필터 ${v.ch0 + 1}번 (8개 중)`;
    } else {
      const cv = a.conv[S.kernel];
      map = { img: a.img.conv[S.kernel], rows: cv.rows, cols: cv.cols, best: cv.best, padTop: cv.padTop, padLeft: cv.padLeft };
      kernel = EXAMPLE_KERNELS[S.kernel].w;
      kname = `예시 필터: ${EXAMPLE_KERNELS[S.kernel].name}`;
    }
    const x0 = pad;
    const w = W - 2 * pad;
    const yA = H * 0.05;
    const hA = H * 0.33;
    const yC = H * 0.62;
    const hC = H * 0.33;
    roundImage(ctx, this.bandImg(a), x0, yA, w, hA, 14);

    const run = clamp((t - 0.6) / 4.0);
    const done = t > 4.8;
    let i;
    let j;
    if (done) ({ i, j } = map.best);
    else {
      const pos = Math.floor(run * map.rows * map.cols);
      j = Math.min(map.cols - 1, Math.floor(pos / map.rows));
      i = pos % map.rows;
    }
    roundBox(ctx, x0, yC, w, hC, 14, { fill: "rgba(254,239,197,0.35)" });
    if (done || run >= 1) roundImage(ctx, map.img, x0, yC, w, hC, 14);
    else if (j > 0) roundImage(ctx, map.img, x0, yC, w, hC, 14, [0, 0, j, map.rows, j / map.cols]);

    // the sliding window on the spectrogram (4 x 8 cells, stride 2)
    const kh = kernel.length;
    const kw = kernel[0].length;
    const rx = x0 + ((j * 2 - map.padLeft) / 511) * w;
    const ry = yA + ((i * 2 - map.padTop) / 96) * hA;
    const rw = Math.max(10, (kw / 511) * w);
    const rh = Math.max(8, (kh / 96) * hA);
    if (run > 0 || done) {
      roundBox(ctx, rx - 3, ry - 3, rw + 6, rh + 6, 5, { stroke: C.bg, width: 5 });
      roundBox(ctx, rx - 3, ry - 3, rw + 6, rh + 6, 5, { stroke: C.copper, width: 2.5 });
      dot(ctx, x0 + ((j + 0.5) / map.cols) * w, yC + ((i + 0.5) / map.rows) * hC, 4, C.copper);
    }

    // the filter itself, as soft rounded cells
    const cs = clamp(Math.min((w * 0.36) / kw, (H * 0.13) / kh), 7, 18);
    const gx = x0;
    const gy = yA + hA + 14;
    let m = 1e-9;
    for (const r of kernel) for (const v of r) m = Math.max(m, Math.abs(v));
    for (let r = 0; r < kh; r++) {
      for (let c = 0; c < kw; c++) {
        ctx.fillStyle = diverge(kernel[r][c] / m);
        ctx.beginPath();
        ctx.roundRect(gx + c * cs, gy + r * cs, cs - 2, cs - 2, Math.min(5, cs / 3));
        ctx.fill();
      }
    }
    text(ctx, kname, gx + kw * cs + 12, gy + (kh * cs) / 2 + fs * 0.35, { size: fs, weight: 700, color: C.forest });
    if (done && !this.narrow) text(ctx, "가장 크게 반응한 곳", x0 + w, gy + (kh * cs) / 2 + fs * 0.35, { size: fs - 1, color: C.copperInk, align: "right", alpha: seg(t, 4.8, 5.2) });
  }

  // ---------------------------------------------------------------- 6 vector
  vector(t, a) {
    const { ctx, W, H, pad, fs, S } = this;
    const voice = S.mode === "voice";
    let items;
    let vec;
    if (voice) {
      const v = S.voice;
      items = [["그림", a.img.low, [96, 511]], ...v.out.acts.map((L, k) => [`${k + 1}층`, v.imgs[k], [L.H, L.W]])];
      vec = { values: v.out.vec, max: Math.max(...v.out.vec) || 1, cols: 8, label: "벡터: 숫자 32개" };
    } else {
      const d = S.demo;
      items = [["그림", a.img.high, [96, 511]], ["첫 층", d.conv0Img[0], [48, 256]], ["…", null, [24, 64]], ["…", null, [6, 16]], ["…", null, [1, 6]]];
      vec = { values: d.embedding, max: d.embMax, cols: 32, label: "벡터: 숫자 1,024개" };
    }
    const n = items.length;
    const rowY = H * 0.08;
    const maxH = H * 0.3;
    const cell = (W - 2 * pad) / n;
    items.forEach(([name, img, [hh0, ww0]], k) => {
      const al = seg(t, 0.2 + k * 0.35, 0.6 + k * 0.35);
      if (al <= 0) return;
      const hh = maxH * (0.25 + (0.75 * Math.log(hh0 + 1)) / Math.log(97));
      const ww = cell * 0.8 * (0.25 + (0.75 * Math.log(ww0 + 1)) / Math.log(512));
      const bx = pad + k * cell + (cell - ww) / 2;
      const by = rowY + (maxH - hh) / 2;
      ctx.save();
      ctx.globalAlpha = al;
      if (img) roundImage(ctx, img, bx, by, ww, hh, Math.min(10, ww / 4));
      else roundBox(ctx, bx, by, ww, hh, Math.min(10, ww / 4), { fill: "rgba(193,212,133,0.45)" });
      ctx.restore();
      text(ctx, name, pad + k * cell + cell / 2, rowY + maxH + fs + 8, { size: fs, weight: 600, color: C.forest, align: "center", alpha: al });
      if (k < n - 1) text(ctx, "›", pad + (k + 1) * cell, rowY + maxH / 2 + 6, { size: fs + 6, color: C.gray, align: "center", alpha: seg(t, 0.5 + k * 0.35, 0.8 + k * 0.35) });
    });

    // the vector as soft dots (size = value)
    const va = seg(t, 0.4 + n * 0.35, 0.8 + n * 0.35);
    if (va > 0) {
      const vals = vec.values;
      const rowsN = Math.ceil(vals.length / vec.cols);
      const area = Math.min(W - 2 * pad, (H * 0.36 * vec.cols) / rowsN);
      const gsz = area / vec.cols;
      const gx = (W - area) / 2;
      const gy = H * 0.58;
      const cnt = Math.floor(seg(t, 0.6 + n * 0.35, 1.8 + n * 0.35) * vals.length);
      for (let q = 0; q < cnt; q++) {
        const u = clamp(vals[q] / vec.max);
        const r = Math.max(0.8, (gsz / 2) * 0.92 * Math.sqrt(u));
        dot(ctx, gx + ((q % vec.cols) + 0.5) * gsz, gy + (Math.floor(q / vec.cols) + 0.5) * gsz, r, leafCss(0.35 + 0.65 * u), va);
      }
      text(ctx, vec.label, W / 2, gy - 12, { size: fs + 1, weight: 700, color: C.forest, align: "center", alpha: va });
    }
  }

  // ---------------------------------------------------------------- 7 score (logit -> sigmoid)
  score(t) {
    const { ctx, W, H, pad, fs, S } = this;
    const voice = S.mode === "voice";
    const xL = pad + 14;
    const xR = W - pad - 10;
    const yT = H * 0.16;
    const yB = H * 0.72;
    const Z0 = -8;
    const Z1 = 8;
    const xz = (v) => xL + ((clamp(v, Z0, Z1) - Z0) / (Z1 - Z0)) * (xR - xL);
    const ys = (p) => yB - p * (yB - yT);
    roundBox(ctx, xL - 6, yB, xR - xL + 12, 4, 2, { fill: C.hair });
    line(ctx, xL, ys(1), xR, ys(1), { color: C.hair, dash: [4, 6] });
    this.small("1", xL - 6, ys(1) + 4, { align: "right" });
    this.small("0", xL - 6, yB + 4, { align: "right" });
    const cp = seg(t, 0, 1.1);
    const curve = [];
    for (let s = 0; s <= 120 * cp; s++) {
      const v = Z0 + ((Z1 - Z0) * s) / 120;
      curve.push([xz(v), ys(sigmoid(v))]);
    }
    ctx.save();
    ctx.strokeStyle = C.green;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    smoothPath(ctx, curve);
    ctx.stroke();
    ctx.restore();
    const rise = seg(t, 1.5, 2.4);
    const pa = seg(t, 1.2, 1.5);
    const la = seg(t, 2.4, 2.9);
    if (voice) {
      const o = S.voice.out;
      this.small("여성 목소리 쪽", xL, yB + fs + 14);
      this.small("남성 목소리 쪽", xR, yB + fs + 14, { align: "right" });
      const x = xz(o.logit);
      const y = lerp(yB, ys(o.pMale), rise);
      line(ctx, x, yB, x, y, { color: C.copper, width: 2.5, dash: [4, 5], alpha: pa });
      dot(ctx, x, y, 9, C.copper, pa);
      dot(ctx, x, y, 4, C.bg, pa);
      const right = x > W * 0.6;
      text(ctx, `남성 쪽 점수 ${o.pMale.toFixed(2)}`, right ? x - 14 : x + 14, ys(o.pMale) + (o.pMale > 0.5 ? fs * 2 : -fs), { size: fs + 4, weight: 800, color: C.forest, align: right ? "right" : "left", alpha: la });
    } else {
      const d = S.demo;
      const z = d.logits;
      ctx.save();
      ctx.fillStyle = C.gray;
      ctx.globalAlpha = 0.3 * seg(t, 1.0, 1.4);
      for (let q = 0; q < z.length; q += 2) {
        if (z[q] < Z0) continue;
        ctx.beginPath();
        ctx.arc(xz(z[q]), yB + 12 + hash01(q) * 18, 1.4, 0, 6.283);
        ctx.fill();
      }
      ctx.restore();
      d.top.slice(0, 5).forEach((s, k) => {
        const x = xz(s.logit);
        const y = lerp(yB, ys(s.score), rise);
        dot(ctx, x, y, k === 0 ? 9 : 5, k === 0 ? C.copper : C.gray, pa);
        if (k === 0) dot(ctx, x, y, 4, C.bg, pa);
      });
      const s0 = d.top[0];
      text(ctx, `${this.name(s0)} ${s0.score.toFixed(2)}`, xz(s0.logit) - 14, ys(s0.score) + fs * 2, { size: fs + 4, weight: 800, color: C.forest, align: "right", alpha: la });
      text(ctx, `점수 6,522개를 다 더하면 ${d.score_sum.toFixed(2)}`, W / 2, H - fs, { size: fs, color: C.muted, align: "center", alpha: seg(t, 3.0, 3.5) });
    }
  }

  // ---------------------------------------------------------------- 8 result
  result(t) {
    const { ctx, W, H, pad, fs, S } = this;
    if (S.mode === "voice") {
      const p = S.voice.out.pMale;
      const x0 = pad + 20;
      const w = W - 2 * pad - 40;
      const y = H * 0.56;
      const grad = ctx.createLinearGradient(x0, 0, x0 + w, 0);
      grad.addColorStop(0, "rgba(196,126,76,0.5)");
      grad.addColorStop(0.5, "rgba(254,239,197,0.95)");
      grad.addColorStop(1, "rgba(45,106,79,0.6)");
      roundBox(ctx, x0, y - 14, w, 28, 14, { fill: grad });
      const g = seg(t, 0.3, 1.5);
      const kx = x0 + lerp(0.5, p, g) * w;
      dot(ctx, kx, y, 22, C.bg);
      dot(ctx, kx, y, 22, C.forest, 0.15);
      dot(ctx, kx, y, 14, C.forest);
      text(ctx, "여성", x0, y + 52, { size: fs + 2, weight: 700, color: C.copperInk });
      text(ctx, "남성", x0 + w, y + 52, { size: fs + 2, weight: 700, color: C.forest, align: "right" });
      const va = seg(t, 1.6, 2.2);
      let msg;
      if (p >= 0.6) msg = `남성 목소리 같아요 ${p.toFixed(2)}`;
      else if (p <= 0.4) msg = `여성 목소리 같아요 ${(1 - p).toFixed(2)}`;
      else msg = `잘 모르겠어요 ${p.toFixed(2)}`;
      text(ctx, msg, W / 2, H * 0.28, { size: this.narrow ? 22 : 28, weight: 800, color: C.forest, align: "center", alpha: va });
      return;
    }
    const d = S.demo;
    const thr = S.threshold;
    const rows = d.top.slice(0, 5);
    const lw = Math.min(W * 0.36, 170);
    const xb = pad + lw;
    const bw = W - xb - pad - 52;
    const y0 = H * 0.32;
    const rh = Math.min(44, (H * 0.6) / rows.length);
    rows.forEach((s, k) => {
      const yc = y0 + rh * (k + 0.5);
      text(ctx, this.name(s), xb - 12, yc + fs * 0.35, { size: fs + 1, weight: k === 0 ? 700 : 500, color: C.ink, align: "right" });
      const g = seg(t, 0.2 + k * 0.1, 1.2 + k * 0.1);
      const on = s.score >= thr;
      roundBox(ctx, xb, yc - 9, bw, 18, 9, { fill: "rgba(254,239,197,0.7)" });
      roundBox(ctx, xb, yc - 9, Math.max(18, s.score * bw * g), 18, 9, { fill: on ? C.green : C.gray });
      text(ctx, s.score.toFixed(2), xb + bw + 8, yc + fs * 0.35, { size: fs, weight: on ? 700 : 400, color: on ? C.forest : C.muted, alpha: g });
    });
    const tx = xb + thr * bw;
    const ta = seg(t, 1.3, 1.8);
    line(ctx, tx, y0 - 14, tx, y0 + rh * rows.length + 4, { color: C.copper, width: 2.5, dash: [5, 5], alpha: ta });
    text(ctx, `기준 ${thr.toFixed(2)}`, tx, y0 - 20, { size: fs, weight: 700, color: C.copperInk, align: "center", alpha: ta });
    const picked = rows.filter((s) => s.score >= thr);
    const va = seg(t, 1.9, 2.5);
    const msg = picked.length ? `${picked.map((s) => this.name(s)).join(", ")} 같아요 ${picked[0].score.toFixed(2)}` : "기준을 넘은 종이 없어요";
    text(ctx, msg, W / 2, H * 0.16, { size: this.narrow ? 22 : 28, weight: 800, color: C.forest, align: "center", alpha: va });
  }
}
