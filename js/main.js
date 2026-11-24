// Page wiring. Two inputs:
//   demo clip (bird mode)   - computed DSP + BirdNET V2.4 output precomputed in assets/data/
//   microphone (voice mode) - computed DSP + a tiny male/female voice CNN loaded from Hugging Face
// and a full-screen slide deck: 1 idea, 2 section cover, 3-10 the eight steps, 11 references.
// Navigation: buttons, dots, ← / → / Space / PageUp / PageDown, swipe, #n in the address, F = full screen.

import * as dsp from "./dsp.js";
import * as audio from "./audio.js";
import { Hero } from "./hero.js";
import { Stage } from "./stage.js";
import { percentile, softImage } from "./draw.js";
import { loadVoiceModel, runVoice } from "./voice.js";

const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = (s) => document.querySelector(s);
const QUIET_RMS = 0.003; // about -50 dBFS: too quiet to say anything about the voice

const displayName = (c) => c.ko || c.en;

const state = {
  source: "demo",
  get mode() {
    return this.source === "mic" ? "voice" : "bird";
  },
  a: null, // analysis of the current input
  demo: null, // BirdNET output for the demo clip (precomputed)
  voice: { status: "idle", out: null }, // voice-model output for the microphone recording
  rec: { active: false, buf: null, n: 0, rate: dsp.SR },
  play: null,
  kernel: "rise",
  threshold: 0.25,
};

// ------------------------------------------------------------------ analysis
function analyze(raw, mainBand) {
  const xn = dsp.normalize(raw);
  const high = dsp.spectrogram(xn, dsp.BANDS.high);
  const low = dsp.spectrogram(xn, dsp.BANDS.low);
  const range = (s) => [percentile(s.data, 0.05), percentile(s.data, 0.995)];
  const imgRange = { high: range(high), low: range(low) };
  const img = {
    high: softImage(high.data, high.rows, high.cols, ...imgRange.high),
    low: softImage(low.data, low.rows, low.cols, ...imgRange.low),
    conv: {},
  };

  // FFT step: the frame with the most energy in the main band
  const band = dsp.BANDS[mainBand];
  const spec = mainBand === "low" ? low : high;
  let idx = 0;
  let best = -Infinity;
  for (let c = 0; c < spec.cols; c++) {
    let s = 0;
    for (let r = 0; r < spec.rows; r++) s += spec.data[r * spec.cols + c];
    if (s > best) {
      best = s;
      idx = c;
    }
  }
  const start = idx * band.hop;
  const w = dsp.hann(band.nfft);
  const frameRaw = xn.slice(start, start + band.nfft);
  const windowed = frameRaw.map((v, i) => v * w[i]);
  let peak = 1e-9;
  for (const v of frameRaw) peak = Math.max(peak, Math.abs(v));
  const power = dsp.powerFrame(xn, start, w);
  const comp = Array.from(power, (p) => Math.pow(p, band.exponent));
  const cmax = Math.max(...comp);
  const melCol = dsp.melColumn(power, dsp.melMatrix(band), band.exponent);
  const [lo, hi] = imgRange[mainBand];
  const melU = Array.from(melCol, (v) => (v - lo) / (hi - lo));

  // example convolutions (bird mode) on the standardised high band
  const std = dsp.standardize(high);
  const conv = {};
  for (const [k, K] of Object.entries(dsp.EXAMPLE_KERNELS)) {
    const r = dsp.conv2dStride2(std, K.w);
    let m = -Infinity;
    let at = 0;
    for (let i = 0; i < r.relu.length; i++) {
      if (r.relu[i] > m) {
        m = r.relu[i];
        at = i;
      }
    }
    r.best = { i: Math.floor(at / r.cols), j: at % r.cols };
    conv[k] = r;
    img.conv[k] = softImage(r.relu, r.rows, r.cols, 0, Math.max(1e-6, percentile(r.relu, 0.995)), { sr: 0.8, sc: 1 });
  }

  // zoom target for the waveform step: loudest sample away from the edges
  let c = 120;
  let cm = 0;
  for (let i = 120; i < xn.length - 120; i++) {
    const v = Math.abs(xn[i]);
    if (v > cm) {
      cm = v;
      c = i;
    }
  }
  let rawPeak = 1e-6;
  let ss = 0;
  for (const v of raw) {
    rawPeak = Math.max(rawPeak, Math.abs(v));
    ss += v * v;
  }

  // mel band edges (Hz) of the main band, for drawing the humps
  const m = (f) => 1127 * Math.log1p(f / 700);
  const mi = (v) => 700 * Math.expm1(v / 1127);
  const e0 = m(band.fmin);
  const e1 = m(band.fmax);
  const melEdgesHz = Array.from({ length: band.nmel + 2 }, (_, i) => mi(e0 + ((e1 - e0) * i) / (band.nmel + 1)));

  return {
    raw, xn, high, low, img, imgRange, conv, melEdgesHz,
    peak: rawPeak,
    rms: Math.sqrt(ss / raw.length),
    zoom: { c },
    frame: { idx, start, raw: frameRaw, windowed, peak, bars: comp.map((v) => v / cmax), melU },
  };
}

// ------------------------------------------------------------------ voice model
function argmaxMean(act) {
  const hw = act.H * act.W;
  let bestC = 0;
  let bestM = -Infinity;
  for (let ch = 0; ch < act.C; ch++) {
    let s = 0;
    for (let k = 0; k < hw; k++) s += act.data[ch * hw + k];
    if (s > bestM) {
      bestM = s;
      bestC = ch;
    }
  }
  return bestC;
}

function channel(act, ch) {
  const hw = act.H * act.W;
  return act.data.subarray(ch * hw, (ch + 1) * hw);
}

async function classifyVoice() {
  const a = state.a;
  state.voice = { status: "loading", out: null };
  stage.dataChanged();
  if (a.rms < QUIET_RMS) {
    state.voice.status = "quiet";
    stage.dataChanged();
    return;
  }
  try {
    const model = await loadVoiceModel();
    if (state.a !== a) return; // input changed meanwhile
    const out = runVoice(model, a.low);
    const imgs = out.acts.map((act) => {
      const m = channel(act, argmaxMean(act));
      return softImage(m, act.H, act.W, 0, Math.max(1e-6, percentile(m, 0.995)), { sr: 0.8, sc: 1 });
    });
    const ch0 = argmaxMean(out.acts[0]);
    const map0 = channel(out.acts[0], ch0);
    let at = 0;
    for (let i = 1; i < map0.length; i++) if (map0[i] > map0[at]) at = i;
    const [, , kh, kw] = out.kernels.shape;
    const k0 = Array.from({ length: kh }, (_, r) => Array.from({ length: kw }, (_, c) => out.kernels.data[(ch0 * kh + r) * kw + c]));
    state.voice = {
      status: "ready", out, imgs, ch0, kernel0: k0,
      best0: { i: Math.floor(at / out.acts[0].W), j: at % out.acts[0].W },
    };
    const p = out.pMale;
    setStatus(`<strong>내 목소리 3초</strong> · ${p >= 0.6 ? "남성 목소리 같아요" : p <= 0.4 ? "여성 목소리 같아요" : "잘 모르겠어요"} (마지막 단계에서 자세히)`);
  } catch (e) {
    console.error(e);
    state.voice = { status: "error", out: null };
  }
  stage.dataChanged();
}

// ------------------------------------------------------------------ demo data
function b64bytes(s) {
  const bin = atob(s);
  const u = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
  return u;
}

async function loadDemo() {
  const [json, wav] = await Promise.all([
    fetch("assets/data/demo-birdnet.json").then((r) => {
      if (!r.ok) throw new Error(`demo json ${r.status}`);
      return r.json();
    }),
    fetch("assets/audio/demo-kookmin-bulbul-3s.wav").then((r) => {
      if (!r.ok) throw new Error(`demo wav ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  const { rate, samples } = dsp.parseWav16(wav);
  if (rate !== dsp.SR || samples.length < dsp.N) throw new Error("unexpected demo wav format");
  const bytes = b64bytes(json.logits_x100_b64);
  const dv = new DataView(bytes.buffer);
  const logits = new Float32Array(bytes.length / 2);
  for (let i = 0; i < logits.length; i++) logits[i] = dv.getInt16(i * 2, true) / 100;
  const [rows, cols] = json.conv0.shape;
  const conv0Img = json.conv0.maps_u8_b64.map((s) => softImage(Float32Array.from(b64bytes(s), (v) => v / 255), rows, cols, 0, 1, { sr: 0.8, sc: 1 }));
  return {
    ...json,
    samples: samples.slice(0, dsp.N),
    logits,
    conv0Img,
    embMax: Math.max(...json.embedding),
  };
}

// ------------------------------------------------------------------ UI
const statusEl = $("#status");
const sourceEl = $("#stage-source");
function setStatus(html, warn = false) {
  statusEl.innerHTML = html;
  statusEl.classList.toggle("warn", warn);
}
function updateSource() {
  document.body.classList.toggle("mode-voice", state.mode === "voice");
  sourceEl.textContent = state.rec.active
    ? "녹음 중…"
    : state.mode === "voice"
      ? "입력: 내 목소리 3초 · 남/여 목소리 모델 (Hugging Face)"
      : "입력: 새소리 데모 · 국민대 직박구리 · BirdNET V2.4";
}

let stage;

function setInput(raw, source) {
  audio.stop();
  state.play = null;
  state.source = source;
  state.voice = { status: "idle", out: null };
  state.a = analyze(raw, source === "mic" ? "low" : "high");
  updateSource();
  stage?.dataChanged();
}

function useDemo(message) {
  if (!state.demo) return;
  setInput(state.demo.samples, "demo");
  setStatus(message || "<strong>새소리 데모</strong> · 국민대 직박구리 3초");
}

const dbfs = (v) => `${(20 * Math.log10(Math.max(v, 1e-9))).toFixed(0)} dBFS`;

async function onMic() {
  if (state.rec.active) return;
  const sup = audio.micSupport();
  if (sup !== "ok") {
    useDemo(`<strong>${sup === "insecure" ? "HTTPS가 아니어서" : "이 브라우저에서는"} 마이크를 쓸 수 없어요.</strong> 새소리 데모를 보여 줄게요.`);
    statusEl.classList.add("warn");
    return;
  }
  loadVoiceModel().catch(() => {}); // fetch the small model while recording
  const btn = $("#btn-mic");
  const label = btn.querySelector("span");
  state.rec = { active: true, buf: null, n: 0, rate: dsp.SR };
  btn.classList.add("recording");
  btn.disabled = true;
  label.textContent = "녹음 중…";
  updateSource();
  setStatus("마이크를 허락하면 3초 동안 녹음해요.");
  show(FIRST_STEP);
  try {
    const x = await audio.record3s((buf, n, rate) => {
      if (!state.rec.buf) setStatus("<strong>녹음 중</strong> · 아무 말이나 해 보세요.");
      state.rec.buf = buf;
      state.rec.n = n;
      state.rec.rate = rate;
    });
    state.rec.active = false;
    setInput(x, "mic");
    if (state.a.rms < QUIET_RMS) setStatus(`<strong>소리가 너무 작아요</strong> (평균 ${dbfs(state.a.rms)}). 다시 녹음해 보세요.`, true);
    else setStatus("<strong>내 목소리 3초</strong> · 목소리 모델이 계산하는 중…");
    classifyVoice();
  } catch (e) {
    state.rec.active = false;
    const why = e && e.name === "NotAllowedError" ? "마이크 권한이 없어요" : e && e.name === "NotFoundError" ? "마이크를 찾지 못했어요" : "마이크를 열지 못했어요";
    useDemo(`<strong>${why}.</strong> 새소리 데모를 보여 줄게요.`);
    statusEl.classList.add("warn");
    console.warn("microphone:", e);
  } finally {
    btn.classList.remove("recording");
    btn.disabled = false;
    label.textContent = "내 목소리 3초 녹음";
    updateSource();
  }
}

// ------------------------------------------------------------------ slides
const steps = [...document.querySelectorAll(".step")];
const FIRST_STEP = 2;
const SLIDES = [
  { screen: "hero", label: "우리가 하고 싶은 것" },
  { screen: "divider", label: "새소리를 분석하는 세 단계" },
  ...steps.map((s, k) => ({ screen: "steps", step: k, phase: s.dataset.phase, label: s.dataset.title })),
  { screen: "refs", label: "참고문헌" },
];
const PHASE = { 1: "1단계 소리 → 그림", 2: "2단계 AI(합성곱 신경망)의 판단", 3: "3단계 결과 도출" };
const screens = Object.fromEntries([...document.querySelectorAll(".screen")].map((el) => [el.dataset.screen, el]));
const panel = $("#step-panel");
const dots = [];
let cur = -1;
let hero;

function buildDots() {
  const box = $("#dots");
  let prevGroup = null;
  SLIDES.forEach((sl, i) => {
    const group = sl.screen === "steps" ? `p${sl.phase}` : sl.screen;
    if (prevGroup !== null && group !== prevGroup) box.appendChild(Object.assign(document.createElement("span"), { className: "gap" }));
    prevGroup = group;
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-label", `${i + 1}. ${sl.label}`);
    b.title = `${i + 1}. ${sl.label}`;
    b.addEventListener("click", () => show(i));
    box.appendChild(b);
    dots.push(b);
  });
}

function show(i, { fromHash = false } = {}) {
  const n = Math.max(0, Math.min(SLIDES.length - 1, i));
  if (n === cur) return;
  cur = n;
  const sl = SLIDES[n];
  for (const [name, el] of Object.entries(screens)) el.hidden = name !== sl.screen;
  if (sl.screen === "steps") {
    const s = steps[sl.step];
    steps.forEach((x, k) => (x.hidden = k !== sl.step));
    $("#slide-title").textContent = s.dataset.title;
    $("#phase-name").textContent = `${PHASE[s.dataset.phase]} · ${sl.step + 1} / ${steps.length}`;
    panel.scrollTop = 0;
    stage?.setScene(s.dataset.scene);
  }
  if (sl.screen === "hero") hero?.replay();
  const count = `${n + 1} / ${SLIDES.length}`;
  $("#nav-count").textContent = count;
  dots.forEach((d, k) => (k === n ? d.setAttribute("aria-current", "step") : d.removeAttribute("aria-current")));
  $("#btn-prev").disabled = n === 0;
  $("#btn-next").disabled = n === SLIDES.length - 1;
  $("#btn-replay").disabled = !(sl.screen === "hero" || sl.screen === "steps");
  document.title = `${sl.label} · BirdNET 따라가기`;
  if (!fromHash) history.replaceState(null, "", `#${n + 1}`);
}

function slideFromHash() {
  const m = /^#(\d+)$/.exec(location.hash);
  return m ? Number(m[1]) - 1 : 0;
}

function toggleFullscreen() {
  if (!document.fullscreenEnabled) return;
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch(() => {});
}

function onKey(e) {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
  const t = e.target;
  const tag = t.tagName;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || t.isContentEditable) return;
  const onControl = tag === "BUTTON" || tag === "A" || tag === "SUMMARY";
  const k = e.key;
  if (k === "ArrowRight" || k === "PageDown" || (k === " " && !e.shiftKey && !onControl)) {
    e.preventDefault();
    show(cur + 1);
  } else if (k === "ArrowLeft" || k === "PageUp" || (k === " " && e.shiftKey && !onControl)) {
    e.preventDefault();
    show(cur - 1);
  } else if (k === "Home") {
    e.preventDefault();
    show(0);
  } else if (k === "End") {
    e.preventDefault();
    show(SLIDES.length - 1);
  } else if ((k === "f" || k === "F") && !e.shiftKey) {
    e.preventDefault();
    toggleFullscreen();
  }
}

function enableSwipe(el) {
  let x0 = null;
  let y0 = 0;
  el.addEventListener("touchstart", (e) => {
    if (e.touches.length !== 1 || e.target.closest("input")) {
      x0 = null;
      return;
    }
    x0 = e.touches[0].clientX;
    y0 = e.touches[0].clientY;
  }, { passive: true });
  el.addEventListener("touchend", (e) => {
    if (x0 === null) return;
    const dx = e.changedTouches[0].clientX - x0;
    const dy = e.changedTouches[0].clientY - y0;
    x0 = null;
    if (Math.abs(dx) > 60 && Math.abs(dx) > 1.5 * Math.abs(dy)) show(cur + (dx < 0 ? 1 : -1));
  }, { passive: true });
}

// A mouse click should not leave focus on a button, so Space keeps meaning "next".
document.addEventListener("click", (e) => {
  const b = e.target.closest("button, summary");
  if (b && e.detail > 0) b.blur();
});

// ------------------------------------------------------------------ start
window.__birdnetPage = { state, slides: SLIDES, show: (i) => show(i) }; // hook for tools/shoot.py

async function init() {
  hero = new Hero($("#hero-canvas"), $("#hero-steps"), { reduce });
  window.__birdnetPage.hero = hero;
  stage = new Stage($("#stage-canvas"), state, { reduce, displayName });
  window.__birdnetPage.stage = stage;
  buildDots();
  show(slideFromHash());
  window.addEventListener("hashchange", () => show(slideFromHash(), { fromHash: true }));
  updateSource();
  setStatus("새소리 데모를 불러오는 중…");

  $("#btn-mic").addEventListener("click", onMic);
  $("#btn-demo").addEventListener("click", () => useDemo());
  $("#btn-play").addEventListener("click", () => {
    if (!state.a || state.rec.active) return;
    state.play = audio.play(state.a.raw);
    stage.loop();
  });
  $("#btn-prev").addEventListener("click", () => show(cur - 1));
  $("#btn-next").addEventListener("click", () => show(cur + 1));
  $("#btn-replay").addEventListener("click", () => (SLIDES[cur].screen === "hero" ? hero.replay() : stage.replay()));
  const full = $("#btn-full");
  if (!document.fullscreenEnabled) full.hidden = true;
  else full.addEventListener("click", toggleFullscreen);
  document.querySelectorAll("[data-kernel]").forEach((b) =>
    b.addEventListener("click", () => {
      state.kernel = b.dataset.kernel;
      document.querySelectorAll("[data-kernel]").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      stage.replay();
    }),
  );
  const thr = $("#threshold");
  const thrOut = $("#threshold-out");
  thr.addEventListener("input", () => {
    state.threshold = Number(thr.value);
    thrOut.textContent = state.threshold.toFixed(2);
  });
  document.addEventListener("keydown", onKey);
  enableSwipe($("#deck"));

  try {
    state.demo = await loadDemo();
    if (!state.a) useDemo();
  } catch (e) {
    console.error(e);
    setStatus("<strong>데모를 불러오지 못했어요.</strong> 웹 서버로 열어야 해요 (예: python3 -m http.server).", true);
  }
}

init();
