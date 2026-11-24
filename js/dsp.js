// BirdNET V2.4 front end, re-implemented for the browser.
// Parameters come from the V2.4 docs and were checked against the TFLite model graph
// (MEL_SPEC1 / MEL_SPEC2 layers): see tools/check_dsp.mjs.
//
//   3 s at 48 kHz = 144,000 samples, min-max scaled to [-1, 1]
//   low  band: frame 2048, hop 278, 96 mel bands, 0-3000 Hz,   magnitude^2 -> ^0.2295
//   high band: frame 1024, hop 280, 96 mel bands, 500-15000 Hz, magnitude^2 -> ^0.1905
//   periodic Hann window, no padding -> 511 frames each, image rows flipped (row 0 = high)

export const SR = 48000;
export const N = 144000;

export const BANDS = {
  high: { nfft: 1024, hop: 280, nmel: 96, fmin: 500, fmax: 15000, exponent: 0.1905273 },
  low: { nfft: 2048, hop: 278, nmel: 96, fmin: 0, fmax: 3000, exponent: 0.22952409 },
};

// BirdNET scales each 3-s chunk by its own min and max.
export function normalize(x) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of x) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const out = new Float32Array(x.length);
  const scale = 1 / (hi - lo + 1e-6);
  for (let i = 0; i < x.length; i++) out[i] = ((x[i] - lo) * scale - 0.5) * 2;
  return out;
}

export function hann(n) {
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n); // periodic
  return w;
}

// In-place iterative radix-2 FFT on (re, im).
function fftInPlace(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wr = Math.cos(ang);
    const wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1;
      let ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k;
        const b = a + len / 2;
        const tr = re[b] * cr - im[b] * ci;
        const ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr;
        im[b] = im[a] - ti;
        re[a] += tr;
        im[a] += ti;
        const t = cr * wr - ci * wi;
        ci = cr * wi + ci * wr;
        cr = t;
      }
    }
  }
}

// FFT of frame x[start .. start+nfft) times window w; returns bins k = 0..nfft/2.
export function fftFrame(x, start, w) {
  const n = w.length;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = x[start + i] * w[i];
  fftInPlace(re, im);
  return { re: re.subarray(0, n / 2 + 1), im: im.subarray(0, n / 2 + 1) };
}

// Power spectrum |X_k|^2.
export function powerFrame(x, start, w) {
  const { re, im } = fftFrame(x, start, w);
  const p = new Float64Array(re.length);
  for (let k = 0; k < re.length; k++) p[k] = re[k] * re[k] + im[k] * im[k];
  return p;
}

// tf.signal.linear_to_mel_weight_matrix (HTK mel, DC row zero), as rows [mel][bin].
export function melMatrix({ nfft, nmel, fmin, fmax }) {
  const nbins = nfft / 2 + 1;
  const hz2mel = (f) => 1127 * Math.log1p(f / 700);
  const lo = hz2mel(fmin);
  const hi = hz2mel(fmax);
  const edges = Array.from({ length: nmel + 2 }, (_, i) => lo + ((hi - lo) * i) / (nmel + 1));
  const m = Array.from({ length: nmel }, () => new Float64Array(nbins));
  for (let k = 1; k < nbins; k++) {
    const mk = hz2mel((k * SR) / 2 / (nbins - 1));
    for (let b = 0; b < nmel; b++) {
      const up = (mk - edges[b]) / (edges[b + 1] - edges[b]);
      const down = (edges[b + 2] - mk) / (edges[b + 2] - edges[b + 1]);
      m[b][k] = Math.max(0, Math.min(up, down));
    }
  }
  return m;
}

// Centre frequency of each mel band (Hz), bottom band first.
export function melCenters({ nmel, fmin, fmax }) {
  const hz2mel = (f) => 1127 * Math.log1p(f / 700);
  const mel2hz = (m) => 700 * Math.expm1(m / 1127);
  const lo = hz2mel(fmin);
  const hi = hz2mel(fmax);
  return Array.from({ length: nmel }, (_, b) => mel2hz(lo + ((hi - lo) * (b + 1)) / (nmel + 1)));
}

export function melColumn(power, mel, exponent) {
  const col = new Float32Array(mel.length);
  for (let b = 0; b < mel.length; b++) {
    let s = 0;
    const row = mel[b];
    for (let k = 0; k < row.length; k++) if (row[k]) s += row[k] * power[k];
    col[b] = Math.pow(s, exponent);
  }
  return col;
}

// Returns { data: Float32Array(nmel * frames), rows: nmel, cols: frames }, row 0 = highest band.
// mode "power" (shown on the page): (mel . |X|^2)^e, the textbook mel spectrogram.
// mode "model": what the V2.4 TFLite graph computes. Its tf.cast(complex -> float32)
// keeps only the real part, so it is ((mel . Re X)^2)^e. Used to check against the model.
export function spectrogram(xn, band, mode = "power") {
  const { nfft, hop, nmel, exponent } = band;
  const w = hann(nfft);
  const mel = melMatrix(band);
  const frames = 1 + Math.floor((xn.length - nfft) / hop);
  const data = new Float32Array(nmel * frames);
  for (let t = 0; t < frames; t++) {
    let col;
    if (mode === "model") {
      const { re } = fftFrame(xn, t * hop, w);
      col = new Float32Array(nmel);
      for (let b = 0; b < nmel; b++) {
        let s = 0;
        for (let k = 0; k < re.length; k++) if (mel[b][k]) s += mel[b][k] * re[k];
        col[b] = Math.pow(s * s, exponent);
      }
    } else {
      col = melColumn(powerFrame(xn, t * hop, w), mel, exponent);
    }
    for (let b = 0; b < nmel; b++) data[(nmel - 1 - b) * frames + t] = col[b];
  }
  return { data, rows: nmel, cols: frames };
}

// --- Example convolution (NOT BirdNET's learned filters) --------------------------------
// Kernels are 4 x 8 (frequency x time) like BirdNET V2.4's first conv layer; stride 2,
// "same" padding as in TensorFlow, so a 96 x 511 input gives a 48 x 256 map.
export const EXAMPLE_KERNELS = {
  flat: {
    name: "가로선",
    hint: "일정한 높이로 이어지는 소리",
    w: [
      [-1, -1, -1, -1, -1, -1, -1, -1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [1, 1, 1, 1, 1, 1, 1, 1],
      [-1, -1, -1, -1, -1, -1, -1, -1],
    ],
  },
  rise: {
    name: "올라가는 선",
    hint: "음이 점점 높아지는 소리",
    w: [
      [-1, -1, -1, -1, -1, -0.5, 1.5, 1.5],
      [-1, -1, -1, -0.5, 1.5, 1.5, -0.5, -1],
      [-1, -0.5, 1.5, 1.5, -0.5, -1, -1, -1],
      [1.5, 1.5, -0.5, -1, -1, -1, -1, -1],
    ],
  },
  fall: {
    name: "내려가는 선",
    hint: "음이 점점 낮아지는 소리",
    w: [
      [1.5, 1.5, -0.5, -1, -1, -1, -1, -1],
      [-1, -0.5, 1.5, 1.5, -0.5, -1, -1, -1],
      [-1, -1, -1, -0.5, 1.5, 1.5, -0.5, -1],
      [-1, -1, -1, -1, -1, -0.5, 1.5, 1.5],
    ],
  },
};

// Make each example kernel zero-mean (so flat regions give 0, not a response to loudness)
// and scale it to max |w| = 1.
for (const K of Object.values(EXAMPLE_KERNELS)) {
  const flat = K.w.flat();
  const mean = flat.reduce((a, v) => a + v, 0) / flat.length;
  const m = Math.max(...flat.map((v) => Math.abs(v - mean)));
  K.w = K.w.map((row) => row.map((v) => (v - mean) / m));
}

// Zero-mean, unit-variance copy of a spectrogram (stands in for BirdNET's batch norm).
export function standardize(spec) {
  const { data } = spec;
  let m = 0;
  for (const v of data) m += v;
  m /= data.length;
  let s = 0;
  for (const v of data) s += (v - m) ** 2;
  s = Math.sqrt(s / data.length) + 1e-9;
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = (data[i] - m) / s;
  return { data: out, rows: spec.rows, cols: spec.cols };
}

// TF "same" padding for stride 2: pad_before = floor(total / 2).
function samePad(n, k, s) {
  const out = Math.ceil(n / s);
  const total = Math.max((out - 1) * s + k - n, 0);
  return { out, before: Math.floor(total / 2) };
}

// Returns pre-activation sums and the ReLU map, both rows x cols = 48 x 256 for 96 x 511.
export function conv2dStride2(spec, kernel) {
  const kh = kernel.length;
  const kw = kernel[0].length;
  const r = samePad(spec.rows, kh, 2);
  const c = samePad(spec.cols, kw, 2);
  const sum = new Float32Array(r.out * c.out);
  for (let i = 0; i < r.out; i++) {
    for (let j = 0; j < c.out; j++) {
      let s = 0;
      for (let a = 0; a < kh; a++) {
        const y = i * 2 - r.before + a;
        if (y < 0 || y >= spec.rows) continue;
        for (let b = 0; b < kw; b++) {
          const x = j * 2 - c.before + b;
          if (x < 0 || x >= spec.cols) continue;
          s += spec.data[y * spec.cols + x] * kernel[a][b];
        }
      }
      sum[i * c.out + j] = s;
    }
  }
  const relu = sum.map((v) => (v > 0 ? v : 0));
  return { sum, relu, rows: r.out, cols: c.out, padTop: r.before, padLeft: c.before };
}

export function sigmoid(z) {
  const c = Math.max(-15, Math.min(15, z));
  return 1 / (1 + Math.exp(-c));
}

// Parse a 16-bit PCM mono WAV (the demo clip) without resampling.
export function parseWav16(buf) {
  const dv = new DataView(buf);
  let p = 12;
  let rate = 0;
  let ch = 1;
  while (p < dv.byteLength - 8) {
    const id = String.fromCharCode(dv.getUint8(p), dv.getUint8(p + 1), dv.getUint8(p + 2), dv.getUint8(p + 3));
    const size = dv.getUint32(p + 4, true);
    if (id === "fmt ") {
      ch = dv.getUint16(p + 10, true);
      rate = dv.getUint32(p + 12, true);
    } else if (id === "data") {
      const n = size / 2 / ch;
      const x = new Float32Array(n);
      for (let i = 0; i < n; i++) x[i] = dv.getInt16(p + 8 + i * 2 * ch, true) / 32768;
      return { rate, samples: x };
    }
    p += 8 + size + (size % 2);
  }
  throw new Error("WAV data chunk not found");
}
