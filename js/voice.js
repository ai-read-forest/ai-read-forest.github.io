// Tiny voice classifier (male / female voice), run in plain JavaScript so every layer can be drawn.
// Weights: Hugging Face model repo below (trained on LibriSpeech by tools/voice_gender/train.py).
// Input: the page's own low-band spectrogram (js/dsp.js, BANDS.low, "power" mode), 96 x 511.

export const VOICE_REPO = "hojinlab/ai-read-forest-voice-cnn";
const BASE = `https://huggingface.co/${VOICE_REPO}/resolve/main/`;

let loading = null;

function parseSafetensors(buf) {
  const dv = new DataView(buf);
  const n = Number(dv.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, n)));
  const out = {};
  for (const [k, v] of Object.entries(header)) {
    if (k === "__metadata__") continue;
    if (v.dtype !== "F32") throw new Error(`unexpected dtype ${v.dtype}`);
    const [a, b] = v.data_offsets;
    out[k] = { shape: v.shape, data: new Float32Array(buf.slice(8 + n + a, 8 + n + b)) };
  }
  return out;
}

export function modelFromBuffers(configJson, safetensorsBuf) {
  return { cfg: configJson, w: parseSafetensors(safetensorsBuf) };
}

export function loadVoiceModel(base = BASE) {
  loading ??= (async () => {
    const [cfg, buf] = await Promise.all([
      fetch(`${base}config.json`).then((r) => {
        if (!r.ok) throw new Error(`config ${r.status}`);
        return r.json();
      }),
      fetch(`${base}model.safetensors`).then((r) => {
        if (!r.ok) throw new Error(`weights ${r.status}`);
        return r.arrayBuffer();
      }),
    ]);
    return modelFromBuffers(cfg, buf);
  })();
  loading.catch(() => {
    loading = null;
  });
  return loading;
}

function samePad(n, k, s) {
  const out = Math.ceil(n / s);
  const total = Math.max((out - 1) * s + k - n, 0);
  return { out, before: Math.floor(total / 2) };
}

// x: Float32Array [C, H, W]; w: [O, C, kh, kw]; returns ReLU(conv + b) as Float32Array [O, Ho, Wo]
function convRelu(x, C, H, W, weight, bias, stride) {
  const [O, , kh, kw] = weight.shape;
  const wd = weight.data;
  const r = samePad(H, kh, stride);
  const c = samePad(W, kw, stride);
  const Ho = r.out;
  const Wo = c.out;
  const y = new Float32Array(O * Ho * Wo);
  for (let o = 0; o < O; o++) {
    const b = bias.data[o];
    for (let i = 0; i < Ho; i++) {
      for (let j = 0; j < Wo; j++) {
        let s = b;
        for (let ch = 0; ch < C; ch++) {
          const xoff = ch * H * W;
          const woff = ((o * C + ch) * kh) * kw;
          for (let a = 0; a < kh; a++) {
            const yy = i * stride - r.before + a;
            if (yy < 0 || yy >= H) continue;
            for (let bb = 0; bb < kw; bb++) {
              const xx = j * stride - c.before + bb;
              if (xx < 0 || xx >= W) continue;
              s += x[xoff + yy * W + xx] * wd[woff + a * kw + bb];
            }
          }
        }
        y[(o * Ho + i) * Wo + j] = s > 0 ? s : 0;
      }
    }
  }
  return { data: y, C: O, H: Ho, W: Wo };
}

// spec: { data: Float32Array(rows * cols), rows, cols } (row 0 = highest band)
export function runVoice(model, spec) {
  const { data, rows, cols } = spec;
  let m = 0;
  for (const v of data) m += v;
  m /= data.length;
  let v2 = 0;
  for (const v of data) v2 += (v - m) ** 2;
  const sd = Math.sqrt(v2 / data.length) + 1e-5;
  let x = { data: Float32Array.from(data, (v) => (v - m) / sd), C: 1, H: rows, W: cols };
  const acts = [];
  model.cfg.layers.forEach((L, i) => {
    x = convRelu(x.data, x.C, x.H, x.W, model.w[`conv${i + 1}.weight`], model.w[`conv${i + 1}.bias`], L.stride);
    acts.push(x);
  });
  const vec = new Float32Array(x.C);
  const hw = x.H * x.W;
  for (let c = 0; c < x.C; c++) {
    let s = 0;
    for (let k = 0; k < hw; k++) s += x.data[c * hw + k];
    vec[c] = s / hw;
  }
  let logit = model.w["fc.bias"].data[0];
  for (let c = 0; c < vec.length; c++) logit += model.w["fc.weight"].data[c] * vec[c];
  const pMale = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, logit))));
  return { acts, vec, logit, pMale, kernels: model.w["conv1.weight"] };
}
