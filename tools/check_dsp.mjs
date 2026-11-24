// Compare js/dsp.js with numpy (FFT power per frame) and, in "model" mode, with BirdNET
// V2.4's own mel spectrogram tensors for the demo clip. Run tools/make_dsp_refs.py first.
// Usage: node tools/check_dsp.mjs <debug dir>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { BANDS, hann, normalize, powerFrame, spectrogram } from "../js/dsp.js";

const dir = process.argv[2];
const f32 = (name) => {
  const b = readFileSync(join(dir, name));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const relErr = (a, b) => {
  let num = 0, den = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - b[i]) ** 2;
    den += b[i] ** 2;
    max = Math.max(max, Math.abs(a[i] - b[i]));
  }
  return { rel: Math.sqrt(num / den), maxAbs: max };
};

const xn = normalize(f32("seg.f32"));
let ok = true;
for (const name of ["high", "low"]) {
  const band = BANDS[name];
  const w = hann(band.nfft);
  for (const t of [0, 255, 510]) {
    const e = relErr(powerFrame(xn, t * band.hop, w), f32(`np_power_${name}_${t}.f32`));
    console.log(`FFT power  ${name} frame ${t}: rel L2 ${e.rel.toExponential(2)}`);
    ok &&= e.rel < 1e-4;
  }
  const s = spectrogram(xn, band, "model");
  const ref = f32(`mel_${name}.f32`);
  const e = relErr(s.data, ref);
  let refMax = 0;
  for (const v of ref) refMax = Math.max(refMax, v);
  console.log(`mel ${name} ${s.rows}x${s.cols} vs BirdNET tensor: rel L2 ${e.rel.toExponential(2)}, max abs ${e.maxAbs.toExponential(2)} (ref max ${refMax.toFixed(3)})`);
  ok &&= s.rows === 96 && s.cols === 511 && e.rel < 1e-3;
  // The page draws the textbook version; report how far it is from the model's input.
  const p = spectrogram(xn, band, "power");
  let num = 0, den = 0, corr = 0, pa = 0, pb = 0;
  const ma = p.data.reduce((a, v) => a + v, 0) / p.data.length;
  const mb = ref.reduce((a, v) => a + v, 0) / ref.length;
  for (let i = 0; i < ref.length; i++) {
    corr += (p.data[i] - ma) * (ref[i] - mb);
    pa += (p.data[i] - ma) ** 2;
    pb += (ref[i] - mb) ** 2;
  }
  console.log(`  page (|X|^2) version vs BirdNET tensor: Pearson r ${(corr / Math.sqrt(pa * pb)).toFixed(3)}`);
}
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
