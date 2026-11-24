// Check js/voice.js (+ js/dsp.js features) against PyTorch outputs written by
// tools/voice_gender/train.py: spectrogram, first conv layer and logit for one female and one male
// held-out clip. Usage: node tools/check_voice.mjs <train.py out dir>
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { BANDS, normalize, spectrogram } from "../js/dsp.js";
import { modelFromBuffers, runVoice } from "../js/voice.js";

const dir = process.argv[2];
const f32 = (name) => {
  const b = readFileSync(join(dir, name));
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};
const rel = (a, b) => {
  let n = 0;
  let d = 0;
  for (let i = 0; i < a.length; i++) {
    n += (a[i] - b[i]) ** 2;
    d += b[i] ** 2;
  }
  return Math.sqrt(n / d);
};

const buf = readFileSync(join(dir, "model.safetensors"));
const model = modelFromBuffers(JSON.parse(readFileSync(join(dir, "config.json"), "utf8")), buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const refs = JSON.parse(readFileSync(join(dir, "refs.json"), "utf8"));
let ok = true;
for (const name of ["female", "male"]) {
  const spec = spectrogram(normalize(f32(`ref_${name}_wave.f32`)), BANDS.low);
  const eSpec = rel(spec.data, f32(`ref_${name}_spec.f32`));
  const out = runVoice(model, spec);
  const eConv = rel(out.acts[0].data, f32(`ref_${name}_conv1.f32`));
  const dLogit = Math.abs(out.logit - refs[name].logit);
  console.log(`${name}: spec rel ${eSpec.toExponential(2)}, conv1 rel ${eConv.toExponential(2)}, logit js ${out.logit.toFixed(4)} vs torch ${refs[name].logit.toFixed(4)}, P(male) ${out.pMale.toFixed(3)}`);
  ok &&= eSpec < 1e-4 && eConv < 1e-3 && dLogit < 1e-3;
}
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
