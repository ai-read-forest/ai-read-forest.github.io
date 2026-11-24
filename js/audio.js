// Microphone recording (3 s) and playback. Everything stays in the browser.

import { N, SR } from "./dsp.js";

let ctx = null;
let workletReady = null;

export function audioContext() {
  if (!ctx) {
    const AC = window.AudioContext || window.webkitAudioContext;
    ctx = new AC();
  }
  if (ctx.state === "suspended") ctx.resume();
  return ctx;
}

export function micSupport() {
  if (!window.isSecureContext) return "insecure";
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return "unsupported";
  return "ok";
}

// Records 3 s. onLive(buf, n, rate) is called as samples arrive (for the live waveform).
// Resolves to Float32Array(144000) at 48 kHz.
export async function record3s(onLive) {
  const ac = audioContext();
  await ac.resume();
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
  });
  const rate = ac.sampleRate;
  const need = Math.ceil(rate * (N / SR));
  const buf = new Float32Array(need);
  let n = 0;
  const src = ac.createMediaStreamSource(stream);
  const sink = ac.createGain();
  sink.gain.value = 0;
  sink.connect(ac.destination);

  let node;
  const done = new Promise((resolve) => {
    const push = (chunk) => {
      if (n >= need) return;
      const k = Math.min(chunk.length, need - n);
      buf.set(chunk.subarray(0, k), n);
      n += k;
      onLive?.(buf, n, rate);
      if (n >= need) resolve();
    };
    const useWorklet = async () => {
      workletReady ??= ac.audioWorklet.addModule(new URL("./recorder-worklet.js", import.meta.url));
      await workletReady;
      node = new AudioWorkletNode(ac, "recorder-tap");
      node.port.onmessage = (e) => push(e.data);
    };
    const useScript = () => {
      node = ac.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (e) => push(new Float32Array(e.inputBuffer.getChannelData(0)));
    };
    (ac.audioWorklet ? useWorklet().catch(useScript) : Promise.resolve(useScript())).then(() => {
      src.connect(node);
      node.connect(sink);
    });
  });

  await done;
  stream.getTracks().forEach((tr) => tr.stop());
  try {
    src.disconnect();
    node.disconnect();
    sink.disconnect();
  } catch {
    /* already disconnected */
  }
  return resampleTo48k(buf, rate);
}

async function resampleTo48k(x, rate) {
  if (rate === SR) return x.slice(0, N);
  const OAC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const off = new OAC(1, N, SR);
  const b = off.createBuffer(1, x.length, rate);
  b.copyToChannel(x, 0);
  const s = off.createBufferSource();
  s.buffer = b;
  s.connect(off.destination);
  s.start();
  const out = await off.startRendering();
  return out.getChannelData(0).slice(0, N);
}

let current = null;

// Plays samples at 48 kHz; returns { start (AudioContext time), dur }.
export function play(samples) {
  const ac = audioContext();
  stop();
  const b = ac.createBuffer(1, samples.length, SR);
  b.copyToChannel(samples, 0);
  const s = ac.createBufferSource();
  s.buffer = b;
  s.connect(ac.destination);
  s.start();
  current = s;
  return { start: ac.currentTime, dur: samples.length / SR };
}

export function stop() {
  if (current) {
    try {
      current.stop();
    } catch {
      /* not started */
    }
    current = null;
  }
}

export function now() {
  return ctx ? ctx.currentTime : 0;
}
