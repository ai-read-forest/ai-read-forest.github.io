// Concept animation for the idea (not a tested system): a flock enters from the right
// calling (teal arcs), the microphone hears it and names a species, a speaker plays a
// predator call (indigo arcs) and the flock curves away from its planned path (dashed).

import { C, clamp, hash01, line, seg, text } from "./draw.js";

const LOOP = 13; // seconds
const T_LISTEN = 3.0;
const T_LABEL = 4.0;
const T_PLAY = 4.8;
const T_TURN = 5.7;
const N_BIRDS = 13;
const STILL = 6.9; // reduced motion: one frame that tells the whole story

export class Hero {
  constructor(canvas, stepsEl, { reduce = false } = {}) {
    this.cv = canvas;
    this.ctx = canvas.getContext("2d");
    this.steps = stepsEl ? [...stepsEl.children] : [];
    this.reduce = reduce;
    this.t0 = performance.now();
    this.visible = false;
    this.resize();
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    new IntersectionObserver(([e]) => {
      const was = this.visible;
      this.visible = e.isIntersecting;
      if (this.visible && !was) {
        if (!this.started) {
          this.started = true;
          this.t0 = performance.now();
        }
        this.loop();
      }
    }).observe(canvas);
  }

  resize() {
    const r = this.cv.parentElement.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.W = Math.max(1, r.width);
    this.H = Math.max(1, r.height);
    this.cv.width = Math.round(this.W * dpr);
    this.cv.height = Math.round(this.H * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.layout();
    this.draw(this.reduce ? STILL : this.time());
  }

  layout() {
    const { W, H } = this;
    this.s = clamp(W / 95, 6, 12); // bird half-span
    this.fs = clamp(W / 68, 11.5, 15); // font size
    this.mic = { x: W * 0.5, y: H * 0.64 };
    this.ground = H * 0.9;
    this.spk = { x: W * 0.5 + Math.max(34, W * 0.045), y: H * 0.8 };
    // flock centre path, integrated at 60 Hz
    const v0 = W * 0.088;
    const dt = 1 / 60;
    let x = W * 1.12;
    let y = H * 0.27;
    let th = Math.PI;
    this.path = [];
    for (let t = 0; t <= LOOP + 1; t += dt) {
      const k = seg(t, T_TURN, T_TURN + 2.4);
      th = Math.PI + k * 0.78 * Math.PI; // left -> up -> up-right
      const v = v0 * (1 + 0.9 * seg(t, T_TURN, T_TURN + 1.2));
      x += Math.cos(th) * v * dt;
      y += Math.sin(th) * v * dt;
      this.path.push([x, y]);
    }
    // loose cluster offsets
    this.birds = Array.from({ length: N_BIRDS }, (_, i) => {
      const a = hash01(i * 7 + 1);
      const b = hash01(i * 13 + 5);
      const ox = (a - 0.35) * Math.min(W * 0.2, 190);
      const oy = (b - 0.5) * Math.min(H * 0.22, 90);
      return { ox, oy, lag: Math.max(0, ox) / (W * 0.09), ph: hash01(i * 3 + 2) * 6.28 };
    });
    // which bird calls when
    this.calls = [];
    for (let t = 0.4, i = 0; t < 4.3; t += 0.38, i++) this.calls.push({ t, b: Math.floor(hash01(i * 17 + 3) * N_BIRDS) });
  }

  centre(t) {
    const i = clamp(Math.round(t * 60), 0, this.path.length - 1);
    return this.path[i];
  }

  birdPos(b, t) {
    const [x, y] = this.centre(t - b.lag);
    return [x + b.ox, y + b.oy + Math.sin(t * 1.7 + b.ph) * 3];
  }

  time() {
    return (performance.now() - this.t0) / 1000;
  }

  replay() {
    this.t0 = performance.now();
    this.started = true;
    if (this.reduce) this.reduceOnce = true;
    this.loop();
  }

  loop() {
    if (this.raf) return;
    const tick = () => {
      this.raf = null;
      let t = this.time();
      if (this.reduce && !this.reduceOnce) t = STILL;
      if (t > LOOP) {
        if (this.reduceOnce) {
          this.reduceOnce = false;
          t = STILL;
        } else {
          this.t0 = performance.now();
          t = 0;
        }
      }
      this.draw(t);
      if (this.visible && (!this.reduce || this.reduceOnce)) this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }

  draw(t) {
    const { ctx, W, H, mic, spk, fs } = this;
    ctx.clearRect(0, 0, W, H);
    const fade = 1 - seg(t, LOOP - 0.6, LOOP);
    ctx.save();
    ctx.globalAlpha = fade;

    // ground
    line(ctx, 0, this.ground, W, this.ground, { color: C.hair, width: 1 });

    // planned path (dashed) and travelled path
    const [cx0, cy0] = this.path[0];
    line(ctx, cx0, cy0, -20, cy0, { color: C.gray, width: 1.5, dash: [6, 6], alpha: 0.9 });
    const upto = clamp(Math.round(t * 60), 0, this.path.length - 1);
    if (t > T_TURN - 0.5) {
      ctx.save();
      ctx.strokeStyle = C.ink;
      ctx.globalAlpha *= 0.28;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i <= upto; i += 2) {
        const [x, y] = this.path[i];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // bird calls: teal arcs travelling toward the microphone
    const still = this.reduce && !this.reduceOnce;
    for (const c of this.calls) {
      // in the still frame, show a few calls where they were made, on the planned path
      const age = still ? (c.t > 1.4 && c.t < 3.6 ? 0.55 : -1) : t - c.t;
      if (age < 0 || age > 1.8) continue;
      const [bx, by] = this.birdPos(this.birds[c.b], c.t);
      const ang = Math.atan2(mic.y - by, mic.x - bx);
      this.arcs(bx, by, ang, 0.55, age, { color: C.green, speed: Math.min(W, 900) * 0.14, life: 1.8 });
    }

    // predator call: indigo arcs from the speaker toward the flock
    for (let k = 0; k < 4; k++) {
      const te = T_PLAY + k * 0.5;
      const age = t - te;
      if (age < 0 || age > 2.6) continue;
      const [fx, fy] = this.centre(te + 0.6);
      const ang = Math.atan2(fy - spk.y, fx - spk.x);
      this.arcs(spk.x + 10, spk.y - 8, ang, 0.7, age, { color: C.copper, speed: Math.min(W, 900) * 0.2, life: 2.6, width: 2.4 });
    }

    // birds
    for (const b of this.birds) {
      const [x, y] = this.birdPos(b, t);
      const flap = Math.sin(t * (t > T_TURN ? 15 : 9) + b.ph);
      this.bird(x, y, flap);
    }

    // microphone and speaker
    const listening = seg(t, T_LISTEN - 0.4, T_LISTEN) * (1 - seg(t, T_PLAY + 0.4, T_PLAY + 1));
    this.micGlyph(mic.x, mic.y, listening, t);
    this.speakerGlyph(spk.x, spk.y, seg(t, T_PLAY - 0.2, T_PLAY));

    // labels
    const la = seg(t, T_LABEL, T_LABEL + 0.5);
    const lx = mic.x - 22;
    text(ctx, "종 추정", lx, mic.y - 2, { size: fs * 0.85, color: C.muted, align: "right", alpha: la });
    text(ctx, "직박구리", lx, mic.y + fs * 1.15, { size: fs * 1.15, weight: 700, color: C.forest, align: "right", alpha: la });
    const pa = seg(t, T_PLAY, T_PLAY + 0.5);
    text(ctx, "천적 소리 재생", spk.x + 26, spk.y - 4, { size: fs * 0.85, color: C.muted, alpha: pa });
    text(ctx, "예: 맹금류", spk.x + 26, spk.y + fs * 1.1, { size: fs, weight: 700, color: C.copperInk, alpha: pa });
    const ma = seg(t, T_LISTEN - 0.4, T_LISTEN) * (1 - la);
    text(ctx, "듣는 중", lx, mic.y + fs * 0.4, { size: fs * 0.85, color: C.muted, align: "right", alpha: ma });
    text(ctx, "원래 가던 길", 12, cy0 - 8, { size: fs * 0.8, color: C.muted, alpha: seg(t, 0.6, 1.4) * 0.9 });
    ctx.restore();

    // step list under the canvas
    const stage = t < T_LISTEN ? 0 : t < T_PLAY ? 1 : t < T_TURN ? 2 : 3;
    this.steps.forEach((li, i) => li.classList.toggle("on", this.reduce && !this.reduceOnce ? true : i === stage));
  }

  arcs(x, y, ang, spread, age, { color, speed, life, width = 2 }) {
    const { ctx } = this;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    for (let k = 0; k < 3; k++) {
      const a = age - k * 0.14;
      if (a < 0) continue;
      const r = 5 + k * 2 + a * speed;
      ctx.globalAlpha = clamp(1 - a / life) * 0.9;
      ctx.beginPath();
      ctx.arc(x, y, r, ang - spread, ang + spread);
      ctx.stroke();
    }
    ctx.restore();
  }

  bird(x, y, flap) {
    const { ctx, s } = this;
    const tip = flap * s * 0.45;
    ctx.save();
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = Math.max(1.6, s * 0.2);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(x - s, y - tip);
    ctx.quadraticCurveTo(x - s * 0.45, y - s * 0.55 - tip * 0.3, x, y);
    ctx.quadraticCurveTo(x + s * 0.45, y - s * 0.55 - tip * 0.3, x + s, y - tip);
    ctx.stroke();
    ctx.restore();
  }

  micGlyph(x, y, glow, t) {
    const { ctx, ground } = this;
    const w = 12;
    const h = 22;
    if (glow > 0) {
      ctx.save();
      ctx.strokeStyle = C.green;
      ctx.lineWidth = 2.5;
      ctx.globalAlpha = glow * (0.55 + 0.45 * Math.sin(t * 8));
      ctx.beginPath();
      ctx.arc(x, y, 22, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
    ctx.save();
    ctx.strokeStyle = C.ink;
    ctx.fillStyle = C.bg;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(x - w / 2, y - h / 2, w, h, w / 2);
    ctx.fill();
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(x, y + 1, 11, 0.1 * Math.PI, 0.9 * Math.PI);
    ctx.moveTo(x, y + 12);
    ctx.lineTo(x, ground);
    ctx.stroke();
    ctx.restore();
  }

  speakerGlyph(x, y, on) {
    const { ctx, ground } = this;
    ctx.save();
    ctx.strokeStyle = on > 0 ? C.copper : C.ink;
    ctx.fillStyle = C.bg;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.rect(x - 6, y - 14, 10, 12);
    ctx.moveTo(x + 4, y - 14);
    ctx.lineTo(x + 14, y - 21);
    ctx.lineTo(x + 14, y + 5);
    ctx.lineTo(x + 4, y - 2);
    ctx.fill();
    ctx.stroke();
    ctx.strokeStyle = C.ink;
    ctx.beginPath();
    ctx.moveTo(x - 1, y - 2);
    ctx.lineTo(x - 1, ground);
    ctx.stroke();
    ctx.restore();
  }
}

