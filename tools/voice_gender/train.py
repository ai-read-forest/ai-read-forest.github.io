"""Train the page's tiny voice classifier (male / female voice) on LibriSpeech.

The model reads exactly the spectrogram the page draws for a microphone recording:
BirdNET V2.4's low band (3 s at 48 kHz scaled to -1..1, Hann 2048 / hop 278, |X|^2, 96 TF-style
mel bands 0-3000 Hz, ^0.2295, row 0 = high) -> 96 x 511. So the page's convolution, vector and
sigmoid steps can be drawn from this model's real layer outputs (js/voice.js re-runs it).

Data: LibriSpeech (CC BY 4.0) from the Hugging Face copy `openslr/librispeech_asr` (parquet) and
the corpus' SPEAKERS.TXT for the reader's sex. Speakers never overlap between train / val / test:
train = train-clean-100 (all 251 readers, at most 40 utterances each), val = dev-clean (model
selection), test = test-clean (held out, reported).

Usage:
    uv run --no-project --with torch --index https://download.pytorch.org/whl/cpu \
        --with pyarrow --with soundfile --with scipy --with safetensors --with numpy \
        python tools/voice_gender/train.py <librispeech dir> <out dir>
"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

SR, N = 48000, 144000
NFFT, HOP, NMEL, FMIN, FMAX, EXPONENT = 2048, 278, 96, 0.0, 3000.0, 0.22952409  # BirdNET V2.4 low band
FRAMES = 1 + (N - NFFT) // HOP  # 511
SEED = 7


# ----------------------------------------------------------------------------- features
def mel_matrix() -> np.ndarray:
    """tf.signal.linear_to_mel_weight_matrix (HTK mel, DC row zero) as [mel, bin]; same as js/dsp.js."""
    nbins = NFFT // 2 + 1
    hz2mel = lambda f: 1127.0 * np.log1p(np.asarray(f) / 700.0)  # noqa: E731
    lin = np.linspace(0, SR / 2, nbins)[1:]
    bm = hz2mel(lin)[:, None]
    edges = np.linspace(hz2mel(FMIN), hz2mel(FMAX), NMEL + 2)
    lo, c, hi = edges[:-2], edges[1:-1], edges[2:]
    w = np.maximum(0, np.minimum((bm - lo) / (c - lo), (hi - bm) / (hi - c)))
    return np.vstack([np.zeros((1, NMEL)), w]).T.astype(np.float32)


MEL = torch.from_numpy(mel_matrix())  # [96, 1025]
HANN = (0.5 - 0.5 * torch.cos(2 * torch.pi * torch.arange(NFFT) / NFFT)).float()  # periodic


def features(x48: torch.Tensor) -> torch.Tensor:
    """[B, 144000] float32 at 48 kHz -> [B, 96, 511], row 0 = highest band."""
    lo = x48.min(dim=1, keepdim=True).values
    hi = x48.max(dim=1, keepdim=True).values
    x = ((x48 - lo) / (hi - lo + 1e-6) - 0.5) * 2
    fr = x.unfold(1, NFFT, HOP) * HANN  # [B, 511, 2048]
    spec = torch.fft.rfft(fr.double(), dim=-1)
    power = (spec.real**2 + spec.imag**2).float()  # [B, 511, 1025]
    mel = power @ MEL.T  # [B, 511, 96]
    return mel.clamp_min(0).pow(EXPONENT).transpose(1, 2).flip(1).contiguous()


# ----------------------------------------------------------------------------- model
def same_pad(n: int, k: int, s: int) -> tuple[int, int]:
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2


LAYERS = [  # (in, out, (kh, kw), stride) ; first layer is 4 x 8 / stride 2 like BirdNET V2.4's CONV_0
    (1, 8, (4, 8), 2),
    (8, 16, (3, 3), 2),
    (16, 32, (3, 3), 2),
    (32, 32, (3, 3), 2),
]


class VoiceCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv2d(i, o, k, s, bias=False) for i, o, k, s in LAYERS)
        self.bns = nn.ModuleList(nn.BatchNorm2d(o) for _, o, _, _ in LAYERS)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(LAYERS[-1][1], 1)

    @staticmethod
    def standardize(s: torch.Tensor) -> torch.Tensor:
        m = s.mean(dim=(1, 2), keepdim=True)
        sd = s.std(dim=(1, 2), keepdim=True, unbiased=False)
        return (s - m) / (sd + 1e-5)

    def forward(self, spec: torch.Tensor, keep: bool = False):
        x = self.standardize(spec).unsqueeze(1)
        acts = []
        for conv, bn in zip(self.convs, self.bns):
            kh, kw = conv.kernel_size
            st = conv.stride[0]
            ph = same_pad(x.shape[2], kh, st)
            pw = same_pad(x.shape[3], kw, st)
            x = F.relu(bn(conv(F.pad(x, (pw[0], pw[1], ph[0], ph[1])))))
            acts.append(x)
        vec = x.mean(dim=(2, 3))
        logit = self.fc(self.drop(vec)).squeeze(1)
        return (logit, vec, acts) if keep else logit


def folded_state(model: VoiceCNN) -> dict[str, torch.Tensor]:
    """Fold each BatchNorm into its conv (inference form used by js/voice.js)."""
    out = {}
    for i, (conv, bn) in enumerate(zip(model.convs, model.bns), start=1):
        g = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        out[f"conv{i}.weight"] = (conv.weight * g[:, None, None, None]).detach().contiguous()
        out[f"conv{i}.bias"] = (bn.bias - bn.running_mean * g).detach().contiguous()
    out["fc.weight"] = model.fc.weight.detach().contiguous()
    out["fc.bias"] = model.fc.bias.detach().contiguous()
    return out


def folded_forward(st: dict[str, torch.Tensor], spec: torch.Tensor):
    x = VoiceCNN.standardize(spec).unsqueeze(1)
    acts = []
    for i, (_, _, k, s) in enumerate(LAYERS, start=1):
        ph, pw = same_pad(x.shape[2], k[0], s), same_pad(x.shape[3], k[1], s)
        x = F.relu(F.conv2d(F.pad(x, (pw[0], pw[1], ph[0], ph[1])), st[f"conv{i}.weight"], st[f"conv{i}.bias"], s))
        acts.append(x)
    vec = x.mean(dim=(2, 3))
    return (vec @ st["fc.weight"].T).squeeze(1) + st["fc.bias"], vec, acts


# ----------------------------------------------------------------------------- data
def read_sex(speakers_txt: Path) -> dict[int, int]:
    sex = {}
    for line in speakers_txt.read_text().splitlines():
        if line.startswith(";") or "|" not in line:
            continue
        sid, s = (p.strip() for p in line.split("|")[:2])
        sex[int(sid)] = 1 if s == "M" else 0
    return sex


def load_split(files: list[Path], sex: dict[int, int], per_speaker: int | None = None, seed: int = SEED):
    """Rows (speaker, sex, int16 audio at 16 kHz); optionally at most `per_speaker` utterances each."""
    rng = np.random.default_rng(seed)
    rows = []
    for f in files:
        spk_col = pq.read_table(f, columns=["speaker_id"]).column("speaker_id").to_numpy()
        keep = np.arange(len(spk_col))
        if per_speaker:
            keep = np.concatenate([rng.permutation(np.flatnonzero(spk_col == s))[:per_speaker] for s in np.unique(spk_col)])
            keep.sort()
        t = pq.read_table(f, columns=["audio", "speaker_id"]).take(keep)
        for a, spk in zip(t.column("audio").to_pylist(), t.column("speaker_id").to_pylist()):
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="int16")
            assert sr == 16000, sr
            rows.append((int(spk), sex[int(spk)], x))
    return rows


def crops(rows, per_utt: int, rng: np.random.Generator | None, noise: bool):
    """3-s crops at 16 kHz -> upsampled to 48 kHz (float32), labels, speaker ids."""
    n16 = N // 3
    xs, ys, spk = [], [], []
    for sid, y, x in rows:
        x = x.astype(np.float32) / 32768
        if len(x) < n16:
            x = np.pad(x, (0, n16 - len(x)))
        starts = (
            rng.integers(0, len(x) - n16 + 1, per_utt)
            if rng is not None
            else np.arange(0, max(1, len(x) - n16 + 1), n16)[:per_utt]
        )
        for s0 in starts:
            c = x[s0:s0 + n16]
            if noise and rng is not None and rng.random() < 0.5:
                snr = rng.uniform(5, 40)
                p = np.mean(c**2) + 1e-10
                c = c + rng.standard_normal(n16).astype(np.float32) * np.sqrt(p / 10 ** (snr / 10))
            xs.append(resample_poly(c, 3, 1).astype(np.float32)[:N])
            ys.append(y)
            spk.append(sid)
    return xs, np.array(ys, dtype=np.float32), np.array(spk)


def featurize(xs: list[np.ndarray], bs: int = 256) -> torch.Tensor:
    out = []
    for i in range(0, len(xs), bs):
        out.append(features(torch.from_numpy(np.stack(xs[i:i + bs]))).half())
    return torch.cat(out)


def crop_features(rows, per_utt: int, rng, noise: bool, chunk: int = 400):
    """Crops -> float16 features, a chunk of utterances at a time (waveforms are not kept)."""
    feats, ys, spk = [], [], []
    for i in range(0, len(rows), chunk):
        xs, y, s = crops(rows[i:i + chunk], per_utt, rng, noise)
        feats.append(featurize(xs))
        ys.append(y)
        spk.append(s)
    return torch.cat(feats), np.concatenate(ys), np.concatenate(spk)


def time_mask(spec: torch.Tensor, rng, n: int = 2, width: int = 40) -> torch.Tensor:
    spec = spec.clone()
    lo = spec.amin(dim=(1, 2), keepdim=True)
    for b in range(spec.shape[0]):
        for _ in range(n):
            w = int(rng.integers(0, width + 1))
            t0 = int(rng.integers(0, FRAMES - w + 1))
            spec[b, :, t0:t0 + w] = lo[b]
    return spec


@torch.no_grad()
def evaluate(model, feats, ys, spk):
    model.eval()
    logits = torch.cat([model(feats[i:i + 512].float()) for i in range(0, len(feats), 512)]).numpy()
    pred = (logits > 0).astype(np.float32)
    clip_acc = float((pred == ys).mean())
    per_spk = {}
    for s, p, y in zip(spk, pred, ys):
        per_spk.setdefault(s, [y, []])[1].append(p)
    spk_acc = float(np.mean([float((np.mean(v) > 0.5) == y) for y, v in per_spk.values()]))
    by_sex = {k: float((pred[ys == v] == v).mean()) for k, v in (("female", 0), ("male", 1))}
    return {"clip_acc": clip_acc, "speaker_acc": spk_acc, "n_clips": int(len(ys)), "n_speakers": len(per_spk), **{f"clip_acc_{k}": v for k, v in by_sex.items()}}


def main(data: Path, out: Path) -> None:
    torch.manual_seed(SEED)
    torch.set_num_threads(16)
    rng = np.random.default_rng(SEED)
    out.mkdir(parents=True, exist_ok=True)
    sex = read_sex(data / "meta/SPEAKERS.TXT")
    split = {
        "train": sorted((data / "all/train.clean.100").glob("*.parquet")),
        "val": sorted((data / "all/validation.clean").glob("*.parquet")),
        "test": sorted((data / "all/test.clean").glob("*.parquet")),
    }
    t0 = time.time()
    rows = {k: load_split(v, sex, per_speaker=40 if k == "train" else None) for k, v in split.items()}
    spk = {k: {r[0] for r in v} for k, v in rows.items()}
    assert not (spk["train"] & spk["val"] or spk["train"] & spk["test"] or spk["val"] & spk["test"]), "speaker overlap"
    info = {k: {"utterances": len(v), "speakers": len(spk[k]), "female": sum(1 for s in spk[k] if sex[s] == 0), "male": sum(1 for s in spk[k] if sex[s] == 1)} for k, v in rows.items()}
    print("loaded", json.dumps(info), f"{time.time() - t0:.0f}s", flush=True)

    ftr, ytr, _ = crop_features(rows["train"], 2, rng, noise=True)
    fva, yva, sva = crop_features(rows["val"], 3, None, noise=False)
    fte, yte, ste = crop_features(rows["test"], 3, None, noise=False)
    print("features", ftr.shape, fva.shape, fte.shape, f"{time.time() - t0:.0f}s", flush=True)

    model = VoiceCNN()
    print("params", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    epochs = 16
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * ((len(ftr) + 127) // 128))
    ytr_t = torch.from_numpy(ytr)
    best, best_state = -1.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ftr))
        tot = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            spec = ftr[idx].float()
            # random time shift (roll) as light augmentation
            spec = time_mask(torch.roll(spec, int(rng.integers(0, FRAMES)), dims=2), rng)
            loss = F.binary_cross_entropy_with_logits(model(spec), ytr_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(idx)
        val = evaluate(model, fva, yva, sva)
        print(f"epoch {ep + 1:2d} loss {tot / len(ftr):.4f} val clip {val['clip_acc']:.4f} speaker {val['speaker_acc']:.4f}", flush=True)
        if val["clip_acc"] > best:
            best, best_state = val["clip_acc"], {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    val, test = evaluate(model, fva, yva, sva), evaluate(model, fte, yte, ste)
    print("val", json.dumps(val), "\ntest", json.dumps(test), flush=True)

    # folded (BatchNorm-free) weights must give the same logits
    model.eval()
    st = folded_state(model)
    with torch.no_grad():
        ref = model(fte[:64].float())
        fl, _, _ = folded_forward(st, fte[:64].float())
    assert torch.allclose(ref, fl, atol=1e-4), (ref - fl).abs().max()

    from safetensors.torch import save_file

    save_file(st, out / "model.safetensors", metadata={"format": "pt", "positive_class": "male"})
    config = {
        "model": "VoiceCNN (tiny CNN, folded BatchNorm)",
        "positive_class": "male",
        "labels": {"0": "female", "1": "male"},
        "input": {"sample_rate": SR, "samples": N, "normalize": "min-max to [-1, 1] per 3-s clip",
                  "nfft": NFFT, "hop": HOP, "window": "periodic Hann", "power": "|X|^2",
                  "mel": {"bands": NMEL, "fmin": FMIN, "fmax": FMAX, "scale": "HTK (tf.signal.linear_to_mel_weight_matrix)"},
                  "exponent": EXPONENT, "shape": [NMEL, FRAMES], "row0": "highest band",
                  "standardize": "per clip: (s - mean) / (std + 1e-5)"},
        "layers": [{"in": i, "out": o, "kernel": list(k), "stride": s, "padding": "TF SAME", "activation": "relu"} for i, o, k, s in LAYERS],
        "head": "global average pooling -> linear(32, 1) -> sigmoid = P(male voice)",
        "data": info,
        "metrics": {"val_dev_clean": val, "test_test_clean": test},
        "params": int(sum(v.numel() for v in st.values())),
    }
    (out / "config.json").write_text(json.dumps(config, indent=1))

    # reference clips for tools/check_voice.mjs and the fake-microphone test: one random held-out
    # (test-clean) speaker per sex, seeded; 9 s of that speaker's speech for Chrome's fake mic
    pick = np.random.default_rng(SEED + 1)
    refs = {}
    for name, want in (("female", 0), ("male", 1)):
        spks = sorted({r[0] for r in rows["test"] if r[1] == want})
        sid = int(pick.choice(spks))
        utts = [r[2] for r in rows["test"] if r[0] == sid]
        speech = np.concatenate(utts).astype(np.float32) / 32768
        x16 = speech[: 9 * 16000]
        x48 = resample_poly(x16, 3, 1).astype(np.float32)
        x = x48[:N]
        logit, vec, acts = folded_forward(st, features(torch.from_numpy(x[None])))
        x.astype("<f4").tofile(out / f"ref_{name}_wave.f32")
        features(torch.from_numpy(x[None]))[0].numpy().astype("<f4").tofile(out / f"ref_{name}_spec.f32")
        acts[0][0].numpy().astype("<f4").tofile(out / f"ref_{name}_conv1.f32")
        # every 3-s window of the fake-mic file, for reporting
        wins = np.stack([x48[k * SR: k * SR + N] for k in range(0, 7)])
        wl = folded_forward(st, features(torch.from_numpy(wins)))[0]
        refs[name] = {"speaker": sid, "logit": float(logit[0]), "vec": [float(v) for v in vec[0]],
                      "p_male_per_1s_window": [round(float(torch.sigmoid(v)), 3) for v in wl]}
        sf.write(out / f"fake_mic_{name}_{sid}.wav", x48, SR, subtype="PCM_16")
    (out / "refs.json").write_text(json.dumps(refs, indent=1))
    print("refs", json.dumps({k: {"speaker": v["speaker"], "logit": round(v["logit"], 3), "p_male_windows": v["p_male_per_1s_window"]} for k, v in refs.items()}))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
