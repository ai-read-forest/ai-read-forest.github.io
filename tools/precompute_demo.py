"""Precompute the real BirdNET V2.4 output for the page's demo clip.

Cuts the 3-s demo window out of the team's Kookmin recording, runs the BirdNET V2.4
acoustic model (TFLite FP32, as downloaded by the `birdnet` package) on it, and writes
what the page shows for the demo clip:

  assets/audio/demo-kookmin-bulbul-3s.wav   3 s, 48 kHz, 16-bit mono
  assets/data/demo-birdnet.json             logits, scores, 1024-d embedding, 3 first-layer maps

The model weights are not part of this repository (CC BY-NC-SA 4.0). The script reads the
copy the `birdnet` package keeps in ~/.local/share/birdnet. It also cross-checks the scores
against the `birdnet` package's own predict() so the page shows what BirdNET itself reports.

Usage (from ~/korea-bird-sounds, whose venv has birdnet==1.4.0):
    .venv/bin/python ~/ai-read-forest.github.io/tools/precompute_demo.py <src wav> <repo dir> [<debug dir>]
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = Path.home() / ".local/share/birdnet/acoustic-models/v2.4/tf/model-fp32.tflite"
LABELS = MODEL.parent / "labels"
SR, N = 48000, 144000
START_S = 2.0  # 2.0-5.0 s of the 7-s clip (29.5-36.5 s of the original recording)
CONV0_SHOW = (15, 10, 21)  # first-layer channels drawn on the page (picked by eye: smooth energy, edges, clicks)
TOP_K = 12


def sigmoid(z: np.ndarray) -> np.ndarray:
    # birdnet's flat_sigmoid with sensitivity 1.0: clip logits to +-15, then a plain sigmoid
    return 1.0 / (1.0 + np.exp(-np.clip(z, -15.0, 15.0)))


def run_model(seg: np.ndarray) -> dict[str, np.ndarray]:
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(str(MODEL), experimental_preserve_all_tensors=True)
    it.allocate_tensors()
    it.set_tensor(it.get_input_details()[0]["index"], seg[None, :].astype(np.float32))
    it.invoke()
    idx = {d["name"]: d["index"] for d in it.get_tensor_details()}
    get = lambda name: it.get_tensor(idx[name])  # noqa: E731
    return {
        "mel_low": get("model/MEL_SPEC1/ExpandDims")[0, :, :, 0],  # 96 x 511, row 0 = high
        "mel_high": get("model/MEL_SPEC2/ExpandDims")[0, :, :, 0],
        "conv0": get("model/ACT_0/Relu;model/BNORM_0/FusedBatchNormV3;model/CONV_0/Conv2D")[0],
        "embedding": get("model/GLOBAL_AVG_POOL/Mean")[0],
        "logits": it.get_tensor(it.get_output_details()[0]["index"])[0],
    }


def package_scores(wav: Path) -> dict[str, float]:
    import birdnet

    model = birdnet.load("acoustic", "2.4", "tf", library="litert", lang="en_us")
    df = model.predict(wav, default_confidence_threshold=0.01).to_dataframe()
    df["species_name"] = df["species_name"].astype(str)
    return {r.species_name.split("_")[0]: float(r.confidence) for r in df.itertuples()}


def u8_b64(a: np.ndarray) -> str:
    a = np.clip(a / (np.percentile(a, 99.5) + 1e-12), 0, 1)  # a few hot pixels should not wash out the map
    return base64.b64encode(np.round(a * 255).astype(np.uint8).tobytes()).decode()


def main(src: Path, repo: Path, debug: Path | None) -> None:
    x, sr = sf.read(src, dtype="float32")
    assert sr == SR and x.ndim == 1, (sr, x.shape)
    seg = x[int(START_S * SR):int(START_S * SR) + N]
    wav = repo / "assets/audio/demo-kookmin-bulbul-3s.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav, seg, SR, subtype="PCM_16")
    seg = sf.read(wav, dtype="float32")[0]  # use exactly the samples the page will load

    out = run_model(seg)
    z = out["logits"].astype(np.float64)
    s = sigmoid(z)
    sci = [line.split("_", 1)[0] for line in (LABELS / "en_us.txt").read_text().splitlines()]
    en = [line.split("_", 1)[1] for line in (LABELS / "en_us.txt").read_text().splitlines()]
    ko = [line.split("_", 1)[1] for line in (LABELS / "ko.txt").read_text().splitlines()]
    order = np.argsort(-z)

    pkg = package_scores(wav)
    print("birdnet.predict (>= 0.01):", {k: round(v, 4) for k, v in pkg.items()})
    for i in order[:TOP_K]:
        if s[i] >= 0.011:  # every class the package reports must match ours
            assert abs(pkg[sci[i]] - s[i]) < 1e-3, (sci[i], pkg.get(sci[i]), s[i])

    top = [{"rank": r + 1, "index": int(i), "sci": sci[i], "en": en[i],
            "ko": ko[i] if ko[i] != en[i] else None,
            "logit": round(float(z[i]), 4), "score": round(float(s[i]), 4)}
           for r, i in enumerate(order[:TOP_K])]
    data = {
        "model": "BirdNET V2.4 acoustic (TFLite FP32), birdnet==1.4.0, sigmoid sensitivity 1.0, no location filter",
        "clip": {"file": wav.name, "source": src.name, "start_s_in_source": START_S, "dur_s": 3.0,
                 "recorded": "2026-11-09, Kookmin University, Seoul (team recording)"},
        "n_classes": int(z.size),
        "top": top,
        "score_sum": round(float(s.sum()), 3),
        "n_above": {str(t): int((s >= t).sum()) for t in (0.1, 0.25, 0.5)},
        # all logits x100 as int16 (for the 6,522-dot strip)
        "logits_x100_b64": base64.b64encode(np.round(z * 100).astype("<i2").tobytes()).decode(),
        "embedding": [round(float(v), 3) for v in out["embedding"]],
        "conv0": {"shape": list(out["conv0"].shape[:2]), "channels": list(CONV0_SHOW),
                  "maps_u8_b64": [u8_b64(out["conv0"][:, :, c]) for c in CONV0_SHOW]},
        "shapes": [
            ["입력 파형", [144000]], ["스펙트로그램 2장", [96, 511, 2]], ["첫 conv 4×8, 24개", [48, 256, 24]],
            ["pooling", [48, 128, 24]], ["블록 1 (3개)", [24, 64, 36]], ["블록 2 (4개)", [12, 32, 72]],
            ["블록 3 (5개)", [6, 16, 108]], ["블록 4 (4개)", [3, 8, 192]], ["3×3 conv", [1, 6, 1024]],
            ["평균 pooling", [1024]], ["dense", [6522]],
        ],
    }
    out_json = repo / "assets/data/demo-birdnet.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {wav} ({wav.stat().st_size} B), {out_json} ({out_json.stat().st_size} B)")
    for t in top[:5]:
        print(f"  {t['rank']}. {t['sci']} / {t['ko'] or t['en']}  logit {t['logit']:+.3f}  score {t['score']:.3f}")
    print("score sum", data["score_sum"], "n_above", data["n_above"])

    if debug:
        debug.mkdir(parents=True, exist_ok=True)
        for k in ("mel_low", "mel_high", "conv0"):
            np.save(debug / f"{k}.npy", out[k])
        np.save(debug / "seg.npy", seg)


if __name__ == "__main__":  # required: BirdNET starts worker processes with spawn
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
