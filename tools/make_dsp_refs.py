"""Write reference arrays for tools/check_dsp.mjs.

Inputs: the debug dir written by precompute_demo.py (seg.npy = the 3-s demo samples,
mel_low.npy / mel_high.npy = BirdNET V2.4's own MEL_SPEC1 / MEL_SPEC2 tensors).
Adds numpy references for one FFT frame per band. All outputs are little-endian float32.

Usage: python tools/make_dsp_refs.py <debug dir>
"""
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
seg = np.load(d / "seg.npy").astype(np.float64)
xn = ((seg - seg.min()) / (seg.max() - seg.min() + 1e-6) - 0.5) * 2
for name, nfft, hop in (("high", 1024, 280), ("low", 2048, 278)):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(nfft) / nfft)
    for t in (0, 255, 510):
        p = np.abs(np.fft.rfft(xn[t * hop:t * hop + nfft] * w)) ** 2
        p.astype("<f4").tofile(d / f"np_power_{name}_{t}.f32")
for k in ("mel_low", "mel_high"):
    np.load(d / f"{k}.npy").astype("<f4").tofile(d / f"{k}.f32")
seg.astype("<f4").tofile(d / "seg.f32")
print("refs written to", d)
