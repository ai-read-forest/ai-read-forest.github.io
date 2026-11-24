"""Write the Hugging Face model card (README.md) from train.py's config.json.

Usage: python tools/voice_gender/make_card.py <train.py out dir>
"""

import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
cfg = json.loads((out / "config.json").read_text())
refs = json.loads((out / "refs.json").read_text())
te = cfg["metrics"]["test_test_clean"]
va = cfg["metrics"]["val_dev_clean"]
d = cfg["data"]
size_kb = (out / "model.safetensors").stat().st_size / 1024
pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731

card = f"""---
license: cc-by-4.0
language:
- en
tags:
- audio
- audio-classification
- spectrogram
- education
datasets:
- openslr/librispeech_asr
pipeline_tag: audio-classification
---

# ai-read-forest-voice-cnn

A **tiny** CNN ({cfg['params']:,} parameters, {size_kb:.0f} KB) that guesses whether a 3-second
recording sounds like a **male or female voice**. It was made for a non-commercial university
course page ("AI로 읽는 숲", 상어 팀: <https://ai-read-forest.github.io/>) that explains, step by
step, how BirdNET turns sound into a label. Visitors record their own voice and watch the same
steps (waveform → spectrogram → convolution → vector → sigmoid) run on it in the browser.

이 모델은 수업 페이지에서 "BirdNET과 같은 원리로 내 목소리를 분류해 보는" 체험용이다.
목소리 높낮이 같은 특징으로 추정하므로 틀릴 수 있고, 한국어 목소리로는 시험하지 않았다.

## Input

The model reads the page's own spectrogram, computed in JavaScript:
BirdNET V2.4's low band — 3 s at 48 kHz scaled to −1…1, periodic Hann window of 2048 samples,
hop 278, power |X|², 96 mel bands 0–3000 Hz (TF / HTK mel), raised to the power
{cfg['input']['exponent']}, row 0 = highest band → a 96 × 511 image, standardised per clip.

## Architecture

| Layer | Output |
| --- | --- |
| input spectrogram | 1 × 96 × 511 |
| conv 4×8, stride 2, ReLU | 8 × 48 × 256 |
| conv 3×3, stride 2, ReLU | 16 × 24 × 128 |
| conv 3×3, stride 2, ReLU | 32 × 12 × 64 |
| conv 3×3, stride 2, ReLU | 32 × 6 × 32 |
| global average pooling | 32 |
| linear + sigmoid | P(male voice) |

BatchNorm is folded into the convolutions in `model.safetensors` (float32). `config.json` lists the
exact feature and layer settings. The page runs the network in plain JavaScript
(`js/voice.js` in the site repository) so every layer can be drawn.

## Training data

[LibriSpeech](https://www.openslr.org/12) (Panayotov et al., 2015, CC BY 4.0), read English
audiobooks, via the Hugging Face copy `openslr/librispeech_asr`. The reader's sex comes from the
corpus' `SPEAKERS.TXT` (labels F / M). Speakers never overlap between splits.

| Split | Subset | Speakers (F / M) | Utterances |
| --- | --- | --- | --- |
| train | train-clean-100 (≤ 40 utterances per reader) | {d['train']['speakers']} ({d['train']['female']} / {d['train']['male']}) | {d['train']['utterances']:,} |
| validation (model selection) | dev-clean | {d['val']['speakers']} ({d['val']['female']} / {d['val']['male']}) | {d['val']['utterances']:,} |
| test (held out) | test-clean | {d['test']['speakers']} ({d['test']['female']} / {d['test']['male']}) | {d['test']['utterances']:,} |

Training used random 3-s crops with added noise (5–40 dB SNR, half of the crops), random time
shifts and time masking.

## Results on held-out speakers (test-clean)

| Metric | Value |
| --- | --- |
| 3-s clip accuracy | {pct(te['clip_acc'])} ({te['n_clips']:,} clips) |
| female clips correct | {pct(te['clip_acc_female'])} |
| male clips correct | {pct(te['clip_acc_male'])} |
| speaker accuracy (majority of a speaker's clips) | {pct(te['speaker_acc'])} ({te['n_speakers']} speakers) |

Validation (dev-clean): clip {pct(va['clip_acc'])}, speaker {pct(va['speaker_acc'])}.

## Limitations

- Trained only on read English speech from audiobook recordings. **Not tested on Korean speakers**,
  children, singing, whispering, phone microphones in noisy rooms, or other languages.
- It guesses from acoustic cues such as voice pitch. Voices do not map neatly onto two classes, and
  the labels are the corpus' recorded F / M reader metadata. The output says what a voice *sounds
  like* to this small model, nothing about a person's gender identity. It can be wrong.
- Scores are not calibrated probabilities. Use it for the classroom demo it was built for, not to
  identify or profile people.
- Very quiet input is amplified by the per-clip −1…1 scaling; the page refuses to classify
  recordings quieter than about −50 dBFS.

## Files

- `model.safetensors` — folded float32 weights (`conv1`…`conv4`, `fc`)
- `config.json` — feature settings, layers, data counts and metrics
- `train.py` — the training script (PyTorch)

## Citation

Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). Librispeech: An ASR corpus based on
public domain audio books. *ICASSP 2015*, 5206–5210. https://doi.org/10.1109/ICASSP.2015.7178964
"""
(out / "README.md").write_text(card)
print(card[:400])
print("... written", out / "README.md")
print("refs", json.dumps({k: {"speaker": v["speaker"], "p_male_windows": v.get("p_male_per_1s_window")} for k, v in refs.items()}))
