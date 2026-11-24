# ai-read-forest.github.io

"AI로 읽는 숲" 상어 팀 기말 과제 페이지: <https://ai-read-forest.github.io/>

3초의 소리가 파형 → FFT → 스펙트로그램 → 합성곱 신경망 → 벡터 → sigmoid 점수 → 결과로 바뀌는 과정을 전체 화면 슬라이드 11장으로 가볍게 보여 준다. 1장은 팀 아이디어(새 무리의 종을 알아내고 천적 소리로 경로를 바꾸게 하기)의 개념 애니메이션으로, 그 장이 열릴 때 재생된다. 이 아이디어는 확인하지 않은 구상이다. 2장은 "03 새소리를 분석하는 세 단계" 표지, 3–10장은 여덟 단계, 11장은 참고문헌이다. 단계와 색은 팀 발표 자료 "소리로 숲의 새를 알아보는 AI, BirdNET"에 맞췄다.

입력은 두 가지다.

| 입력 | 1–4단계 (소리 → 그림) | 5–8단계 (신경망 → 결과) |
| --- | --- | --- |
| 새소리 데모 (국민대 직박구리 3초) | 브라우저에서 실제 계산 (BirdNET V2.4 설정) | 실제 BirdNET V2.4 출력, 미리 계산 (`assets/data/demo-birdnet.json`). 5단계 필터만 설명용 예시 |
| 내 목소리 (마이크 3초) | 브라우저에서 실제 계산 (저음용 0–3 kHz 그림) | 남/여 목소리 모델의 실제 층 출력, 브라우저에서 계산 (`js/voice.js`) |

목소리 모델은 이 페이지를 위해 LibriSpeech로 학습한 작은 CNN이다(15,441개 파라미터, 약 60 KB). 가중치는 Hugging Face [hojinlab/ai-read-forest-voice-cnn](https://huggingface.co/hojinlab/ai-read-forest-voice-cnn)에 있고, 페이지가 녹음 직후 거기서 불러온다. 정확도와 한계는 모델 카드에 있다. 녹음 오디오는 브라우저 밖으로 나가지 않는다. BirdNET 모델 가중치는 이 저장소에 없다.

## 보기

빌드가 없는 정적 페이지다. `getUserMedia`는 HTTPS나 localhost에서만 동작하므로 웹 서버로 연다.

```bash
python3 -m http.server 8765
```

<http://localhost:8765/>를 연다. 넘기기는 이전/다음 버튼, 아래 점, `←`/`→`/`Space`/`PageUp`/`PageDown`, 휴대폰은 좌우 스와이프다. 주소의 `#3`처럼 장 번호로 바로 열 수 있고, `F`(또는 오른쪽 아래 버튼)로 전체 화면을 켠다. 장은 모두 화면 높이(`100dvh`) 안에 들어가고, 참고문헌 장만 그 안에서 스크롤한다.

## 파일

| 경로 | 내용 |
| --- | --- |
| `index.html`, `css/style.css` | 페이지. 색은 팀 발표 자료에서 뽑았다(숲 초록 `#183F2D`, 아이보리 `#FAFAF7`, 연두 `#C1D485`, 버터 노랑 `#FEEFC5`, 회녹색 `#66736A`, 구리색 `#C47E4C`) |
| `js/dsp.js` | BirdNET V2.4 전처리 재구현: −1~1 정규화, Hann 창, FFT, TF 방식 mel 필터(고음 0.5–15 kHz·nfft 1024·hop 280, 저음 0–3 kHz·nfft 2048·hop 278, 96띠), 압축 지수, 예시 합성곱 |
| `js/voice.js` | 목소리 모델을 Hugging Face에서 받아(safetensors) 순수 JS로 실행하고 층마다 출력을 돌려준다 |
| `js/stage.js`, `js/draw.js` | 8단계 장면. 번진 초록–버터 스펙트로그램, 매끈한 곡선, 둥근 막대 |
| `js/hero.js` | 아이디어 개념 애니메이션 |
| `js/audio.js`, `js/recorder-worklet.js` | 3초 녹음(AudioWorklet), 48 kHz 재표본화, 재생 |
| `js/main.js` | 입력 분석, 두 모드, 슬라이드 넘기기(키·스와이프·`#n` 주소·전체 화면) |
| `assets/audio/demo-kookmin-bulbul-3s.wav` | 데모: 팀 녹음(국민대, 2026-11-09) 원본 31.5–34.5초, 48 kHz 16-bit mono |
| `assets/data/demo-birdnet.json` | 데모의 실제 BirdNET V2.4 출력: logit 6,522개, 상위 12개 점수, 1,024차원 벡터, 첫 층 출력 3장 |
| `tools/precompute_demo.py` | 위 데모 파일을 만든다(BirdNET V2.4 TFLite를 직접 실행하고 `birdnet` 패키지 `predict()`와 대조) |
| `tools/make_dsp_refs.py`, `tools/check_dsp.mjs` | `js/dsp.js`를 numpy FFT, BirdNET 모델 내부 mel 텐서와 비교 |
| `tools/voice_gender/train.py`, `make_card.py` | 목소리 모델 학습(LibriSpeech, 화자 분리)과 모델 카드 생성 |
| `tools/check_voice.mjs` | `js/voice.js`를 PyTorch 출력(스펙트로그램, 첫 층, logit)과 비교 |
| `tools/shoot.py` | 5개 화면 크기(1440×900, 1280×720, 1920×1080, 390×844, 360×740)에서 모든 장 스크린샷과 넘침 검사, Chrome 가짜 마이크로 녹음 → 남/여 판정 경로 검사 |

## 다시 만들기와 검증

데모 출력(`birdnet==1.4.0`이 있는 환경, 예: `~/korea-bird-sounds/.venv`):

```bash
~/korea-bird-sounds/.venv/bin/python tools/precompute_demo.py ~/Documents/ai-forest-presentation/birdnet/media/1-kookmin-jikbakguri.wav . /tmp/birdnet-debug
```

```bash
uv run --no-project --with numpy python tools/make_dsp_refs.py /tmp/birdnet-debug && node tools/check_dsp.mjs /tmp/birdnet-debug
```

목소리 모델(LibriSpeech parquet은 `hf download openslr/librispeech_asr --repo-type dataset`, `SPEAKERS.TXT`는 `s3prl/librispeech_metadata`):

```bash
uv run --no-project --with torch --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --with pyarrow --with soundfile --with scipy --with safetensors --with numpy python tools/voice_gender/train.py /path/to/librispeech /tmp/voice_out
```

```bash
node tools/check_voice.mjs /tmp/voice_out
```

```bash
uv run --no-project --with playwright python tools/shoot.py http://127.0.0.1:8765/ /tmp/shots /tmp/voice_out/fake_mic_female_*.wav /tmp/voice_out/fake_mic_male_*.wav
```

`check_dsp.mjs`: FFT는 numpy와 상대오차 1e-8, 모델 방식 mel 스펙트로그램은 BirdNET V2.4 내부 텐서와 상대오차 1e-4 이하다. V2.4 그래프는 FFT 결과의 실수부만 쓰고, 페이지 그림은 교과서대로 |X|²를 쓴다. 목소리 모델은 페이지와 같은 |X|² 저음용 그림으로 학습했고, `check_voice.mjs`에서 JS와 PyTorch의 logit이 소수점 넷째 자리까지 같다.

## 라이선스

코드는 MIT(`LICENSE`). 목소리 모델은 CC BY 4.0(LibriSpeech, CC BY 4.0로 학습). 데모 오디오는 팀이 직접 녹음했다.
