# Voice-Controlled IR Remote on ESP32-C3 (Edge AI)

Offline Vietnamese voice-command recognition to replace a TV / air-conditioner remote. This repo contains the **Python training pipeline**; the classifier is exported as a C header (via [emlearn](https://github.com/emlearn/emlearn)) for an ESP32-C3. ESP32 firmware (INMP441 I2S microphone + IR LED output) is **in progress**.

## Results

| | |
|---|---|
| Dataset | 7,348 one-second clips, 6 speakers, including noisy conditions |
| Split | Speaker-independent (test speakers never seen in training) |
| Random Forest | **89.1%** accuracy on 6 actions, 81.1% on 7 classes (6,917 test clips) |

Confusion matrices: [`reports/`](reports/). Random Forest was chosen over an RBF SVM because emlearn exports trees to compact C, while an SVM would carry its support vectors onto the MCU.

## Pipeline

```
raw .m4a  ->  step1_split.py        energy-based word splitting into 1 s clips
          ->  step1b/c/d            augmentation, validation, noisy-clip flagging
          ->  step2_dataset.py      MFCC (13) + delta + ZCR + RMS features
          ->  step3_train.py        Random Forest vs SVM, per-class report
          ->  step4_export.py       emlearn -> voice_model.h (C header)
```

Feature parameters (16 kHz, n_mfcc=13, n_fft=512, hop=160, win=400) are in [`src/config.py`](src/config.py) and must match exactly in the ESP32 feature extractor.

An ablation showed pitch (YIN) added no measurable accuracy, so it was dropped to save MCU compute.

## Run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                      # ffmpeg needed for .m4a
cd src
python step1_split.py && python step2_dataset.py && python step3_train.py && python step4_export.py
```

The voice recordings are not published (they contain other people's voices); put your own files under `data/raw/<label>/<speaker>_<take>.m4a`. More detailed notes (Vietnamese): [`docs/pipeline_notes_vi.md`](docs/pipeline_notes_vi.md).

## Stack

Python, librosa, scikit-learn, emlearn, ESP32-C3 (Embedded C, in progress)
