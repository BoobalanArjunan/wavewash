# WaveWash

[![tests](https://github.com/BoobalanArjunan/wavewash/actions/workflows/tests.yml/badge.svg)](https://github.com/BoobalanArjunan/wavewash/actions/workflows/tests.yml)
![status](https://img.shields.io/badge/status-alpha-orange)
![license](https://img.shields.io/badge/license-MIT-blue)

**Check, repair and clean audio datasets before you train on them.**

Point WaveWash at a folder of audio and tell it what you are training (speech recognition, voice synthesis, speaker ID, music generation). It checks every file, repairs what it can *verify* is fixed, moves the rest to an error folder with a labeled reason, sets duplicates aside, and writes a full audit log. Your original files are never modified.

> **Status: alpha (0.2.0a1).** It works on the cases in its test suite, but it has not yet been validated on large real-world speech corpora. Treat its verdicts as a first pass and listen to a sample of what it keeps and rejects. See [Limitations](#limitations).

![WaveWash app](https://raw.githubusercontent.com/BoobalanArjunan/wavewash/main/docs/screenshot-setup.png)

## Why

Models inherit the flaws of their audio: clipped or distorted takes, DC offset, background noise, low-quality copies of better recordings, duplicates that quietly inflate test scores, empty or broken files. These are usually found by hand (listening to thousands of files) or not at all. WaveWash puts the common checks in one tool and tells you *why* each file was kept, repaired or rejected.

## Install

```bash
python3 -m venv ~/wavewash-env
source ~/wavewash-env/bin/activate      # Windows: wavewash-env\Scripts\activate
pip install wavewash
wavewash doctor                         # confirms the install works
```

Python 3.10 or newer. No Node.js needed: the web app ships inside the package.

Optional extras: `pip install "wavewash[lang]"` (transcript language check), `[plots]` (spectrograms in review samples), `[vad]` (Silero VAD, pulls in PyTorch), `[whisper]` (WER check).

## Use it

**Web app** (runs only on your machine, at `http://127.0.0.1:8420`):

```bash
wavewash app
```

Pick a purpose, choose an input folder (subfolders are included) and an output folder, click run.

**Command line:**

```bash
wavewash run --input ./raw_audio --output ./out/clean --purpose asr
wavewash run --help
```

Purposes: `asr`, `tts`, `speaker_id`, `music_gen`, `custom`.

## Using WaveWash in Python Code (Developer API)

You can import WaveWash directly into your PyTorch datasets, audio preprocessing pipelines, or model ingestion scripts:

```python
from pathlib import Path
from wavewash import PROFILES, analyze_file, apply_fix_pipeline, load_audio, save_audio

# 1. Analyze an audio file in-memory
file_path = Path("sample.wav")
report = analyze_file(file_path, PROFILES["asr"])

print(f"Passed QC: {report.passed}")
print(f"SNR: {report.snr_db:.1f} dB  |  LUFS: {report.lufs:.1f}")
print(f"Issues found: {report.issues}")
print(f"Advisory warnings: {report.warnings}")

# 2. Repair damaged audio in-memory
if not report.passed:
    audio, sr = load_audio(file_path)
    # Automatically removes DC offset, reconstructs clipped runs, fixes dead stereo channels, etc.
    repaired_audio, sr, fixes = apply_fix_pipeline(audio, sr, PROFILES["asr"], report.issues)
    print(f"Fixes applied: {fixes}")
    save_audio(Path("sample_clean.wav"), repaired_audio, sr)

# 3. Fine-tune threshold limits in code
import copy
custom_profile = copy.deepcopy(PROFILES["tts"])
custom_profile.min_snr_db = 32.0   # enforce stricter studio silence floor
custom_profile.max_repairable_clipped_ratio = 0.03
```

## What you get

```
out/
  clean/          audio that passed, or was repaired and re-checked
  _error/         rejected files, in folders named by reason (low_snr/, clipping/, corrupt_or_unreadable/ ...)
  _duplicates/    duplicates, copied here (nothing is deleted)
  _logs/          audit_log_*.csv and .json (one row per file: what was found, what was fixed, why rejected), summary_*.md
```

## Music Pack (`--purpose music_gen`, new in 0.2)

Extra checks for music files: a dead channel (rejected, and repaired by copying the live channel), plus warnings for fake
stereo, channels that cancel in mono, digital dropouts mid-song, and files cut mid-sound (click at start or end). Only the
dead channel is an issue; the rest could be intentional, so they are listed in the report and the file is kept.
The thresholds are judgement defaults, not measured on real music yet.

## What it checks and repairs

| Checks | Repairs (only kept if a re-check shows they helped) |
|---|---|
| Unreadable, empty or truncated files | DC offset removal |
| Too short / too long for the purpose | De-clipping of short clipped runs (up to 8 ms, spline interpolation that never goes below the clipped value) |
| Signal-to-noise ratio (estimate) | Noise reduction (spectral subtraction, kept only if the SNR estimate improves) |
| Clipping, DC offset, loudness | Loudness normalisation, resampling, channel conversion to the purpose's targets |
| Silence: mostly-silent files, long gaps | Trimming silence at the edges |
| Lossy or low-pass sources (sharp spectral cliff) | |
| Exact duplicates, same-samples duplicates, near-duplicates (same take, different level or format) | |
| Multiple speakers (heuristic) | |
| Optional: transcript language, WER against Whisper, speaker/session train-validation split | |

Repair is classical signal processing, **not** machine learning.

## Limitations

Please read these before trusting a result.

- **Alpha, tested on synthetic data.** The test suite plants known defects (DC offset, clipping, duplicates, broken, empty and stereo files, other sample rates) and checks the routing, that originals are never modified, the local server's safety rules and the command line. It has *not* been validated on large real speech corpora; thresholds are reasoned defaults.
- **"Repaired" means "passed the re-check", not "restored".** De-clipping interpolates over saturated samples. Since 0.2 a repaired sample is never lowered below the clipped value (0.1 lowered peaks, which on synthetic speech made light clipping 5 to 18 dB *worse* than leaving it), and files with more than 5% saturated samples (`max_repairable_clipped_ratio`) are rejected as `CLIPPING_HEAVY` instead of "repaired". On synthetic speech-like signals the repair is about neutral at 0.1 to 0.5% clipping and 1 to 13 dB better from 2% to 10%. The 5% limit is provisional: run `scripts/real_speech_check.py` on LibriSpeech (see `docs/REAL_DATA_CHECK.md`) and tune it. Restored peaks can exceed full scale, so the whole file is lowered together and the log says by how many dB. Listen to repaired files.
- **Clipping is only detected at full scale.** Audio clipped at a lower ceiling (for example a limiter at -3 dB) is not detected. A plateau-based detector was tried and dropped: it false-alarmed on 16-bit low bass tones and on speech-like test audio.
- **The settings page of the web UI does not yet show `max_repairable_clipped_ratio`;** change it in a profile or in code for now.
- **Reverb (RT60) is a warning only.** Measured against synthetic rooms of known reverb time, the blind estimate could not tell dry from reverberant audio, so it never rejects a file.
- **Heuristics, not trained models:** the SNR estimate, the multi-speaker check and the near-duplicate threshold. Check their verdicts on your own data.
- **Music:** mid-track silence is not flagged, and moderate noise can go unnoticed.
- **Not yet measured:** whether cleaned data improves a trained model; speed and memory on very large datasets; real Whisper and Silero runs.
- The web app has **no login** and binds to `127.0.0.1` only. Do not expose it to a network.

## Development

```bash
git clone https://github.com/BoobalanArjunan/wavewash.git
cd wavewash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest -q
```

The front-end source is in `frontend/` (React + Vite). The built app is committed in `src/wavewash/static/`; to rebuild it: `cd frontend && npm ci && npm run build`, then copy `dist/` over `src/wavewash/static/`.

Documentation: [step-by-step guide](docs/STEP_BY_STEP.md) · [user manual](docs/USER_MANUAL.md)

## Contributing

Bug reports with a small example file are the most useful contribution. If WaveWash rejects a good file or passes a bad one, please open an issue with the log row for that file (from `_logs/audit_log_*.csv`).

## License

MIT
