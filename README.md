# WaveWash

[![tests](https://github.com/BoobalanArjunan/wavewash/actions/workflows/tests.yml/badge.svg)](https://github.com/BoobalanArjunan/wavewash/actions/workflows/tests.yml)
![version](https://img.shields.io/badge/version-0.2.0-blue)
![status](https://img.shields.io/badge/status-production--ready-green)
![license](https://img.shields.io/badge/license-MIT-blue)

**Deterministic, mathematically validated audio quality control, verification, repair, and deduplication for Speech & Audio Machine Learning (ASR, TTS, Speaker ID, Music Generation).**

Point WaveWash at a folder of audio or embed it directly into your Python ML pipelines. It inspects every file, repairs what it can *verify* is fixed, segregates rejected files into categorized error folders with clear reasons, sets duplicates aside, and writes full audit logs. Original audio files are never modified.

> **Status: v0.2.0 (Production-Ready Engine).** Mathematically validated on real speech (LibriSpeech: **71.8% MSE distortion reduction** on clipping repair, two-tier `CLIPPING_HEAVY` guardrails) and real multi-track music (MUSDB18-HQ: dead-channel auto-repair, zero false rejections). Features a first-class in-memory Python Developer API for PyTorch and ML pipelines.

![WaveWash app](https://raw.githubusercontent.com/BoobalanArjunan/wavewash/main/docs/screenshot-setup.png)

---

## Why WaveWash

Machine learning models inherit the acoustic flaws of their training data:
* **Clipped / saturated waveforms** create harsh non-linear harmonic distortion in neural vocoders.
* **DC offsets** distort zero-crossings and waste dynamic range.
* **Acoustic duplicates and resampled takes** quietly leak across train/test splits, inflating test scores.
* **Silent stereo channels and phase cancellations** corrupt generative music models.

WaveWash automates dataset curation through deterministic Digital Signal Processing (DSP) rather than opaque black-box models. Every repair is rigorously re-checked before being accepted.

---

## Installation

```bash
pip install wavewash
wavewash doctor    # confirms all audio backends and DSP dependencies are healthy
```

Python 3.10 or newer. Zero configuration required — the embedded React Web UI is pre-compiled inside the package.

**Optional Extras:**
```bash
pip install "wavewash[vad]"      # Silero neural VAD (pulls torch + torchaudio)
pip install "wavewash[whisper]"  # Whisper ASR transcription & WER verification
pip install "wavewash[plots]"    # Spectrogram rendering in review folders
pip install "wavewash[all]"      # All extras
```

---

## Developer API: Embedding WaveWash in Python Code

WaveWash provides a first-class, in-memory Python API designed for direct embedding into PyTorch datasets, ingestion workers, and streaming preprocessing pipelines.

### 1. In-Memory Audio Analysis (`analyze_file`)
Inspect audio files programmatically before passing them into your models:

```python
from pathlib import Path
from wavewash import PROFILES, analyze_file

# Load the purpose-driven profile (asr, tts, speaker_id, music_gen, or custom)
profile = PROFILES["asr"]
report = analyze_file(Path("speech_sample.wav"), profile)

if report.passed:
    print(f"File is clean! SNR: {report.snr_db:.1f} dB, Loudness: {report.lufs:.1f} LUFS")
else:
    print(f"Rejected for issues: {report.issues}")
    print(f"Advisory warnings: {report.warnings}")
```

### 2. In-Memory Automatic Audio Repair (`apply_fix_pipeline`)
Automatically repair fixable acoustic defects (DC offset, reconstructible clipping, dead stereo channels, loudness normalization) directly on audio buffers:

```python
from pathlib import Path
from wavewash import PROFILES, analyze_file, apply_fix_pipeline, load_audio, save_audio

file_path = Path("distorted_sample.wav")
profile = PROFILES["tts"]
report = analyze_file(file_path, profile)

if not report.passed:
    audio, sr = load_audio(file_path)
    # Automatically repairs DC displacement, clipped runs, dead channels, and normalizes LUFS
    repaired_audio, sr, fixes = apply_fix_pipeline(audio, sr, profile, report.issues)
    
    print(f"Repairs applied: {fixes}")
    save_audio(Path("cleaned_sample.wav"), repaired_audio, sr)
```

### 3. Embedding in a PyTorch `Dataset`
Clean or filter training samples dynamically in your PyTorch DataLoader:

```python
import torch
from torch.utils.data import Dataset
from pathlib import Path
from wavewash import PROFILES, analyze_file, apply_fix_pipeline, load_audio

class CleanAudioDataset(Dataset):
    def __init__(self, file_paths, profile_name="asr", auto_fix=True):
        self.file_paths = [Path(p) for p in file_paths]
        self.profile = PROFILES[profile_name]
        self.auto_fix = auto_fix

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        report = analyze_file(path, self.profile)
        audio, sr = load_audio(path)
        
        # Apply deterministic DSP repair on-the-fly if needed
        if not report.passed and self.auto_fix:
            audio, sr, fixes = apply_fix_pipeline(audio, sr, self.profile, report.issues)
            
        tensor = torch.from_numpy(audio).float()
        return tensor, sr, path.name
```

### 4. Customizing Threshold Profiles in Code
Tune parameters to match your custom studio standards:

```python
from dataclasses import replace
from wavewash import PROFILES

# Create a customized high-precision TTS profile
custom_tts = replace(
    PROFILES["tts"],
    min_snr_db=32.0,                    # enforce stricter studio noise floor
    max_repairable_clipped_ratio=0.03,  # reject if > 3% clipped
    target_lufs=-20.0                   # custom target loudness
)
```

---

## Running via CLI & Local Web UI

### Local Web Dashboard
Launch the zero-config web dashboard on loopback interface:

```bash
wavewash app [--port 8420]
```
Open `http://127.0.0.1:8420` in your browser. Choose an input folder, pick your training purpose, configure targets, and monitor live progress with full visual auditing.

### Batch Processing via CLI
Process entire multi-gigabyte corpora in a single command:

```bash
# Speech Recognition dataset
wavewash run --input ./raw_speech --output ./clean_speech --purpose asr

# Studio Text-to-Speech (strict SNR & multi-speaker rejection)
wavewash run --input ./tts_recordings --output ./clean_tts --purpose tts

# Music Generation (Full bandwidth 44.1kHz stereo with Music Pack integrity suite)
wavewash run --input ./music_mixes --output ./clean_music --purpose music_gen

# Dry-run mode (audit and report without writing files)
wavewash run --input ./raw_audio --output ./audit_only --purpose asr --dry-run
```

---

## Output Architecture

WaveWash processes data with zero destructive writes. Original audio is never modified:

```
output_dir/
  clean/          Audio that passed as-is, or was successfully repaired and verified
  _error/         Rejected files segregated into labeled reason folders:
                  ├── low_snr/
                  ├── clipping_heavy/
                  ├── dead_channel/
                  ├── corrupt_or_unreadable/
                  └── duration_out_of_range/
  _duplicates/    Exact duplicates and near-duplicate takes (nothing deleted)
  _review_sample/ The most challenging edge cases auto-sampled for human spot-checking
  _logs/          Audit trail (audit_log_*.csv, audit_log_*.json, summary_*.md)
```

---

## What WaveWash Checks & Repairs

| Acoustic / Signal Defect | Detection Mechanism | Automated DSP Repair |
|---|---|---|
| **DC Displacement** | Mean amplitude offset per channel ($> 10^{-5}$) | Subtracts DC displacement per channel |
| **Clipping & Saturation** | Ceiling flatline detection at 0.999 threshold | **v0.2 Non-Destructive Repair:** Proportional peak interpolation (up to 8 ms runs) preserving natural peak floors |
| **Severe Distortion** | Files exceeding 5% saturated samples | **Two-Tier Threshold:** Categorized as `CLIPPING_HEAVY` and rejected cleanly to protect model training |
| **SNR & Noise Floor** | Blind spectral noise floor estimation | Spectral subtraction filtering (retained only if post-check verifies SNR improvement) |
| **Loudness Dynamics** | EBU R128 integrated loudness (LUFS) | Non-destructive ITU-R BS.1770 gain normalization |
| **Dead Stereo Channels** | One channel silent while other active | **Auto-Repair:** Clones active channel to dead channel (clean dual-mono conversion) |
| **Spectral Cliffs** | Steep STFT magnitude dropoff ($>36\text{ dB/oct}$) | Identifies hidden low-bitrate MP3 transcodes |
| **Acoustic Duplicates** | Spectral peak hashing & MFCC embeddings | Isolates exact takes, resampled files, and re-encoded duplicates |
| **Phase & Stereo Integrity** | Phase cancellation & mono summing loss | Flags `MONO_CANCELLATION`, `FAKE_STEREO`, and `DIGITAL_DROPOUT` |
| **Edge Click / Pop Risk** | Instantaneous non-zero onset amplitude | Flags `ABRUPT_EDGE` warnings on truncated tracks |

---

## Empirical Benchmark Results (v0.2.0)

WaveWash 0.2 has been validated against real-world speech and multi-track music corpora:

* **LibriSpeech (`dev-clean`) Benchmark:**  
  * On real speech with 2% clipping, WaveWash 0.2 achieved an objective **71.8% reduction in Mean Squared Error (MSE)** and a **$+2.57\text{ dB}$ median SDR improvement** (compared to v0.1 legacy repair which degraded speech by $-0.53\text{ dB}$).
  * The two-tier cutoff cleanly segregated 100% of severe clipping ($>5\%$) into `CLIPPING_HEAVY` without applying false repairs.
* **MUSDB18-HQ Multi-Track Music Benchmark:**  
  * Zero false rejections across 15 commercial multi-track songs.
  * 100% mathematical precision on dead-channel stereo repair ($\Delta = 0.000000$).

---

## Contributing & Bug Reports

Contributions and real-world audio edge cases are welcome! If WaveWash rejects a valid audio file or passes a defective one, please open an issue with the corresponding row from `_logs/audit_log_*.csv`.

---

## License

MIT License. Designed and maintained by **Boobalan Arjunan**.
