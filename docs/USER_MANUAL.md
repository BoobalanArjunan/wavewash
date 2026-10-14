# WaveWash User Manual

A plain-language guide to using every feature. For code/architecture details, see `README.md` in the package — this manual is task-focused: "I want to do X, here's exactly what to type."

---

## 1. Before you start

**Install once:**
```bash
pip install wavewash        # or: pip install ./wavewash-0.1.0-py3-none-any.whl
wavewash doctor             # confirms everything is installed
wavewash app                # optional: the point-and-click web UI
```

**Every command follows this shape:**
```bash
wavewash run --input <where your audio is> --output <where clean audio should go> --purpose <asr|tts|speaker_id|music_gen|custom>
```

That's the whole tool, at minimum. Everything below adds optional power on top.

---

## 2. Quick start — the 3 things everyone needs to decide

1. **Where is my audio?** → `--input`
   - A single file: `--input myfile.wav`
   - A folder (flat or with subfolders, any depth): `--input ./my_dataset`
2. **What am I training?** → `--purpose`
   - `asr` — speech recognition
   - `tts` — text-to-speech voice data
   - `speaker_id` — speaker recognition / diarization
   - `music_gen` — music generation
   - `custom` — anything else (loose defaults, tune in `config.py`)
3. **Where should the clean result go?** → `--output`

**Example — the simplest possible run:**
```bash
wavewash run --input ./raw_recordings --output ./clean_dataset --purpose asr
```

**Don't know the purpose yet?** Just omit `--purpose` and it'll ask you interactively:
```bash
wavewash run --input ./raw_recordings --output ./clean_dataset
```

---

## 3. What happens when you run it

1. It finds every `.wav` file under `--input`, however deeply nested.
2. It removes duplicate/near-duplicate files.
3. It checks every remaining file for problems (noise, clipping, silence, etc.).
4. It **auto-fixes** what can be fixed (see §7 for exactly what).
5. It **rejects** what can't be fixed, with a clear reason.
6. It writes the good files to `--output`, and everything else to labeled folders next to it.
7. It writes full logs so you can see exactly what happened to every file.

You don't need to do anything else — steps 1–7 all happen from that one command.

---

## 4. Where to find your results afterward

Say you ran:
```bash
wavewash run --input ./raw --output ./clean_dataset --purpose asr
```

You'll get this folder layout:
```
clean_dataset/              <- your good, ready-to-train audio
clean_dataset/../_error/    <- rejected files, sorted into subfolders by reason
clean_dataset/../_duplicates/  <- removed duplicate files
clean_dataset/../_logs/     <- everything about what happened (see §10)
clean_dataset/../_review_sample/  <- a random sample of both good and bad files, for you to listen to yourself
```

("`../`" means one level above `clean_dataset` — e.g. if `--output` was `./project/clean_dataset`, the error folder is `./project/_error`.)

---

## 5. Common commands, copy-paste ready

**Basic run:**
```bash
wavewash run --input ./raw --output ./clean --purpose asr
```

**Let it accept mp3/flac/m4a/ogg too (auto-converted to wav):**
```bash
wavewash run --input ./raw --output ./clean --purpose tts --include-extra-formats
```

**Just want a report first, without moving/changing any files?**
```bash
wavewash run --input ./raw --output ./clean --purpose music_gen --dry-run
```
This tells you what *would* happen — nothing gets written. Good for a first look at a new dataset.

**Skip duplicate detection (if you already know there are none, to save time):**
```bash
wavewash run --input ./raw --output ./clean --purpose asr --skip-dedup
```

**Turn off the manual-review sample (if you don't want it):**
```bash
wavewash run --input ./raw --output ./clean --purpose asr --review-sample-size 0
```

---

## 6. Using the extra features (all optional)

### 6.1 — I have transcripts and want them checked (ASR/TTS)

Make a CSV like this:
```csv
filename,transcript
utt001.wav,hello there how are you
utt002.wav,this is a test recording
```

Then:
```bash
wavewash run --input ./raw --output ./clean --purpose asr \
  --transcripts ./transcripts.csv --expected-language en
```

This checks three things automatically:
- **Does the transcript's language match what you expect?** (`--expected-language en`, `es`, etc.)
- **Is the transcript's word count plausible for the clip's length?** (catches "this transcript belongs to a different file" mistakes)
- **Does the transcript actually match what's said?** — only if you also add real speech-to-text checking:
```bash
pip install openai-whisper
wavewash run --input ./raw --output ./clean --purpose asr \
  --transcripts ./transcripts.csv --expected-language en --asr-model whisper
```
(First run downloads the Whisper model — needs internet, one-time.)

### 6.2 — I know which speaker/session each file belongs to (Speaker ID)

Make a CSV like this:
```csv
filename,speaker_id,session_id
spk1_001.wav,speaker_01,session_A
spk1_002.wav,speaker_01,session_A
spk2_001.wav,speaker_02,session_B
```

Then:
```bash
wavewash run --input ./raw --output ./clean --purpose speaker_id \
  --metadata ./metadata.csv --make-split --train-ratio 0.9
```

This gives you:
- A **speaker balance report** (are some speakers wildly over/under-represented?)
- **Mislabel warnings** (a file that sounds like a different speaker than its label says)
- **`train_manifest.csv` / `val_manifest.csv`** — a train/validation split that never puts the same recording session on both sides (prevents inflated eval scores)

### 6.3 — I want more accurate silence/speech detection

By default it uses a fast built-in method. For better accuracy on noisy or quiet speech:
```bash
pip install torch torchaudio
wavewash run --input ./raw --output ./clean --purpose asr --use-real-vad
```

---

## 7. What gets fixed automatically vs. rejected

**Fixed automatically (no action needed from you):**
| Problem | What happens |
|---|---|
| Wrong sample rate | Resampled to the target rate for your purpose |
| Wrong channel count (mono/stereo) | Converted |
| Silent lead-in/lead-out | Trimmed |
| DC offset (waveform not centered on zero) | Corrected |
| Wrong loudness | Normalized to target |
| A few clicks/pops from tiny clipped moments | Smoothed out |

**Rejected — moved to `_error/<reason>/` instead:**
| Reason folder | What it means |
|---|---|
| `low_snr` | Too much background noise relative to the signal |
| `clipping` | Audio distortion too severe to repair |
| `mostly_silence` | Barely any actual sound in the clip |
| `corrupt_or_unreadable` | The file itself is broken |
| `high_reverb` | Room echo too strong (ASR/TTS purposes) |
| `low_dynamic_range` | Over-compressed/"loudness war" mastering (Music Gen) |
| `likely_lossy_source` | Looks like it was compressed (e.g. MP3) at some point, even if saved as .wav now |
| `multi_speaker_suspected` | Possible second voice/crosstalk (TTS purpose) |
| `high_wer` | Doesn't match its transcript well (only if you used `--asr-model whisper`) |
| `language_mismatch` | Transcript looks like the wrong language |
| `alignment_suspect` | Transcript word count doesn't fit the clip's length |
| `too_short` / `too_long` | Outside the acceptable duration range |

These aren't things the tool "gave up" on — they genuinely can't be fixed by processing audio (you can't invent signal that isn't there), so a human needs to look at them.

---

## 8. Reading the review sample

After a run, open `_review_sample/`:
```
_review_sample/
  passed/     <- a few files that passed, so you can spot-check the tool's judgment
  failed/     <- a few files that failed, with the reason noted
  manifest.md <- explains what's there
```
Each file may also have a `_spectrogram.png` next to it (if you installed `matplotlib`) — a quick visual way to sanity-check what the numbers are telling you.

**Do this after every new dataset**, especially the first time — automated checks can miss things a five-second listen catches instantly.

---

## 9. Reading the diversity report

`_logs/diversity_report.md` tells you, at a glance:
- Total usable hours of audio
- Duration distribution (a text histogram)
- Original sample rate / channel mix
- Speaker balance (if you provided metadata)
- A breakdown of why files failed, most common reason first

Use this to catch dataset-level problems a per-file check can't see — e.g. "90% of my audio is under 2 seconds" or "one speaker has 10x more data than everyone else."

---

## 10. Reading the logs (for troubleshooting or auditing)

`_logs/` contains:
| File | What it's for |
|---|---|
| `pipeline_run.log` | Plain-English play-by-play of the run — read this first if something looks wrong |
| `audit_log_<time>.csv` | One row per file — every measurement and decision, open in Excel/Sheets |
| `audit_log_<time>.json` | Same data, for scripts/programs |
| `summary_<time>.md` | Quick pass/fail/fixed/duplicate counts |
| `diversity_report.md` | Dataset-level stats (see §9) |
| `speaker_contamination_flags.md` | Only appears if `--metadata` found possible mislabels |

**If a file failed and you don't know why:** open the CSV, find the filename, look at the `unfixable_reasons` column — it's in plain language.

---

## 11. Full flag reference

```
--input <path>              REQUIRED. Single file, folder, or nested folder tree.
--output <path>              REQUIRED. Where clean, passed audio goes.
--purpose <name>              asr | tts | speaker_id | music_gen | custom
                               (omit to be asked interactively)

--error-dir <path>            Default: next to --output, named _error
--dupes-dir <path>            Default: next to --output, named _duplicates
--log-dir <path>               Default: next to --output, named _logs

--include-extra-formats        Also accept mp3/flac/m4a/ogg as input
--skip-dedup                   Don't check for duplicates
--dedup-mode {off,exact,near}   # default near: exact copies + the same recording re-exported/trimmed/resampled.
                                    # Different recordings that only share tempo/groove are never flagged.

--transcripts <path>           CSV or JSON: filename -> transcript text
--expected-language <code>      e.g. 'en', 'es' -- flags mismatched transcript language
--wer-threshold <0-1>          Default 0.5 -- reject if transcript mismatch exceeds this
--asr-model none|whisper       Default none. 'whisper' enables real transcript-matching (needs
                                `pip install openai-whisper`)
--whisper-model-size <size>    tiny/base/small/medium/large (default: base)

--metadata <path>              CSV: filename,speaker_id,session_id
--make-split                   Write a train/val split that respects session boundaries
--train-ratio <0-1>            Default 0.9

--use-real-vad                 Use a more accurate (but heavier) silence/speech detector
--review-sample-size <number>  How many passed + failed files to sample for manual review
                                (default 5, set to 0 to disable)
--no-diversity-report           Skip generating the dataset-level report

--dry-run                      Show what would happen without writing/moving anything
```

---

## 12. Troubleshooting

**"No audio files found"** — Check your `--input` path is correct and actually contains `.wav` files (or add `--include-extra-formats` if they're mp3/flac/etc.).

**Everything is failing on `low_snr`** — Your source recordings may genuinely be noisy, or (if you generated test/synthetic audio) a perfectly steady tone can confuse the noise estimator, since it expects natural pauses like real speech has. Real recordings usually behave correctly.

**A file I expected to pass ended up in `_error/`** — Open the audit CSV, find that filename, read the `issues_found` and `unfixable_reasons` columns — they explain exactly which check it failed and by how much. You can also loosen that purpose's thresholds in `config.py` if your standards are different.

**WER / language checks aren't doing anything** — Make sure you passed `--transcripts` (required for both) and, for WER specifically, `--asr-model whisper` plus `pip install openai-whisper`.

**Spectrograms aren't showing up in `_review_sample/`** — Install matplotlib: `pip install matplotlib`.

---

## 13. One-line cheat sheet

```bash
# The one command most people need:
wavewash run --input ./raw --output ./clean --purpose asr

# With everything turned on:
wavewash run --input ./raw --output ./clean --purpose asr \
  --transcripts ./transcripts.csv --expected-language en --asr-model whisper \
  --metadata ./metadata.csv --make-split --include-extra-formats
```
