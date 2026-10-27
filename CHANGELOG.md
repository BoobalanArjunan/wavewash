# Changelog

## 0.2.0 (2026-10-27)

Clipping, fixed properly. Music Pack added.

- **Music Pack** (only for `--purpose music_gen`; `enable_music_checks`). New file `music_checks.py`:
  - `DEAD_CHANNEL` (issue): one channel is empty while another has sound. Fixed by copying the live channel; the re-check must pass.
  - Warnings only, never a rejection, because real music can do these on purpose: `FAKE_STEREO` (both channels identical),
    `MONO_CANCELLATION` (loses more than `music_min_mono_sum_db`, default -6 dB, when summed to mono),
    `DIGITAL_DROPOUT` (exact-zero run of `music_min_dropout_ms`, default 20 ms, with sound on both sides within 200 ms),
    `ABRUPT_EDGE` (file starts or ends mid-sound, which clicks).
  - Thresholds are reasoned defaults, NOT measured on real music. The web settings page does not show the new fields yet.
  - Tests: 42 to 64. Planted-bug check on the new code: 8 of 9 caught; the survivor (`DEAD_CHANNEL` in the pipeline's
    fatal list) is a safety net that cannot trigger because the repair always works.

- **Bug fix: the 0.1 de-clip repair made light clipping worse.** It interpolated over clipped samples and lowered the peaks. On
  known-clipped synthetic speech the result was 18 dB (0.1% clipped) to 5 dB (2%) further from the original than leaving the file
  alone, while the re-check reported "fixed". The repair now never goes below the clipped value.
- Only samples at the file's own full-scale ceiling count as clipped, not natural peaks just below it.
- Restored peaks above full scale: the whole file is lowered together so the peak is 0.999, and the log says by how many dB.
- New setting `max_repairable_clipped_ratio` (default 0.05, provisional). Above it a file is rejected as `CLIPPING_HEAVY`
  and no repair is attempted. `CLIPPING` now means between `max_clipped_ratio` and this limit.
- De-clip runs only for `CLIPPING:` issues (not `CLIPPING_HEAVY`).
- `scripts/real_speech_check.py`: LibriSpeech check of false rejections per profile and of the clipping repair (new vs 0.1).
- Tests: 23 to 42 (clipping). A 22-case deliberate-bug check on the changed code: all caught.
- Tried and dropped: detecting clipping at any ceiling through flat runs. It false-alarmed on bass tones and speech-like audio.
- Known gaps: web settings page does not show the new setting; the 5% limit needs tuning from real speech.

## 0.1.0a1
First release.
