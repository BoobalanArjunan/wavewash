# Real-speech check for WaveWash 0.2 (about 20 minutes)

The test suite uses synthetic audio. This check uses real speech (LibriSpeech) and answers two things:
1. Does WaveWash wrongly reject or warn on clean speech (per profile)?
2. Does the clipping repair really help on real speech, and is the 5% heavy-clipping limit in the right place?

## Tasks
1. Install: `pip install -e ".[dev]"` from the repo (or `pip install --pre wavewash`), then `pytest -q` (expect 64 passed).
2. Download LibriSpeech `dev-clean` from openslr.org/12 and unpack it.
3. Run: `python scripts/real_speech_check.py --librispeech ~/LibriSpeech/dev-clean --out real_check`
4. Open `real_check/results.md` and send it to me.
5. Listen (3 minutes) to the files in `real_check/listen/` (original, clipped, 0.2 repair, 0.1 repair at 2% and 10%).
6. Tell me: is the 0.2 repair closer to the original than the clipped file? Does the 0.1 repair sound worse at 2%?

## How the numbers get used
- Profile with more than a few % of clean files rejected: its thresholds are too tight; I lower them from the measured spread.
- Clipping table: raise `max_repairable_clipped_ratio` only if the 0.2 median is still clearly positive and almost no file is worse; lower it otherwise.

## Music Pack on real music (about 10 minutes)
The Music Pack thresholds are unmeasured. With any folder of real, healthy songs (for example MUSDB18 `mixture.wav` files):
1. `wavewash run --input <songs> --output out_music --purpose music_gen --dry-run`
2. Count how many songs get `FAKE_STEREO`, `MONO_CANCELLATION`, `DIGITAL_DROPOUT` or `ABRUPT_EDGE`. On healthy songs these should be rare;
   send me the counts and the song names that fire. I will tune `music_min_mono_sum_db` and `music_min_dropout_ms` from them.
3. Damage one song on purpose (silence one channel in an editor) and check it is reported as `DEAD_CHANNEL` and repaired.
