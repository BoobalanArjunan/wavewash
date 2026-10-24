"""
config.py
---------
Central configuration for the WaveWash — audio QC & data prep pipeline.

Every "purpose" (ASR, TTS, Speaker ID, Music Generation, Custom) has its own
threshold profile. Edit these numbers to match your own project's standards --
they are starting points based on common practice, not absolute laws.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PurposeProfile:
    name: str
    description: str

    # --- format targets (fixes will convert TOWARD these) ---
    target_sample_rate: int = 16000
    target_channels: int = 1              # 1 = mono, 2 = stereo
    target_bit_depth: int = 16            # 16 or 24 (used when re-exporting)

    # --- hard rejection thresholds (file goes to error/ if it can't be fixed) ---
    min_duration_sec: float = 0.3
    max_duration_sec: float = 30.0
    min_snr_db: float = 15.0
    max_clipped_ratio: float = 0.001      # fraction of samples allowed at full scale
    max_dc_offset: float = 0.02
    # Above this share of saturated samples the file is rejected as heavily clipped and no repair is attempted:
    # interpolation cannot honestly restore that much (see audio_fixer.fix_declip). Provisional default.
    max_repairable_clipped_ratio: float = 0.05
    min_speech_ratio: Optional[float] = None   # fraction of clip that must be "active" audio
    reject_multi_speaker: bool = False    # TTS: reject if diarization finds >1 speaker
    max_silence_ratio: float = 0.85       # reject if clip is almost entirely silence
    max_internal_silence_sec: Optional[float] = None  # reject if any single mid-clip gap exceeds this

    # --- loudness ---
    target_lufs: float = -23.0
    loudness_tolerance_db: float = 3.0

    # --- spectral / codec-artifact detection ---
    # expressed as a FRACTION of Nyquist (sr/2) of the file's ORIGINAL sample rate,
    # not an absolute Hz value -- this makes the same check meaningful at 16kHz,
    # 22kHz or 44.1kHz instead of only working for one target rate.
    min_spectral_rolloff_fraction: Optional[float] = 0.70
    # min_spectral_rolloff_fraction is now a WARNING threshold only: audio whose energy
    # stops below this fraction of Nyquist gets a "bandwidth limited" note, never a
    # rejection. (Low-frequency instruments -- frame drums, tabla, bass -- legitimately
    # have little high-frequency energy; rejecting them was a false-positive factory.)
    #
    # Rejection as "likely lossy/filtered" is driven by evidence of an actual filter or
    # codec: a SHARP spectral cliff. Measured on real loop-pack stems, natural instrument
    # roll-off is 9-22 dB/octave; a low-pass filter or codec cutoff is 49-130+ dB/octave.
    lossy_cliff_db_per_octave: float = 36.0
    enable_rolloff_check: bool = True

    # --- reverb (RT60 estimate) ---
    enable_reverb_check: bool = False
    max_rt60_ms: Optional[float] = None   # reject/flag if estimated RT60 exceeds this

    # --- dynamic range (mainly Music Gen) ---
    enable_crest_factor_check: bool = False
    min_crest_factor_db: Optional[float] = None   # reject over-compressed "loudness war" masters

    # --- music-only checks (see music_checks.py) ---
    enable_music_checks: bool = False     # dead channel, fake stereo, mono cancellation, digital dropouts, abrupt edges
    music_min_mono_sum_db: float = -6.0   # warn if averaging the two channels into mono loses this many dB or more
    music_min_dropout_ms: float = 20.0    # exact-zero stretch (with sound on both sides) at least this long = dropout

    # --- fixable-with-effort tuning ---
    declip_max_overshoot: float = 1.5  # a restored peak may rise at most this many times above the clip level (1.0 = never above it).
                                       # Reasoned limit, not tuned: without it one wild interpolation turned a whole file down by 8 dB.
    declip_max_run_ms: float = 8.0   # widest clipped run (in ms) fix_declip() will attempt to
                                       # repair via interpolation. Longer runs than this mean too
                                       # much of the original waveform is genuinely gone -- interpolating
                                       # across them would fabricate audio, not repair it, so those are
                                       # left for the SNR/clipped_ratio check to reject instead.

    # --- noise repair ---
    enable_denoise: bool = False          # spectral-gating noise reduction (repair for LOW_SNR)
    denoise_strength: float = 0.8         # 0..1 -- how much of the estimated noise to remove

    # --- behavior flags ---
    trim_silence_edges: bool = True
    auto_resample: bool = True
    auto_fix_dc_offset: bool = True
    auto_normalize_loudness: bool = True
    attempt_declip: bool = True
    run_duplicate_detection: bool = True
    use_real_vad: bool = False            # try Silero VAD if available (falls back silently)


PROFILES = {
    "asr": PurposeProfile(
        enable_denoise=True,
        name="ASR (Automatic Speech Recognition)",
        description="Clean speech, accurate alignment, moderate noise tolerance.",
        target_sample_rate=16000,
        target_channels=1,
        min_duration_sec=0.5,
        max_duration_sec=30.0,
        min_snr_db=15.0,
        max_clipped_ratio=0.002,
        min_speech_ratio=0.4,
        max_silence_ratio=0.85,
        max_internal_silence_sec=4.0,
        target_lufs=-23.0,
        min_spectral_rolloff_fraction=0.65,
        enable_rolloff_check=True,
        enable_reverb_check=True,
        max_rt60_ms=900.0,
    ),
    "tts": PurposeProfile(
        enable_denoise=True,
        name="TTS (Text-to-Speech)",
        description="Studio-clean single-speaker audio; strict noise/reverb tolerance.",
        target_sample_rate=22050,
        target_channels=1,
        min_duration_sec=0.8,
        max_duration_sec=20.0,
        min_snr_db=28.0,
        max_clipped_ratio=0.0005,
        reject_multi_speaker=True,
        max_silence_ratio=0.6,
        max_internal_silence_sec=2.5,
        target_lufs=-23.0,
        loudness_tolerance_db=2.0,
        min_spectral_rolloff_fraction=0.75,
        enable_rolloff_check=True,
        enable_reverb_check=True,
        max_rt60_ms=500.0,          # TTS needs near-dry recordings
    ),
    "speaker_id": PurposeProfile(
        enable_denoise=True,
        name="Speaker Recognition / Diarization",
        description="Speaker-distinguishing features preserved; some noise is fine.",
        target_sample_rate=16000,
        target_channels=1,
        min_duration_sec=1.0,
        max_duration_sec=60.0,
        min_snr_db=10.0,
        max_clipped_ratio=0.003,
        max_silence_ratio=0.9,
        max_internal_silence_sec=8.0,
        target_lufs=-23.0,
        min_spectral_rolloff_fraction=0.55,
        enable_rolloff_check=True,
        enable_reverb_check=False,
    ),
    "music_gen": PurposeProfile(
        name="Music Generation",
        description="High fidelity, full bandwidth, minimal lossy-compression history.",
        target_sample_rate=44100,
        target_channels=2,
        target_bit_depth=24,
        min_duration_sec=2.0,
        max_duration_sec=600.0,
        min_snr_db=5.0,
        max_clipped_ratio=0.001,
        max_silence_ratio=0.95,
        max_internal_silence_sec=None,   # music legitimately has quiet passages
        min_spectral_rolloff_fraction=0.72,   # ~16kHz at 44.1kHz source -- classic MP3 tell
        enable_rolloff_check=True,
        target_lufs=-14.0,               # streaming-loudness-ish target
        loudness_tolerance_db=4.0,
        enable_crest_factor_check=True,
        min_crest_factor_db=6.0,         # flag heavily brick-walled/over-compressed masters
        enable_music_checks=True,
    ),
    "custom": PurposeProfile(
        name="Custom / General Purpose",
        description="Loose defaults -- edit config.py to tune for your own use case.",
        target_sample_rate=16000,
        target_channels=1,
        min_duration_sec=0.2,
        max_duration_sec=120.0,
        min_snr_db=8.0,
        max_clipped_ratio=0.01,
        max_silence_ratio=0.95,
        target_lufs=-23.0,
        loudness_tolerance_db=6.0,
        min_spectral_rolloff_fraction=0.5,
        enable_rolloff_check=True,
    ),
}

SUPPORTED_EXTENSIONS = {".wav", ".wave"}
EXTRA_READABLE_EXTENSIONS = {".flac", ".mp3", ".m4a", ".ogg"}

DUPLICATE_HASH_THRESHOLD = 12   # hamming distance <= this (out of 256 bits) => candidate duplicate
                                 # (final decision also requires duration match + MFCC correlation,
                                 # see dedup_detector.find_duplicate_groups)

DEFAULT_REVIEW_SAMPLE_SIZE = 5

# --- optional runtime overrides (used by the local desktop app's "advanced settings")
# If WAVEWASH_CONFIG_OVERRIDES points to a JSON file of the form
# {"purpose": "asr", "overrides": {"min_snr_db": 12.0, ...}}, apply it to that
# profile now, before anything else in the app reads PROFILES. This lets the
# UI adjust thresholds per-run without editing this file by hand -- the CLI
# and config.py itself are completely unaffected if this env var isn't set.
import os as _os
import json as _json

_override_path = _os.environ.get("WAVEWASH_CONFIG_OVERRIDES")
if _override_path and _os.path.exists(_override_path):
    try:
        with open(_override_path) as _f:
            _data = _json.load(_f)
        _purpose = _data.get("purpose")
        _overrides = _data.get("overrides", {})
        if _purpose in PROFILES:
            for _key, _value in _overrides.items():
                if hasattr(PROFILES[_purpose], _key):
                    setattr(PROFILES[_purpose], _key, _value)
    except Exception as _e:
        print(f"Warning: could not apply config overrides from {_override_path}: {_e}")
