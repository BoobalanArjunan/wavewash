"""
quality_checker.py
-------------------
All the "is this audio good?" measurements live here. Nothing in this file
modifies audio -- it only inspects it and reports numbers + issue strings.
Fixing happens in audio_fixer.py.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pyloudnorm as pyln
    HAVE_PYLOUDNORM = True
except ImportError:
    HAVE_PYLOUDNORM = False

import librosa

from .config import PurposeProfile
from .vad_module import get_speech_segments
from . import music_checks


@dataclass
class QualityReport:
    ok_to_load: bool = True
    load_error: Optional[str] = None

    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    duration_sec: Optional[float] = None

    clipped_ratio: Optional[float] = None
    dc_offset: Optional[float] = None
    snr_db: Optional[float] = None
    silence_ratio: Optional[float] = None
    max_internal_silence_sec: Optional[float] = None
    lufs: Optional[float] = None
    spectral_rolloff_hz: Optional[float] = None
    spectral_rolloff_fraction: Optional[float] = None
    rt60_ms: Optional[float] = None
    crest_factor_db: Optional[float] = None
    num_speakers_estimate: Optional[int] = None
    vad_backend: Optional[str] = None

    mono_sum_loss_db: Optional[float] = None
    dropout_count: Optional[int] = None
    spectral_cliff_db_per_oct: Optional[float] = None
    spectral_cliff_hz: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    issues: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Returns True if the file loaded successfully and has zero blocking issues."""
        return self.ok_to_load and len(self.issues) == 0

    @property
    def is_clean(self) -> bool:
        """Alias for passed."""
        return self.passed


def load_audio(path: Path):
    audio, sr = sf.read(str(path), always_2d=False)
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    return audio, sr, channels


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def estimate_snr_db(mono_audio: np.ndarray, sr: int) -> float:
    frame_len = int(0.032 * sr) or 512
    hop = frame_len // 2
    if len(mono_audio) < frame_len:
        frame_len = len(mono_audio)
        hop = max(1, frame_len // 2)

    energies = []
    for start in range(0, max(1, len(mono_audio) - frame_len), hop):
        frame = mono_audio[start:start + frame_len]
        energies.append(np.mean(frame ** 2) + 1e-12)

    if not energies:
        return 0.0

    energies = np.array(energies)
    noise_floor = np.percentile(energies, 10)
    signal_level = np.percentile(energies, 90)
    snr = 10 * np.log10((signal_level + 1e-12) / (noise_floor + 1e-12))
    return float(max(snr, -20.0))


def _frame_db_envelope(mono_audio: np.ndarray, sr: int, frame_ms: float = 25.0):
    frame_len = int(frame_ms / 1000 * sr) or 400
    hop = frame_len // 2
    if len(mono_audio) < frame_len:
        rms = np.sqrt(np.mean(mono_audio ** 2) + 1e-12)
        return np.array([20 * np.log10(rms + 1e-12)]), frame_len, hop
    dbs = []
    for start in range(0, len(mono_audio) - frame_len, hop):
        frame = mono_audio[start:start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        dbs.append(20 * np.log10(rms + 1e-12))
    return np.array(dbs), frame_len, hop


def estimate_silence_ratio(mono_audio: np.ndarray, sr: int,
                            energy_threshold_db: float = -40.0) -> float:
    dbs, _, _ = _frame_db_envelope(mono_audio, sr)
    if len(dbs) == 0:
        return 1.0
    return float(np.mean(dbs < energy_threshold_db))


def estimate_max_internal_silence_sec(mono_audio: np.ndarray, sr: int,
                                       energy_threshold_db: float = -40.0,
                                       edge_guard_sec: float = 0.15) -> float:
    """
    Longest continuous silent stretch that is NOT at the very start/end of the
    clip (those are handled by trimming). Flags dead-air gaps mid-recording
    that waste training capacity or break alignment.
    """
    dbs, frame_len, hop = _frame_db_envelope(mono_audio, sr)
    if len(dbs) < 3:
        return 0.0
    guard_frames = max(1, int(edge_guard_sec * sr / hop))
    interior = dbs[guard_frames:-guard_frames] if len(dbs) > 2 * guard_frames else dbs
    is_silent = interior < energy_threshold_db

    longest_run = 0
    current_run = 0
    for s in is_silent:
        if s:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return float(longest_run * hop / sr)


def estimate_clipping(audio: np.ndarray, threshold: float = 0.999) -> float:
    return float(np.mean(np.abs(audio) >= threshold))


def estimate_dc_offset(mono_audio: np.ndarray) -> float:
    return float(np.mean(mono_audio))


def estimate_lufs(audio: np.ndarray, sr: int) -> Optional[float]:
    if not HAVE_PYLOUDNORM:
        return None
    try:
        meter = pyln.Meter(sr)
        data = audio if audio.ndim > 1 else audio.reshape(-1, 1)
        return float(meter.integrated_loudness(data))
    except Exception:
        return None


def estimate_crest_factor_db(audio: np.ndarray) -> Optional[float]:
    """Peak-to-RMS ratio in dB. Low crest factor = heavily compressed/limited
    ('loudness war') master with little dynamic range left."""
    try:
        flat = audio if audio.ndim == 1 else audio.flatten()
        peak = np.max(np.abs(flat))
        rms = np.sqrt(np.mean(flat ** 2))
        if rms < 1e-9 or peak < 1e-9:
            return None
        return float(20 * np.log10(peak / rms))
    except Exception:
        return None


def estimate_spectral_rolloff(mono_audio: np.ndarray, sr: int) -> Optional[float]:
    """Frequency (Hz) below which ~99% of spectral energy is contained."""
    try:
        rolloff = librosa.feature.spectral_rolloff(
            y=mono_audio.astype(np.float32), sr=sr, roll_percent=0.99
        )
        return float(np.median(rolloff))
    except Exception:
        return None


def estimate_spectral_cliff(mono_audio: np.ndarray, sr: int):
    """
    Detects a filter/codec-style spectral cliff: the steepest drop-off in the
    long-term average spectrum, in dB per octave, between 3 kHz and 92% of Nyquist.
    Returns (steepness_db_per_octave, cutoff_hz) or (0.0, None).

    Why this instead of "rolloff frequency": low-frequency instruments (frame drums,
    tabla, bass) simply have little high-frequency energy, which says nothing about
    compression. What compression/low-pass filtering leaves behind is a *steep* edge.
    Measured on real percussion stems: natural roll-off 9-22 dB/oct; low-pass or
    codec cutoffs 49-130+ dB/oct.
    """
    try:
        n_fft = 4096
        S = np.abs(librosa.stft(mono_audio.astype(np.float32), n_fft=n_fft, hop_length=n_fft // 2)) ** 2
        p = S.mean(axis=1) + 1e-20
        db = 10 * np.log10(p)
        f = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        peak = db.max()
        nyq = sr / 2.0
        best_slope, best_fc = 0.0, None
        fc = 3000.0
        while fc < 0.92 * nyq:
            lo = (f >= fc / 1.19) & (f < fc)
            hi = (f >= fc) & (f < fc * 1.19)
            if lo.sum() > 2 and hi.sum() > 2:
                lvl_lo, lvl_hi = db[lo].mean(), db[hi].mean()
                slope = (lvl_lo - lvl_hi) / 0.5          # window centres are ~0.5 octave apart
                if lvl_lo > peak - 70 and slope > best_slope:   # ignore cliffs inside pure silence
                    best_slope, best_fc = slope, fc
            fc *= 1.05
        return float(best_slope), (float(best_fc) if best_fc else None)
    except Exception:
        return 0.0, None


def estimate_rt60_ms(mono_audio: np.ndarray, sr: int) -> Optional[float]:
    """
    Blind, approximate RT60 (reverberation time) estimate from speech/audio
    alone, no impulse response available. Method: find energy 'offsets'
    (moments where level drops sharply and keeps decaying), fit a line to the
    decay in dB over the following ~150-400ms, extrapolate to -60dB, average
    across multiple offsets found in the clip. This is a rough heuristic
    (a real RT60 measurement needs a proper impulse response / sine sweep) --
    good enough to flag "this is clearly a live/boomy room" vs "this is dry",
    not to certify acoustic treatment.
    """
    try:
        dbs, frame_len, hop = _frame_db_envelope(mono_audio, sr, frame_ms=20.0)
        if len(dbs) < 20:
            return None
        frame_sec = hop / sr

        decay_window_frames = max(5, int(0.25 / frame_sec))
        estimates = []
        for i in range(2, len(dbs) - decay_window_frames - 1):
            drop = dbs[i - 1] - dbs[i]
            if drop < 8:   # look for a sharp offset (onset->silence transition)
                continue
            window = dbs[i:i + decay_window_frames]
            if len(window) < 4:
                continue
            x = np.arange(len(window)) * frame_sec
            # only use points that are actually decaying (monotonic-ish, above noise floor)
            valid = window > (np.min(window) + 1.0)
            if valid.sum() < 4:
                continue
            slope, intercept = np.polyfit(x[valid], window[valid], 1)
            if slope >= -0.5:   # not a real decay
                continue
            rt60 = -60.0 / slope   # seconds to fall 60dB at this slope
            if 0.05 < rt60 < 5.0:  # sanity bounds
                estimates.append(rt60 * 1000)

        if not estimates:
            return None
        return float(np.median(estimates))
    except Exception:
        return None


def estimate_speaker_count(mono_audio: np.ndarray, sr: int) -> Optional[int]:
    try:
        from sklearn.cluster import KMeans
        mfcc = librosa.feature.mfcc(y=mono_audio.astype(np.float32), sr=sr, n_mfcc=13)
        if mfcc.shape[1] < 20:
            return 1
        feats = mfcc.T
        km = KMeans(n_clusters=2, n_init=4, random_state=0).fit(feats)
        c0, c1 = km.cluster_centers_
        separation = np.linalg.norm(c0 - c1)
        counts = np.bincount(km.labels_)
        balanced_enough = min(counts) / max(counts) > 0.15
        if separation > 25 and balanced_enough:
            return 2
        return 1
    except Exception:
        return None


def speaker_embedding(mono_audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Cheap 'speaker embedding' proxy (mean+std of MFCCs) used only for the
    metadata cross-contamination check -- NOT a real speaker-verification
    embedding. Swap in a proper model (e.g. SpeechBrain ECAPA-TDNN) for
    production-grade speaker-ID label auditing."""
    try:
        mfcc = librosa.feature.mfcc(y=mono_audio.astype(np.float32), sr=sr, n_mfcc=20)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
    except Exception:
        return None


def analyze_file(path: Path, profile: PurposeProfile,
                  check_speaker_count: bool = False) -> QualityReport:
    report = QualityReport()

    try:
        audio, sr, channels = load_audio(path)
    except Exception as e:
        report.ok_to_load = False
        report.load_error = str(e)
        report.issues.append(f"CORRUPT_OR_UNREADABLE: {e}")
        return report

    if audio.size == 0:
        report.ok_to_load = False
        report.load_error = "empty file (0 samples)"
        report.issues.append("CORRUPT_OR_UNREADABLE: empty file")
        return report

    mono = _to_mono(audio.astype(np.float32))
    duration = len(mono) / sr if sr else 0.0

    report.sample_rate = sr
    report.channels = channels
    report.duration_sec = duration

    # --- duration ---
    if duration < profile.min_duration_sec:
        report.issues.append(f"TOO_SHORT: {duration:.2f}s < min {profile.min_duration_sec}s")
    if duration > profile.max_duration_sec:
        report.issues.append(f"TOO_LONG: {duration:.2f}s > max {profile.max_duration_sec}s")

    # --- clipping ---
    clipped = estimate_clipping(audio)
    report.clipped_ratio = clipped
    if clipped > profile.max_repairable_clipped_ratio:
        report.issues.append(f"CLIPPING_HEAVY: {clipped*100:.2f}% samples saturated, more than the "
                              f"{profile.max_repairable_clipped_ratio*100:.1f}% that can be repaired honestly")
    elif clipped > profile.max_clipped_ratio:
        report.issues.append(f"CLIPPING: {clipped*100:.3f}% samples saturated "
                              f"(max allowed {profile.max_clipped_ratio*100:.3f}%)")

    # --- DC offset ---
    dc = estimate_dc_offset(mono)
    report.dc_offset = dc
    if abs(dc) > profile.max_dc_offset:
        report.issues.append(f"DC_OFFSET: {dc:.4f} exceeds {profile.max_dc_offset}")

    # --- SNR ---
    snr = estimate_snr_db(mono, sr)
    report.snr_db = snr
    if profile.min_snr_db is not None and snr < profile.min_snr_db:
        report.issues.append(f"LOW_SNR: {snr:.1f}dB < min {profile.min_snr_db}dB")

    # --- silence ratio (VAD-aware where available) ---
    segments, vad_backend = get_speech_segments(mono, sr, use_real_vad=profile.use_real_vad)
    report.vad_backend = vad_backend
    if segments is not None and len(mono) > 0:
        speech_samples = sum(e - s for s, e in segments)
        silence_ratio = 1.0 - (speech_samples / len(mono))
    else:
        silence_ratio = estimate_silence_ratio(mono, sr)
    report.silence_ratio = silence_ratio
    if silence_ratio > profile.max_silence_ratio:
        report.issues.append(f"MOSTLY_SILENCE: {silence_ratio*100:.1f}% of clip is silent")
    if profile.min_speech_ratio is not None and (1 - silence_ratio) < profile.min_speech_ratio:
        report.issues.append(
            f"INSUFFICIENT_SPEECH: only {(1-silence_ratio)*100:.1f}% active "
            f"(need >= {profile.min_speech_ratio*100:.0f}%)")

    # --- mid-clip dead-air gaps ---
    if profile.max_internal_silence_sec is not None:
        max_gap = estimate_max_internal_silence_sec(mono, sr)
        report.max_internal_silence_sec = max_gap
        if max_gap > profile.max_internal_silence_sec:
            report.issues.append(
                f"LONG_INTERNAL_SILENCE: {max_gap:.1f}s dead-air gap mid-clip "
                f"> max {profile.max_internal_silence_sec}s")

    # --- loudness ---
    lufs = estimate_lufs(audio, sr)
    report.lufs = lufs
    if lufs is not None:
        if abs(lufs - profile.target_lufs) > profile.loudness_tolerance_db + 15:
            report.issues.append(f"LOUDNESS_EXTREME: {lufs:.1f} LUFS vs target {profile.target_lufs}")

    # --- crest factor (dynamic range) ---
    if profile.enable_crest_factor_check:
        cf = estimate_crest_factor_db(audio)
        report.crest_factor_db = cf
        if cf is not None and profile.min_crest_factor_db is not None and cf < profile.min_crest_factor_db:
            report.issues.append(
                f"LOW_DYNAMIC_RANGE: crest factor {cf:.1f}dB < min "
                f"{profile.min_crest_factor_db}dB (over-compressed/limited master?)")

    # --- music-only checks ---
    if profile.enable_music_checks:
        _music_checks(audio, sr, profile, report)

    # --- lossy / filtered source detection ---
    # REJECT only on evidence of a filter/codec (a sharp spectral cliff).
    # A merely low rolloff (bass-heavy instruments) is reported as a WARNING.
    if profile.enable_rolloff_check:
        steepness, cliff_hz = estimate_spectral_cliff(mono, sr)
        report.spectral_cliff_db_per_oct = steepness
        report.spectral_cliff_hz = cliff_hz
        if cliff_hz is not None and steepness >= profile.lossy_cliff_db_per_octave:
            report.issues.append(
                f"LIKELY_LOSSY_SOURCE: sharp spectral cutoff at {cliff_hz:.0f}Hz "
                f"({steepness:.0f} dB/octave, limit {profile.lossy_cliff_db_per_octave:.0f}) -- "
                f"typical of lossy compression or a low-pass filter")

        rolloff = estimate_spectral_rolloff(mono, sr)
        report.spectral_rolloff_hz = rolloff
        nyquist = sr / 2.0
        if rolloff is not None and nyquist > 0:
            frac = rolloff / nyquist
            report.spectral_rolloff_fraction = frac
            if profile.min_spectral_rolloff_fraction is not None and frac < profile.min_spectral_rolloff_fraction:
                report.warnings.append(
                    f"BANDWIDTH_LIMITED: little energy above {rolloff:.0f}Hz "
                    f"({frac*100:.0f}% of Nyquist) -- normal for low-frequency instruments, "
                    f"but worth knowing if you need full-range audio")

    # --- reverb (RT60) ---
    if profile.enable_reverb_check:
        rt60 = estimate_rt60_ms(mono, sr)
        report.rt60_ms = rt60
        if rt60 is not None and profile.max_rt60_ms is not None and rt60 > profile.max_rt60_ms:
            # WARNING only, never a rejection: calibration against synthetic rooms of known
            # reverb time (tests/test_calibration.py) showed this blind estimate cannot tell
            # dry audio from reverberant audio (dry clips read 400-4400 ms, 0.3-1.5 s rooms
            # often returned nothing). Rejecting files on it would discard clean data at random.
            report.warnings.append(
                f"REVERB_ESTIMATE_HIGH: rough RT60 ~{rt60:.0f}ms vs {profile.max_rt60_ms:.0f}ms target "
                f"-- low confidence, please listen to this file if reverb matters for your purpose")

    # --- speaker count (TTS mainly) ---
    if check_speaker_count and profile.reject_multi_speaker:
        n_spk = estimate_speaker_count(mono, sr)
        report.num_speakers_estimate = n_spk
        if n_spk is not None and n_spk > 1:
            report.issues.append("MULTI_SPEAKER_SUSPECTED: possible crosstalk/second speaker")

    return report


def _music_checks(audio: np.ndarray, sr: int, profile: PurposeProfile, report: QualityReport) -> None:
    dead = music_checks.dead_channels(audio)
    if dead:
        names = ", ".join(str(c + 1) for c in dead)
        report.issues.append(f"DEAD_CHANNEL: channel {names} is empty while the other channel has sound "
                              f"(the music is in one ear only)")
    elif music_checks.is_dual_mono(audio):
        report.warnings.append("FAKE_STEREO: both channels are identical (mono saved as stereo); "
                                "there is no stereo information to learn from")
    else:
        loss = music_checks.mono_sum_loss_db(audio)
        report.mono_sum_loss_db = loss
        if loss is not None and loss <= profile.music_min_mono_sum_db:
            report.warnings.append(f"MONO_CANCELLATION: averaging the channels into mono loses {-loss:.1f} dB "
                                    f"(limit {-profile.music_min_mono_sum_db:.0f} dB): the channels partly cancel, "
                                    f"which is a phase problem and will sound thin or empty as mono")
    drops = music_checks.find_dropouts(audio, sr, profile.music_min_dropout_ms)
    report.dropout_count = len(drops)
    if drops:
        s0, ms0 = drops[0]
        report.warnings.append(f"DIGITAL_DROPOUT: {len(drops)} stretch(es) of exact digital silence in the middle of the music "
                                f"(first at {s0:.2f} s, {ms0:.0f} ms): a lost-data glitch, unless the break is intentional")
    starts, ends = music_checks.abrupt_edges(audio)
    if starts or ends:
        which = " and ".join(w for w, f in (("starts", starts), ("ends", ends)) if f)
        report.warnings.append(f"ABRUPT_EDGE: the file {which} mid-sound, which leaves a click "
                                f"(normal for a chunk cut from a longer song)")
