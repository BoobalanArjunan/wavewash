"""
audio_fixer.py
---------------
Attempts to automatically repair the issues quality_checker.py finds.
Every fix function takes (audio, sr) and returns (new_audio, new_sr, fix_description or None).
If a fix isn't possible/safe, it returns the audio unchanged and no description --
the caller decides whether the *remaining* issue is fatal.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, List

try:
    import pyloudnorm as pyln
    HAVE_PYLOUDNORM = True
except ImportError:
    HAVE_PYLOUDNORM = False

from .config import PurposeProfile


def fix_resample(audio: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int, Optional[str]]:
    if sr == target_sr:
        return audio, sr, None
    if audio.ndim == 1:
        new_audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    else:
        chans = [librosa.resample(audio[:, c].astype(np.float32), orig_sr=sr, target_sr=target_sr)
                 for c in range(audio.shape[1])]
        new_audio = np.stack(chans, axis=1)
    return new_audio, target_sr, f"resampled {sr}Hz -> {target_sr}Hz"


def fix_channels(audio: np.ndarray, target_channels: int) -> Tuple[np.ndarray, Optional[str]]:
    cur_channels = 1 if audio.ndim == 1 else audio.shape[1]
    if cur_channels == target_channels:
        return audio, None
    if target_channels == 1 and cur_channels > 1:
        return np.mean(audio, axis=1), f"downmixed {cur_channels}ch -> mono"
    if target_channels == 2 and cur_channels == 1:
        return np.stack([audio, audio], axis=1), "upmixed mono -> stereo (duplicated channel)"
    return audio, None


def fix_dc_offset(audio: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    if audio.ndim == 1:
        offset = np.mean(audio)
        if abs(offset) < 1e-5:
            return audio, None
        return audio - offset, f"removed DC offset ({offset:.4f})"
    offsets = np.mean(audio, axis=0)
    if np.max(np.abs(offsets)) < 1e-5:
        return audio, None
    return audio - offsets, f"removed DC offset (per-channel)"


def fix_trim_silence(audio: np.ndarray, sr: int, top_db: float = 35.0) -> Tuple[np.ndarray, Optional[str]]:
    mono_for_detection = audio if audio.ndim == 1 else np.mean(audio, axis=1)
    try:
        _, index = librosa.effects.trim(mono_for_detection.astype(np.float32), top_db=top_db)
    except Exception:
        return audio, None
    start, end = index
    if start == 0 and end == len(mono_for_detection):
        return audio, None
    trimmed = audio[start:end] if audio.ndim == 1 else audio[start:end, :]
    if len(trimmed) < 0.05 * sr:  # don't trim down to almost nothing
        return audio, None
    return trimmed, f"trimmed leading/trailing silence ({start/sr:.2f}s / " \
                     f"{(len(mono_for_detection)-end)/sr:.2f}s removed)"


def fix_loudness(audio: np.ndarray, sr: int, target_lufs: float) -> Tuple[np.ndarray, Optional[str]]:
    if not HAVE_PYLOUDNORM:
        return audio, None
    try:
        meter = pyln.Meter(sr)
        data = audio if audio.ndim > 1 else audio.reshape(-1, 1)
        current_lufs = meter.integrated_loudness(data)
        if current_lufs == float("-inf"):
            return audio, None
        normalized = pyln.normalize.loudness(data, current_lufs, target_lufs)
        # prevent new clipping introduced by normalization
        peak = np.max(np.abs(normalized))
        if peak > 0.999:
            normalized = normalized * (0.999 / peak)
        result = normalized if audio.ndim > 1 else normalized.reshape(-1)
        return result, f"normalized loudness {current_lufs:.1f} -> {target_lufs:.1f} LUFS"
    except Exception:
        return audio, None


def fix_declip(audio: np.ndarray, sr: int, max_run_ms: float = 8.0, threshold: float = 0.999) -> Tuple[np.ndarray, Optional[str]]:
    """
    De-clip by interpolating across clipped runs up to `max_run_ms` long.
    Runs longer than that are left alone -- too much of the original waveform
    is genuinely gone at that point; interpolating across it would be
    fabricating audio, not repairing it, and the file's overall clipped_ratio
    should fail the quality check and get rejected instead.

    Uses cubic interpolation (via scipy) when there are enough surrounding
    samples to fit one -- a meaningfully smoother, more natural-sounding
    repair than a straight line for anything beyond a couple of samples.
    Falls back to linear interpolation for very short runs or right at the
    start/end of the file, where a cubic fit isn't well-defined.

    NOTE ON SCOPE: this is still parameter-free interpolation, not a real
    declipping model (e.g. sparse-reconstruction or consistent-signal-recovery
    algorithms). It's a solid, fast improvement over pure linear interpolation
    within the run lengths it targets -- not a substitute for a dedicated
    restoration tool on heavily clipped material.
    """
    max_run_samples = max(1, int(max_run_ms / 1000 * sr))

    def declip_channel(x: np.ndarray) -> Tuple[np.ndarray, int, int]:
        clipped_mask = np.abs(x) >= threshold
        if not clipped_mask.any():
            return x, 0, 0
        x = x.copy()
        n = len(x)
        i = 0
        fixed_runs = 0
        skipped_runs = 0
        while i < n:
            if clipped_mask[i]:
                j = i
                while j < n and clipped_mask[j]:
                    j += 1
                run_len = j - i
                if run_len > max_run_samples:
                    skipped_runs += 1
                    i = j
                    continue

                # gather a few good samples on either side for a cubic fit
                context = 3
                left_start = max(0, i - context)
                right_end = min(n, j + context)
                have_left = i - left_start
                have_right = right_end - j

                if run_len >= 3 and have_left >= 2 and have_right >= 2:
                    try:
                        from scipy.interpolate import CubicSpline
                        xs = np.concatenate([np.arange(left_start, i), np.arange(j, right_end)])
                        ys = np.concatenate([x[left_start:i], x[j:right_end]])
                        cs = CubicSpline(xs, ys)
                        x[i:j] = cs(np.arange(i, j))
                    except Exception:
                        left = x[i - 1] if i > 0 else 0.0
                        right = x[j] if j < n else 0.0
                        x[i:j] = np.linspace(left, right, run_len)
                else:
                    left = x[i - 1] if i > 0 else 0.0
                    right = x[j] if j < n else 0.0
                    x[i:j] = np.linspace(left, right, run_len)

                # a repaired stretch shouldn't itself re-clip -- keep it just under scale
                x[i:j] = np.clip(x[i:j], -0.999, 0.999)
                fixed_runs += 1
                i = j
            else:
                i += 1
        return x, fixed_runs, skipped_runs

    if audio.ndim == 1:
        fixed, runs, skipped = declip_channel(audio)
    else:
        fixed = audio.copy()
        runs = skipped = 0
        for c in range(audio.shape[1]):
            fixed[:, c], r, s = declip_channel(audio[:, c])
            runs += r
            skipped += s

    if runs == 0:
        return audio, None
    desc = f"interpolated {runs} clipped run(s) (up to {max_run_ms:.0f}ms each)"
    if skipped:
        desc += f"; {skipped} longer run(s) left untouched (too long to repair honestly)"
    return fixed, desc


def fix_denoise(audio: np.ndarray, sr: int, strength: float = 0.8) -> Tuple[np.ndarray, Optional[str]]:
    """
    Spectral-subtraction noise reduction (repair for LOW_SNR). The noise profile is
    measured from the quietest ~15% of the clip, so it adapts to each recording.
    Pure numpy/librosa -- no extra dependency.

    Honest limits: classical DSP, not a trained model. Good on steady background noise
    (hiss, hum, fans, room tone); weak on non-stationary noise (traffic, other voices,
    music bleed). The result is only kept if it measurably improves the estimated SNR --
    otherwise the original is returned and NO fix is claimed, so a failed repair can
    never masquerade as a success.
    """
    from scipy.ndimage import uniform_filter
    from .quality_checker import estimate_snr_db

    n_fft, hop = 2048, 512
    min_gain = 10 ** (-25 / 20)          # never remove more than ~25 dB in any bin

    def process(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        S = librosa.stft(x, n_fft=n_fft, hop_length=hop)
        mag, phase = np.abs(S), np.angle(S)
        # noise profile from the quietest frames
        frame_energy = mag.mean(axis=0)
        k = max(2, int(0.15 * mag.shape[1]))
        quiet = mag[:, np.argsort(frame_energy)[:k]]
        noise_est = quiet.mean(axis=1, keepdims=True) + 0.5 * quiet.std(axis=1, keepdims=True)
        gain = 1.0 - float(strength) * (noise_est / (mag + 1e-9))
        gain = np.clip(gain, min_gain, 1.0)
        gain = uniform_filter(gain, size=(3, 3))            # smooth to avoid "musical noise"
        y = librosa.istft(mag * gain * np.exp(1j * phase), hop_length=hop, length=len(x))
        return y.astype(np.float32)

    mono_before = audio if audio.ndim == 1 else audio.mean(axis=1)
    snr_before = estimate_snr_db(mono_before.astype(np.float32), sr)

    if audio.ndim == 1:
        cleaned = process(audio)
    else:
        cleaned = np.stack([process(audio[:, c]) for c in range(audio.shape[1])], axis=1)

    mono_after = cleaned if cleaned.ndim == 1 else cleaned.mean(axis=1)
    snr_after = estimate_snr_db(mono_after, sr)
    if snr_after < snr_before + 1.0:
        return audio, None
    return cleaned, f"reduced background noise (estimated SNR {snr_before:.1f} -> {snr_after:.1f} dB)"


def apply_fix_pipeline(audio: np.ndarray, sr: int, profile: PurposeProfile,
                        issues: List[str]) -> Tuple[np.ndarray, int, List[str]]:
    """
    Runs the relevant fixers in a sensible order based on which issues were found
    (and the profile's on/off flags). Returns (fixed_audio, fixed_sr, list_of_fix_descriptions).
    """
    applied = []

    if profile.attempt_declip and any("CLIPPING" in i for i in issues):
        audio, desc = fix_declip(audio, sr, max_run_ms=profile.declip_max_run_ms)
        if desc:
            applied.append(desc)

    if profile.auto_fix_dc_offset and any("DC_OFFSET" in i for i in issues):
        audio, desc = fix_dc_offset(audio)
        if desc:
            applied.append(desc)

    if getattr(profile, "enable_denoise", False) and any("LOW_SNR" in i for i in issues):
        audio, desc = fix_denoise(audio, sr, strength=getattr(profile, "denoise_strength", 0.8))
        if desc:
            applied.append(desc)

    if profile.trim_silence_edges:
        audio, desc = fix_trim_silence(audio, sr)
        if desc:
            applied.append(desc)

    if profile.auto_resample:
        audio, sr, desc = fix_resample(audio, sr, profile.target_sample_rate)
        if desc:
            applied.append(desc)

    audio, desc = fix_channels(audio, profile.target_channels)
    if desc:
        applied.append(desc)

    if profile.auto_normalize_loudness:
        audio, desc = fix_loudness(audio, sr, profile.target_lufs)
        if desc:
            applied.append(desc)

    return audio, sr, applied


def save_audio(audio: np.ndarray, sr: int, out_path: Path, bit_depth: int = 16):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subtype = "PCM_16" if bit_depth == 16 else "PCM_24"
    sf.write(str(out_path), audio, sr, subtype=subtype)
