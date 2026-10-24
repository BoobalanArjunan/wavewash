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


def fix_declip(audio: np.ndarray, sr: int, max_run_ms: float = 8.0, threshold: float = 0.999,
               max_overshoot: float = 1.5) -> Tuple[np.ndarray, Optional[str]]:
    """
    Repair clipped runs up to `max_run_ms` long by cubic interpolation, under one rule:
    **a clipped sample's true value is at least the clip level.** The interpolated value is
    therefore never allowed below the sample it replaces.

    Why that rule matters (measured on known-clipped speech-like audio, error against the
    unclipped original, median of 5 signals): plain interpolation lowered the peaks, so after
    "repair" the audio was 18 dB (at 0.1% clipped samples) to 5 dB (at 2%) FURTHER from the
    original than the untouched clipped file, while the re-check reported the clipping as fixed.
    With the rule the repair is about neutral at 0.1% to 0.5% clipping (within 0.4 dB either way)
    and better above: +1.1 dB at 2%, +8 dB at 5%, +13 dB at 10%. Real speech may differ; see
    docs/REAL_DATA_CHECK.md.

    Restored peaks can rise above full scale, but never above `max_overshoot` times the clip level
    (default 1.5): an unconstrained cubic curve can swing far past the true peak (seen on real speech:
    a true peak 3% above the clip level was restored to 2.6 times it), and because the whole file is
    scaled to fit the highest peak, one such swing used to turn the whole file down by 8 dB and made it
    32 dB worse than the untouched clipped file. The cap also limits that turn-down to about 3.5 dB.
    The whole file (all channels together) is then scaled down just enough that the highest peak sits
    at 0.999, and the description says by how much. 1.5 is a reasoned limit, not a measured optimum;
    `max_overshoot=1.0` restores the shape but never lifts a peak above the clip level. Runs longer than `max_run_ms`, and runs too near the start or end of the file to fit a
    curve, are left alone.

    Still parameter-free interpolation, not a trained declipping model: it is not a substitute
    for a dedicated restoration tool on heavily clipped material (the pipeline rejects those, see
    `max_repairable_clipped_ratio`).
    """
    if not max_overshoot >= 1.0:
        raise ValueError("max_overshoot must be at least 1.0 (1.0 = never lift a peak above the clip level)")
    max_run_samples = max(1, int(max_run_ms / 1000 * sr))
    from scipy.interpolate import CubicSpline
    ctx = 4

    def declip_channel(x: np.ndarray):
        # A clipped sample sits exactly at the ceiling. Samples that are merely near full scale are natural
        # peaks, and lifting those would overshoot the truth, so only samples within 1e-4 of the highest
        # positive (or lowest negative) value, and at or above `threshold`, count as clipped.
        x = x.astype(np.float64)
        pos, neg = float(x.max()), float(-x.min())
        clipped_mask = (x >= max(pos - 1e-4, threshold)) | (-x >= max(neg - 1e-4, threshold))
        if not clipped_mask.any():
            return x, 0, 0, 0
        n = len(x)
        i = 0
        fixed_runs = skipped_runs = fixed_samples = 0
        while i < n:
            if not clipped_mask[i]:
                i += 1
                continue
            j = i
            while j < n and clipped_mask[j]:
                j += 1
            left_start, right_end = max(0, i - ctx), min(n, j + ctx)
            if j - i > max_run_samples or i - left_start < 2 or right_end - j < 2:
                skipped_runs += 1
            else:
                xs = np.concatenate([np.arange(left_start, i), np.arange(j, right_end)])
                ys = np.concatenate([x[left_start:i], x[j:right_end]])
                est = CubicSpline(xs, ys)(np.arange(i, j))
                orig = x[i:j]
                # never below the clipped value, and never more than `max_overshoot` times it
                x[i:j] = np.sign(orig) * np.minimum(np.maximum(np.abs(est), np.abs(orig)), max_overshoot * np.abs(orig))
                fixed_runs += 1
                fixed_samples += j - i
            i = j
        return x, fixed_runs, skipped_runs, fixed_samples

    if audio.ndim == 1:
        fixed, runs, skipped, nsamp = declip_channel(audio)
        total = len(audio)
    else:
        fixed = np.empty(audio.shape, dtype=np.float64)
        runs = skipped = nsamp = 0
        for c in range(audio.shape[1]):
            fixed[:, c], r, sk, ns = declip_channel(audio[:, c])
            runs += r
            skipped += sk
            nsamp += ns
        total = audio.size

    if runs == 0:
        return audio, None
    peak = float(np.max(np.abs(fixed)))
    scale_note = ""
    if peak > 0.999:
        fixed = fixed * (0.999 / peak)
        lowered_db = -20 * np.log10(0.999 / peak)
        if lowered_db >= 0.05:          # the 1.0 -> 0.999 trim of an untouched ceiling is not worth mentioning
            scale_note = f"; level lowered {lowered_db:.1f} dB so the restored peaks fit"
    fixed = fixed.astype(audio.dtype if np.issubdtype(audio.dtype, np.floating) else np.float32)
    desc = (f"restored {runs} clipped run(s) ({100 * nsamp / total:.2f}% of samples, up to {max_run_ms:.0f}ms each) "
            f"by interpolation, never below the clip level{scale_note}")
    if skipped:
        desc += f"; {skipped} run(s) left untouched (too long to repair honestly, or at the file edge)"
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


def fix_dead_channel(audio: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    """A channel that is empty while another carries the music (the music is in one ear only): copy the live channel into the empty
    one. This restores a balanced file but invents no stereo information, and the description says so."""
    from .music_checks import dead_channels
    dead = dead_channels(audio)
    if not dead:
        return audio, None
    live = [c for c in range(audio.shape[1]) if c not in dead]
    src = max(live, key=lambda c: float(np.sum(np.square(audio[:, c], dtype=np.float64))))
    fixed = audio.copy()
    for c in dead:
        fixed[:, c] = audio[:, src]
    names = ", ".join(str(c + 1) for c in dead)
    return fixed, f"filled empty channel {names} from channel {src + 1} (the result is dual mono: no stereo information was restored)"


def apply_fix_pipeline(audio: np.ndarray, sr: int, profile: PurposeProfile,
                        issues: List[str]) -> Tuple[np.ndarray, int, List[str]]:
    """
    Runs the relevant fixers in a sensible order based on which issues were found
    (and the profile's on/off flags). Returns (fixed_audio, fixed_sr, list_of_fix_descriptions).
    """
    applied = []

    if any(i.startswith("DEAD_CHANNEL") for i in issues):
        audio, desc = fix_dead_channel(audio)
        if desc:
            applied.append(desc)

    if profile.attempt_declip and any(i.startswith("CLIPPING:") for i in issues):
        audio, desc = fix_declip(audio, sr, max_run_ms=profile.declip_max_run_ms, max_overshoot=profile.declip_max_overshoot)
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
