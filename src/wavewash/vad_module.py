"""
vad_module.py
--------------
Voice Activity Detection with a real-model option and a zero-dependency
fallback, so the pipeline works out of the box but can be upgraded.

- If `use_real_vad=True` AND `torch` + Silero VAD are available (Silero is
  fetched via torch.hub on first use, which needs internet access once),
  we use Silero VAD -- much more accurate than an energy threshold,
  especially on noisy or low-level speech.
- Otherwise we fall back to a simple energy-threshold VAD (same logic the
  rest of the pipeline already used), so nothing breaks if torch isn't
  installed or there's no network access to fetch the model.

To enable real VAD:
    pip install torch torchaudio
    # then pass use_real_vad=True in your purpose profile (config.py),
    # or run: python pipeline.py ... --use-real-vad
"""

import numpy as np
from typing import List, Tuple, Optional

_SILERO_MODEL = None
_SILERO_UTILS = None
_SILERO_LOAD_FAILED = False


def _try_load_silero():
    global _SILERO_MODEL, _SILERO_UTILS, _SILERO_LOAD_FAILED
    if _SILERO_MODEL is not None or _SILERO_LOAD_FAILED:
        return
    try:
        import torch
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad",
            force_reload=False, onnx=False, trust_repo=True,
        )
        _SILERO_MODEL = model
        _SILERO_UTILS = utils
    except Exception:
        _SILERO_LOAD_FAILED = True


def _silero_speech_segments(mono_audio: np.ndarray, sr: int) -> Optional[List[Tuple[int, int]]]:
    _try_load_silero()
    if _SILERO_MODEL is None:
        return None
    try:
        import torch
        (get_speech_timestamps, _, _, _, _) = _SILERO_UTILS
        target_sr = 16000
        if sr != target_sr:
            import librosa
            resampled = librosa.resample(mono_audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        else:
            resampled = mono_audio.astype(np.float32)
        wav_tensor = torch.from_numpy(resampled)
        timestamps = get_speech_timestamps(wav_tensor, _SILERO_MODEL, sampling_rate=target_sr)
        scale = sr / target_sr
        segments = [(int(t["start"] * scale), int(t["end"] * scale)) for t in timestamps]
        return segments
    except Exception:
        return None


def _energy_speech_segments(mono_audio: np.ndarray, sr: int,
                             threshold_db: float = -40.0,
                             frame_ms: float = 25.0) -> List[Tuple[int, int]]:
    frame_len = int(frame_ms / 1000 * sr) or 400
    hop = frame_len // 2
    if len(mono_audio) < frame_len:
        return [(0, len(mono_audio))] if np.max(np.abs(mono_audio)) > 1e-4 else []

    segments = []
    in_speech = False
    seg_start = 0
    for start in range(0, len(mono_audio) - frame_len, hop):
        frame = mono_audio[start:start + frame_len]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        db = 20 * np.log10(rms + 1e-12)
        active = db >= threshold_db
        if active and not in_speech:
            in_speech = True
            seg_start = start
        elif not active and in_speech:
            in_speech = False
            segments.append((seg_start, start))
    if in_speech:
        segments.append((seg_start, len(mono_audio)))
    return segments


def get_speech_segments(mono_audio: np.ndarray, sr: int,
                         use_real_vad: bool = False) -> Tuple[Optional[List[Tuple[int, int]]], str]:
    """
    Returns (segments, backend_used).
    segments: list of (start_sample, end_sample) tuples of detected speech.
    backend_used: "silero" or "energy".
    """
    if use_real_vad:
        segs = _silero_speech_segments(mono_audio, sr)
        if segs is not None:
            return segs, "silero"
    return _energy_speech_segments(mono_audio, sr), "energy"
