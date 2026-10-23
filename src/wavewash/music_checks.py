"""
music_checks.py
---------------
Checks that only make sense for MUSIC files (the `music_gen` purpose turns them on). They look for damage that
speech checks cannot see: a dead channel, fake stereo, channels that cancel when summed to mono, digital dropouts
in the middle of a song, and files that start or end with a click because they were cut mid-sound.

All thresholds are reasoned defaults, not measured on a real music corpus (see docs/REAL_DATA_CHECK.md); anything
that could be intentional in real music (a digital-silence break in electronic music, a chunk cut from a longer
song) is a WARNING, never a rejection. A dead channel is the only rejection, because it is fixable and verified.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

DEAD_RMS = 1e-4          # a channel quieter than this (about -80 dBFS) is empty
LIVE_RMS = 1e-3          # while another channel is louder than this (about -60 dBFS)
DUAL_MONO_TOL = 1e-4     # channels equal to within this share of the peak count as identical
DROPOUT_CONTEXT_MS = 200.0
DROPOUT_CONTEXT_RMS = 3e-3   # sound this loud (about -50 dBFS) must surround a run of exact zeros for it to be a dropout
EDGE_FRACTION = 0.10     # the first/last sample being this share of the file's loud level or more means a hard cut


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) if x.size else 0.0


def dead_channels(audio: np.ndarray) -> List[int]:
    """Indices of channels that are empty while at least one other channel carries sound. [] for mono or healthy files."""
    if audio.ndim != 2 or audio.shape[1] < 2:
        return []
    rms = [_rms(audio[:, c]) for c in range(audio.shape[1])]
    if max(rms) < LIVE_RMS:
        return []
    return [c for c, r in enumerate(rms) if r < DEAD_RMS]


def is_dual_mono(audio: np.ndarray) -> bool:
    """True for a multi-channel file whose channels are identical (stereo that carries no stereo information)."""
    if audio.ndim != 2 or audio.shape[1] < 2:
        return False
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return False
    return all(float(np.max(np.abs(audio[:, 0] - audio[:, c]))) <= DUAL_MONO_TOL * peak for c in range(1, audio.shape[1]))


def mono_sum_loss_db(audio: np.ndarray) -> Optional[float]:
    """How much quieter (dB, 0 or negative) the file gets when its two channels are averaged into mono.
    0 = no loss; about -3 = unrelated channels; -6 or lower = the channels largely cancel. None unless 2 channels."""
    if audio.ndim != 2 or audio.shape[1] != 2:
        return None
    l, r = audio[:, 0].astype(np.float64), audio[:, 1].astype(np.float64)
    p_avg = (np.mean(l * l) + np.mean(r * r)) / 2.0
    if p_avg <= 0.0:
        return None
    p_mono = np.mean(((l + r) / 2.0) ** 2)
    return float(max(10.0 * np.log10(max(p_mono / p_avg, 1e-6)), -60.0))


def find_dropouts(audio: np.ndarray, sr: int, min_ms: float = 20.0) -> List[Tuple[float, float]]:
    """Runs of exact digital zero (all channels) of at least `min_ms` that have sound on BOTH sides, as (start_s, length_ms).
    Silence at the start or end of the file is not a dropout, and neither is a quiet passage (it has a noise floor, not exact zeros)."""
    x = audio if audio.ndim == 2 else audio[:, None]
    n = x.shape[0]
    zero = np.all(x == 0, axis=1)
    if not zero.any():
        return []
    min_n = max(1, int(min_ms / 1000.0 * sr))
    ctx = max(1, int(DROPOUT_CONTEXT_MS / 1000.0 * sr))
    d = np.diff(np.concatenate([[0], zero.astype(np.int8), [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    out = []
    for s, e in zip(starts, ends):
        if e - s < min_n or s == 0 or e >= n:
            continue
        before, after = x[max(0, s - ctx):s], x[e:e + ctx]
        if _rms(before) >= DROPOUT_CONTEXT_RMS and _rms(after) >= DROPOUT_CONTEXT_RMS:
            out.append((s / sr, (e - s) / sr * 1000.0))
    return out


def abrupt_edges(audio: np.ndarray) -> Tuple[bool, bool]:
    """(starts abruptly, ends abruptly): the very first / last sample is far from zero compared with the file's loud level,
    which is what a cut through the middle of a sound leaves (it clicks). A fade or a cut at a zero crossing is not flagged."""
    x = np.abs(audio) if audio.ndim == 1 else np.max(np.abs(audio), axis=1)
    if x.size < 16:
        return False, False
    level = float(np.percentile(x, 95))
    if level <= 0.0:
        return False, False
    return bool(x[0] > EDGE_FRACTION * level), bool(x[-1] > EDGE_FRACTION * level)
