"""Real-speech check for WaveWash 0.2, using LibriSpeech (or any folder of clean speech).

It answers two questions:

  1. Which checks fire on CLEAN speech, per purpose profile? (Each one is a false rejection or false warning, since
     LibriSpeech dev-clean is clean read speech. It also prints how the measured values are spread, so thresholds
     can be set from data instead of guesses.)
  2. Does the clipping repair help on REAL speech? We clip real utterances by a known amount, repair them, and measure
     how far the result is from the original, comparing the 0.1 repair with the 0.2 repair. This is where the
     heavy-clipping limit (max_repairable_clipped_ratio) gets its evidence.

    python scripts/real_speech_check.py --librispeech ~/LibriSpeech/dev-clean --out real_check

Your LibriSpeech folder is never modified. Everything is written under --out.
"""
from __future__ import annotations

import argparse
import collections
import statistics as st
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    from wavewash.audio_fixer import fix_declip
    from wavewash.config import PROFILES
    from wavewash.quality_checker import analyze_file
except ImportError:
    sys.exit("wavewash is not installed (pip install --pre wavewash, or pip install -e . from a checkout)")


def legacy_declip(audio, sr, max_run_ms=8.0, threshold=0.999):
    """The 0.1 repair (plain interpolation, no 'never below the clip level' rule), kept here only for comparison."""
    from scipy.interpolate import CubicSpline
    x = audio.astype(np.float64).copy()
    n, maxrun = len(x), max(1, int(max_run_ms / 1000 * sr))
    m = np.abs(x) >= threshold
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        if j - i <= maxrun:
            ls, re = max(0, i - 3), min(n, j + 3)
            if j - i >= 3 and i - ls >= 2 and re - j >= 2:
                xs = np.concatenate([np.arange(ls, i), np.arange(j, re)])
                x[i:j] = CubicSpline(xs, np.concatenate([x[ls:i], x[j:re]]))(np.arange(i, j))
            else:
                x[i:j] = np.linspace(x[i - 1] if i > 0 else 0.0, x[j] if j < n else 0.0, j - i)
            x[i:j] = np.clip(x[i:j], -0.999, 0.999)
        i = j
    return x


def clip_to_ratio(x, target):
    """Scale x so about `target` of the samples reach full scale, then clip. Returns (unclipped truth, clipped)."""
    a = np.abs(x)
    if a.max() == 0:
        return None, None
    # gain so that the (1 - target) quantile of |x| lands exactly at full scale
    q = float(np.quantile(a, 1.0 - target))
    if q <= 0:
        return None, None
    g = 1.0 / q
    t = x * g
    return t, np.clip(t, -1.0, 1.0)


def best_gain_error(a, truth):
    g = float(np.dot(a, truth) / max(np.dot(a, a), 1e-30))
    return float(np.sum((g * a - truth) ** 2))


def pct(v, qs=(5, 50, 95)):
    v = [x for x in v if x is not None]
    return ", ".join(f"p{q} {np.percentile(v, q):.3g}" for q in qs) if v else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--librispeech", type=Path, required=True)
    ap.add_argument("--files", type=int, default=200, help="utterances for the false-rejection check (default 200)")
    ap.add_argument("--clip-files", type=int, default=40, help="utterances for the clipping repair test (default 40)")
    ap.add_argument("--out", type=Path, default=Path("real_check"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    files = sorted(a.librispeech.rglob("*.flac"))
    if len(files) < 20:
        sys.exit(f"found only {len(files)} .flac files in {a.librispeech}")
    rng = np.random.RandomState(a.seed)
    rng.shuffle(files)
    a.out.mkdir(parents=True, exist_ok=True)
    L = ["# WaveWash 0.2: real-speech check (LibriSpeech)", ""]

    # ---------------------------------------------------------------- 1. clean speech through the checks
    sample = files[: a.files]
    L += [f"## 1. Clean speech through the checks ({len(sample)} utterances)", "",
          "LibriSpeech dev-clean is clean read speech, so every issue below is a **false rejection** (an `issue`) or a "
          "**false warning**. A good profile has near zero issues here.", ""]
    for key in ("asr", "tts", "speaker_id", "custom"):
        prof = PROFILES[key]
        issues = collections.Counter()
        warns = collections.Counter()
        rejected = 0
        vals = collections.defaultdict(list)
        for i, p in enumerate(sample, 1):
            r = analyze_file(p, prof, check_speaker_count=False)
            if r.issues:
                rejected += 1
            for x in r.issues:
                issues[x.split(":")[0]] += 1
            for x in r.warnings:
                warns[x.split(":")[0]] += 1
            for name in ("snr_db", "silence_ratio", "lufs", "clipped_ratio", "dc_offset", "duration_sec"):
                vals[name].append(getattr(r, name))
            if i % 50 == 0:
                print(f"  {key}: {i}/{len(sample)}", flush=True)
        L += [f"### profile `{key}`", "",
              f"- files with at least one issue: **{rejected}/{len(sample)}** ({100 * rejected / len(sample):.0f}%)",
              "- issues: " + (", ".join(f"{k} x{v}" for k, v in issues.most_common()) or "none"),
              "- warnings: " + (", ".join(f"{k} x{v}" for k, v in warns.most_common()) or "none"), ""]
        if key == "asr":
            L += ["Measured spread on clean speech (to set thresholds from):", "",
                  "| value | spread | profile limit |", "|---|---|---|",
                  f"| SNR (dB) | {pct(vals['snr_db'])} | min {prof.min_snr_db} |",
                  f"| silence ratio | {pct(vals['silence_ratio'])} | max {prof.max_silence_ratio} |",
                  f"| loudness (LUFS) | {pct(vals['lufs'])} | target {prof.target_lufs} |",
                  f"| clipped ratio | {pct(vals['clipped_ratio'])} | max {prof.max_clipped_ratio} |",
                  f"| DC offset | {pct(vals['dc_offset'])} | max {prof.max_dc_offset} |",
                  f"| duration (s) | {pct(vals['duration_sec'])} | {prof.min_duration_sec} to {prof.max_duration_sec} |", ""]

    # ---------------------------------------------------------------- 2. clipping repair on real speech
    L += [f"## 2. Clipping repair on real speech ({a.clip_files} utterances)", "",
          "Each utterance is clipped by a known share of samples. 'Improvement' is how much closer to the original "
          "(unclipped) waveform the repaired audio is than the untouched clipped audio, after the best overall gain. "
          "**Positive = the repair helped; negative = the repair made it worse.** The 0.1 repair is shown for comparison.", "",
          "| clipped share | 0.2 repair: median [worst .. best] dB | files where 0.2 is worse by > 0.5 dB | 0.1 repair: median [worst .. best] dB |",
          "|---|---|---|---|"]
    prof = PROFILES["asr"]
    clip_sample = [p for p in files if sf.info(str(p)).duration >= 3.0][: a.clip_files]
    listen = a.out / "listen"
    listen.mkdir(exist_ok=True)
    wrote_listen = set()
    for ratio in (0.001, 0.005, 0.02, 0.05, 0.10, 0.20):
        new, old, worse = [], [], 0
        for p in clip_sample:
            x, sr = sf.read(str(p), dtype="float64")
            truth, clipped = clip_to_ratio(x, ratio)
            if truth is None:
                continue
            e0 = best_gain_error(clipped, truth)
            fixed, _ = fix_declip(clipped.astype(np.float32), sr)
            e_new = best_gain_error(np.asarray(fixed, dtype=np.float64), truth)
            e_old = best_gain_error(legacy_declip(clipped.astype(np.float32), sr), truth)
            g_new, g_old = 10 * np.log10(e0 / max(e_new, 1e-30)), 10 * np.log10(e0 / max(e_old, 1e-30))
            new.append(g_new)
            old.append(g_old)
            worse += g_new < -0.5
            if ratio in (0.02, 0.10) and ratio not in wrote_listen:
                tag = f"{int(ratio * 100)}pct"
                peak = max(np.abs(truth).max(), 1.0)
                sf.write(str(listen / f"{tag}_1_original.wav"), (truth / peak).astype(np.float32), sr)
                sf.write(str(listen / f"{tag}_2_clipped.wav"), (clipped / peak).astype(np.float32), sr)
                sf.write(str(listen / f"{tag}_3_repaired_0.2.wav"), np.asarray(fixed, dtype=np.float32), sr)
                sf.write(str(listen / f"{tag}_4_repaired_0.1.wav"),
                         np.clip(legacy_declip(clipped.astype(np.float32), sr), -1, 1).astype(np.float32), sr)
                wrote_listen.add(ratio)
        if new:
            L.append(f"| {ratio * 100:g}% | {st.median(new):+.2f} [{min(new):+.2f} .. {max(new):+.2f}] | {worse}/{len(new)} | "
                     f"{st.median(old):+.2f} [{min(old):+.2f} .. {max(old):+.2f}] |")
        print(f"  clipping {ratio * 100:g}%: median improvement {st.median(new):+.2f} dB", flush=True)
    L += ["", "How to read it: the heavy-clipping limit (`max_repairable_clipped_ratio`, default 5%) should sit where the "
          "0.2 improvement stops being clearly positive for most files. If the median is still well above +3 dB at 10% "
          "and almost no file is worse, the limit can be raised; if it is near zero or many files are worse at 5%, lower it.", ""]

    # what the checker does with each level
    L += ["### What the quality check says at each level (profile `asr`)", "",
          "| clipped share | CLIPPING | CLIPPING_HEAVY | neither |", "|---|---|---|---|"]
    for ratio in (0.0005, 0.001, 0.005, 0.02, 0.05, 0.10, 0.20):
        c = h = n = 0
        tmpdir = a.out / "_tmp"
        tmpdir.mkdir(exist_ok=True)
        for p in clip_sample[:15]:
            x, sr = sf.read(str(p), dtype="float64")
            _, clipped = clip_to_ratio(x, ratio)
            if clipped is None:
                continue
            tp = tmpdir / "c.wav"
            sf.write(str(tp), clipped.astype(np.float32), sr)
            r = analyze_file(tp, prof)
            f = [i.split(":")[0] for i in r.issues if i.startswith("CLIPPING")]
            c += "CLIPPING" in f
            h += "CLIPPING_HEAVY" in f
            n += not f
        L.append(f"| {ratio * 100:g}% | {c} | {h} | {n} |")
    L += ["", "## Listen (3 minutes)", "",
          f"Open the files in `{listen}`: for 2% and 10% clipping you have the original, the clipped version, the 0.2 repair "
          "and the 0.1 repair. Is the 0.2 repair closer to the original than the clipped one? Does the 0.1 repair sound "
          "worse than the clipped one at 2%? Tell me what you hear.", ""]
    (a.out / "results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwritten to {a.out / 'results.md'}")


if __name__ == "__main__":
    main()
