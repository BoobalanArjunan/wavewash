"""
review_sampler.py
--------------------
Closes the "manual spot-check / spectrogram inspection" gap. Automated
metrics can miss things a human catches in five seconds of listening or one
glance at a spectrogram -- this pulls a random sample of passed AND failed
files into a review folder so you actually do that check, instead of
trusting thresholds blindly.

For each sampled file, if matplotlib is available, also renders a PNG
spectrogram for fast visual triage (banding, aliasing, missing bands, odd
cutoffs -- the things automated rolloff/SNR checks can only approximate).
"""

import random
import shutil
from pathlib import Path
from typing import List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False


def _save_spectrogram(wav_path: Path, out_png: Path):
    try:
        y, sr = librosa.load(str(wav_path), sr=None, mono=True)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        fig, ax = plt.subplots(figsize=(8, 3))
        librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(wav_path.name)
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
    except Exception:
        pass


def build_review_sample(records: list, review_dir: Path, sample_size: int = 5,
                          seed: int = 42, make_spectrograms: bool = True) -> Path:
    """
    records: list of FileRecord (needs .status, .source_path, .final_destination, .filename)
    Copies up to `sample_size` passed/fixed files and up to `sample_size` failed
    files into review_dir/{passed,failed}/, with spectrogram PNGs if matplotlib
    is installed, plus a manifest.md explaining what's there.
    """
    rng = random.Random(seed)
    review_dir.mkdir(parents=True, exist_ok=True)

    passed = [r for r in records if r.status in ("passed", "fixed") and r.final_destination]
    failed = [r for r in records if r.status == "failed" and r.final_destination]

    sample_passed = rng.sample(passed, min(sample_size, len(passed)))
    sample_failed = rng.sample(failed, min(sample_size, len(failed)))

    manifest_lines = ["# Manual Review Sample", "",
                       "Automated checks can miss things a human catches instantly. "
                       "Listen to these (and look at the spectrograms, if generated) "
                       "before fully trusting the pipeline's output on a new dataset.",
                       ""]

    for label, sample in (("passed", sample_passed), ("failed", sample_failed)):
        subdir = review_dir / label
        subdir.mkdir(parents=True, exist_ok=True)
        manifest_lines.append(f"## {label.capitalize()} sample ({len(sample)} files)")
        manifest_lines.append("")
        for r in sample:
            src = Path(r.final_destination)
            if not src.exists():
                continue
            dest_audio = subdir / src.name
            try:
                shutil.copy2(src, dest_audio)
            except Exception:
                continue
            note = ""
            if label == "failed" and r.unfixable_reasons:
                note = f" -- reason: {', '.join(r.unfixable_reasons)}"
            manifest_lines.append(f"- `{src.name}`{note}")

            if make_spectrograms and HAVE_MATPLOTLIB:
                png_path = subdir / (src.stem + "_spectrogram.png")
                _save_spectrogram(src, png_path)

        manifest_lines.append("")

    if not HAVE_MATPLOTLIB:
        manifest_lines.append("_(matplotlib not installed -- spectrogram images were skipped; "
                               "`pip install matplotlib` to enable them.)_")

    (review_dir / "manifest.md").write_text("\n".join(manifest_lines), encoding="utf-8")
    return review_dir
