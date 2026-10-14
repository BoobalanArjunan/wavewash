"""
alignment_checker.py
----------------------
Partial stand-in for real forced-alignment timestamp verification.

What this IS: a fast, dependency-light sanity check that flags transcripts
whose word count is wildly implausible for the clip's duration (e.g. a
40-word transcript on a 1-second clip, or 2 words on a 30-second clip).
This catches the most common real-world label bug -- a transcript that
belongs to a completely different, wrongly-matched audio file -- without
needing an aligner installed.

What this is NOT: real forced alignment. It does not tell you where each
word falls in time, and it will not catch a transcript that's plausible in
length but wrong in content (that's what the WER check in
transcript_checker.py is for) or off-by-a-few-hundred-ms boundary drift.

If you need real timestamp-level alignment QA (e.g. verifying segment
boundaries for TTS training), install the Montreal Forced Aligner (MFA) and
call it per-file; the hook point is `run_external_aligner()` below -- fill
in the subprocess call for your installed `mfa align` command and wire its
output (word-level timestamps) into whatever boundary-drift check you need.
Not bundled here because MFA needs its own conda environment, pretrained
acoustic models, and pronunciation dictionaries (multi-GB, per-language) --
too heavy to ship inside a general-purpose pipeline.
"""

from pathlib import Path
from typing import Optional, Tuple

# Typical sustainable speaking rates for read/conversational speech.
# Outside this range for a whole clip almost always means a duration/transcript
# mismatch rather than genuinely fast/slow speech.
MIN_PLAUSIBLE_WORDS_PER_SEC = 0.8
MAX_PLAUSIBLE_WORDS_PER_SEC = 5.0


def speaking_rate_sanity_check(transcript: str, duration_sec: float) -> Tuple[Optional[float], list]:
    """
    Returns (words_per_second, issues). issues is empty if the rate looks
    plausible; contains an ALIGNMENT_SUSPECT string if not.
    """
    if not transcript or duration_sec is None or duration_sec <= 0:
        return None, []

    word_count = len(transcript.strip().split())
    if word_count == 0:
        return None, []

    rate = word_count / duration_sec
    issues = []
    if rate < MIN_PLAUSIBLE_WORDS_PER_SEC:
        issues.append(
            f"ALIGNMENT_SUSPECT: {word_count} words over {duration_sec:.1f}s "
            f"({rate:.2f} words/sec) is implausibly slow -- transcript may belong "
            f"to a different/longer recording"
        )
    elif rate > MAX_PLAUSIBLE_WORDS_PER_SEC:
        issues.append(
            f"ALIGNMENT_SUSPECT: {word_count} words over {duration_sec:.1f}s "
            f"({rate:.2f} words/sec) is implausibly fast -- transcript may belong "
            f"to a different/shorter recording"
        )
    return rate, issues


def run_external_aligner(wav_path: Path, transcript: str, aligner_cmd: Optional[str] = None):
    """
    Hook for plugging in a real forced aligner (e.g. Montreal Forced Aligner).
    Not implemented by default -- raises NotImplementedError so it's obvious
    this needs to be filled in for your environment rather than silently
    doing nothing. Example sketch for MFA (once installed and configured):

        import subprocess, json
        result = subprocess.run(
            ["mfa", "align_one", str(wav_path), transcript_file, dictionary, acoustic_model, out_dir],
            capture_output=True, text=True
        )
        # parse out_dir/*.TextGrid for word-level timestamps, then compare
        # against your expected segment boundaries.
    """
    raise NotImplementedError(
        "run_external_aligner() is a hook, not an implementation -- wire in "
        "your installed forced aligner (e.g. MFA) here if you need true "
        "word-level timestamp verification. speaking_rate_sanity_check() "
        "above is the lightweight alternative that ships and runs by default."
    )
