"""
transcript_checker.py
----------------------
Closes the "label/transcript alignment" gap: if you have paired transcripts
for your audio, this module verifies them against the audio.

Two independent checks, both optional and only run if you provide the data:

1. WER cross-check -- if you supply a `transcribe_fn` (any callable that
   takes a wav path and returns text -- e.g. a wrapper around openai-whisper,
   vosk, or your own ASR), we transcribe each file and compute Word Error
   Rate against your provided transcript. High WER flags either a bad
   recording or a wrong/misaligned label. No ASR model ships with this
   pipeline (keeps it lightweight) -- wire in whatever you already use.

2. Language-ID check -- if you pass `expected_language` (an ISO 639-1 code
   like "en"), we run a lightweight text-based language detector
   (`langdetect`) on the *transcript text* and flag mismatches. This catches
   mislabeled language tags even without an ASR model, as long as you have
   the transcript text.

Transcript mapping file formats supported:
  CSV : filename,transcript                 (header optional, auto-detected)
  JSON: {"file1.wav": "transcript text", ...}   OR   [{"filename":.., "transcript":..}, ...]
"""

import csv
import json
from pathlib import Path
from typing import Dict, Optional, Callable, Tuple

try:
    from langdetect import detect as _langdetect_detect
    HAVE_LANGDETECT = True
except ImportError:
    HAVE_LANGDETECT = False


def load_transcript_map(path: str) -> Dict[str, str]:
    """Returns {filename (basename): transcript text}."""
    p = Path(path)
    mapping = {}

    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            mapping = {Path(k).name: v for k, v in data.items()}
        elif isinstance(data, list):
            for row in data:
                fname = row.get("filename") or row.get("file") or row.get("wav")
                text = row.get("transcript") or row.get("text")
                if fname and text is not None:
                    mapping[Path(fname).name] = text
        return mapping

    # CSV (default)
    with open(p, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        reader = csv.reader(f)
        rows = list(reader)
    start_idx = 1 if has_header else 0
    for row in rows[start_idx:]:
        if len(row) < 2:
            continue
        fname, text = row[0], row[1]
        mapping[Path(fname.strip()).name] = text.strip()
    return mapping


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Standard WER via Levenshtein edit distance over words:
    WER = (substitutions + deletions + insertions) / len(reference_words)
    Pure Python, no extra dependency needed.
    """
    ref = reference.strip().lower().split()
    hyp = hypothesis.strip().lower().split()
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    # DP edit distance
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost  # substitution / match
            )
    return d[len(ref)][len(hyp)] / len(ref)


def detect_language(text: str) -> Optional[str]:
    if not HAVE_LANGDETECT or not text or not text.strip():
        return None
    try:
        return _langdetect_detect(text)
    except Exception:
        return None


def check_transcript(
    wav_path: Path,
    transcript_map: Dict[str, str],
    transcribe_fn: Optional[Callable[[Path], str]] = None,
    expected_language: Optional[str] = None,
    wer_reject_threshold: float = 0.5,
) -> Tuple[Optional[str], Optional[float], Optional[str], bool, list]:
    """
    Returns (transcript_text, wer, detected_language, lang_ok, issues)
    - transcript_text: the provided reference transcript, if found
    - wer: word error rate vs an ASR hypothesis, if transcribe_fn was given
    - detected_language: langdetect result on the transcript text
    - lang_ok: True/False/None (None = not checked)
    - issues: list of issue strings to merge into the file's report
    """
    issues = []
    transcript = transcript_map.get(wav_path.name)

    if transcript is None:
        issues.append("NO_TRANSCRIPT_FOUND: file not present in transcript mapping")
        return None, None, None, None, issues

    wer = None
    if transcribe_fn is not None:
        try:
            hypothesis = transcribe_fn(wav_path)
            wer = word_error_rate(transcript, hypothesis)
            if wer > wer_reject_threshold:
                issues.append(f"HIGH_WER: {wer*100:.1f}% vs reference transcript "
                               f"(threshold {wer_reject_threshold*100:.0f}%) -- "
                               f"audio/label mismatch or bad recording")
        except Exception as e:
            issues.append(f"ASR_TRANSCRIBE_FAILED: {e}")

    detected_lang = None
    lang_ok = None
    if expected_language:
        detected_lang = detect_language(transcript)
        if detected_lang is None:
            lang_ok = None
            if not HAVE_LANGDETECT:
                issues.append("LANG_CHECK_SKIPPED: langdetect not installed "
                               "(pip install langdetect)")
        else:
            lang_ok = (detected_lang == expected_language)
            if not lang_ok:
                issues.append(f"LANGUAGE_MISMATCH: transcript looks like '{detected_lang}', "
                               f"expected '{expected_language}'")

    return transcript, wer, detected_lang, lang_ok, issues
