#!/usr/bin/env python3
"""
pipeline.py
-----------
Main entry point for the WaveWash — audio QC & data prep pipeline.

WHAT IT DOES
1. Locate wav files: a single file, a flat folder, or a main folder with many
   nested subfolders -- any depth.
2. Ask (or accept via --purpose) what the data is for: ASR / TTS / Speaker ID /
   Music Generation / Custom.
3. Run a full quality check on every file: clipping, DC offset, SNR, silence
   ratio (VAD-based), mid-clip dead-air gaps, sample rate, loudness, spectral
   rolloff (codec-artifact / lossy-source detection), reverb (RT60 estimate),
   dynamic range (crest factor), multi-speaker detection.
4. Auto-fix what can be fixed (resample, trim silence, de-DC-offset, loudness
   normalize, interpolate short clipped runs).
5. Detect and remove duplicate/near-duplicate audio.
6. (Optional) Cross-check against provided transcripts: WER (if you supply an
   ASR callable) and language-ID mismatch detection.
7. (Optional) Cross-check against provided speaker/session metadata: speaker
   balance report, cross-contamination (mislabel) flags, session-safe
   train/val split.
8. Route every file to exactly one place: destination/, error/ (labeled), or
   duplicates/.
9. Write a full audit trail: text log + CSV + JSON, a markdown summary, a
   dataset-level diversity report, and a random manual-review sample.

USAGE
    python pipeline.py --input /path/to/wav_data --output /path/to/clean_dataset \
        --purpose asr

    # interactive purpose selection (omit --purpose):
    python pipeline.py --input /path/to/wav_data --output /path/to/clean_dataset

    # with transcripts (WER + language check):
    python pipeline.py --input ./raw --output ./clean --purpose asr \
        --transcripts ./transcripts.csv --expected-language en

    # with speaker/session metadata:
    python pipeline.py --input ./raw --output ./clean --purpose speaker_id \
        --metadata ./meta.csv --make-split --train-ratio 0.9

    # real VAD (needs `pip install torch torchaudio`, one-time model fetch):
    python pipeline.py --input ./raw --output ./clean --purpose asr --use-real-vad

    # just see what would happen, don't write any files:
    python pipeline.py --input ./raw --output ./clean --purpose music_gen --dry-run
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

from .config import PROFILES, DEFAULT_REVIEW_SAMPLE_SIZE
from .audio_scanner import find_audio_files, describe_input
from .quality_checker import analyze_file, load_audio, speaker_embedding, _to_mono
from .audio_fixer import apply_fix_pipeline, save_audio
from .dedup_detector import (file_md5, audio_content_hash, spectral_signature,
                            normalized_16k, find_near_duplicate_groups)
from .logger_setup import build_text_logger, AuditLog, FileRecord
from .transcript_checker import load_transcript_map, check_transcript
from .alignment_checker import speaking_rate_sanity_check
from .metadata_checker import load_metadata, speaker_balance_report, cross_contamination_check, session_aware_split
from .diversity_report import build_diversity_report
from .review_sampler import build_review_sample

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def prompt_for_purpose() -> str:
    print("\nWhat is this audio data for?")
    options = list(PROFILES.items())
    for i, (key, profile) in enumerate(options, 1):
        print(f"  {i}. {profile.name}  -- {profile.description}")

    while True:
        choice = input(f"Enter a number (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1][0]
        print("Please enter a valid number.")


def run_duplicate_pass(files, logger, audit: AuditLog, purpose: str,
                        mode: str, dupes_dir: Path, dry_run: bool):
    """
    mode: 'exact' -> byte-identical or sample-identical files only
          'near'  -> exact + the same recording re-exported (resampled, gain-changed,
                     trimmed). Different recordings that share tempo/key/groove --
                     e.g. a solo stem vs. the mix containing it -- are NEVER treated
                     as duplicates. See dedup_detector.py for the measured thresholds.
    Nothing is ever deleted: duplicates are copied to dupes_dir and the source is untouched.
    """
    logger.info(f"Running duplicate detection ({mode}) on {len(files)} files...")
    to_remove = {}          # dup_path -> (keeper_path, kind)

    # --- exact: file bytes ---
    md5_groups = {}
    for f in tqdm(files, desc="Hashing (exact dupes)"):
        try:
            md5_groups.setdefault(file_md5(f), []).append(f)
        except Exception as e:
            logger.warning(f"Could not hash {f}: {e}")
    for group in md5_groups.values():
        if len(group) > 1:
            keeper = sorted(group)[0]
            for dup in group:
                if dup != keeper:
                    to_remove[dup] = (keeper, "exact_duplicate")

    # --- decode remaining files once: sample-identical + near-duplicate signatures ---
    remaining = [f for f in files if f not in to_remove]
    content_groups = {}
    signatures = {}
    for f in tqdm(remaining, desc="Comparing audio content"):
        try:
            audio, sr, _ = load_audio(f)
            mono = audio if audio.ndim == 1 else np.mean(audio, axis=1)
            content_groups.setdefault(audio_content_hash(mono, sr), []).append(f)
            if mode == "near":
                signatures[f] = spectral_signature(mono, sr)
        except Exception:
            continue
    for group in content_groups.values():
        if len(group) > 1:
            keeper = sorted(group)[0]
            for dup in group:
                if dup != keeper:
                    to_remove[dup] = (keeper, "exact_duplicate")

    if mode == "near":
        candidates = {f: s for f, s in signatures.items() if f not in to_remove}
        def _load_normalized(path):
            audio, sr, _ = load_audio(path)
            return normalized_16k(audio if audio.ndim == 1 else np.mean(audio, axis=1), sr)

        for group in find_near_duplicate_groups(candidates, _load_normalized):
            keeper = sorted(group)[0]
            for dup in group:
                if dup != keeper:
                    to_remove[dup] = (keeper, "near_duplicate")

    if to_remove:
        logger.info(f"Found {len(to_remove)} duplicate file(s).")
    else:
        logger.info("No duplicates found.")

    for dup_path, (keeper, kind) in to_remove.items():
        record = FileRecord(
            source_path=str(dup_path), filename=dup_path.name, purpose=purpose,
            status="duplicate", duplicate_of=str(keeper),
            issues_found=[f"{kind.upper()} of {keeper.name}"],
            notes=f"Same recording as kept file: {keeper}",
        )
        if not dry_run:
            dupes_dir.mkdir(parents=True, exist_ok=True)
            dest = dupes_dir / dup_path.name
            if dest.exists():
                dest = dupes_dir / f"{dup_path.stem}_{abs(hash(str(dup_path))) % 10000}{dup_path.suffix}"
            try:
                shutil.copy2(dup_path, dest)
                record.final_destination = str(dest)
            except Exception as e:
                record.notes += f" | copy failed: {e}"
        audit.add(record)

    return [f for f in files if f not in to_remove]


def process_file(path: Path, profile, profile_key: str, output_dir: Path,
                  error_dir: Path, logger, dry_run: bool,
                  transcript_map=None, transcribe_fn=None, expected_language=None,
                  wer_reject_threshold=0.5, metadata_map=None) -> FileRecord:
    record = FileRecord(source_path=str(path), filename=path.name, purpose=profile_key)

    if metadata_map and path.name in metadata_map:
        record.speaker_id = metadata_map[path.name].get("speaker_id")
        record.session_id = metadata_map[path.name].get("session_id")

    report = analyze_file(path, profile, check_speaker_count=profile.reject_multi_speaker)
    record.original_sample_rate = report.sample_rate
    record.original_channels = report.channels
    record.original_duration_sec = report.duration_sec
    record.snr_db = report.snr_db
    record.clipped_ratio = report.clipped_ratio
    record.silence_ratio = report.silence_ratio
    record.max_internal_silence_sec = report.max_internal_silence_sec
    record.dc_offset = report.dc_offset
    record.lufs = report.lufs
    record.spectral_rolloff_hz = report.spectral_rolloff_hz
    record.spectral_rolloff_fraction = report.spectral_rolloff_fraction
    record.rt60_ms = report.rt60_ms
    record.crest_factor_db = report.crest_factor_db
    record.vad_backend = report.vad_backend
    record.issues_found = list(report.issues)
    record.warnings = list(report.warnings)

    # --- optional transcript cross-check (does not gate pass/fail on its own
    #     unless WER is catastrophic; it's primarily informational/audit) ---
    if transcript_map is not None:
        transcript, wer, lang, lang_ok, t_issues = check_transcript(
            path, transcript_map, transcribe_fn, expected_language, wer_reject_threshold)
        record.transcript = transcript
        record.wer = wer
        record.detected_language = lang
        record.language_match = lang_ok
        record.issues_found.extend(t_issues)
        report.issues.extend(t_issues)

        if transcript:
            _, rate_issues = speaking_rate_sanity_check(transcript, report.duration_sec)
            record.issues_found.extend(rate_issues)
            report.issues.extend(rate_issues)

    if not report.ok_to_load:
        record.status = "failed"
        record.unfixable_reasons = [f"CORRUPT_OR_UNREADABLE: {report.load_error}"]
        if not dry_run:
            dest = error_dir / "corrupt_or_unreadable" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, dest)
                record.final_destination = str(dest)
            except Exception as e:
                record.notes = f"could not even copy to error folder: {e}"
        return record

    # Two DIFFERENT lists, not one -- this used to be a single list, which was a bug:
    # it caused CLIPPING (among others) to skip the fix attempt entirely even though
    # fix_declip() can genuinely repair moderate clipping. Now:
    #   SKIP_FIX_MARKERS      -- no audio processing could possibly help; don't waste
    #                            time attempting a fix, reject immediately.
    #   FATAL_IF_STILL_PRESENT -- WORTH attempting a fix for; only rejected if the
    #                            problem is still measurably present after fixing.
    SKIP_FIX_MARKERS = ("CORRUPT_OR_UNREADABLE", "MULTI_SPEAKER_SUSPECTED",
                         "TOO_SHORT", "TOO_LONG", "HIGH_WER", "LANGUAGE_MISMATCH",
                         "ALIGNMENT_SUSPECT", "HIGH_REVERB")
    # HIGH_REVERB stays skip-fix: there's no dereverberation step in audio_fixer.py
    # today, so attempting a "fix" would do nothing. (Real dereverberation is a
    # much heavier addition -- see README if you want that added.)

    FATAL_IF_STILL_PRESENT = ("CORRUPT_OR_UNREADABLE", "MULTI_SPEAKER_SUSPECTED",
                               "TOO_SHORT", "TOO_LONG", "HIGH_WER", "LANGUAGE_MISMATCH",
                               "HIGH_REVERB", "LOW_DYNAMIC_RANGE", "LIKELY_LOSSY_SOURCE",
                               "LONG_INTERNAL_SILENCE", "MOSTLY_SILENCE", "INSUFFICIENT_SPEECH",
                               "LOW_SNR", "CLIPPING", "ALIGNMENT_SUSPECT")
    hard_fail_markers = FATAL_IF_STILL_PRESENT  # kept name for the rest of this function

    if not report.issues:
        audio, sr, _ = load_audio(path)
        record.status = "passed"
        if not dry_run:
            dest = output_dir / (path.stem + ".wav")
            if dest.exists():
                dest = output_dir / f"{path.stem}_{abs(hash(str(path)))%10000}.wav"
            save_audio(audio, sr, dest, bit_depth=profile.target_bit_depth)
            record.final_destination = str(dest)
            record.final_duration_sec = report.duration_sec
        return record

    # if every current issue is a skip-fix-only kind (no processing could help),
    # skip the fix attempt entirely -- otherwise, always try fixing before rejecting
    only_hard_fails = all(any(m in i for m in SKIP_FIX_MARKERS) for i in report.issues)

    if only_hard_fails:
        record.status = "failed"
        record.unfixable_reasons = list(report.issues)
        if not dry_run:
            label = report.issues[0].split(":")[0] if report.issues else "UNKNOWN_ISSUE"
            dest = error_dir / label.lower() / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, dest)
                record.final_destination = str(dest)
            except Exception as e:
                record.notes = f"copy to error folder failed: {e}"
        return record

    audio, sr, _ = load_audio(path)
    fixed_audio, fixed_sr, fixes = apply_fix_pipeline(audio, sr, profile, report.issues)
    record.fixes_applied = fixes

    tmp_path = None
    if not dry_run:
        tmp_path = output_dir / f"_tmp_recheck_{path.stem}.wav"
        save_audio(fixed_audio, fixed_sr, tmp_path, bit_depth=profile.target_bit_depth)
        recheck = analyze_file(tmp_path, profile, check_speaker_count=False)
    else:
        recheck = None

    remaining_issues = recheck.issues if recheck else report.issues
    # carry forward transcript/hard-fail issues that fixing audio can't touch
    for i in report.issues:
        if any(m in i for m in ("HIGH_WER", "LANGUAGE_MISMATCH", "ALIGNMENT_SUSPECT",
                                 "MULTI_SPEAKER_SUSPECTED", "TOO_SHORT", "TOO_LONG")):
            if i not in remaining_issues:
                remaining_issues = list(remaining_issues) + [i]

    unresolved_hard_fails = [i for i in remaining_issues if any(m in i for m in hard_fail_markers)]
    fatal = bool(unresolved_hard_fails)

    if fatal:
        record.status = "failed"
        record.unfixable_reasons = remaining_issues
        if not dry_run:
            label = unresolved_hard_fails[0].split(":")[0]
            dest = error_dir / label.lower() / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, dest)
                record.final_destination = str(dest)
            except Exception as e:
                record.notes = f"copy to error folder failed: {e}"
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
    else:
        record.status = "fixed" if fixes else "passed"
        if not dry_run:
            dest = output_dir / (path.stem + ".wav")
            if dest.exists():
                dest = output_dir / f"{path.stem}_{abs(hash(str(path)))%10000}.wav"
            if tmp_path and tmp_path.exists():
                shutil.move(str(tmp_path), str(dest))
            else:
                save_audio(fixed_audio, fixed_sr, dest, bit_depth=profile.target_bit_depth)
            record.final_destination = str(dest)
            record.final_duration_sec = len(fixed_audio) / fixed_sr

    return record


def run_metadata_checks(records, files, logger, metadata_map, output_dir):
    """Speaker balance + cross-contamination check, written as its own report."""
    if not metadata_map:
        return

    logger.info("Running speaker/session metadata checks...")
    speaker_ids = {fn: meta.get("speaker_id") for fn, meta in metadata_map.items() if meta.get("speaker_id")}

    embeddings = {}
    for f in files:
        if f.name not in speaker_ids:
            continue
        try:
            audio, sr, _ = load_audio(f)
            mono = _to_mono(audio.astype(np.float32))
            emb = speaker_embedding(mono, sr)
            if emb is not None:
                embeddings[f.name] = emb
        except Exception:
            continue

    warnings = cross_contamination_check(embeddings, speaker_ids)
    if warnings:
        logger.warning(f"{len(warnings)} file(s) flagged for possible speaker mislabeling.")
        report_lines = ["# Speaker Cross-Contamination Flags", "",
                        "Files whose audio fingerprint looks closer to a different "
                        "speaker than their claimed label. Worth a manual listen.", ""]
        for fname, msg in warnings.items():
            report_lines.append(f"- `{fname}`: {msg}")
        (output_dir.parent / "_logs" / "speaker_contamination_flags.md").write_text(
            "\n".join(report_lines), encoding="utf-8")
    else:
        logger.info("No speaker cross-contamination flags.")


def write_split_manifests(records, metadata_map, output_dir, train_ratio, logger):
    passed_files = [r.filename for r in records if r.status in ("passed", "fixed")]
    session_ids = {fn: meta.get("session_id") for fn, meta in (metadata_map or {}).items()}
    train, val = session_aware_split(passed_files, session_ids, train_ratio=train_ratio)

    with open(output_dir / "train_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename"])
        for fn in train:
            w.writerow([fn])
    with open(output_dir / "val_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename"])
        for fn in val:
            w.writerow([fn])

    logger.info(f"Session-aware split written: {len(train)} train / {len(val)} val "
                f"-> train_manifest.csv, val_manifest.csv")


def build_transcribe_fn(asr_model: str, model_size: str, logger):
    """
    Returns a callable(wav_path) -> str, or None if no ASR backend was requested
    / available. Kept as a separate builder so pipeline.py has zero hard
    dependency on any ASR library unless you actually ask for one.
    """
    if asr_model == "none":
        return None

    if asr_model == "whisper":
        try:
            import whisper
        except ImportError:
            logger.warning("--asr-model whisper requested but `openai-whisper` isn't installed "
                            "(pip install openai-whisper). WER checking will be skipped; "
                            "transcripts are still recorded and the language check still runs.")
            return None
        try:
            logger.info(f"Loading Whisper model '{model_size}' (first run downloads weights, needs internet)...")
            model = whisper.load_model(model_size)
        except Exception as e:
            logger.warning(f"Could not load Whisper model '{model_size}': {e}. "
                            "WER checking will be skipped.")
            return None

        def _transcribe(wav_path: Path) -> str:
            result = model.transcribe(str(wav_path))
            return result.get("text", "")

        logger.info("Whisper ASR backend ready -- WER will be computed against provided transcripts.")
        return _transcribe

    return None


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="WaveWash — audio QC & data prep pipeline -- clean, fix, dedupe, and "
                    "route audio for ML training.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--error-dir", default=None)
    parser.add_argument("--dupes-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--purpose", choices=list(PROFILES.keys()), default=None)
    parser.add_argument("--include-extra-formats", action="store_true")
    parser.add_argument("--skip-dedup", action="store_true")
    parser.add_argument("--dedup-mode", choices=["off", "exact", "near"], default="near",
                        help="off = no duplicate check; exact = identical files/samples only; "
                             "near (default) = also the same recording re-exported/trimmed/resampled. "
                             "Different recordings that only share tempo or groove are never flagged.")
    parser.add_argument("--dup-threshold", type=int, default=None, help=argparse.SUPPRESS)  # legacy, ignored
    parser.add_argument("--dry-run", action="store_true")

    # transcript / WER / language
    parser.add_argument("--transcripts", default=None,
                        help="CSV or JSON mapping filename->transcript text.")
    parser.add_argument("--expected-language", default=None,
                        help="ISO 639-1 code (e.g. 'en'); flags transcripts that don't match.")
    parser.add_argument("--wer-threshold", type=float, default=0.5,
                        help="Reject if WER against ASR hypothesis exceeds this (default 0.5).")
    parser.add_argument("--asr-model", choices=["none", "whisper"], default="none",
                        help="ASR backend for WER checking. 'whisper' requires "
                             "`pip install openai-whisper` (pulls torch + model weights on "
                             "first use, needs internet). 'none' (default) skips WER computation "
                             "but still records provided transcripts and runs the language check.")
    parser.add_argument("--whisper-model-size", default="base",
                        help="Whisper model size if --asr-model whisper (tiny/base/small/medium/large). "
                             "Bigger = more accurate, slower, more RAM.")

    # speaker/session metadata
    parser.add_argument("--metadata", default=None,
                        help="CSV with columns filename,speaker_id,session_id.")
    parser.add_argument("--make-split", action="store_true",
                        help="Write session-aware train/val manifests (requires --metadata for session_id).")
    parser.add_argument("--train-ratio", type=float, default=0.9)

    # VAD / review
    parser.add_argument("--use-real-vad", action="store_true",
                        help="Try Silero VAD (needs torch+torchaudio installed); falls back to energy VAD.")
    parser.add_argument("--review-sample-size", type=int, default=DEFAULT_REVIEW_SAMPLE_SIZE,
                        help="Number of passed + failed files to pull into a manual review folder (0 disables).")
    parser.add_argument("--no-diversity-report", action="store_true")

    args = parser.parse_args(argv)

    output_dir = Path(args.output).expanduser().resolve()
    error_dir = Path(args.error_dir).expanduser().resolve() if args.error_dir else output_dir.parent / "_error"
    dupes_dir = Path(args.dupes_dir).expanduser().resolve() if args.dupes_dir else output_dir.parent / "_duplicates"
    log_dir = Path(args.log_dir).expanduser().resolve() if args.log_dir else output_dir.parent / "_logs"

    logger = build_text_logger(log_dir)
    audit = AuditLog(log_dir)

    purpose_key = args.purpose or prompt_for_purpose()
    profile = PROFILES[purpose_key]
    if args.use_real_vad:
        profile.use_real_vad = True

    logger.info("=== WaveWash pipeline starting ===")
    logger.info(f"Purpose profile: {profile.name}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output (clean data): {output_dir}")
    logger.info(f"Error folder: {error_dir}")
    logger.info(f"Duplicates folder: {dupes_dir}")
    logger.info(f"Log folder: {log_dir}")
    if args.dry_run:
        logger.info("DRY RUN -- no files will be written/copied/moved.")

    transcript_map = None
    if args.transcripts:
        transcript_map = load_transcript_map(args.transcripts)
        logger.info(f"Loaded {len(transcript_map)} transcript(s) from {args.transcripts}")

    transcribe_fn = build_transcribe_fn(args.asr_model, args.whisper_model_size, logger)

    metadata_map = None
    if args.metadata:
        metadata_map = load_metadata(args.metadata)
        logger.info(f"Loaded metadata for {len(metadata_map)} file(s) from {args.metadata}")

    files = find_audio_files(args.input, include_extra_formats=args.include_extra_formats)
    logger.info(describe_input(files, args.input))

    if not files:
        logger.warning("No audio files found at the given input path. Exiting.")
        sys.exit(1)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    dedup_mode = "off" if args.skip_dedup else args.dedup_mode
    if profile.run_duplicate_detection and dedup_mode != "off":
        files = run_duplicate_pass(files, logger, audit, purpose_key,
                                    dedup_mode, dupes_dir, args.dry_run)

    logger.info(f"Processing {len(files)} file(s) through quality checks + fixes...")
    for f in tqdm(files, desc="QC + fix"):
        try:
            record = process_file(
                f, profile, purpose_key, output_dir, error_dir, logger, args.dry_run,
                transcript_map=transcript_map, transcribe_fn=transcribe_fn,
                expected_language=args.expected_language,
                wer_reject_threshold=args.wer_threshold, metadata_map=metadata_map)
            audit.add(record)
            if record.status == "failed":
                logger.warning(f"FAILED  {f.name}: {', '.join(record.unfixable_reasons)}")
            elif record.status == "fixed":
                logger.info(f"FIXED   {f.name}: {', '.join(record.fixes_applied)}")
            else:
                logger.info(f"PASSED  {f.name}")
        except Exception as e:
            logger.error(f"Unhandled error processing {f}: {e}")
            audit.add(FileRecord(
                source_path=str(f), filename=f.name, purpose=purpose_key,
                status="failed", unfixable_reasons=[f"PIPELINE_ERROR: {e}"]
            ))

    if not args.dry_run:
        for tmp in output_dir.glob("_tmp_recheck_*.wav"):
            tmp.unlink(missing_ok=True)

    if metadata_map and not args.dry_run:
        run_metadata_checks(audit.records, files, logger, metadata_map, output_dir)

    if args.make_split and not args.dry_run:
        write_split_manifests(audit.records, metadata_map, output_dir, args.train_ratio, logger)

    if not args.no_diversity_report and not args.dry_run:
        div_path = build_diversity_report(audit.records, profile.name, log_dir / "diversity_report.md")
        logger.info(f"Diversity report: {div_path}")

    if args.review_sample_size > 0 and not args.dry_run:
        review_dir = output_dir.parent / "_review_sample"
        build_review_sample(audit.records, review_dir, sample_size=args.review_sample_size)
        logger.info(f"Manual review sample written to: {review_dir}")

    summary_path = audit.write_summary(profile.name, {
        "input_path": args.input,
        "dry_run": args.dry_run,
        "vad_backend": profile.use_real_vad and "silero (requested)" or "energy",
    })

    logger.info(f"=== Done. Summary written to {summary_path} ===")
    logger.info(f"CSV log: {audit.csv_path}")
    logger.info(f"JSON log: {audit.json_path}")


if __name__ == "__main__":
    main()
