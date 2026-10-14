"""
diversity_report.py
---------------------
Generates a dataset-level markdown report: duration distribution, original
sample-rate/channel mix, pass/fail/fix/duplicate breakdown, and (if metadata
was supplied) speaker balance. This is the "zoom out and look at the whole
dataset" complement to the per-file audit log.

No hard dependency on matplotlib -- charts are simple text-based bar rows so
the report is always readable, even with a minimal install.
"""

from pathlib import Path
from typing import List
from collections import Counter
import numpy as np

from .metadata_checker import speaker_balance_report


def _text_histogram(values: List[float], bins: int = 8, width: int = 30) -> str:
    if not values:
        return "(no data)"
    values = np.array(values)
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int((c / max_count) * width)
        lines.append(f"  {edges[i]:6.1f}-{edges[i+1]:6.1f}s | {bar} {c}")
    return "\n".join(lines)


def build_diversity_report(records: list, purpose_name: str, output_path: Path) -> Path:
    """
    records: list of logger_setup.FileRecord (or objects with the same attrs)
    """
    durations = [r.final_duration_sec or r.original_duration_sec
                 for r in records if (r.final_duration_sec or r.original_duration_sec)]
    orig_srs = Counter(r.original_sample_rate for r in records if r.original_sample_rate)
    orig_channels = Counter(r.original_channels for r in records if r.original_channels)
    statuses = Counter(r.status for r in records)

    lines = [
        f"# Dataset Diversity Report",
        f"",
        f"Purpose: **{purpose_name}**  |  Total files scanned: **{len(records)}**",
        f"",
        f"## Status breakdown",
        f"",
    ]
    for status, count in statuses.most_common():
        lines.append(f"- {status}: {count}")

    if durations:
        total_hours = sum(durations) / 3600
        lines += [
            "",
            "## Duration",
            "",
            f"- Total usable audio: {total_hours:.2f} hours ({sum(durations):.0f}s across {len(durations)} files)",
            f"- Mean duration: {np.mean(durations):.2f}s | Median: {np.median(durations):.2f}s "
            f"| Min: {np.min(durations):.2f}s | Max: {np.max(durations):.2f}s",
            "",
            "```",
            _text_histogram(durations),
            "```",
        ]

    if orig_srs:
        lines += ["", "## Original sample rate distribution", ""]
        for sr, count in sorted(orig_srs.items()):
            lines.append(f"- {sr} Hz: {count} file(s)")

    if orig_channels:
        lines += ["", "## Original channel count distribution", ""]
        for ch, count in sorted(orig_channels.items()):
            label = "mono" if ch == 1 else ("stereo" if ch == 2 else f"{ch}-channel")
            lines.append(f"- {label}: {count} file(s)")

    # speaker balance, if any record has speaker_id set
    speaker_records = [
        {"speaker_id": getattr(r, "speaker_id", None),
         "duration_sec": r.final_duration_sec or r.original_duration_sec or 0.0}
        for r in records if getattr(r, "speaker_id", None)
    ]
    if speaker_records:
        balance_md, _ = speaker_balance_report(speaker_records)
        if balance_md:
            lines += ["", balance_md]

    # failure reason breakdown (repeated here for convenience alongside diversity stats)
    fail_reasons = Counter()
    for r in records:
        if r.status == "failed":
            for reason in (r.unfixable_reasons or []):
                fail_reasons[reason.split(":")[0]] += 1
    if fail_reasons:
        lines += ["", "## Failure reasons", ""]
        for reason, count in fail_reasons.most_common():
            lines.append(f"- {reason}: {count}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
